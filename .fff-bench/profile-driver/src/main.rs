//! July-style process-run driver with cost sensors.
//! Usage: fff-profile-driver.exe <corpus> <query> <iterations> <threads> [jsonl_out]

use fff::file_picker::{FFFMode, FilePicker};
use fff::{
    FileSearchConfig, FuzzySearchOptions, PaginationArgs, QueryParser, SharedFilePicker,
    SharedFrecency,
};
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(windows)]
mod sensors {
    use std::mem::size_of;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    impl FileTime {
        fn as_u64(&self) -> u64 {
            ((self.high as u64) << 32) | self.low as u64
        }
    }

    #[repr(C)]
    struct IoCounters {
        read_op: u64,
        write_op: u64,
        other_op: u64,
        read_bytes: u64,
        write_bytes: u64,
        other_bytes: u64,
    }

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_faults: u32,
        peak_working_set: usize,
        working_set: usize,
        quota_peak_paged: usize,
        quota_paged: usize,
        quota_peak_nonpaged: usize,
        quota_nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }

    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(h: isize, c: *mut FileTime, e: *mut FileTime, k: *mut FileTime, u: *mut FileTime) -> i32;
        fn QueryProcessCycleTime(h: isize, cycles: *mut u64) -> i32;
        fn GetProcessIoCounters(h: isize, io: *mut IoCounters) -> i32;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(h: isize, c: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }

    pub struct Snap {
        pub cpu_100ns: u64,
        pub cycles: u64,
        pub read_bytes: u64,
        pub faults: u32,
        pub rss_mb: f64,
    }

    /// Current working set (not peak) for RSS attribution probes.
    pub fn ws_mb() -> f64 {
        unsafe {
            let h = GetCurrentProcess();
            let mut pm = ProcessMemoryCounters {
                cb: size_of::<ProcessMemoryCounters>() as u32,
                page_faults: 0,
                peak_working_set: 0,
                working_set: 0,
                quota_peak_paged: 0,
                quota_paged: 0,
                quota_peak_nonpaged: 0,
                quota_nonpaged: 0,
                pagefile: 0,
                peak_pagefile: 0,
            };
            GetProcessMemoryInfo(h, &mut pm, pm.cb);
            pm.working_set as f64 / (1024.0 * 1024.0)
        }
    }

    pub fn snap() -> Snap {
        unsafe {
            let h = GetCurrentProcess();
            let mut c = FileTime { low: 0, high: 0 };
            let mut e = FileTime { low: 0, high: 0 };
            let mut k = FileTime { low: 0, high: 0 };
            let mut u = FileTime { low: 0, high: 0 };
            GetProcessTimes(h, &mut c, &mut e, &mut k, &mut u);
            let mut cycles = 0u64;
            QueryProcessCycleTime(h, &mut cycles);
            let mut io = IoCounters {
                read_op: 0,
                write_op: 0,
                other_op: 0,
                read_bytes: 0,
                write_bytes: 0,
                other_bytes: 0,
            };
            GetProcessIoCounters(h, &mut io);
            let mut pm = ProcessMemoryCounters {
                cb: size_of::<ProcessMemoryCounters>() as u32,
                page_faults: 0,
                peak_working_set: 0,
                working_set: 0,
                quota_peak_paged: 0,
                quota_paged: 0,
                quota_peak_nonpaged: 0,
                quota_nonpaged: 0,
                pagefile: 0,
                peak_pagefile: 0,
            };
            GetProcessMemoryInfo(h, &mut pm, pm.cb);
            Snap {
                cpu_100ns: k.as_u64() + u.as_u64(),
                cycles,
                read_bytes: io.read_bytes,
                faults: pm.page_faults,
                rss_mb: pm.peak_working_set as f64 / (1024.0 * 1024.0),
            }
        }
    }
}

#[cfg(not(windows))]
mod sensors {
    pub struct Snap {
        pub cpu_100ns: u64,
        pub cycles: u64,
        pub read_bytes: u64,
        pub faults: u32,
        pub rss_mb: f64,
    }
    pub fn ws_mb() -> f64 { 0.0 }
    pub fn snap() -> Snap {
        Snap { cpu_100ns: 0, cycles: 0, read_bytes: 0, faults: 0, rss_mb: 0.0 }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).expect("corpus path");
    let query = args.get(2).map(String::as_str).unwrap_or("score");
    let iterations: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let threads: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
    let jsonl = args.get(5).cloned();
    let root_path = Path::new(root);

    let shared_picker = SharedFilePicker::default();
    FilePicker::new_with_shared_state(
        shared_picker.clone(),
        SharedFrecency::default(),
        fff::FilePickerOptions {
            base_path: root.clone(),
            enable_mmap_cache: false,
            enable_content_indexing: false,
            mode: FFFMode::Neovim,
            watch: false,
            skip_git: true,
            ..Default::default()
        },
    )
    .expect("open corpus");
    eprintln!("RSS_PROBE before_scan ws_mb={:.2}", sensors::ws_mb());
    let ws_peak = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    {
        let ws_peak = ws_peak.clone();
        std::thread::spawn(move || loop {
            let ws = (sensors::ws_mb() * 100.0) as u64;
            ws_peak.fetch_max(ws, std::sync::atomic::Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(2));
        });
    }
    assert!(
        shared_picker.wait_for_scan(Duration::from_secs(600)),
        "scan timed out"
    );
    std::thread::sleep(std::time::Duration::from_millis(50));
    eprintln!("RSS_PROBE after_scan ws_mb={:.2} polled_peak={:.2}",
        sensors::ws_mb(), ws_peak.load(std::sync::atomic::Ordering::Relaxed) as f64 / 100.0);

    let parser = QueryParser::<FileSearchConfig>::default();
    let parsed = parser.parse(query);
    let search = || {
        let guard = shared_picker.read().expect("picker lock");
        let picker = guard.as_ref().expect("picker");
        let result = picker.fuzzy_search(
            &parsed,
            None,
            FuzzySearchOptions {
                max_threads: threads,
                current_file: None,
                project_path: Some(root_path),
                combo_boost_score_multiplier: 100,
                min_combo_count: 3,
                pagination: PaginationArgs {
                    offset: 0,
                    limit: 100,
                },
            },
        );
        // Rank-aware checksum: total_matched, page size, ordered path lengths
        // and score totals. Catches ranking/page drift the old XOR of counts
        // missed.
        let mut acc = result.total_matched ^ result.items.len();
        for (i, item) in result.items.iter().enumerate() {
            acc = acc
                .wrapping_mul(0x9E3779B97F4A7C15u64 as usize)
                .wrapping_add(item.relative_path_len())
                .wrapping_add(i * 17);
        }
        for s in &result.scores {
            acc = acc.wrapping_add(s.total as usize);
        }
        acc
    };

    // Diagnostic: report the two operands of the checksum.
    {
        let guard = shared_picker.read().expect("picker lock");
        let picker = guard.as_ref().expect("picker");
        let result = picker.fuzzy_search(
            &parsed,
            None,
            FuzzySearchOptions {
                max_threads: threads,
                current_file: None,
                project_path: Some(root_path),
                combo_boost_score_multiplier: 100,
                min_combo_count: 3,
                pagination: PaginationArgs {
                    offset: 0,
                    limit: 100,
                },
            },
        );
        eprintln!(
            "DIAG total_matched={} items_len={} files={}",
            result.total_matched,
            result.items.len(),
            picker.get_files().len()
        );
        eprintln!("RSS_PROBE after_first_search ws_mb={:.2}", sensors::ws_mb());
        if let Ok(_out) = std::env::var("FFF_DUMP_PATHS") {
            use std::io::Write as _;
            let mut f = std::fs::File::create(&_out).expect("dump");
            let picker_ref = picker;
            for item in picker.get_files() {
                let _ = writeln!(f, "{}", item.relative_path(picker_ref));
            }
        }
    }

    // Keystroke wall with warmup: keys:<string> types the string one char at
    // a time (live-narrow engaged across prefixes). The full prefix sequence
    // is run `warmup` times before measurement, then `iters` timed rounds.
    // Reports per-keystroke us_per_op (mean over rounds) — the wall for that
    // keystroke, not a cumulative total.
    if let Some(full) = query.strip_prefix("keys:") {
        let warmup = std::env::var("FFF_KEYS_WARMUP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        let iters = std::env::var("FFF_KEYS_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        eprintln!("SCENARIO keys={full} warmup={warmup} iters={iters}");
        let prefixes: Vec<&str> = (1..=full.len()).map(|n| &full[..n]).collect();
        let run_one = |q: &str| -> (f64, usize, usize) {
            let p2 = parser.parse(q);
            let t0 = Instant::now();
            let (total_matched, items_len) = {
                let guard = shared_picker.read().expect("picker lock");
                let picker = guard.as_ref().expect("picker");
                let r = picker.fuzzy_search(
                    &p2,
                    None,
                    FuzzySearchOptions {
                        max_threads: threads,
                        current_file: None,
                        project_path: Some(root_path),
                        combo_boost_score_multiplier: 100,
                        min_combo_count: 3,
                        pagination: PaginationArgs { offset: 0, limit: 100 },
                    },
                );
                (r.total_matched, r.items.len())
            };
            (t0.elapsed().as_secs_f64() * 1e6, total_matched, items_len)
        };
        // Warmup rounds: let caches, TLS scratch, and live-narrow state settle.
        for _ in 0..warmup {
            for q in &prefixes {
                let _ = run_one(q);
            }
        }
        let mut acc = vec![0.0f64; prefixes.len()];
        let mut tms = vec![0usize; prefixes.len()];
        let mut last_items = 0usize;
        for _ in 0..iters {
            for (i, q) in prefixes.iter().enumerate() {
                let (us, tm, items) = run_one(q);
                acc[i] += us;
                tms[i] = tm;
                last_items = items;
            }
        }
        let n = iters as f64;
        for (i, q) in prefixes.iter().enumerate() {
            eprintln!(
                "KEY q={:?} us_per_op={:.1} total_matched={} items={}",
                q,
                acc[i] / n,
                tms[i],
                last_items
            );
        }
        // Full-query repeat (no live-narrow: same length, can_narrow requires growth).
        let (repeat_us, _, _) = run_one(full);
        eprintln!("KEY repeat_us={:.1}", repeat_us);
        return;
    }

    // Realistic typing scenario: type:<string> times one search per prefix
    // (s, sc, sco, ...) with no cross-prefix warmup — first keystroke after
    // open, then each next character.
    if let Some(full) = query.strip_prefix("type:") {
        eprintln!("SCENARIO typing={full}");
        let mut prev_us = 0.0;
        for n in 1..=full.len() {
            let q = &full[..n];
            let p2 = parser.parse(q);
            let t0 = Instant::now();
            let (total_matched, items_len) = {
                let guard = shared_picker.read().expect("picker lock");
                let picker = guard.as_ref().expect("picker");
                let r = picker.fuzzy_search(
                    &p2,
                    None,
                    FuzzySearchOptions {
                        max_threads: threads,
                        current_file: None,
                        project_path: Some(root_path),
                        combo_boost_score_multiplier: 100,
                        min_combo_count: 3,
                        pagination: PaginationArgs { offset: 0, limit: 100 },
                    },
                );
                (r.total_matched, r.items.len())
            };
            let us = t0.elapsed().as_secs_f64() * 1e6;
            let delta = us - prev_us;
            prev_us = us;
            eprintln!(
                "TYPE q={:?} us={:.1} delta_us={:.1} total_matched={} items={}",
                q, us, delta, total_matched, items_len
            );
        }
        // Second pass: repeat the full query (follow-up / re-query).
        let t0 = Instant::now();
        {
            let guard = shared_picker.read().expect("picker lock");
            let picker = guard.as_ref().expect("picker");
            let _ = picker.fuzzy_search(
                &parsed,
                None,
                FuzzySearchOptions {
                    max_threads: threads,
                    current_file: None,
                    project_path: Some(root_path),
                    combo_boost_score_multiplier: 100,
                    min_combo_count: 3,
                    pagination: PaginationArgs { offset: 0, limit: 100 },
                },
            );
        }
        let us = t0.elapsed().as_secs_f64() * 1e6;
        eprintln!("TYPE repeat_us={:.1}", us);
        return;
    }

    let mut checksum = 0usize;
    for _ in 0..100 {
        checksum ^= search();
    }
    eprintln!("RSS_PROBE after_warmup ws_mb={:.2}", sensors::ws_mb());

    let s0 = sensors::snap();
    let started = Instant::now();
    for _ in 0..iterations {
        checksum = checksum.wrapping_add(search());
    }
    let elapsed = started.elapsed();
    let s1 = sensors::snap();
    eprintln!("RSS_PROBE after_loop ws_mb={:.2} peak_mb={:.2}", sensors::ws_mb(), s1.rss_mb);

    let n = iterations as f64;
    let us_per_op = elapsed.as_secs_f64() * 1_000_000.0 / n;
    let cpu_us_per_op = (s1.cpu_100ns.saturating_sub(s0.cpu_100ns) as f64) / 10.0 / n;
    let cycles_per_op = (s1.cycles.saturating_sub(s0.cycles) as f64) / n;
    let read_b_op = (s1.read_bytes.saturating_sub(s0.read_bytes) as f64) / n;
    let faults_per_100k = (s1.faults.saturating_sub(s0.faults) as f64) / n * 100_000.0;

    println!(
        "query={query} threads={threads} iterations={iterations} us_per_op={us_per_op:.3} cpu_us_per_op={cpu_us_per_op:.3} cycles_per_op={cycles_per_op:.0} read_b_op={read_b_op:.2} faults_per_100k={faults_per_100k:.1} peak_rss_mb={:.2} checksum={checksum}",
        s1.rss_mb
    );

    if let Some(path) = jsonl {
        let line = format!(
            "{{\"query\":\"{query}\",\"threads\":{threads},\"iterations\":{iterations},\"us_per_op\":{us_per_op:.3},\"cpu_us_per_op\":{cpu_us_per_op:.3},\"cycles_per_op\":{cycles_per_op:.0},\"read_bytes\":{},\"faults\":{},\"peak_rss_mb\":{:.2},\"checksum\":{checksum}}}",
            s1.read_bytes.saturating_sub(s0.read_bytes),
            s1.faults.saturating_sub(s0.faults),
            s1.rss_mb
        );
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{line}");
        }
    }
}
