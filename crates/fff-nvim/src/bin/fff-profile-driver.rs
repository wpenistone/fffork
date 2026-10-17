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
        fn GetProcessTimes(
            h: isize,
            c: *mut FileTime,
            e: *mut FileTime,
            k: *mut FileTime,
            u: *mut FileTime,
        ) -> i32;
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
    pub fn snap() -> Snap {
        Snap {
            cpu_100ns: 0,
            cycles: 0,
            read_bytes: 0,
            faults: 0,
            rss_mb: 0.0,
        }
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
    assert!(
        shared_picker.wait_for_scan(Duration::from_secs(600)),
        "scan timed out"
    );

    let parser = QueryParser::<FileSearchConfig>::default();
    let parsed = parser.parse(query);
    let mut last_total = 0usize;
    let mut last_len = 0usize;
    let mut search = || {
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
        last_total = result.total_matched;
        last_len = result.items.len();
        result.total_matched ^ result.items.len()
    };

    let mut checksum = 0usize;
    for _ in 0..100 {
        checksum ^= search();
    }

    let s0 = sensors::snap();
    let started = Instant::now();
    let mut per_op: Vec<f64> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t = Instant::now();
        checksum = checksum.wrapping_add(search());
        per_op.push(t.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let elapsed = started.elapsed();
    let s1 = sensors::snap();

    let n = iterations as f64;
    let us_per_op = elapsed.as_secs_f64() * 1_000_000.0 / n;
    per_op.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |q: f64| per_op[((per_op.len() - 1) as f64 * q).round() as usize];
    let (p50, p90, p99, max_us) = if per_op.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (pct(0.50), pct(0.90), pct(0.99), per_op[per_op.len() - 1])
    };
    let cpu_us_per_op = (s1.cpu_100ns.saturating_sub(s0.cpu_100ns) as f64) / 10.0 / n;
    let cycles_per_op = (s1.cycles.saturating_sub(s0.cycles) as f64) / n;
    let read_b_op = (s1.read_bytes.saturating_sub(s0.read_bytes) as f64) / n;
    let faults_per_100k = (s1.faults.saturating_sub(s0.faults) as f64) / n * 100_000.0;

    println!(
        "query={query} threads={threads} iterations={iterations} us_per_op={us_per_op:.3} p50_us={p50:.3} p90_us={p90:.3} p99_us={p99:.3} max_us={max_us:.3} cpu_us_per_op={cpu_us_per_op:.3} cycles_per_op={cycles_per_op:.0} read_b_op={read_b_op:.2} faults_per_100k={faults_per_100k:.1} peak_rss_mb={:.2} checksum={checksum} total_matched={last_total} items_len={last_len}",
        s1.rss_mb
    );

    if let Some(path) = jsonl {
        let line = format!(
            "{{\"query\":\"{query}\",\"threads\":{threads},\"iterations\":{iterations},\"us_per_op\":{us_per_op:.3},\"p50_us\":{p50:.3},\"p90_us\":{p90:.3},\"p99_us\":{p99:.3},\"max_us\":{max_us:.3},\"cpu_us_per_op\":{cpu_us_per_op:.3},\"cycles_per_op\":{cycles_per_op:.0},\"read_bytes\":{},\"faults\":{},\"peak_rss_mb\":{:.2},\"checksum\":{checksum}}}",
            s1.read_bytes.saturating_sub(s0.read_bytes),
            s1.faults.saturating_sub(s0.faults),
            s1.rss_mb
        );
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{line}");
        }
    }
}
