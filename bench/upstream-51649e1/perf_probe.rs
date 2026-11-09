//! July-style process-run driver with cost sensors.
//! Usage: fff-profile-driver.exe <corpus> <query> <iterations> <threads> [jsonl_out]
//! FFF_DRV_COLD=1 evicts the query cache with the bench's 9 dummy queries
//! before each timed call, so every timed search runs on the full corpus.
//! FFF_DRV_GIT=1 loads git status (default is skip_git). FFF_DRV_LIMIT=N
//! sets the page limit (0 = all items, default 100). hash_ordered is an
//! order-sensitive FNV-1a over every returned item's relative path.

use fff_search::file_picker::{FFFMode, FilePicker};
use fff_search::{
    FFFQuery, FileSearchConfig, FuzzySearchOptions, PaginationArgs, QueryParser, SharedFilePicker,
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

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn dump_results(
    shared_picker: &SharedFilePicker,
    parsed: &FFFQuery<'_>,
    root: &str,
    query: &str,
    threads: usize,
    limit: usize,
    path: &str,
) {
    use std::io::Write;

    let guard = shared_picker.read().expect("picker lock");
    let picker = guard.as_ref().expect("picker");
    let result = picker.fuzzy_search(
        parsed,
        None,
        FuzzySearchOptions {
            max_threads: threads,
            current_file: None,
            project_path: Some(Path::new(root)),
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            pagination: PaginationArgs { offset: 0, limit },
        },
    );
    let label = std::env::var("FFF_DRV_DUMP_QUERY").unwrap_or_else(|_| query.to_string());

    let mut line = String::with_capacity(256 + result.items.len() * 256);
    line.push_str(&format!(
        "{{\"project\":\"{}\",\"query\":\"{}\",\"results\":[",
        json_escape(root),
        json_escape(&label)
    ));
    for (rank, (item, score)) in result.items.iter().zip(result.scores.iter()).enumerate() {
        if rank > 0 {
            line.push(',');
        }
        line.push_str(&format!(
            concat!(
                "{{\"rank\":{},\"path\":\"{}\",\"total\":{},\"components\":{{",
                "\"base_score\":{},\"filename_bonus\":{},\"special_filename_bonus\":{},",
                "\"frecency_boost\":{},\"git_status_boost\":{},\"git_recency_boost\":{},",
                "\"distance_penalty\":{},\"current_file_penalty\":{},\"combo_match_boost\":{},",
                "\"path_alignment_bonus\":{},\"evidence_bonus\":0,\"exact_match\":{},",
                "\"match_type\":\"{}\"}}}}",
            ),
            rank,
            json_escape(&item.relative_path(picker)),
            score.total,
            score.base_score,
            score.filename_bonus,
            score.special_filename_bonus,
            score.frecency_boost,
            score.git_status_boost,
            score.git_recency_boost,
            score.distance_penalty,
            score.current_file_penalty,
            score.combo_match_boost,
            score.path_alignment_bonus,
            score.exact_match,
            score.match_type,
        ));
    }
    line.push_str("]}\n");

    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(line.as_bytes());
    }
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).expect("corpus path");
    let query = args.get(2).map(String::as_str).unwrap_or("score");
    let iterations: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let threads: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
    let jsonl = args.get(5).cloned();
    let root_path = Path::new(root);
    let limit: usize = std::env::var("FFF_DRV_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let shared_picker = SharedFilePicker::default();
    FilePicker::new_with_shared_state(
        shared_picker.clone(),
        SharedFrecency::default(),
        fff_search::FilePickerOptions {
            base_path: root.clone(),
            enable_mmap_cache: false,
            enable_content_indexing: false,
            mode: FFFMode::Neovim,
            watch: false,
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
    let mut hash_ordered = 0u64;
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
                pagination: PaginationArgs { offset: 0, limit },
            },
        );
        last_total = result.total_matched;
        last_len = result.items.len();
        let mut h = 0xcbf29ce484222325u64;
        for item in &result.items {
            h = fnv1a(h, &[0xFF]);
            h = fnv1a(h, item.relative_path(picker).as_bytes());
        }
        hash_ordered = h;
        result.total_matched ^ result.items.len()
    };

    let cold = std::env::var_os("FFF_DRV_COLD").is_some();
    let evict_queries: [&str; 9] = [
        "__fff_bench_cache_evict_0",
        "__fff_bench_cache_evict_1",
        "__fff_bench_cache_evict_2",
        "__fff_bench_cache_evict_3",
        "__fff_bench_cache_evict_4",
        "__fff_bench_cache_evict_5",
        "__fff_bench_cache_evict_6",
        "__fff_bench_cache_evict_7",
        "__fff_bench_cache_evict_8",
    ];
    let evict_parsed: Vec<_> = evict_queries.iter().map(|q| parser.parse(q)).collect();
    let evict = |p: &_| {
        let guard = shared_picker.read().expect("picker lock");
        let picker = guard.as_ref().expect("picker");
        let _ = picker.fuzzy_search(
            p,
            None,
            FuzzySearchOptions {
                max_threads: threads,
                current_file: None,
                project_path: Some(root_path),
                combo_boost_score_multiplier: 100,
                min_combo_count: 3,
                pagination: PaginationArgs { offset: 0, limit },
            },
        );
    };

    let warmup: usize = std::env::var("FFF_DRV_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let trace = std::env::var_os("FFF_DRV_TRACE").is_some();
    let mut checksum = 0usize;
    for i in 0..warmup {
        let s0 = sensors::snap();
        let t = Instant::now();
        checksum ^= search();
        let us = t.elapsed().as_secs_f64() * 1_000_000.0;
        let s1 = sensors::snap();
        if trace {
            eprintln!(
                "WARM i={i} us={us:.1} cyc={}",
                s1.cycles.saturating_sub(s0.cycles)
            );
        }
    }

    let mut per_op: Vec<f64> = Vec::with_capacity(iterations);
    let mut acc_us = 0.0f64;
    let mut acc_cpu_100ns = 0u64;
    let mut acc_cycles = 0u64;
    let mut acc_read = 0u64;
    let mut acc_faults = 0u64;
    let started = Instant::now();
    for i in 0..iterations {
        if cold {
            for p in &evict_parsed {
                evict(p);
            }
        }
        let s0 = sensors::snap();
        let t = Instant::now();
        checksum = checksum.wrapping_add(search());
        let us = t.elapsed().as_secs_f64() * 1_000_000.0;
        let s1 = sensors::snap();
        if trace {
            eprintln!(
                "TIMED i={i} cold={} us={us:.1} cyc={}",
                cold as u8,
                s1.cycles.saturating_sub(s0.cycles)
            );
        }
        per_op.push(us);
        acc_us += us;
        acc_cpu_100ns += s1.cpu_100ns.saturating_sub(s0.cpu_100ns);
        acc_cycles += s1.cycles.saturating_sub(s0.cycles);
        acc_read += s1.read_bytes.saturating_sub(s0.read_bytes);
        acc_faults += s1.faults.saturating_sub(s0.faults) as u64;
    }
    let elapsed = started.elapsed();

    if let Some(path) = std::env::var_os("FFF_DRV_DUMP") {
        let dump_limit: usize = std::env::var("FFF_DRV_DUMP_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        dump_results(
            &shared_picker,
            &parsed,
            root,
            query,
            threads,
            dump_limit,
            &path.to_string_lossy(),
        );
    }

    let n = iterations as f64;
    let us_per_op = acc_us / n;
    per_op.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |q: f64| per_op[((per_op.len() - 1) as f64 * q).round() as usize];
    let (p50, p90, p99, max_us) = if per_op.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (pct(0.50), pct(0.90), pct(0.99), per_op[per_op.len() - 1])
    };
    let cpu_us_per_op = acc_cpu_100ns as f64 / 10.0 / n;
    let cycles_per_op = acc_cycles as f64 / n;
    let read_b_op = acc_read as f64 / n;
    let faults_per_100k = acc_faults as f64 / n * 100_000.0;
    let s_end = sensors::snap();

    println!(
        "query={query} threads={threads} iterations={iterations} cold={} us_per_op={us_per_op:.3} p50_us={p50:.3} p90_us={p90:.3} p99_us={p99:.3} max_us={max_us:.3} cpu_us_per_op={cpu_us_per_op:.3} cycles_per_op={cycles_per_op:.0} read_b_op={read_b_op:.2} faults_per_100k={faults_per_100k:.1} peak_rss_mb={:.2} checksum={checksum} total_matched={last_total} items_len={last_len} hash_ordered={hash_ordered:016x}",
        cold as u8,
        s_end.rss_mb
    );

    if let Some(path) = jsonl {
        let line = format!(
            "{{\"query\":\"{query}\",\"threads\":{threads},\"iterations\":{iterations},\"cold\":{},\"us_per_op\":{us_per_op:.3},\"p50_us\":{p50:.3},\"p90_us\":{p90:.3},\"p99_us\":{p99:.3},\"max_us\":{max_us:.3},\"cpu_us_per_op\":{cpu_us_per_op:.3},\"cycles_per_op\":{cycles_per_op:.0},\"read_bytes\":{acc_read},\"faults\":{acc_faults},\"peak_rss_mb\":{:.2},\"checksum\":{checksum},\"hash_ordered\":\"{hash_ordered:016x}\",\"items_len\":{last_len}}}",
            cold as u8,
            s_end.rss_mb
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
