//! fff-profile-driver <corpus> <query> <iterations> <threads> [jsonl_out]
//! Times the argv query only; FFF_DRV_COLD evicts outside that window. hash_ordered is path-order FNV-1a.

use fff::file_picker::{FFFMode, FilePicker};
use fff::{
    FFFQuery, FileSearchConfig, FuzzySearchOptions, PaginationArgs, QueryParser, SharedFilePicker,
    SharedFrecency,
};
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(windows)]
mod sensors {
    use std::mem::size_of;

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
        fn QueryProcessCycleTime(h: isize, cycles: *mut u64) -> i32;
        fn GetProcessIoCounters(h: isize, io: *mut IoCounters) -> i32;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(h: isize, c: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }

    pub struct Usage {
        pub read_bytes: u64,
        pub faults: u32,
        pub rss_mb: f64,
    }

    pub fn cycles() -> u64 {
        unsafe {
            let mut cycles = 0u64;
            QueryProcessCycleTime(GetCurrentProcess(), &mut cycles);
            cycles
        }
    }

    pub fn usage() -> Usage {
        unsafe {
            let h = GetCurrentProcess();
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
            Usage {
                read_bytes: io.read_bytes,
                faults: pm.page_faults,
                rss_mb: pm.working_set as f64 / (1024.0 * 1024.0),
            }
        }
    }
}

#[cfg(not(windows))]
mod sensors {
    pub struct Usage {
        pub read_bytes: u64,
        pub faults: u32,
        pub rss_mb: f64,
    }
    pub fn cycles() -> u64 {
        0
    }
    pub fn usage() -> Usage {
        Usage {
            read_bytes: 0,
            faults: 0,
            rss_mb: 0.0,
        }
    }
}

fn encode_timed(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    for &b in query.as_bytes() {
        if b.is_ascii_graphic() && b != b'%' {
            out.push(b as char);
        } else {
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0xf) as usize] as char);
        }
    }
    out
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
                "\"path_alignment_bonus\":{},\"evidence_bonus\":{},\"exact_match\":{},",
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
            score.evidence_bonus,
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
        fff::FilePickerOptions {
            base_path: root.clone(),
            enable_mmap_cache: false,
            enable_content_indexing: false,
            mode: FFFMode::Neovim,
            watch: false,
            skip_git: std::env::var_os("FFF_DRV_GIT").is_none(),
            ..Default::default()
        },
    )
    .expect("open corpus");
    assert!(
        shared_picker.wait_for_scan(Duration::from_secs(600)),
        "scan timed out"
    );
    let file_count = {
        let guard = shared_picker.read().expect("picker lock");
        guard.as_ref().expect("picker").get_files().len()
    };

    let parser = QueryParser::<FileSearchConfig>::default();
    let parsed = parser.parse(query);
    let mut last_total = 0usize;
    let mut last_len = 0usize;
    let mut hash_ordered = 0u64;
    let mut path_buf = String::new();
    let mut measure = || {
        let guard = shared_picker.read().expect("picker lock");
        let picker = guard.as_ref().expect("picker");
        let u0 = sensors::usage();
        let c0 = sensors::cycles();
        let t = Instant::now();
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
        let us = t.elapsed().as_secs_f64() * 1_000_000.0;
        let cyc = sensors::cycles().saturating_sub(c0);
        let u1 = sensors::usage();
        let mut h = 0xcbf29ce484222325u64;
        for item in &result.items {
            item.write_relative_path(picker, &mut path_buf);
            h = fnv1a(h, &[0xFF]);
            h = fnv1a(h, path_buf.as_bytes());
        }
        hash_ordered = h;
        (
            us,
            cyc,
            u1.read_bytes.saturating_sub(u0.read_bytes),
            u1.faults.saturating_sub(u0.faults),
            u1.rss_mb,
            result.total_matched ^ result.items.len(),
        )
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
        let (us, cyc, _, _, _, ident) = measure();
        checksum ^= ident;
        if trace {
            eprintln!("WARM i={i} us={us:.1} cyc={cyc}");
        }
    }

    let mut per_op: Vec<f64> = Vec::with_capacity(iterations);
    let mut acc_us = 0.0f64;
    let mut acc_cycles = 0u64;
    let mut acc_read = 0u64;
    let mut acc_faults = 0u64;
    let mut query_rss_mb = 0.0f64;
    for i in 0..iterations {
        if cold {
            for p in &evict_parsed {
                evict(p);
            }
        }
        let (us, cyc, read, faults, rss, ident) = measure();
        checksum = checksum.wrapping_add(ident);
        if trace {
            eprintln!(
                "TIMED i={i} cold={} us={us:.1} cyc={cyc}",
                cold as u8
            );
        }
        per_op.push(us);
        acc_us += us;
        acc_cycles += cyc;
        acc_read += read;
        acc_faults += faults as u64;
        query_rss_mb = query_rss_mb.max(rss);
    }

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
    let cycles_per_op = acc_cycles as f64 / n;
    let read_b_op = acc_read as f64 / n;
    let faults_per_100k = acc_faults as f64 / n * 100_000.0;

    let timed_query = encode_timed(query);
    let query_json = json_escape(query);
    println!(
        "query={query} timed={timed_query} threads={threads} iterations={iterations} cold={} us_per_op={us_per_op:.3} p50_us={p50:.3} p90_us={p90:.3} p99_us={p99:.3} max_us={max_us:.3} cycles_per_op={cycles_per_op:.0} read_b_op={read_b_op:.2} faults_per_100k={faults_per_100k:.1} peak_rss_mb={query_rss_mb:.2} checksum={checksum} total_matched={last_total} items_len={last_len} hash_ordered={hash_ordered:016x} files={file_count}",
        cold as u8
    );

    if let Some(path) = jsonl {
        let line = format!(
            "{{\"query\":\"{query_json}\",\"timed_query\":\"{query_json}\",\"threads\":{threads},\"iterations\":{iterations},\"cold\":{},\"us_per_op\":{us_per_op:.3},\"p50_us\":{p50:.3},\"p90_us\":{p90:.3},\"p99_us\":{p99:.3},\"max_us\":{max_us:.3},\"cycles_per_op\":{cycles_per_op:.0},\"read_bytes\":{acc_read},\"faults\":{acc_faults},\"peak_rss_mb\":{query_rss_mb:.2},\"checksum\":{checksum},\"hash_ordered\":\"{hash_ordered:016x}\",\"items_len\":{last_len}}}",
            cold as u8
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
