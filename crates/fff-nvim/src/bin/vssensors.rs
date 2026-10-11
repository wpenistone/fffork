//! Full-sensor vs harness: scan once, run many queries.
//! Usage: vssensors <corpus> <threads> <iters> <q1> [q2 ...]
//! Per query: wall/cpu/cycles/io/faults/rss + checksum + matched.

use fff::file_picker::FilePicker;
use fff::{FuzzySearchOptions, PaginationArgs, QueryParser};
use std::time::Instant;

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
    let threads: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(4);
    let iterations: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(30);
    let queries: Vec<String> = args.iter().skip(4).cloned().collect();
    assert!(!queries.is_empty(), "pass at least one query");

    let mut picker = FilePicker::new(fff::FilePickerOptions {
        base_path: root.clone(),
        enable_mmap_cache: false,
        enable_content_indexing: false,
        mode: fff::FFFMode::Neovim,
        watch: false,
        ..Default::default()
    })
    .expect("open corpus");
    picker.collect_files().expect("collect files");

    let parser = QueryParser::default();
    for query in &queries {
        let parsed = parser.parse(query);
        let search = || {
            let r = picker.fuzzy_search(
                &parsed,
                None,
                FuzzySearchOptions {
                    max_threads: threads,
                    current_file: None,
                    project_path: None,
                    combo_boost_score_multiplier: 100,
                    min_combo_count: 3,
                    pagination: PaginationArgs {
                        offset: 0,
                        limit: 100,
                    },
                },
            );
            let mut acc: u64 = r.total_matched as u64 ^ ((r.items.len() as u64) << 32);
            for (i, s) in r.scores.iter().enumerate() {
                acc = acc
                    .wrapping_mul(31)
                    .wrapping_add((s.total as u64).wrapping_add(i as u64));
            }
            (acc, r.total_matched)
        };
        let mut checksum = 0u64;
        let mut matched = 0usize;
        for _ in 0..5 {
            let (c, m) = search();
            checksum ^= c;
            matched = m;
        }
        let s0 = sensors::snap();
        let started = Instant::now();
        for _ in 0..iterations {
            let (c, _) = search();
            checksum = checksum.wrapping_add(c);
        }
        let elapsed = started.elapsed();
        let s1 = sensors::snap();
        let n = iterations as f64;
        println!(
            "vssensors query={query} threads={threads} iters={iterations} us_op={:.1} cpu_us_op={:.1} cycles_op={:.0} readb_op={:.0} faults={} rss_mb={:.1} checksum={checksum} matched={matched}",
            elapsed.as_secs_f64() * 1_000_000.0 / n,
            (s1.cpu_100ns.saturating_sub(s0.cpu_100ns) as f64) / 10.0 / n,
            (s1.cycles.saturating_sub(s0.cycles) as f64) / n,
            (s1.read_bytes.saturating_sub(s0.read_bytes) as f64) / n,
            s1.faults.saturating_sub(s0.faults),
            s1.rss_mb,
        );
    }
}
