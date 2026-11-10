//! fff-typing-driver <corpus> <prefixes(csv)> <threads>
//! Each timed key is that argv prefix alone. Checksum is path-order FNV-1a; cycles are QueryProcessCycleTime.

use fff::file_picker::{FFFMode, FilePicker};
use fff::{
    FileSearchConfig, FuzzySearchOptions, PaginationArgs, QueryParser, SharedFilePicker,
    SharedFrecency,
};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).expect("corpus path").clone();
    let prefixes: Vec<String> = args
        .get(2)
        .map(|s| s.split(',').map(ToOwned::to_owned).collect())
        .unwrap_or_else(|| {
            vec![
                "s", "sc", "sco", "scor", "score", "n", "ne", "net", "netc", "netco", "netcon",
                "netconv", "netconve", "netconver", "netconvert",
            ]
            .into_iter()
            .map(ToOwned::to_owned)
            .collect()
        });
    let threads: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(4);
    let limit: usize = std::env::var("FFF_DRV_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let root_path = Path::new(&root);
    let shared_picker = SharedFilePicker::default();
    let opened = Instant::now();
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
    let scan_ms = opened.elapsed().as_secs_f64() * 1000.0;

    let parser = QueryParser::<FileSearchConfig>::default();
    let stream_start = Instant::now();
    let mut stream_us = Vec::with_capacity(prefixes.len());
    let mut stream_cycles = 0u64;
    let mut checksum = 0xcbf29ce484222325u64;
    let mut path_buf = String::new();
    {
        let guard = shared_picker.read().expect("picker lock");
        let picker = guard.as_ref().expect("picker");
        for prefix in &prefixes {
            let parsed = parser.parse(prefix);
            let c0 = process_cycles();
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
            let us = t.elapsed().as_secs_f64() * 1_000_000.0;
            let cycles = process_cycles().saturating_sub(c0);
            stream_us.push(us);
            stream_cycles = stream_cycles.saturating_add(cycles);
            checksum = fnv1a(checksum, &(result.total_matched as u64).to_le_bytes());
            checksum = fnv1a(checksum, &(result.items.len() as u64).to_le_bytes());
            let mut page = 0xcbf29ce484222325u64;
            for item in &result.items {
                item.write_relative_path(picker, &mut path_buf);
                page = fnv1a(page, &[0xFF]);
                page = fnv1a(page, path_buf.as_bytes());
                checksum = fnv1a(checksum, &[0xFF]);
                checksum = fnv1a(checksum, path_buf.as_bytes());
            }
            checksum = fnv1a(checksum, &[0x00]);
            let timed = encode_timed(prefix);
            println!(
                "KEY prefix={prefix} timed={timed} us={us:.1} cyc={cycles} total={} len={} hash={page:016x}",
                result.total_matched,
                result.items.len()
            );
        }
    }
    let total_ms = stream_start.elapsed().as_secs_f64() * 1000.0;
    let sum_us: f64 = stream_us.iter().sum();
    println!(
        "STREAM scan_ms={scan_ms:.1} keys={} total_ms={total_ms:.1} search_us={sum_us:.1} cycles={stream_cycles} checksum={checksum}",
        prefixes.len()
    );
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

#[cfg(windows)]
fn process_cycles() -> u64 {
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn QueryProcessCycleTime(h: isize, cycles: *mut u64) -> i32;
    }
    unsafe {
        let mut cycles = 0u64;
        QueryProcessCycleTime(GetCurrentProcess(), &mut cycles);
        cycles
    }
}

#[cfg(not(windows))]
fn process_cycles() -> u64 {
    0
}
