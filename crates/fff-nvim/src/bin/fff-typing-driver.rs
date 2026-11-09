//! Typing-stream bench for the target-scale matrix.
//! Usage: fff-typing-driver.exe <corpus> <prefixes(csv)> <threads>
//! One process = one picker session; prefixes are typed in order without
//! flushing the query cache, so this measures a realistic key stream.
//! FFF_DRV_LIMIT sets the page limit (default 100).

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
    let suffix = std::env::var("FFF_DRV_TYPING_SUFFIX").unwrap_or_default();

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
    let (mut stream_us, mut stream_cycles) = (Vec::with_capacity(prefixes.len()), 0u64);
    {
        let guard = shared_picker.read().expect("picker lock");
        let picker = guard.as_ref().expect("picker");
        for prefix in &prefixes {
            let query = if suffix.is_empty() {
                prefix.clone()
            } else {
                format!("{prefix} {suffix}")
            };
            let parsed = parser.parse(&query);
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
            stream_us.push(us);
            stream_cycles = stream_cycles.wrapping_add(result.total_matched as u64);
            println!(
                "KEY prefix={prefix} us={us:.1} total={} len={}",
                result.total_matched,
                result.items.len()
            );
        }
    }
    let total_ms = stream_start.elapsed().as_secs_f64() * 1000.0;
    let sum_us: f64 = stream_us.iter().sum();
    println!(
        "STREAM scan_ms={scan_ms:.1} keys={} total_ms={total_ms:.1} search_us={sum_us:.1} checksum={stream_cycles}",
        prefixes.len()
    );
}
