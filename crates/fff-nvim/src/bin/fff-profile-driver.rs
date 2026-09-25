//! July-style process-run driver.
//! Usage: fff-profile-driver.exe <corpus> <query> <iterations> <threads>
//! Prints: query=... threads=... iterations=... us_per_op=... checksum=...

use fff::file_picker::{FFFMode, FilePicker};
use fff::{
    FileSearchConfig, FuzzySearchOptions, PaginationArgs, QueryParser, SharedFilePicker,
    SharedFrecency,
};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).expect("corpus path");
    let query = args.get(2).map(String::as_str).unwrap_or("score");
    let iterations: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let threads: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
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
        result.total_matched ^ result.items.len()
    };

    let mut checksum = 0usize;
    for _ in 0..100 {
        checksum ^= search();
    }
    let started = Instant::now();
    for _ in 0..iterations {
        checksum = checksum.wrapping_add(search());
    }
    let elapsed = started.elapsed();
    println!(
        "query={query} threads={threads} iterations={iterations} us_per_op={:.3} checksum={checksum}",
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
    );
}
