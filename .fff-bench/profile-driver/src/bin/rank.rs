//! Ranked top-k dump + timing for keepers-vs-upstream A/B.
//! Usage: rank <corpus> <iters> <threads> <limit> <query> [<query>...]

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
    let iters: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(50);
    let threads: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let limit: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(10);
    let queries: Vec<&str> = args[5..].iter().map(String::as_str).collect();
    assert!(!queries.is_empty(), "at least one query required");
    let root_path = Path::new(&root);

    let shared_picker = SharedFilePicker::default();
    let scan_t0 = Instant::now();
    FilePicker::new_with_shared_state(
        shared_picker.clone(),
        SharedFrecency::default(),
        fff::FilePickerOptions {
            base_path: root.clone(),
            enable_mmap_cache: false,
            enable_content_indexing: false,
            mode: FFFMode::Neovim,
            watch: false,
            skip_git: false,
            ..Default::default()
        },
    )
    .expect("open corpus");
    assert!(
        shared_picker.wait_for_scan(Duration::from_secs(600)),
        "scan timed out"
    );
    eprintln!("SCAN us={:.0}", scan_t0.elapsed().as_secs_f64() * 1e6);

    let parser = QueryParser::<FileSearchConfig>::default();
    let mk_opts = |limit: usize| FuzzySearchOptions {
        max_threads: threads,
        current_file: None,
        project_path: Some(root_path),
        combo_boost_score_multiplier: 100,
        min_combo_count: 3,
        pagination: PaginationArgs {
            offset: 0,
            limit,
        },
    };

    for query in &queries {
        let parsed = parser.parse(query);
        {
            let guard = shared_picker.read().expect("picker lock");
            let picker = guard.as_ref().expect("picker");
            let result = picker.fuzzy_search(&parsed, None, mk_opts(limit));
            println!("query={query:?} total_matched={}", result.total_matched);
            for (i, (item, score)) in result.items.iter().zip(result.scores.iter()).enumerate() {
                println!(
                    "{i}\t{}\t{}\t{}",
                    score.total,
                    item.relative_path(picker),
                    score.match_type
                );
            }
        }

        let time_opts = mk_opts(100);
        for _ in 0..5 {
            let guard = shared_picker.read().expect("picker lock");
            let picker = guard.as_ref().expect("picker");
            let r = picker.fuzzy_search(&parsed, None, time_opts);
            std::hint::black_box(r.total_matched);
        }
        let mut times = Vec::with_capacity(iters);
        let mut checksum = 0usize;
        for _ in 0..iters {
            let t0 = Instant::now();
            {
                let guard = shared_picker.read().expect("picker lock");
                let picker = guard.as_ref().expect("picker");
                let r = picker.fuzzy_search(&parsed, None, time_opts);
                checksum ^= r.total_matched ^ r.items.len();
            }
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(|a, b| a.partial_cmp(b).expect("f64"));
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let p50 = times[times.len() / 2];
        let p95 = times[(times.len() * 95 / 100).min(times.len() - 1)];
        println!(
            "timing query={query:?} iters={iters} threads={threads} mean_us={mean:.1} p50_us={p50:.1} p95_us={p95:.1} checksum={checksum}"
        );
    }
}
