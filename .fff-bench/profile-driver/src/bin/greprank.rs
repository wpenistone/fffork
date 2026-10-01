//! Grep A/B probe for keepers-vs-upstream.
//! Usage: greprank <corpus> <iters> <page_limit> <query>...
//! Queries may be prefixed: `plain:` (default), `fuzzy:`, `re:`.

use fff::file_picker::{FFFMode, FilePicker};
use fff::grep::{parse_grep_query, GrepMode, GrepSearchOptions};
use fff::{SharedFilePicker, SharedFrecency};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).expect("corpus path").clone();
    let iters: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20);
    let page_limit: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(50);
    let queries: Vec<&str> = args[4..].iter().map(String::as_str).collect();
    assert!(!queries.is_empty(), "at least one query required");

    let scan_t0 = Instant::now();
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

    for raw in queries {
        let (mode, needle) = match raw.split_once(':') {
            Some(("fuzzy", q)) => (GrepMode::Fuzzy, q),
            Some(("re", q)) => (GrepMode::Regex, q),
            Some(("plain", q)) => (GrepMode::PlainText, q),
            _ => (GrepMode::PlainText, raw),
        };
        let parsed = parse_grep_query(needle);
        let opts = GrepSearchOptions {
            page_limit,
            mode,
            ..Default::default()
        };
        let mut times = Vec::with_capacity(iters);
        let mut checksum = 0usize;
        let (mut files_searched, mut matches, mut with_matches, mut searchable) = (0, 0, 0, 0);
        for i in 0..(iters + 3) {
            let t0 = Instant::now();
            let (fs, m, wm, sf) = {
                let guard = shared_picker.read().expect("picker lock");
                let picker = guard.as_ref().expect("picker");
                let r = picker.grep(&parsed, &opts);
                (
                    r.total_files_searched,
                    r.matches.len(),
                    r.files_with_matches,
                    r.filtered_file_count,
                )
            };
            let us = t0.elapsed().as_secs_f64() * 1e6;
            if i >= 3 {
                times.push(us);
                checksum ^= fs ^ m ^ wm;
                files_searched = fs;
                matches = m;
                with_matches = wm;
                searchable = sf;
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).expect("f64"));
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let p50 = times[times.len() / 2];
        let p95 = times[(times.len() * 95 / 100).min(times.len() - 1)];
        println!(
            "grep={raw} mode={mode:?} files_searched={files_searched} searchable={searchable} \
             matches={matches} files_with_matches={with_matches} mean_us={mean:.1} \
             p50_us={p50:.1} p95_us={p95:.1} checksum={checksum}"
        );
    }
}