//! Ordered top-N ranking receipt for one corpus+query, shared byte-for-byte
//! between the upstream and fork trees. Usage: rank_probe <corpus> <query>
//! [limit] [threads]. Writes TSV to $FFF_RP_OUT and prints a one-line summary.

use fff_search::file_picker::{FFFMode, FilePicker, FilePickerOptions};
use fff_search::{
    FileSearchConfig, FuzzySearchOptions, PaginationArgs, QueryParser, SharedFilePicker,
    SharedFrecency,
};
use std::path::Path;
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).expect("corpus path");
    let query = args.get(2).expect("query");
    let limit: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(200);
    let threads: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(4);
    let root_path = Path::new(root);

    let shared_picker = SharedFilePicker::default();
    FilePicker::new_with_shared_state(
        shared_picker.clone(),
        SharedFrecency::default(),
        FilePickerOptions {
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

    let mut out = String::with_capacity(64 * 1024);
    out.push_str(&format!("corpus\t{}\n", root));
    out.push_str(&format!("query\t{}\n", query));
    out.push_str(&format!("limit\t{}\n", limit));
    out.push_str(&format!("threads\t{}\n", threads));
    out.push_str(&format!("total_matched\t{}\n", result.total_matched));
    out.push_str(&format!("total_files\t{}\n", result.total_files));
    out.push_str(&format!("items\t{}\n", result.items.len()));
    for (i, item) in result.items.iter().enumerate() {
        let total = result.scores.get(i).map(|s| s.total).unwrap_or(i32::MIN);
        let base = result
            .scores
            .get(i)
            .map(|s| s.base_score)
            .unwrap_or(i32::MIN);
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            i + 1,
            total,
            base,
            item.relative_path(picker)
        ));
    }
    let out_path = std::env::var_os("FFF_RP_OUT").expect("FFF_RP_OUT");
    std::fs::write(&out_path, out.as_bytes()).expect("write receipt");
    println!(
        "ok corpus={} query={} matched={} items={}",
        root,
        query,
        result.total_matched,
        result.items.len()
    );
}
