use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fff_search::{
    FFFQuery, FilePicker, FilePickerOptions, FileSearchConfig, FuzzySearchOptions, PaginationArgs,
};
use std::{hint::black_box, time::Duration};

fn bench_fuzzy_search(c: &mut Criterion) {
    for count in [1_000, 10_000, 100_000] {
        let (_root, picker) = make_picker(count);
        bench_picker(
            c,
            &picker,
            &format!("fuzzy_search_{count}"),
            "src/components/group_0/controller_0.rs",
        );
    }
}

fn bench_picker(c: &mut Criterion, picker: &FilePicker, name: &str, current_file: &str) {
    let mut group = c.benchmark_group(name);
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));
    for (name, text) in [
        ("empty", ""),
        ("short", "mo"),
        ("filename", "controller"),
        ("path", "src/components"),
        ("multipart", "components controller"),
        ("extension", "*.rs"),
    ] {
        let query = FFFQuery::parse(text, FileSearchConfig);
        for current_file in [None, Some(current_file)] {
            let options = FuzzySearchOptions {
                max_threads: 4,
                current_file,
                project_path: None,
                combo_boost_score_multiplier: 0,
                min_combo_count: 0,
                pagination: PaginationArgs {
                    offset: 0,
                    limit: 50,
                },
            };
            let suffix = if current_file.is_some() {
                "current_file"
            } else {
                "no_current_file"
            };
            group.bench_function(BenchmarkId::new(name, suffix), |b| {
                b.iter(|| black_box(picker.fuzzy_search(black_box(&query), None, options)));
            });
        }
    }
    group.finish();
}

fn make_picker(count: usize) -> (tempfile::TempDir, FilePicker) {
    let root = tempfile::tempdir().unwrap();
    let areas = [
        "src/components",
        "src/core",
        "tests/integration",
        "packages/server",
    ];
    let names = ["controller", "model", "service", "utils", "index"];
    let extensions = ["rs", "ts", "lua", "py"];
    for group in 0..count.div_ceil(100) {
        let directory = root
            .path()
            .join(format!("{}/group_{group}", areas[group % areas.len()]));
        std::fs::create_dir_all(&directory).unwrap();
        for index in group * 100..((group + 1) * 100).min(count) {
            let name = format!(
                "{}_{index}.{}",
                names[index % names.len()],
                extensions[index % extensions.len()]
            );
            let file = std::fs::File::create(directory.join(name)).unwrap();
            let modified =
                std::time::UNIX_EPOCH + Duration::from_secs(1_700_000_000 + (index % 1000) as u64);
            file.set_times(std::fs::FileTimes::new().set_modified(modified))
                .unwrap();
        }
    }
    let mut picker = FilePicker::new(FilePickerOptions {
        base_path: root.path().to_string_lossy().into_owned(),
        enable_mmap_cache: false,
        enable_content_indexing: false,
        watch: false,
        ..Default::default()
    })
    .unwrap();
    picker.collect_files().unwrap();
    assert_eq!(picker.live_file_count(), count);
    (root, picker)
}

fn bench_typing_stream(c: &mut Criterion) {
    // Prefix stream exercises the live-narrow subset path: each query
    // extends the previous one, so all but the first reuse prior hits.
    let (_root, picker) = make_picker(10_000);
    let steps = [
        "c", "co", "con", "cont", "contr", "contro", "control", "controll", "controlle",
        "controller",
    ];
    let mut group = c.benchmark_group("typing_stream_10k");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(2));
    for threads in [1usize, 4] {
        group.bench_function(BenchmarkId::new("prefix", threads), |b| {
            b.iter(|| {
                for text in steps {
                    let query = FFFQuery::parse(text, FileSearchConfig);
                    let options = FuzzySearchOptions {
                        max_threads: threads,
                        current_file: None,
                        project_path: None,
                        combo_boost_score_multiplier: 0,
                        min_combo_count: 0,
                        pagination: PaginationArgs {
                            offset: 0,
                            limit: 50,
                        },
                    };
                    black_box(picker.fuzzy_search(black_box(&query), None, options));
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fuzzy_search, bench_typing_stream);
criterion_main!(benches);
