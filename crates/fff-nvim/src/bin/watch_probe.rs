//! Watcher readiness and add/remove latency probe for the target-scale matrix.
//! Usage: watch_probe.exe <corpus> [timeout_secs]
//! The probe creates and deletes `._fff_watch_probe_<pid>.tmp` in the corpus
//! root. Run it outside a timing panel and re-pin afterwards.

use fff::file_picker::{FFFMode, FilePicker};
use fff::{SharedFilePicker, SharedFrecency};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).expect("corpus path").clone();
    let timeout_secs: u64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(120);
    let root_path = Path::new(&root);
    let probe = root_path.join(format!("._fff_watch_probe_{}.tmp", std::process::id()));

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
            watch: true,
            skip_git: std::env::var_os("FFF_DRV_GIT").is_none(),
            ..Default::default()
        },
    )
    .expect("open corpus");
    assert!(
        shared_picker.wait_for_scan(Duration::from_secs(timeout_secs)),
        "scan timed out"
    );
    let scan_ms = opened.elapsed().as_secs_f64() * 1000.0;

    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let mut watch_ready_ms = f64::NAN;
    while Instant::now() < deadline {
        let ready = shared_picker
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(|p| p.is_watcher_ready()))
            .unwrap_or(false);
        if ready {
            watch_ready_ms = opened.elapsed().as_secs_f64() * 1000.0;
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }

    let live = |sp: &SharedFilePicker| -> usize {
        sp.read()
            .ok()
            .and_then(|g| g.as_ref().map(|p| p.live_file_count()))
            .unwrap_or(0)
    };
    let initial = live(&shared_picker);
    let name = probe
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let present = |sp: &SharedFilePicker| -> bool {
        sp.read()
            .ok()
            .and_then(|g| {
                g.as_ref().map(|p| {
                    p.get_files()
                        .iter()
                        .filter(|f| !f.is_deleted())
                        .any(|f| f.relative_path(p).ends_with(&name))
                })
            })
            .unwrap_or(false)
    };
    let wait_present = |sp: &SharedFilePicker, want: bool, secs: u64| -> f64 {
        let deadline = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < deadline {
            if present(sp) == want {
                return 1.0;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        0.0
    };

    let t = Instant::now();
    std::fs::write(&probe, b"watch probe\n").expect("write probe");
    let added = wait_present(&shared_picker, true, timeout_secs);
    let add_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    std::fs::remove_file(&probe).expect("remove probe");
    let removed = wait_present(&shared_picker, false, timeout_secs);
    let remove_ms = t.elapsed().as_secs_f64() * 1000.0;

    let final_count = live(&shared_picker);
    println!(
        "WATCH scan_ms={scan_ms:.1} watch_ready_ms={watch_ready_ms:.1} initial={initial} added={added} add_ms={add_ms:.1} removed={removed} remove_ms={remove_ms:.1} final={final_count}"
    );
}
