//! Mutation / watcher correctness gate (FFF_REFERENCE_PACK).

use crate::adapter::{AdapterConfig, CallKind, FffAdapter, SearchSpec};
use crate::fixture;
use crate::gates::GateEntry;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MutationReport {
    pub passed: bool,
    pub cases: Vec<MutationCase>,
    pub propagation_us: Vec<u64>,
    pub deadline_failures: usize,
    #[serde(default)]
    pub entries: Vec<GateEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationCase {
    pub id: String,
    pub op: String,
    pub passed: bool,
    pub propagation_us: Option<u64>,
    pub message: String,
}

const SETTLE_DEADLINE: Duration = Duration::from_secs(5);

/// Run mutation gate on an isolated copy of public-v1 (or any corpus path).
pub fn run_mutation_gate(source_corpus: &Path) -> Result<MutationReport, String> {
    let fixture_root = if source_corpus.join("src/file_picker.rs").exists()
        && source_corpus.join("mutation").exists()
    {
        source_corpus.to_path_buf()
    } else {
        let cache = dirs_cache();
        fixture::materialize_public_v1(&cache)?.0
    };
    let work = fixture::clone_for_mutation(&fixture_root)?;
    let result = run_on_worktree(&work);
    let _ = fs::remove_dir_all(&work);
    result
}

fn dirs_cache() -> PathBuf {
    if let Ok(c) = std::env::var("FFF_BENCH_CACHE") {
        return PathBuf::from(c);
    }
    std::env::temp_dir().join("fff-bench-cache")
}

fn run_on_worktree(work: &Path) -> Result<MutationReport, String> {
    let mut cfg = AdapterConfig::default();
    cfg.timeout_secs = 30;
    // Watcher enabled for settle contract.
    let adapter = FffAdapter::open_corpus_watch(work, cfg, true)?;

    let mut cases = Vec::new();
    let mut prop = Vec::new();
    let mut deadline_failures = 0usize;

    // CREATE
    let create_rel = PathBuf::from("mutation").join("created_by_bench.rs");
    let create_abs = work.join(&create_rel);
    fs::write(&create_abs, b"pub fn created_by_bench() {}\n").map_err(|e| e.to_string())?;
    notify_create(&adapter, &create_abs);
    let t0 = Instant::now();
    let seen = wait_until(
        &adapter,
        "created_by_bench",
        true,
        SETTLE_DEADLINE,
    );
    let pus = t0.elapsed().as_micros() as u64;
    prop.push(pus);
    if !seen {
        deadline_failures += 1;
    }
    cases.push(MutationCase {
        id: "mutation.create".into(),
        op: "create".into(),
        passed: seen,
        propagation_us: Some(pus),
        message: if seen {
            format!("visible after {pus}µs")
        } else {
            "create not visible within settle deadline".into()
        },
    });

    // RENAME
    let renamed_rel = PathBuf::from("mutation").join("renamed_by_bench.rs");
    let renamed_abs = work.join(&renamed_rel);
    let _ = fs::rename(&create_abs, &renamed_abs);
    notify_delete(&adapter, &create_abs);
    notify_create(&adapter, &renamed_abs);
    let t0 = Instant::now();
    let new_seen = wait_until(&adapter, "renamed_by_bench", true, SETTLE_DEADLINE);
    let old_gone = wait_until(&adapter, "created_by_bench", false, SETTLE_DEADLINE);
    let pus = t0.elapsed().as_micros() as u64;
    prop.push(pus);
    let ok = new_seen && old_gone;
    if !ok {
        deadline_failures += 1;
    }
    cases.push(MutationCase {
        id: "mutation.rename".into(),
        op: "rename".into(),
        passed: ok,
        propagation_us: Some(pus),
        message: format!("new_seen={new_seen} old_gone={old_gone} {pus}µs"),
    });

    // MODIFY (touch content)
    fs::write(&renamed_abs, b"pub fn renamed_by_bench() { let _x = 1; }\n")
        .map_err(|e| e.to_string())?;
    notify_modify(&adapter, &renamed_abs);
    let t0 = Instant::now();
    let still = wait_until(&adapter, "renamed_by_bench", true, SETTLE_DEADLINE);
    let pus = t0.elapsed().as_micros() as u64;
    prop.push(pus);
    cases.push(MutationCase {
        id: "mutation.modify".into(),
        op: "modify".into(),
        passed: still,
        propagation_us: Some(pus),
        message: if still {
            "still visible after modify".into()
        } else {
            "lost after modify".into()
        },
    });
    if !still {
        deadline_failures += 1;
    }

    // DELETE
    let _ = fs::remove_file(&renamed_abs);
    notify_delete(&adapter, &renamed_abs);
    let t0 = Instant::now();
    let gone = wait_until(&adapter, "renamed_by_bench", false, SETTLE_DEADLINE);
    let pus = t0.elapsed().as_micros() as u64;
    prop.push(pus);
    if !gone {
        deadline_failures += 1;
    }
    cases.push(MutationCase {
        id: "mutation.delete".into(),
        op: "delete".into(),
        passed: gone,
        propagation_us: Some(pus),
        message: if gone {
            format!("absent after {pus}µs")
        } else {
            "still visible after delete deadline".into()
        },
    });

    let passed = cases.iter().all(|c| c.passed);
    let entry = GateEntry {
        gate_id: "gate.mutation.public".into(),
        status: if passed { "pass" } else { "fail" }.into(),
        required: true,
        reason_code: if passed {
            None
        } else {
            Some("mutation_visibility".into())
        },
        evidence: Some(serde_json::json!({
            "cases": cases.len(),
            "deadline_failures": deadline_failures,
            "propagation_us": prop,
        })),
    };

    Ok(MutationReport {
        passed,
        cases,
        propagation_us: prop,
        deadline_failures,
        entries: vec![entry],
    })
}

fn notify_create(adapter: &FffAdapter, path: &Path) {
    adapter.notify_path_changed(path);
}

fn notify_modify(adapter: &FffAdapter, path: &Path) {
    adapter.notify_path_changed(path);
}

fn notify_delete(adapter: &FffAdapter, path: &Path) {
    // Tombstone via modify on missing path.
    adapter.notify_path_changed(path);
}

fn wait_until(adapter: &FffAdapter, query: &str, expect_present: bool, deadline: Duration) -> bool {
    let start = Instant::now();
    loop {
        let r = adapter.search(&SearchSpec {
            query: query.into(),
            limit: 20,
            offset: 0,
            threads: 1,
            call: CallKind::File,
        });
        let hit = adapter.rank_of(&r, query).is_some()
            || r.top_k.iter().any(|h| {
                h.path
                    .replace('\\', "/")
                    .to_lowercase()
                    .contains(&query.to_lowercase())
            });
        if hit == expect_present {
            return true;
        }
        if start.elapsed() >= deadline {
            return false;
        }
        // Nudge the index again on each poll.
        std::thread::sleep(Duration::from_millis(25));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutation_gate_on_fixture() {
        let cache = std::env::temp_dir().join(format!("fff-mut-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&cache);
        let (fx, _) = fixture::materialize_public_v1(&cache).unwrap();
        let report = run_mutation_gate(&fx).unwrap();
        assert!(
            report.passed,
            "mutation cases: {:?}",
            report.cases
        );
        let _ = fs::remove_dir_all(&cache);
    }
}
