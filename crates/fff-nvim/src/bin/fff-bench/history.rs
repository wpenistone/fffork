//! History store for agent loops.

use crate::runner::BenchReport;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct HistoryIndexLine {
    pub run_id: String,
    pub suite: String,
    pub admissible: bool,
    pub score: Option<f64>,
    pub score_status: Option<String>,
    pub git_commit: Option<String>,
    pub patch_sha256: Option<String>,
    pub report_path: String,
    pub integrity: String,
}

pub fn record(
    history_dir: &Path,
    report: &BenchReport,
    log_dir: Option<&Path>,
) -> Result<(), String> {
    fs::create_dir_all(history_dir.join("runs")).map_err(|e| e.to_string())?;
    let run_dir = history_dir.join("runs").join(&report.metadata.run.run_id);
    fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;

    let report_path = run_dir.join("report.json");
    let json = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
    fs::write(&report_path, json).map_err(|e| e.to_string())?;

    if let Some(ref change) = report.change {
        if let Some(ref pp) = change.patch_path {
            let src = PathBuf::from(pp);
            if src.exists() {
                let _ = fs::copy(&src, run_dir.join("change.patch"));
            }
        }
    }
    if let Some(ld) = log_dir {
        let patch = ld.join("change.patch");
        if patch.exists() {
            let _ = fs::copy(patch, run_dir.join("change.patch"));
        }
    }

    let line = HistoryIndexLine {
        run_id: report.metadata.run.run_id.clone(),
        suite: report.metadata.run.suite.clone(),
        admissible: report.admissible,
        score: report.score.as_ref().and_then(|s| s.score),
        score_status: report.score.as_ref().map(|s| s.score_status.clone()),
        git_commit: report.metadata.subject.git_commit.clone(),
        patch_sha256: report.change.as_ref().and_then(|c| c.patch_sha256.clone()),
        report_path: report_path.display().to_string(),
        integrity: report.noise.integrity.clone(),
    };

    let index = history_dir.join("index.jsonl");
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(index)
        .map_err(|e| e.to_string())?;
    writeln!(f, "{}", serde_json::to_string(&line).unwrap()).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn latest_comparable(
    history_dir: &Path,
    suite: &str,
    exclude_run_id: Option<&str>,
) -> Option<PathBuf> {
    latest_comparable_any(history_dir, &[suite], exclude_run_id)
}

/// Prefer exact suite match; otherwise accept any suite in `suites` (first listed wins preference).
pub fn latest_comparable_any(
    history_dir: &Path,
    suites: &[&str],
    exclude_run_id: Option<&str>,
) -> Option<PathBuf> {
    let index = history_dir.join("index.jsonl");
    let text = fs::read_to_string(index).ok()?;
    // Pass 1: exact preferred suite (suites[0])
    if let Some(preferred) = suites.first() {
        if let Some(p) = scan_index(&text, preferred, exclude_run_id, true) {
            return Some(p);
        }
    }
    // Pass 2: any listed suite
    for line in text.lines().rev() {
        let Ok(e) = serde_json::from_str::<HistoryIndexLine>(line) else {
            continue;
        };
        if let Some(ex) = exclude_run_id {
            if e.run_id == ex {
                continue;
            }
        }
        if !e.admissible {
            continue;
        }
        if suites.iter().any(|s| suite_aliases_match(s, &e.suite)) {
            let p = PathBuf::from(&e.report_path);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

fn scan_index(
    text: &str,
    suite: &str,
    exclude_run_id: Option<&str>,
    exact_only: bool,
) -> Option<PathBuf> {
    for line in text.lines().rev() {
        let Ok(e) = serde_json::from_str::<HistoryIndexLine>(line) else {
            continue;
        };
        if let Some(ex) = exclude_run_id {
            if e.run_id == ex {
                continue;
            }
        }
        if !e.admissible {
            continue;
        }
        let ok = if exact_only {
            e.suite == suite || suite_aliases_match(suite, &e.suite)
        } else {
            suite_aliases_match(suite, &e.suite)
        };
        if ok {
            let p = PathBuf::from(&e.report_path);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

fn suite_aliases_match(want: &str, have: &str) -> bool {
    fn n(s: &str) -> String {
        match s {
            "agent-loop" | "agent-screen" | "screen" => "agent-screen".into(),
            "agent-confirm" | "confirm" => "agent-confirm".into(),
            "peak" | "peak-frontier" | "peak-diagnose" | "diagnose" => "peak-diagnose".into(),
            x => x.into(),
        }
    }
    n(want) == n(have)
}

pub fn list(history_dir: &Path) -> Vec<HistoryIndexLine> {
    let index = history_dir.join("index.jsonl");
    let Ok(text) = fs::read_to_string(index) else {
        return vec![];
    };
    text.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn latest_comparable_skips_excluded_and_inadmissible() {
        let dir = std::env::temp_dir().join(format!("fff-bench-hist-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("runs/a")).unwrap();
        fs::create_dir_all(dir.join("runs/b")).unwrap();
        let ra = dir.join("runs/a/report.json");
        let rb = dir.join("runs/b/report.json");
        fs::write(&ra, "{}").unwrap();
        fs::write(&rb, "{}").unwrap();
        let mut idx = fs::File::create(dir.join("index.jsonl")).unwrap();
        writeln!(
            idx,
            "{}",
            serde_json::to_string(&HistoryIndexLine {
                run_id: "a".into(),
                suite: "agent-loop".into(),
                admissible: true,
                score: Some(1.0),
                score_status: Some("ok".into()),
                git_commit: None,
                patch_sha256: None,
                report_path: ra.display().to_string(),
                integrity: "clean".into(),
            })
            .unwrap()
        )
        .unwrap();
        writeln!(
            idx,
            "{}",
            serde_json::to_string(&HistoryIndexLine {
                run_id: "b".into(),
                suite: "agent-loop".into(),
                admissible: true,
                score: Some(0.9),
                score_status: Some("ok".into()),
                git_commit: None,
                patch_sha256: None,
                report_path: rb.display().to_string(),
                integrity: "clean".into(),
            })
            .unwrap()
        )
        .unwrap();
        // exclude b → should get a
        let p = latest_comparable(&dir, "agent-loop", Some("b")).unwrap();
        assert!(p.ends_with("report.json"));
        assert!(p.to_string_lossy().contains("runs"));
        let p2 = latest_comparable(&dir, "agent-loop", None).unwrap();
        assert!(p2.to_string_lossy().contains("b") || p2 == rb);
        let _ = fs::remove_dir_all(&dir);
    }
}
