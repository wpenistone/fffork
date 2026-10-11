//! validate / lint / explain / diff / export / init / calibrate / replay.

use crate::compare::CompareOut;
use crate::evaluator;
use crate::manifest::Manifest;
use crate::runner::BenchReport;
use crate::scenarios;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidateOut {
    pub ok: bool,
    pub status: String,
    pub issues: Vec<Issue>,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Issue {
    pub code: String,
    pub severity: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExplainOut {
    pub verdict: Option<String>,
    pub admissible: bool,
    pub layers: Vec<ExplainLayer>,
    pub reason_codes: Vec<String>,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExplainLayer {
    pub layer: String,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiffOut {
    pub comparable_level: String,
    pub differences: Vec<DiffItem>,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiffItem {
    pub kind: String,
    pub field: String,
    pub baseline: serde_json::Value,
    pub candidate: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalibrateOut {
    pub status: String,
    pub clock_resolution_ns: u64,
    pub snapshot_overhead_us: f64,
    pub host_cpu_busy_pct: Option<f64>,
    pub digest: String,
    pub summary: String,
    /// Per-pack exterior sensor availability on this host.
    #[serde(default)]
    pub sensor_packs: Vec<crate::sensors::PackProbe>,
}

pub fn validate_manifest(path: &Path) -> ValidateOut {
    let mut issues = Vec::new();
    match Manifest::load(path) {
        Ok(m) => {
            if m.repos.is_empty() {
                issues.push(Issue {
                    code: "empty_corpora".into(),
                    severity: "error".into(),
                    message: "manifest has no repos/corpora".into(),
                    path: Some(path.display().to_string()),
                });
            }
            for r in &m.repos {
                if r.id.is_empty() {
                    issues.push(Issue {
                        code: "empty_corpus_id".into(),
                        severity: "error".into(),
                        message: "corpus id is empty".into(),
                        path: Some(path.display().to_string()),
                    });
                }
            }
            let ok = !issues.iter().any(|i| i.severity == "error");
            ValidateOut {
                ok,
                status: if ok { "valid".into() } else { "invalid".into() },
                issues,
                summary: format!(
                    "manifest {} repos={} workflow={}/{}/{}",
                    if ok { "ok" } else { "failed" },
                    m.repos.len(),
                    m.workflow.screen_suite,
                    m.workflow.confirm_suite,
                    m.workflow.diagnose_suite
                ),
            }
        }
        Err(e) => ValidateOut {
            ok: false,
            status: "invalid".into(),
            issues: vec![Issue {
                code: "parse_error".into(),
                severity: "error".into(),
                message: e,
                path: Some(path.display().to_string()),
            }],
            summary: "manifest parse failed".into(),
        },
    }
}

pub fn validate_report(path: &Path) -> ValidateOut {
    let mut issues = Vec::new();
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            return ValidateOut {
                ok: false,
                status: "invalid".into(),
                issues: vec![Issue {
                    code: "read_error".into(),
                    severity: "error".into(),
                    message: e.to_string(),
                    path: Some(path.display().to_string()),
                }],
                summary: "cannot read report".into(),
            };
        }
    };
    let report: BenchReport = match serde_json::from_str(&raw) {
        Ok(r) => r,
        Err(e) => {
            return ValidateOut {
                ok: false,
                status: "invalid".into(),
                issues: vec![Issue {
                    code: "schema_error".into(),
                    severity: "error".into(),
                    message: e.to_string(),
                    path: Some(path.display().to_string()),
                }],
                summary: "report JSON does not match BenchReport".into(),
            };
        }
    };
    if report.metadata.run.run_id.is_empty() {
        issues.push(Issue {
            code: "missing_run_id".into(),
            severity: "error".into(),
            message: "run_id empty".into(),
            path: None,
        });
    }
    if report.metadata.run.suite.is_empty() {
        issues.push(Issue {
            code: "missing_suite".into(),
            severity: "error".into(),
            message: "suite empty".into(),
            path: None,
        });
    }
    if report.evaluator.is_none() {
        issues.push(Issue {
            code: "missing_evaluator".into(),
            severity: "warning".into(),
            message: "evaluator identity absent (pre-digest report)".into(),
            path: None,
        });
    }
    let ok = !issues.iter().any(|i| i.severity == "error");
    ValidateOut {
        ok,
        status: if ok { "valid".into() } else { "invalid".into() },
        issues,
        summary: format!(
            "report {} suite={} cells={} admissible={}",
            if ok { "ok" } else { "failed" },
            report.metadata.run.suite,
            report.cells.len(),
            report.admissible
        ),
    }
}

pub fn lint_suite(suite: &str, repo_id: &str) -> ValidateOut {
    let mut issues = Vec::new();
    let cells = scenarios::expand_suite(suite, repo_id);
    if cells.is_empty() {
        issues.push(Issue {
            code: "empty_suite".into(),
            severity: "error".into(),
            message: format!("suite {suite} expands to 0 cells"),
            path: None,
        });
    }
    if cells.len() > 2000 {
        issues.push(Issue {
            code: "unbounded_matrix".into(),
            severity: "warning".into(),
            message: format!("suite expands to {} cells — review sparsity", cells.len()),
            path: None,
        });
    }
    let mut keys = std::collections::HashSet::new();
    for c in &cells {
        if !keys.insert(c.cell_key.clone()) {
            issues.push(Issue {
                code: "duplicate_cell_key".into(),
                severity: "error".into(),
                message: format!("duplicate cell_key {}", c.cell_key.replace('\0', "|")),
                path: None,
            });
        }
        if c.family == "cold" && c.scenario_id.contains("cold") {
            // soft check: cold claims should use qualified state when axes have state
            if c.axes.get("state_id").is_none() && c.axes.get("state").is_none() {
                // informational only for now
            }
        }
    }
    let ok = !issues.iter().any(|i| i.severity == "error");
    ValidateOut {
        ok,
        status: if ok { "clean".into() } else { "hazards".into() },
        issues,
        summary: format!(
            "lint suite={suite} cells={} status={}",
            cells.len(),
            if ok { "clean" } else { "hazards" }
        ),
    }
}

pub fn explain_report(report: &BenchReport, cmp: Option<&CompareOut>) -> ExplainOut {
    let mut layers = Vec::new();
    layers.push(ExplainLayer {
        layer: "infrastructure".into(),
        status: "pass".into(),
        detail: "report loaded".into(),
    });
    layers.push(ExplainLayer {
        layer: "evaluator".into(),
        status: if report.evaluator.is_some() {
            "pass"
        } else {
            "warn"
        }
        .into(),
        detail: report
            .evaluator
            .as_ref()
            .map(|e| e.evaluator_digest.clone())
            .unwrap_or_else(|| "missing".into()),
    });
    layers.push(ExplainLayer {
        layer: "gates".into(),
        status: if report.admissible { "pass" } else { "fail" }.into(),
        detail: if report.admissible {
            "all required gates passed".into()
        } else {
            format!("failed: {:?}", report.gates.failed)
        },
    });
    if let Some(st) = &report.stability {
        layers.push(ExplainLayer {
            layer: "stability".into(),
            status: st.stability.clone(),
            detail: format!(
                "within={} host={} tail={} max_rmad={:?}",
                st.within_cell_dispersion, st.host_noise, st.tail_uncertainty, st.max_relative_mad
            ),
        });
    }
    let mut reason_codes = Vec::new();
    if !report.admissible {
        reason_codes.push("gates_failed".into());
    }
    let verdict = if let Some(c) = cmp {
        reason_codes.extend(c.reason_codes.clone());
        layers.push(ExplainLayer {
            layer: "comparability".into(),
            status: if c.comparable { "pass" } else { "fail" }.into(),
            detail: format!("comparable={}", c.comparable),
        });
        layers.push(ExplainLayer {
            layer: "objective".into(),
            status: c.verdict.clone(),
            detail: c.summary_for_agent.clone(),
        });
        Some(c.verdict.clone())
    } else if !report.admissible {
        Some("inadmissible".into())
    } else {
        report.score.as_ref().map(|s| s.score_status.clone())
    };

    ExplainOut {
        verdict: verdict.clone(),
        admissible: report.admissible,
        layers,
        reason_codes,
        summary: format!(
            "verdict={:?} admissible={} cells={}",
            verdict,
            report.admissible,
            report.cells.len()
        ),
    }
}

pub fn diff_reports(a: &BenchReport, b: &BenchReport) -> DiffOut {
    let mut diffs = Vec::new();
    if a.metadata.run.suite != b.metadata.run.suite {
        diffs.push(DiffItem {
            kind: "evaluator".into(),
            field: "suite".into(),
            baseline: serde_json::json!(a.metadata.run.suite),
            candidate: serde_json::json!(b.metadata.run.suite),
        });
    }
    match (&a.evaluator, &b.evaluator) {
        (Some(ea), Some(eb)) if ea.evaluator_digest != eb.evaluator_digest => {
            diffs.push(DiffItem {
                kind: "evaluator".into(),
                field: "evaluator_digest".into(),
                baseline: serde_json::json!(ea.evaluator_digest),
                candidate: serde_json::json!(eb.evaluator_digest),
            });
        }
        (None, Some(_)) | (Some(_), None) => {
            diffs.push(DiffItem {
                kind: "evaluator".into(),
                field: "evaluator_present".into(),
                baseline: serde_json::json!(a.evaluator.is_some()),
                candidate: serde_json::json!(b.evaluator.is_some()),
            });
        }
        _ => {}
    }
    if a.metadata.stats.iterations != b.metadata.stats.iterations {
        diffs.push(DiffItem {
            kind: "plan".into(),
            field: "iterations".into(),
            baseline: serde_json::json!(a.metadata.stats.iterations),
            candidate: serde_json::json!(b.metadata.stats.iterations),
        });
    }
    if a.admissible != b.admissible {
        diffs.push(DiffItem {
            kind: "evidence".into(),
            field: "admissible".into(),
            baseline: serde_json::json!(a.admissible),
            candidate: serde_json::json!(b.admissible),
        });
    }
    let a_keys: std::collections::HashSet<_> =
        a.cells.iter().map(|c| c.cell_key.as_str()).collect();
    let b_keys: std::collections::HashSet<_> =
        b.cells.iter().map(|c| c.cell_key.as_str()).collect();
    if a_keys != b_keys {
        diffs.push(DiffItem {
            kind: "plan".into(),
            field: "cell_set".into(),
            baseline: serde_json::json!(a_keys.len()),
            candidate: serde_json::json!(b_keys.len()),
        });
    }
    for c in &b.cells {
        if let Some(ab) = a.cells.iter().find(|x| x.cell_key == c.cell_key) {
            let rel = if ab.cost.wall_us.p95 > 0.0 {
                (c.cost.wall_us.p95 - ab.cost.wall_us.p95) / ab.cost.wall_us.p95
            } else {
                0.0
            };
            if rel.abs() > 0.05 {
                diffs.push(DiffItem {
                    kind: "evidence".into(),
                    field: format!("wall_p95:{}", c.cell_key.replace('\0', "|")),
                    baseline: serde_json::json!(ab.cost.wall_us.p95),
                    candidate: serde_json::json!(c.cost.wall_us.p95),
                });
            }
        }
    }
    let level = if diffs
        .iter()
        .any(|d| d.kind == "evaluator" || d.kind == "plan")
    {
        "incomparable"
    } else if diffs.is_empty() {
        "exact"
    } else {
        "informational"
    };
    DiffOut {
        comparable_level: level.into(),
        summary: format!("{} differences; level={level}", diffs.len()),
        differences: diffs,
    }
}

pub fn export_csv(report: &BenchReport) -> String {
    let mut out = String::from(
        "cell_key,scenario_id,family,repo_id,wall_p50,wall_p95,wall_relative_mad,n,\
         cpu_total_p50,cpu_total_p95,rss_peak_mb,cpu_cycles_p50,threads_peak,rank_target,throughput_qps\n",
    );
    for c in &report.cells {
        let key = c.cell_key.replace('\0', "|").replace(',', ";");
        let cpu = c.cost.cpu_total_us.as_ref();
        let cycles = c.cost.cpu_cycles.as_ref();
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            key,
            c.scenario_id,
            c.family,
            c.repo_id,
            c.cost.wall_us.p50,
            c.cost.wall_us.p95,
            c.cost.wall_us.relative_mad,
            c.cost.wall_us.n,
            cpu.map(|d| format!("{:.0}", d.p50)).unwrap_or_default(),
            cpu.map(|d| format!("{:.0}", d.p95)).unwrap_or_default(),
            c.cost
                .rss_peak_mb
                .map(|v| format!("{v:.2}"))
                .unwrap_or_default(),
            cycles.map(|d| format!("{:.0}", d.p50)).unwrap_or_default(),
            c.cost
                .threads_observed_peak
                .map(|t| t.to_string())
                .unwrap_or_default(),
            c.rank_target.map(|r| r.to_string()).unwrap_or_default(),
            c.throughput_qps
                .map(|q| format!("{q:.2}"))
                .unwrap_or_default(),
        ));
    }
    out
}

pub fn init_pack(dir: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let manifest = dir.join("fff-bench.json");
    let body = serde_json::json!({
        "schema_version": 1,
        "workflow": {
            "screen_suite": "agent-screen",
            "confirm_suite": "agent-confirm",
            "diagnose_suite": "peak-diagnose"
        },
        "suite": "agent-confirm",
        "adapter": {
            "enable_content_indexing": false,
            "enable_mmap_cache": false,
            "score_breakdown": true,
            "top_k_log": 10
        },
        "repos": [
            {
                "id": "self",
                "role": ["gates", "confirm", "screen"],
                "source": { "kind": "path", "path": "." }
            }
        ],
        "history": {
            "directory": ".fff-bench/history",
            "record": true
        }
    });
    fs::write(&manifest, serde_json::to_string_pretty(&body).unwrap())
        .map_err(|e| e.to_string())?;
    let readme = dir.join("README.bench.md");
    let _ = fs::write(
        readme,
        "# FFF bench pack\n\n```\ncargo build -p fff-nvim --bin fff-bench --release\nfff-bench doctor --path . --json\nfff-bench baseline --path . --json\nfff-bench check --baseline auto --path . --json\nfff-bench run --path . --suite public-v1 -n 5 --json\n```\n\nSee `fff-bench --help` for sensor packs, suites, and load profiles.\n",
    );
    Ok(manifest)
}

pub fn calibrate() -> CalibrateOut {
    use std::time::Instant;
    // Clock resolution: tight spin measuring Instant deltas.
    let mut min_ns = u64::MAX;
    for _ in 0..10_000 {
        let t0 = Instant::now();
        let dt = t0.elapsed().as_nanos() as u64;
        if dt > 0 && dt < min_ns {
            min_ns = dt;
        }
    }
    if min_ns == u64::MAX {
        min_ns = 1;
    }
    // Snapshot overhead: empty sensor snap pair wall.
    let sensors = crate::sensors::Sensors::from_packs("mem,cpu");
    let mut overs = Vec::new();
    for _ in 0..50 {
        let t0 = Instant::now();
        let a = sensors.snapshot();
        let b = sensors.snapshot();
        let _ = sensors.delta(&a, &b);
        overs.push(t0.elapsed().as_micros() as f64);
    }
    let overhead = overs.iter().sum::<f64>() / overs.len() as f64;
    let busy = crate::noise::host_cpu_busy_pct();
    let sensor_packs = crate::sensors::probe_packs();
    let packs_digest: String = sensor_packs
        .iter()
        .map(|p| format!("{}:{}", p.pack, p.available as u8))
        .collect::<Vec<_>>()
        .join(",");
    let digest = crate::identity::sha256_hex(
        format!("clock={min_ns}|overhead_us={overhead:.3}|busy={busy:?}|packs={packs_digest}")
            .as_bytes(),
    );
    let avail: Vec<&str> = sensor_packs
        .iter()
        .filter(|p| p.available)
        .map(|p| p.pack.as_str())
        .collect();
    CalibrateOut {
        status: "complete".into(),
        clock_resolution_ns: min_ns.max(1),
        snapshot_overhead_us: overhead,
        host_cpu_busy_pct: busy,
        digest: format!("sha256:{digest}"),
        summary: format!(
            "clock_resolution_ns≈{min_ns} snapshot_overhead_us≈{overhead:.1} host_cpu={busy:?} sensors=[{}]",
            avail.join(",")
        ),
        sensor_packs,
    }
}

pub fn current_evaluator() -> evaluator::EvaluatorIdentity {
    evaluator::current("fff")
}
