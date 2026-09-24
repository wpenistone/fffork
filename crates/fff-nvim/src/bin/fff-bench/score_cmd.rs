//! Constrained agent score (lower is better) with family breakdowns.

use crate::runner::BenchReport;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyBreakdown {
    pub family: String,
    pub cells: usize,
    pub wall_p95_mean: f64,
    pub wall_p50_mean: f64,
    pub throughput_qps_mean: Option<f64>,
    /// Contribution weight used in combined score (0 if quality-only).
    pub weight: f64,
    /// Family-local score vs baseline (1.0 = parity); None if no baseline.
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreOut {
    pub schema_version: u32,
    pub suite: String,
    pub score_status: String,
    pub score: Option<f64>,
    pub score_baseline: Option<f64>,
    pub score_delta: Option<f64>,
    pub lower_is_better: bool,
    pub admissible: bool,
    pub comparable: bool,
    pub integrity: String,
    pub summary_for_agent: String,
    #[serde(default)]
    pub breakdown: Vec<FamilyBreakdown>,
    /// Typical path: mean p95 wall of `search` family (interactive latency).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typical_wall_p95_us: Option<f64>,
    /// Typical throughput: mean burst qps when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typical_throughput_qps: Option<f64>,
}

fn suite_compatible(a: &str, b: &str) -> bool {
    fn n(s: &str) -> String {
        match s {
            "agent-loop" | "agent-screen" | "screen" => "agent-screen".into(),
            "agent-confirm" | "confirm" => "agent-confirm".into(),
            "peak" | "peak-frontier" | "peak-diagnose" | "diagnose" => "peak-diagnose".into(),
            x => x.into(),
        }
    }
    let na = n(a);
    let nb = n(b);
    na == nb
        || (na == "agent-screen" && nb == "agent-confirm")
        || (na == "agent-confirm" && nb == "agent-screen")
}

/// Family weights for combined score (must sum meaningfully; missing families skipped).
fn family_weight(family: &str) -> f64 {
    match family {
        "quality" => 0.0, // gates only — hard admissible
        "search" => 1.0,
        "call" => 0.55,
        "pagination" => 0.25,
        "constraint" => 0.35,
        "grep" => 0.4,
        "throughput" => 0.5,
        "typing" => 0.55,
        "b2b" => 0.5,
        "interleave" => 0.5,
        "load" => 0.45,
        "cold" => 0.6,
        "edge" => 0.2,
        "combo" => 0.35,
        "unique" => 0.3,
        "stability" => 0.4,
        "stress" => 0.35,
        _ => 0.3,
    }
}

pub fn compute(report: &BenchReport, baseline: Option<&BenchReport>) -> ScoreOut {
    let suite = report.metadata.run.suite.clone();
    let admissible = report.admissible;
    let integrity = report.noise.integrity.clone();

    if !admissible {
        return ScoreOut {
            schema_version: 1,
            suite,
            score_status: "inadmissible".into(),
            score: None,
            score_baseline: None,
            score_delta: None,
            lower_is_better: true,
            admissible: false,
            comparable: false,
            integrity,
            summary_for_agent: format!(
                "gates failed: {:?} — score null",
                report.gates.failed
            ),
            breakdown: vec![],
            typical_wall_p95_us: None,
            typical_throughput_qps: None,
        };
    }

    // Diagnostic walls always computed (even when score is withheld for noise).
    let typical_wall = mean_p95_family(report, "search")
        .or_else(|| mean_p95_any_cost(report));
    let typical_tp = mean_throughput(report);

    if integrity == "noisy" {
        return ScoreOut {
            schema_version: 1,
            suite,
            score_status: "noisy".into(),
            score: None,
            score_baseline: None,
            score_delta: None,
            lower_is_better: true,
            admissible: true,
            comparable: false,
            integrity,
            summary_for_agent: format!(
                "host/iter noise — do not accept as win; diagnostic typical_wall_p95={:?}µs host_cpu={:?}",
                typical_wall,
                report.noise.host_cpu_busy_pct
            ),
            breakdown: build_breakdown(report, None),
            typical_wall_p95_us: typical_wall,
            typical_throughput_qps: typical_tp,
        };
    }

    // Drop baseline if suite differs (true incomparable) or n differs (use self-norm).
    let baseline = baseline.and_then(|b| {
        if !suite_compatible(&b.metadata.run.suite, &report.metadata.run.suite) {
            None
        } else if b.metadata.stats.iterations != report.metadata.stats.iterations {
            None
        } else {
            Some(b)
        }
    });
    let comparable = true;

    // All cost cells except pure quality gates (gates already in admissible)
    let cost_cells: Vec<_> = report
        .cells
        .iter()
        .filter(|c| c.family != "quality" && c.cost.wall_us.n > 0)
        .collect();

    if cost_cells.is_empty() {
        return ScoreOut {
            schema_version: 1,
            suite,
            score_status: "partial".into(),
            score: None,
            score_baseline: None,
            score_delta: None,
            lower_is_better: true,
            admissible: true,
            comparable,
            integrity,
            summary_for_agent: "no cost cells to score".into(),
            breakdown: vec![],
            typical_wall_p95_us: typical_wall,
            typical_throughput_qps: typical_tp,
        };
    }

    let breakdown = build_breakdown(report, baseline);

    if baseline.is_none() {
        return ScoreOut {
            schema_version: 1,
            suite,
            score_status: "ok".into(),
            score: Some(1.0),
            score_baseline: None,
            score_delta: None,
            lower_is_better: true,
            admissible: true,
            comparable: true,
            integrity,
            summary_for_agent: format!(
                "score 1.0000 (no baseline); families={}; typical_wall_p95={:?}µs",
                breakdown.len(),
                typical_wall
            ),
            breakdown,
            typical_wall_p95_us: typical_wall,
            typical_throughput_qps: typical_tp,
        };
    }

    let mut score = 0.0;
    let mut weight_sum = 0.0;
    for c in &cost_cells {
        let wall = c.cost.wall_us.p95;
        let ref_wall = baseline
            .and_then(|b| {
                b.cells
                    .iter()
                    .find(|x| x.cell_key == c.cell_key)
                    .map(|x| x.cost.wall_us.p95)
            })
            .filter(|v| *v > 0.0)
            .unwrap_or(wall.max(1.0));
        let fw = family_weight(&c.family);
        if fw <= 0.0 {
            continue;
        }
        score += fw * (wall / ref_wall);
        weight_sum += fw;

        if let Some(ref cpu) = c.cost.cpu_total_us {
            if cpu.p95 > 0.0 {
                let ref_cpu = baseline
                    .and_then(|b| {
                        b.cells
                            .iter()
                            .find(|x| x.cell_key == c.cell_key)
                            .and_then(|x| x.cost.cpu_total_us.as_ref().map(|d| d.p95))
                    })
                    .filter(|v| *v > 0.0)
                    .unwrap_or(cpu.p95);
                let w = fw * 0.15;
                score += w * (cpu.p95 / ref_cpu);
                weight_sum += w;
            }
        }
    }

    let score = if weight_sum > 0.0 {
        score / weight_sum
    } else {
        1.0
    };

    let mut capped = false;
    if let Some(b) = baseline {
        for c in &cost_cells {
            if let (Some(cpu), Some(bc)) = (
                c.cost.cpu_total_us.as_ref(),
                b.cells
                    .iter()
                    .find(|x| x.cell_key == c.cell_key)
                    .and_then(|x| x.cost.cpu_total_us.as_ref()),
            ) {
                if bc.p95 > 0.0 && cpu.p95 / bc.p95 > 1.5 {
                    capped = true;
                }
            }
        }
    }

    let score_baseline = baseline.map(|_| 1.0);
    let score_delta = score_baseline.map(|b| score - b);

    if capped && score_delta.map(|d| d < 0.0).unwrap_or(false) {
        return ScoreOut {
            schema_version: 1,
            suite,
            score_status: "capped".into(),
            score: Some(score),
            score_baseline,
            score_delta,
            lower_is_better: true,
            admissible: true,
            comparable: true,
            integrity,
            summary_for_agent: format!(
                "score {score:.4} but CPU +50% cap hit — reject free-lunch wall win"
            ),
            breakdown,
            typical_wall_p95_us: typical_wall,
            typical_throughput_qps: typical_tp,
        };
    }

    let summary = match score_delta {
        Some(d) if d < -0.01 => format!(
            "score {score:.4} (Δ {d:.4}) improvement; typical_wall_p95={typical_wall:?}µs"
        ),
        Some(d) if d > 0.01 => {
            format!("score {score:.4} (Δ {d:.4}) regression; typical_wall_p95={typical_wall:?}µs")
        }
        Some(_) => format!("score {score:.4} ≈ baseline; typical_wall_p95={typical_wall:?}µs"),
        None => format!("score {score:.4}; typical_wall_p95={typical_wall:?}µs"),
    };

    ScoreOut {
        schema_version: 1,
        suite,
        score_status: "ok".into(),
        score: Some(score),
        score_baseline,
        score_delta,
        lower_is_better: true,
        admissible: true,
        comparable: baseline.is_none() || comparable,
        integrity,
        summary_for_agent: summary,
        breakdown,
        typical_wall_p95_us: typical_wall,
        typical_throughput_qps: typical_tp,
    }
}

fn mean_p95_family(report: &BenchReport, family: &str) -> Option<f64> {
    let vals: Vec<f64> = report
        .cells
        .iter()
        .filter(|c| c.family == family && c.cost.wall_us.n > 0)
        .map(|c| c.cost.wall_us.p95)
        .collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

/// Mean p95 wall across any non-empty cost cell (fallback when no search family).
fn mean_p95_any_cost(report: &BenchReport) -> Option<f64> {
    let vals: Vec<f64> = report
        .cells
        .iter()
        .filter(|c| c.cost.wall_us.n > 0 && c.family != "quality")
        .map(|c| c.cost.wall_us.p95)
        .collect();
    if vals.is_empty() {
        // Include quality/gate cells if that is all we have (e.g. pure public-v1 gates).
        let all: Vec<f64> = report
            .cells
            .iter()
            .filter(|c| c.cost.wall_us.n > 0)
            .map(|c| c.cost.wall_us.p95)
            .collect();
        if all.is_empty() {
            None
        } else {
            Some(all.iter().sum::<f64>() / all.len() as f64)
        }
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

fn mean_throughput(report: &BenchReport) -> Option<f64> {
    let vals: Vec<f64> = report
        .cells
        .iter()
        .filter_map(|c| c.throughput_qps)
        .collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

fn build_breakdown(report: &BenchReport, baseline: Option<&BenchReport>) -> Vec<FamilyBreakdown> {
    let mut by: BTreeMap<String, Vec<&crate::runner::CellReport>> = BTreeMap::new();
    for c in &report.cells {
        if c.cost.wall_us.n == 0 {
            continue;
        }
        by.entry(c.family.clone()).or_default().push(c);
    }
    let mut out = Vec::new();
    for (family, cells) in by {
        let n = cells.len() as f64;
        let wall_p95_mean = cells.iter().map(|c| c.cost.wall_us.p95).sum::<f64>() / n;
        let wall_p50_mean = cells.iter().map(|c| c.cost.wall_us.p50).sum::<f64>() / n;
        let tps: Vec<f64> = cells.iter().filter_map(|c| c.throughput_qps).collect();
        let throughput_qps_mean = if tps.is_empty() {
            None
        } else {
            Some(tps.iter().sum::<f64>() / tps.len() as f64)
        };
        let weight = family_weight(&family);
        let score = baseline.map(|b| {
            let mut s = 0.0;
            let mut wsum = 0.0;
            for c in &cells {
                let ref_wall = b
                    .cells
                    .iter()
                    .find(|x| x.cell_key == c.cell_key)
                    .map(|x| x.cost.wall_us.p95)
                    .filter(|v| *v > 0.0)
                    .unwrap_or(c.cost.wall_us.p95.max(1.0));
                s += c.cost.wall_us.p95 / ref_wall;
                wsum += 1.0;
            }
            if wsum > 0.0 {
                s / wsum
            } else {
                1.0
            }
        });
        out.push(FamilyBreakdown {
            family,
            cells: cells.len(),
            wall_p95_mean,
            wall_p50_mean,
            throughput_qps_mean,
            weight,
            score,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{CostSample, CostSummary, Dist};
    use crate::gates::GatesReport;
    use crate::identity::{
        ChangeMeta, HarnessMeta, MachineMeta, NoiseMeta, RepoMeta, RunMeta, StatsMeta, SubjectMeta,
    };
    use crate::runner::{BenchReport, CellReport, ReportMetadata};

    fn empty_gates(passed: bool) -> GatesReport {
        GatesReport {
            passed,
            failed: if passed {
                vec![]
            } else {
                vec!["x".into()]
            },
            hit_at_1: 1.0,
            hit_at_5: 1.0,
            mrr: 1.0,
            cases: vec![],
            invariants_ok: true,
            entries: vec![],
        }
    }

    fn report_with_wall(suite: &str, wall_p95: f64, admissible: bool) -> BenchReport {
        let mut sample = CostSample::with_wall(wall_p95 as u64, 4);
        sample.finalize_derived();
        let cost = CostSummary {
            wall_us: Dist {
                n: 5,
                min: wall_p95,
                p50: wall_p95,
                p90: wall_p95,
                p95: wall_p95,
                p99: wall_p95,
                max: wall_p95,
                mean: wall_p95,
                stddev: 0.0,
                sum: wall_p95 * 5.0,
                mad: 0.0,
                relative_mad: 0.0,
            },
            threads_configured: 4,
            primary_span: "harness.op_total".into(),
            qps_derived: 1000.0,
            ..Default::default()
        };
        BenchReport {
            schema_version: 1,
            admissible,
            metadata: ReportMetadata {
                run: RunMeta {
                    run_id: "t1".into(),
                    cmdline: "test".into(),
                    started_at_unix: 0,
                    finished_at_unix: Some(1),
                    duration_ms: Some(1),
                    log_dir: None,
                    peak: false,
                    sample_packs: "mem,cpu".into(),
                    primary_span: "harness.op_total".into(),
                    suite: suite.into(),
                },
                subject: SubjectMeta {
                    adapter_id: "fff".into(),
                    git_commit: None,
                    git_dirty: false,
                    version: "0".into(),
                    cargo_profile: "test".into(),
                },
                harness: HarnessMeta {
                    name: "fff-bench".into(),
                    git_commit: None,
                    version: "0".into(),
                },
                machine: MachineMeta {
                    hostname: "t".into(),
                    os: "test".into(),
                    arch: "test".into(),
                    cpu_count: 1,
                    total_ram_mb: 1,
                },
                repos: vec![RepoMeta {
                    id: "default".into(),
                    path: ".".into(),
                    role: vec![],
                    optional: false,
                    file_count_live: Some(1),
                    path_list_hash: None,
                    git_commit: None,
                    git_dirty: false,
                    pin: "path".into(),
                }],
                stats: StatsMeta {
                    warmup: 1,
                    iterations: 5,
                    threads: 4,
                    rng_seed: 1,
                    primary_span: "harness.op_total".into(),
                },
            },
            gates: empty_gates(admissible),
            cells: vec![CellReport {
                cell_key: "profile.search.matrix\0default\0{}".into(),
                scenario_id: "profile.search.matrix".into(),
                repo_id: "default".into(),
                axes: serde_json::json!({}),
                kind: "Search".into(),
                family: "search".into(),
                tags: vec!["search".into()],
                cost,
                last_result: Default::default(),
                rank_target: None,
                iterations_logged: 5,
                throughput_qps: None,
                throughput_qps_dist: None,
                request_wall_us: None,
                operations_per_iteration: None,
                client_concurrency: None,
                arrival_model: None,
                request_errors: None,
                ordered_digest_consistent: true,
                state_id: Some("search_reused_index_warm".into()),
                warmup_converged: Some(true),
                warmups_run: Some(3),
                last_metrics: None,
            }],
            noise: NoiseMeta {
                integrity: "clean".into(),
                host_cpu_busy_pct: Some(5.0),
                background_cpu_pct: None,
                iter_wall_cv: Some(0.1),
                policy: "allow".into(),
            },
            change: Some(ChangeMeta {
                kind: "none".into(),
                base_ref: None,
                head_commit: None,
                dirty: false,
                patch_path: None,
                patch_sha256: None,
                patch_bytes: None,
                summary: None,
            }),
            score: None,
            profile: None,
            evaluator: None,
            plan_digest: None,
            stability: None,
            mutation: None,
            metric_coverage: None,
            agent_usage: None,
            treatment: None,
            network: None,
        }
    }

    #[test]
    fn inadmissible_null_score() {
        let r = report_with_wall("agent-loop", 100.0, false);
        let s = compute(&r, None);
        assert_eq!(s.score_status, "inadmissible");
        assert!(s.score.is_none());
    }

    #[test]
    fn noisy_keeps_diagnostic_walls() {
        let mut r = report_with_wall("agent-screen", 250.0, true);
        r.noise.integrity = "noisy".into();
        r.noise.host_cpu_busy_pct = Some(70.0);
        let s = compute(&r, None);
        assert_eq!(s.score_status, "noisy");
        assert!(s.score.is_none());
        assert!(s.typical_wall_p95_us.is_some());
        assert!(s.typical_wall_p95_us.unwrap() > 0.0);
        assert!(s.summary_for_agent.contains("noise"));
    }

    #[test]
    fn no_baseline_self_normalized_one() {
        let r = report_with_wall("agent-loop", 500.0, true);
        let s = compute(&r, None);
        assert_eq!(s.score_status, "ok");
        assert_eq!(s.score, Some(1.0));
        assert!(!s.breakdown.is_empty());
        assert!(s.typical_wall_p95_us.is_some());
    }

    #[test]
    fn baseline_lower_wall_is_improvement() {
        let base = report_with_wall("agent-loop", 200.0, true);
        let mut cur = report_with_wall("agent-loop", 100.0, true);
        cur.metadata.run.run_id = "t2".into();
        let s = compute(&cur, Some(&base));
        assert_eq!(s.score_status, "ok");
        let score = s.score.unwrap();
        assert!(score < 1.0, "score={score}");
        assert!(s.score_delta.unwrap() < 0.0);
    }

    #[test]
    fn score_exit_codes() {
        assert_eq!(crate::runner::score_exit_code("ok"), 0);
        assert_eq!(crate::runner::score_exit_code("partial"), 0);
        assert_eq!(crate::runner::score_exit_code("noisy"), 4);
        assert_eq!(crate::runner::score_exit_code("unstable"), 4);
        assert_eq!(crate::runner::score_exit_code("inadmissible"), 1);
        assert_eq!(crate::runner::score_exit_code("capped"), 1);
        assert_eq!(crate::runner::score_exit_code("incomparable"), 3);
    }
}
