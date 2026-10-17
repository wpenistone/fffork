//! Compare two reports (COMPARABILITY + METHODOLOGY decision layers).

use crate::bootstrap::{self, RatioCi};
use crate::runner::BenchReport;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareOut {
    pub schema_version: u32,
    pub comparable: bool,
    pub admissible_current: bool,
    pub admissible_baseline: bool,
    /// improve|equivalent|mixed|regress|inadmissible|incomparable|unstable|promising|screen_no_signal
    pub verdict: String,
    pub summary_for_agent: String,
    #[serde(default)]
    pub reason_codes: Vec<String>,
    pub cell_deltas: Vec<CellDelta>,
    /// Relative wall change: mean of overlapping cells (negative = faster).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_p95_rel_mean: Option<f64>,
    /// evidence_grade of the comparison when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_grade: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_ratio_ci95: Option<RatioCi>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caps_failed: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellDelta {
    pub cell_key: String,
    pub wall_us_p95_rel: Option<f64>,
    pub wall_us_p95_abs: Option<f64>,
    pub cpu_total_us_p95_rel: Option<f64>,
    /// Process cycle delta (Windows QueryProcessCycleTime); drift-immune vs wall.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_cycles_p95_rel: Option<f64>,
    pub rank_target_base: Option<usize>,
    pub rank_target_cur: Option<usize>,
    pub quality_regressed: bool,
    /// pass | fail | skip
    #[serde(default = "cap_pass")]
    pub cap: String,
}

fn cap_pass() -> String {
    "pass".into()
}

/// Materiality thresholds (relative wall p95).
const IMPROVE_THRESH: f64 = -0.05; // 5% faster
const REGRESS_THRESH: f64 = 0.10; // 10% slower
const EQUIV_BAND: f64 = 0.05; // within ±5% → equivalent
const MATERIALITY_DELTA: f64 = 0.05;
const BOOTSTRAP_N: usize = 2_000; // lighter than 10k for agent loop; seed recorded
const BOOTSTRAP_SEED: u64 = 0xFFF_BEEF;

/// Screen "promising" needs clearer wall signal than confirm.
const SCREEN_PROMISING: f64 = -0.08;

/// Per-cell wall regression cap (relative). Search family stricter than stress.
fn cell_cap_rel(family: &str) -> f64 {
    match family {
        "search" | "call" | "typing" => 0.20,
        "grep" => 0.30,
        "cold" => 0.40,
        _ => 0.50,
    }
}

/// Absolute floor for caps: cells faster than this are timer-noise
/// dominated (scheduling jitter is tens of us), so a relative
/// regression there is not evidence. Must also exceed the rel cap.
const CELL_CAP_ABS_FLOOR_US: f64 = 250.0;

pub fn compare(baseline: &BenchReport, current: &BenchReport) -> CompareOut {
    compare_with_grade(baseline, current, "confirm")
}

pub fn compare_screen(baseline: &BenchReport, current: &BenchReport) -> CompareOut {
    compare_with_grade(baseline, current, "screen")
}

fn compare_with_grade(baseline: &BenchReport, current: &BenchReport, grade: &str) -> CompareOut {
    let mut reasons = Vec::new();
    let mut comparable = true;

    // Suite mismatch: allow screen vs confirm only for directional screen grade.
    let suite_match = baseline.metadata.run.suite == current.metadata.run.suite
        || suites_compatible(&baseline.metadata.run.suite, &current.metadata.run.suite);
    if !suite_match {
        comparable = false;
        reasons.push("suite_mismatch".into());
    }

    // Evaluator drift is never a product win (COMPARABILITY).
    match (&baseline.evaluator, &current.evaluator) {
        (Some(a), Some(b)) if a.evaluator_digest != b.evaluator_digest => {
            comparable = false;
            reasons.push("evaluator_drift".into());
        }
        (Some(_), None) | (None, Some(_)) => {
            // Legacy reports without digests: allow informational compare.
            reasons.push("evaluator_identity_partial".into());
        }
        _ => {}
    }

    if let (Some(a), Some(b)) = (&baseline.plan_digest, &current.plan_digest) {
        if a != b && grade != "screen" {
            // Screen may use a subset plan vs confirm baseline.
            comparable = false;
            reasons.push("plan_digest_mismatch".into());
        }
    }

    // Treatment validity + paired delivery equivalence (controlled load).
    if let Some(t) = &current.treatment {
        if !t.valid {
            comparable = false;
            reasons.push("treatment_delivery_invalid".into());
        }
        if let Some(bt) = &baseline.treatment {
            let (equiv, mut eq_reasons) = crate::load_actor::treatments_delivery_equivalent(bt, t);
            if !equiv {
                comparable = false;
                reasons.append(&mut eq_reasons);
            }
        }
    }

    // Network environment: external/shared is informational only.
    if let Some(n) = &current.network {
        if n.decision_grade == "informational" {
            comparable = false;
            reasons.push("network_informational_only".into());
        }
        if !n.valid {
            comparable = false;
            reasons.push("network_treatment_invalid".into());
        }
        if let Some(bn) = &baseline.network {
            if bn.profile_id != n.profile_id {
                comparable = false;
                reasons.push("network_profile_mismatch".into());
            }
        }
    }

    if current
        .stability
        .as_ref()
        .map(|s| s.stability == "unstable")
        .unwrap_or(false)
        || baseline
            .stability
            .as_ref()
            .map(|s| s.stability == "unstable")
            .unwrap_or(false)
    {
        let mut out = bare_verdict(
            "unstable",
            comparable,
            current.admissible,
            baseline.admissible,
            "stability rollup is unstable",
            {
                let mut r = reasons.clone();
                r.push("unstable_dispersion_or_host".into());
                r
            },
            grade,
        );
        out.cell_deltas = cell_deltas(baseline, current, &mut comparable);
        return out;
    }

    if baseline.noise.integrity == "noisy" || current.noise.integrity == "noisy" {
        let mut out = bare_verdict(
            "unstable",
            false,
            current.admissible,
            baseline.admissible,
            "host noise prevents a decision",
            vec!["unstable_host_noise".into()],
            grade,
        );
        out.cell_deltas = cell_deltas(baseline, current, &mut comparable);
        return out;
    }

    if !current.admissible {
        let mut out = bare_verdict(
            "inadmissible",
            comparable,
            false,
            baseline.admissible,
            &format!("gates failed: {:?}", current.gates.failed),
            vec!["gates_failed".into()],
            grade,
        );
        out.cell_deltas = cell_deltas(baseline, current, &mut comparable);
        return out;
    }

    if !baseline.admissible {
        reasons.push("baseline_inadmissible".into());
    }

    let deltas = cell_deltas(baseline, current, &mut comparable);
    if deltas.is_empty() && comparable {
        // no overlapping cost cells — still may be gates-only
        reasons.push("no_overlapping_cost_cells".into());
    }

    let quality_regressed = deltas.iter().any(|d| d.quality_regressed);
    if quality_regressed {
        reasons.push("quality_regressed".into());
        return bare_with_deltas(
            "regress",
            comparable,
            current.admissible,
            baseline.admissible,
            "quality regressed on at least one cell",
            reasons,
            deltas,
            None,
            grade,
            None,
            None,
        );
    }

    let caps_failed = deltas.iter().filter(|d| d.cap == "fail").count();
    if caps_failed > 0 && grade != "screen" {
        reasons.push("per_cell_cap_failed".into());
        return bare_with_deltas(
            "regress",
            comparable,
            current.admissible,
            baseline.admissible,
            &format!("{caps_failed} cells failed per-cell wall caps"),
            reasons,
            deltas,
            None,
            grade,
            None,
            Some(caps_failed),
        );
    }

    if !comparable {
        reasons.push("incomparable_cells_or_suite".into());
        return bare_with_deltas(
            "incomparable",
            false,
            current.admissible,
            baseline.admissible,
            "baseline and candidate are not comparable",
            reasons,
            deltas,
            None,
            grade,
            None,
            None,
        );
    }

    // Metric coverage mismatch cannot be a win.
    if grade != "screen" {
        if let (Some(a), Some(b)) = (&baseline.metric_coverage, &current.metric_coverage) {
            if a.matched && !b.matched {
                reasons.push("metric_coverage_missing".into());
                return bare_with_deltas(
                    "incomparable",
                    false,
                    current.admissible,
                    baseline.admissible,
                    "candidate missing required metrics present in baseline",
                    reasons,
                    deltas,
                    None,
                    grade,
                    None,
                    None,
                );
            }
        }
    }

    let rels: Vec<f64> = deltas.iter().filter_map(|d| d.wall_us_p95_rel).collect();
    let wall_mean = if rels.is_empty() {
        None
    } else {
        Some(rels.iter().sum::<f64>() / rels.len() as f64)
    };
    // Cycle delta is drift-immune; reported alongside wall when the pack is on.
    let cyc_rels: Vec<f64> = deltas.iter().filter_map(|d| d.cpu_cycles_p95_rel).collect();
    let cycles_mean = if cyc_rels.is_empty() {
        None
    } else {
        Some(cyc_rels.iter().sum::<f64>() / cyc_rels.len() as f64)
    };

    // Bootstrap on p95 pairs (history baseline acts as paired observation proxy).
    let pairs: Vec<(f64, f64)> = deltas
        .iter()
        .filter_map(|d| {
            let b = baseline.cells.iter().find(|c| c.cell_key == d.cell_key)?;
            let c = current.cells.iter().find(|c| c.cell_key == d.cell_key)?;
            let bv = b.cost.wall_us.p95;
            let cv = c.cost.wall_us.p95;
            if bv > 0.0 && cv > 0.0 {
                Some((bv, cv))
            } else {
                None
            }
        })
        .collect();
    let wall_ci = if grade == "confirm" && pairs.len() >= 3 {
        bootstrap::ratio_ci95(&pairs, BOOTSTRAP_SEED, BOOTSTRAP_N)
    } else {
        None
    };

    let improved = rels.iter().filter(|r| **r < IMPROVE_THRESH).count();
    let regressed = rels.iter().filter(|r| **r > REGRESS_THRESH).count();
    let all_near_zero = rels.iter().all(|r| r.abs() <= EQUIV_BAND);

    let mut verdict = if grade == "screen" {
        screen_verdict(wall_mean, improved, regressed, &mut reasons)
    } else {
        confirm_verdict(wall_mean, improved, regressed, all_near_zero, &mut reasons)
    };

    // CI can override point-estimate improve/regress when decisive.
    if let Some(ref ci) = wall_ci {
        let ci_v = bootstrap::classify_ratio(ci, MATERIALITY_DELTA);
        reasons.push(format!("bootstrap_ci_{ci_v}"));
        if grade != "screen" {
            match (verdict, ci_v) {
                ("improve", "regress") | ("regress", "improve") => {
                    verdict = "mixed";
                    reasons.push("point_ci_disagreement".into());
                }
                ("improve", "equivalent") | ("equivalent", "improve") => {
                    // Prefer CI for decision-grade
                    if ci_v == "equivalent" {
                        verdict = "equivalent";
                    }
                }
                (_, "improve") if verdict == "mixed" || verdict == "equivalent" => {
                    if ci.high < 1.0 - MATERIALITY_DELTA {
                        verdict = "improve";
                    }
                }
                (_, "regress") => {
                    if ci.low > 1.0 + MATERIALITY_DELTA {
                        verdict = "regress";
                    }
                }
                _ => {}
            }
        }
    }

    let summary = format!(
        "verdict={verdict}; wall_p95_rel_mean={wall_mean:?}; cycles_p95_rel_mean={cycles_mean:?}; improved_cells={improved}; regressed_cells={regressed}; caps_failed={caps_failed}; grade={grade}; ci={:?}",
        wall_ci.as_ref().map(|c| (c.low, c.high, c.point))
    );

    bare_with_deltas(
        verdict,
        true,
        current.admissible,
        baseline.admissible,
        &summary,
        reasons,
        deltas,
        wall_mean,
        grade,
        wall_ci,
        Some(caps_failed),
    )
}

fn suite_norm(s: &str) -> String {
    match s {
        "agent-loop" | "agent-screen" | "screen" => "agent-screen".into(),
        "agent-confirm" | "confirm" => "agent-confirm".into(),
        "peak" | "peak-frontier" | "peak-diagnose" | "diagnose" => "peak-diagnose".into(),
        x => x.into(),
    }
}

fn suites_compatible(a: &str, b: &str) -> bool {
    let na = suite_norm(a);
    let nb = suite_norm(b);
    na == nb
        || (na == "agent-screen" && nb == "agent-confirm")
        || (na == "agent-confirm" && nb == "agent-screen")
}

fn screen_verdict(
    wall_mean: Option<f64>,
    improved: usize,
    regressed: usize,
    reasons: &mut Vec<String>,
) -> &'static str {
    if regressed > 0 && improved == 0 {
        reasons.push("screen_regress_signal".into());
        return "regress";
    }
    if let Some(m) = wall_mean {
        if m <= SCREEN_PROMISING && regressed == 0 {
            reasons.push("screen_promising_wall".into());
            return "promising";
        }
    }
    if improved > 0 && regressed == 0 {
        reasons.push("screen_promising_cells".into());
        return "promising";
    }
    reasons.push("screen_no_promising_signal".into());
    "screen_no_signal"
}

fn confirm_verdict(
    wall_mean: Option<f64>,
    improved: usize,
    regressed: usize,
    all_near_zero: bool,
    reasons: &mut Vec<String>,
) -> &'static str {
    if regressed > 0 && improved == 0 {
        reasons.push("confirmation_material_regression".into());
        return "regress";
    }
    if improved > 0 && regressed > 0 {
        reasons.push("mixed_cell_deltas".into());
        return "mixed";
    }
    if improved > 0 && regressed == 0 {
        if let Some(m) = wall_mean {
            if m <= IMPROVE_THRESH {
                reasons.push("confirmation_material_improvement".into());
                reasons.push("gates_passed".into());
                return "improve";
            }
        }
        reasons.push("confirmation_material_improvement".into());
        reasons.push("gates_passed".into());
        return "improve";
    }
    if all_near_zero || wall_mean.map(|m| m.abs() <= EQUIV_BAND).unwrap_or(true) {
        reasons.push("no_material_difference".into());
        reasons.push("gates_passed".into());
        return "equivalent";
    }
    if let Some(m) = wall_mean {
        if m > REGRESS_THRESH {
            reasons.push("confirmation_material_regression".into());
            return "regress";
        }
        if m < IMPROVE_THRESH {
            reasons.push("confirmation_material_improvement".into());
            return "improve";
        }
    }
    reasons.push("no_material_difference".into());
    "equivalent"
}

fn cell_deltas(
    baseline: &BenchReport,
    current: &BenchReport,
    comparable: &mut bool,
) -> Vec<CellDelta> {
    let mut deltas = Vec::new();
    for c in &current.cells {
        if c.scenario_id.starts_with("gate") || c.family == "quality" {
            continue;
        }
        let b = baseline.cells.iter().find(|x| x.cell_key == c.cell_key);
        let (wall_rel, wall_abs, cpu_rel, cyc_rel) = if let Some(b) = b {
            let wr = if b.cost.wall_us.p95 > 0.0 {
                Some((c.cost.wall_us.p95 - b.cost.wall_us.p95) / b.cost.wall_us.p95)
            } else {
                None
            };
            let wa = Some(c.cost.wall_us.p95 - b.cost.wall_us.p95);
            let cr = match (c.cost.cpu_total_us.as_ref(), b.cost.cpu_total_us.as_ref()) {
                (Some(cc), Some(bc)) if bc.p95 > 0.0 => Some((cc.p95 - bc.p95) / bc.p95),
                _ => None,
            };
            let cy = match (c.cost.cpu_cycles.as_ref(), b.cost.cpu_cycles.as_ref()) {
                (Some(cc), Some(bc)) if bc.p95 > 0.0 => Some((cc.p95 - bc.p95) / bc.p95),
                _ => None,
            };
            (wr, wa, cr, cy)
        } else {
            // Missing baseline cell: for screen grade we skip; for strict confirm mark incomplete.
            continue;
        };

        let quality_regressed = match (b.and_then(|x| x.rank_target), c.rank_target) {
            (Some(br), Some(cr)) => cr > br,
            (Some(_), None) => true,
            _ => false,
        };

        let cap = match (wall_rel, wall_abs) {
            (Some(r), Some(a))
                if r > cell_cap_rel(&c.family) && a > CELL_CAP_ABS_FLOOR_US =>
            {
                "fail".into()
            }
            (Some(_), _) => "pass".into(),
            _ => "skip".into(),
        };

        deltas.push(CellDelta {
            cell_key: c.cell_key.clone(),
            wall_us_p95_rel: wall_rel,
            wall_us_p95_abs: wall_abs,
            cpu_total_us_p95_rel: cpu_rel,
            cpu_cycles_p95_rel: cyc_rel,
            rank_target_base: b.and_then(|x| x.rank_target),
            rank_target_cur: c.rank_target,
            quality_regressed,
            cap,
        });
    }

    // If current has cost cells but zero overlap, incomparable.
    let cur_cost_cells = current
        .cells
        .iter()
        .filter(|c| !c.scenario_id.starts_with("gate") && c.family != "quality")
        .count();
    if cur_cost_cells > 0 && deltas.is_empty() {
        *comparable = false;
    }
    deltas
}

fn bare_verdict(
    verdict: &str,
    comparable: bool,
    admissible_current: bool,
    admissible_baseline: bool,
    summary: &str,
    reasons: Vec<String>,
    grade: &str,
) -> CompareOut {
    bare_with_deltas(
        verdict,
        comparable,
        admissible_current,
        admissible_baseline,
        summary,
        reasons,
        vec![],
        None,
        grade,
        None,
        None,
    )
}

fn bare_with_deltas(
    verdict: &str,
    comparable: bool,
    admissible_current: bool,
    admissible_baseline: bool,
    summary: &str,
    reasons: Vec<String>,
    deltas: Vec<CellDelta>,
    wall_mean: Option<f64>,
    grade: &str,
    wall_ci: Option<RatioCi>,
    caps_failed: Option<usize>,
) -> CompareOut {
    CompareOut {
        schema_version: 1,
        comparable,
        admissible_current,
        admissible_baseline,
        verdict: verdict.into(),
        summary_for_agent: summary.into(),
        reason_codes: reasons,
        cell_deltas: deltas,
        wall_p95_rel_mean: wall_mean,
        evidence_grade: Some(grade.into()),
        wall_ratio_ci95: wall_ci,
        caps_failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{CostSummary, Dist};
    use crate::gates::GatesReport;
    use crate::identity::*;
    use crate::runner::{BenchReport, CellReport, ReportMetadata};

    fn bare(suite: &str, run_id: &str, wall: f64, admissible: bool) -> BenchReport {
        let cost = CostSummary {
            wall_us: Dist {
                n: 3,
                min: wall,
                p50: wall,
                p90: wall,
                p95: wall,
                p99: wall,
                max: wall,
                mean: wall,
                stddev: 0.0,
                sum: wall * 3.0,
                mad: 0.0,
                relative_mad: 0.0,
            },
            primary_span: "harness.op_total".into(),
            qps_derived: 1.0,
            threads_configured: 4,
            ..Default::default()
        };
        BenchReport {
            schema_version: 1,
            admissible,
            metadata: ReportMetadata {
                run: RunMeta {
                    run_id: run_id.into(),
                    cmdline: "t".into(),
                    started_at_unix: 0,
                    finished_at_unix: Some(1),
                    duration_ms: Some(1),
                    log_dir: None,
                    peak: false,
                    sample_packs: "mem".into(),
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
                    os: "t".into(),
                    arch: "t".into(),
                    cpu_count: 1,
                    total_ram_mb: 1,
                },
                repos: vec![],
                stats: StatsMeta {
                    warmup: 1,
                    iterations: 5,
                    threads: 4,
                    rng_seed: 1,
                    primary_span: "harness.op_total".into(),
                },
                calibration: None,
            },
            gates: GatesReport {
                passed: admissible,
                failed: vec![],
                hit_at_1: 1.0,
                hit_at_5: 1.0,
                mrr: 1.0,
                cases: vec![],
                invariants_ok: true,
                entries: vec![],
            },
            cells: vec![CellReport {
                cell_key: "k1".into(),
                scenario_id: "profile.search.matrix".into(),
                repo_id: "default".into(),
                axes: serde_json::json!({}),
                kind: "Search".into(),
                family: "search".into(),
                tags: vec![],
                cost,
                last_result: Default::default(),
                rank_target: None,
                iterations_logged: 3,
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
                host_cpu_busy_pct: None,
                background_cpu_pct: None,
                iter_wall_cv: None,
                policy: "allow".into(),
            },
            change: None,
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
    fn faster_wall_is_improve() {
        let base = bare("agent-confirm", "a", 200.0, true);
        let cur = bare("agent-confirm", "b", 100.0, true);
        let out = compare(&base, &cur);
        assert!(out.comparable);
        assert_eq!(out.verdict, "improve");
        assert!(out.cell_deltas[0].wall_us_p95_rel.unwrap() < 0.0);
    }

    #[test]
    fn inadmissible_current_verdict() {
        let base = bare("agent-confirm", "a", 200.0, true);
        let cur = bare("agent-confirm", "b", 100.0, false);
        let out = compare(&base, &cur);
        assert_eq!(out.verdict, "inadmissible");
    }

    #[test]
    fn near_parity_is_equivalent() {
        let base = bare("agent-confirm", "a", 100.0, true);
        let cur = bare("agent-confirm", "b", 102.0, true);
        let out = compare(&base, &cur);
        assert_eq!(out.verdict, "equivalent");
    }

    #[test]
    fn screen_promising_on_clear_gain() {
        let base = bare("agent-confirm", "a", 200.0, true);
        let cur = bare("agent-screen", "b", 150.0, true);
        let out = compare_screen(&base, &cur);
        assert_eq!(out.verdict, "promising");
    }

    #[test]
    fn screen_no_signal_on_tiny_delta() {
        let base = bare("agent-confirm", "a", 100.0, true);
        let cur = bare("agent-screen", "b", 98.0, true);
        let out = compare_screen(&base, &cur);
        assert_eq!(out.verdict, "screen_no_signal");
    }

    #[test]
    fn sub_floor_rel_regression_passes_cap() {
        // +58% rel but +38us abs on a 65us cell: timer-noise dominated.
        let base = bare("agent-confirm", "a", 65.0, true);
        let cur = bare("agent-confirm", "b", 103.0, true);
        let out = compare(&base, &cur);
        assert_eq!(out.cell_deltas[0].cap, "pass");
    }

    #[test]
    fn above_floor_rel_regression_fails_cap() {
        // +58% rel and +580us abs: real evidence.
        let base = bare("agent-confirm", "a", 1000.0, true);
        let cur = bare("agent-confirm", "b", 1580.0, true);
        let out = compare(&base, &cur);
        assert_eq!(out.cell_deltas[0].cap, "fail");
    }
}
