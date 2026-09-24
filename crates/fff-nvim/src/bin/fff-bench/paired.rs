//! Counterbalanced dual-artifact paired confirmation (METHODOLOGY).

use crate::bootstrap::{self, RatioCi};
use crate::compare::{self, CompareOut};
use crate::cost::{CostSummary, Dist};
use crate::runner::BenchReport;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Counterbalanced block subject order (METHODOLOGY).
const BLOCK_ORDER: &[[&str; 2]] = &[
    ["baseline", "candidate"],
    ["candidate", "baseline"],
    ["candidate", "baseline"],
    ["baseline", "candidate"],
];

pub const DEFAULT_BLOCKS: usize = 4;
pub const DEFAULT_ITERS_PER_BLOCK: usize = 12;
pub const DEFAULT_WARMUP: usize = 3;
pub const PAIRED_BOOTSTRAP_N: usize = 10_000;
pub const PAIRED_SEED: u64 = 0xFA17_ED01;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockObs {
    pub block: usize,
    pub role: String,
    pub run_id: String,
    pub admissible: bool,
    pub report_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedRunOut {
    pub schema_version: u32,
    pub pairing: String,
    pub blocks: usize,
    pub iterations_per_block: usize,
    pub seed: u64,
    pub baseline_artifact: String,
    pub candidate_artifact: String,
    pub suite: String,
    #[serde(default)]
    pub load_profile: String,
    pub block_log: Vec<BlockObs>,
    pub baseline_report: BenchReport,
    pub candidate_report: BenchReport,
    pub comparison: CompareOut,
    /// Block-level wall pairs for primary cells: (baseline_p95, candidate_p95).
    pub block_wall_pairs: Vec<(f64, f64)>,
    pub wall_ratio_ci95: Option<RatioCi>,
}

#[derive(Debug, Clone)]
pub struct PairedConfig {
    pub baseline_bin: PathBuf,
    pub candidate_bin: PathBuf,
    pub corpus: PathBuf,
    pub suite: String,
    pub blocks: usize,
    pub iterations_per_block: usize,
    pub warmup: usize,
    pub log_dir: PathBuf,
    pub seed: u64,
    /// Applied to every subprocess measure (`--load-profile`).
    pub load_profile: String,
}

pub fn run_paired(cfg: &PairedConfig) -> Result<PairedRunOut, String> {
    fs::create_dir_all(&cfg.log_dir).map_err(|e| e.to_string())?;
    let blocks = cfg.blocks.max(1).min(BLOCK_ORDER.len());
    let mut block_log = Vec::new();
    // role -> cell_key -> vec of wall p95 per block
    let mut base_walls: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut cand_walls: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut base_meta: Option<BenchReport> = None;
    let mut cand_meta: Option<BenchReport> = None;
    // Per-block geometric mean ratio inputs across cells
    let mut block_pairs: Vec<(f64, f64)> = Vec::new();

    for b in 0..blocks {
        let order = BLOCK_ORDER[b % BLOCK_ORDER.len()];
        let mut block_base: Option<BenchReport> = None;
        let mut block_cand: Option<BenchReport> = None;

        for role_ref in order {
            let role: &str = role_ref;
            let bin = if role == "baseline" {
                &cfg.baseline_bin
            } else {
                &cfg.candidate_bin
            };
            let out = cfg
                .log_dir
                .join(format!("block{b}_{role}.json"));
            let report = measure_subject(
                bin,
                &cfg.corpus,
                &cfg.suite,
                cfg.iterations_per_block,
                cfg.warmup,
                &out,
                &cfg.load_profile,
            )?;
            block_log.push(BlockObs {
                block: b,
                role: role.into(),
                run_id: report.metadata.run.run_id.clone(),
                admissible: report.admissible,
                report_path: out.display().to_string(),
            });
            if role == "baseline" {
                ingest(&mut base_walls, &report);
                block_base = Some(report.clone());
                base_meta = Some(report);
            } else {
                ingest(&mut cand_walls, &report);
                block_cand = Some(report.clone());
                cand_meta = Some(report);
            }
        }

        // Within-block paired cell means → one (base, cand) observation.
        if let (Some(br), Some(cr)) = (block_base, block_cand) {
            if let Some(pair) = block_mean_pair(&br, &cr) {
                block_pairs.push(pair);
            }
        }
    }

    let mut baseline_report = base_meta.ok_or("no baseline block completed")?;
    let mut candidate_report = cand_meta.ok_or("no candidate block completed")?;
    // Replace cell costs with distributions over block p95s.
    rewrite_cells_from_blocks(&mut baseline_report, &base_walls);
    rewrite_cells_from_blocks(&mut candidate_report, &cand_walls);
    baseline_report.metadata.run.suite = cfg.suite.clone();
    candidate_report.metadata.run.suite = cfg.suite.clone();
    // Mark pairing in plan digest space so both share structure.
    let pair_tag = format!("paired-blocks-{blocks}-{}", cfg.suite);
    baseline_report.plan_digest = Some(format!(
        "sha256:{}",
        crate::identity::sha256_hex(pair_tag.as_bytes())
    ));
    candidate_report.plan_digest = baseline_report.plan_digest.clone();

    let mut comparison = compare::compare(&baseline_report, &candidate_report);
    let wall_ci = if block_pairs.len() >= 2 {
        bootstrap::ratio_ci95(&block_pairs, cfg.seed, PAIRED_BOOTSTRAP_N)
    } else {
        None
    };
    if let Some(ref ci) = wall_ci {
        comparison.wall_ratio_ci95 = Some(ci.clone());
        let cls = bootstrap::classify_ratio(ci, 0.05);
        comparison.reason_codes.push(format!("paired_block_ci_{cls}"));
        // Decision-grade: prefer CI when decisive.
        match cls {
            "improve" if comparison.verdict == "equivalent" || comparison.verdict == "mixed" => {
                if ci.high < 0.95 {
                    comparison.verdict = "improve".into();
                }
            }
            "regress" => {
                if ci.low > 1.05 {
                    comparison.verdict = "regress".into();
                }
            }
            "equivalent" if comparison.verdict == "improve" => {
                comparison.verdict = "equivalent".into();
            }
            _ => {}
        }
        comparison.summary_for_agent = format!(
            "{}; paired_blocks={} wall_ci=[{:.4},{:.4}] point={:.4}",
            comparison.summary_for_agent, blocks, ci.low, ci.high, ci.point
        );
        comparison.evidence_grade = Some("confirm".into());
    }

    // METHODOLOGY between-block drift from counterbalanced block aggregates.
    let cand_block_medians: Vec<f64> = block_pairs.iter().map(|(_, c)| *c).collect();
    let base_block_medians: Vec<f64> = block_pairs.iter().map(|(b, _)| *b).collect();
    let iters = cfg.iterations_per_block;
    candidate_report.stability = Some(crate::stability::from_cells_with_drift(
        &candidate_report.cells,
        &candidate_report.noise,
        iters,
        Some(&cand_block_medians),
    ));
    baseline_report.stability = Some(crate::stability::from_cells_with_drift(
        &baseline_report.cells,
        &baseline_report.noise,
        iters,
        Some(&base_block_medians),
    ));
    // Re-compare so unstable drift is visible in verdict path.
    comparison = compare::compare(&baseline_report, &candidate_report);
    if let Some(ref ci) = wall_ci {
        comparison.wall_ratio_ci95 = Some(ci.clone());
        comparison.evidence_grade = Some("confirm".into());
        comparison.reason_codes.push(format!(
            "paired_block_ci_{}",
            bootstrap::classify_ratio(ci, 0.05)
        ));
    }

    // Candidate must remain admissible.
    if !candidate_report.admissible {
        comparison.verdict = "inadmissible".into();
        comparison.reason_codes.push("candidate_gates_failed".into());
    }
    if candidate_report
        .stability
        .as_ref()
        .map(|s| s.stability == "unstable")
        .unwrap_or(false)
    {
        comparison.verdict = "unstable".into();
        comparison.reason_codes.push("between_block_or_host_unstable".into());
    }

    Ok(PairedRunOut {
        schema_version: 1,
        pairing: "counterbalanced_blocks".into(),
        blocks,
        iterations_per_block: cfg.iterations_per_block,
        seed: cfg.seed,
        baseline_artifact: cfg.baseline_bin.display().to_string(),
        candidate_artifact: cfg.candidate_bin.display().to_string(),
        suite: cfg.suite.clone(),
        load_profile: cfg.load_profile.clone(),
        block_log,
        baseline_report,
        candidate_report,
        comparison,
        block_wall_pairs: block_pairs,
        wall_ratio_ci95: wall_ci,
    })
}

fn measure_subject(
    bin: &Path,
    corpus: &Path,
    suite: &str,
    iterations: usize,
    warmup: usize,
    output: &Path,
    load_profile: &str,
) -> Result<BenchReport, String> {
    if !bin.exists() {
        return Err(format!("artifact not found: {}", bin.display()));
    }
    let profile = if load_profile.is_empty() {
        "quiet_v1"
    } else {
        load_profile
    };
    let status = Command::new(bin)
        .arg("run")
        .arg("--path")
        .arg(corpus)
        .arg("--suite")
        .arg(suite)
        .arg("-n")
        .arg(iterations.to_string())
        .arg("--warmup")
        .arg(warmup.to_string())
        .arg("--sample")
        .arg("mem,cpu")
        .arg("--load-profile")
        .arg(profile)
        .arg("--output")
        .arg(output)
        .arg("--patch")
        .arg("none")
        .status()
        .map_err(|e| format!("spawn {}: {e}", bin.display()))?;
    if !status.success() {
        // Still try to load report if written (gates may fail with exit 1).
        if !output.exists() {
            return Err(format!(
                "subject {} exited {} without report",
                bin.display(),
                status
            ));
        }
    }
    let s = fs::read_to_string(output).map_err(|e| format!("read report: {e}"))?;
    serde_json::from_str(&s).map_err(|e| format!("parse report: {e}"))
}

fn ingest(map: &mut BTreeMap<String, Vec<f64>>, report: &BenchReport) {
    for c in &report.cells {
        if c.family == "quality" || c.scenario_id.starts_with("gate") {
            continue;
        }
        map.entry(c.cell_key.clone())
            .or_default()
            .push(c.cost.wall_us.p95);
    }
}

fn block_mean_pair(base: &BenchReport, cand: &BenchReport) -> Option<(f64, f64)> {
    let mut bsum = 0.0;
    let mut csum = 0.0;
    let mut n = 0usize;
    for c in &cand.cells {
        if c.family == "quality" {
            continue;
        }
        if let Some(b) = base.cells.iter().find(|x| x.cell_key == c.cell_key) {
            if b.cost.wall_us.p95 > 0.0 && c.cost.wall_us.p95 > 0.0 {
                bsum += b.cost.wall_us.p95;
                csum += c.cost.wall_us.p95;
                n += 1;
            }
        }
    }
    if n == 0 {
        return None;
    }
    Some((bsum / n as f64, csum / n as f64))
}

fn rewrite_cells_from_blocks(report: &mut BenchReport, walls: &BTreeMap<String, Vec<f64>>) {
    for c in &mut report.cells {
        if let Some(vs) = walls.get(&c.cell_key) {
            let dist = Dist::from_values(vs.clone());
            c.cost = CostSummary {
                wall_us: dist,
                threads_configured: c.cost.threads_configured,
                primary_span: c.cost.primary_span.clone(),
                qps_derived: c.cost.qps_derived,
                cpu_total_us: c.cost.cpu_total_us.clone(),
                ..c.cost.clone()
            };
            c.iterations_logged = vs.len();
        }
    }
}

/// Resolve artifact path: absolute, relative, or `self` = current executable.
pub fn resolve_artifact(spec: &str) -> Result<PathBuf, String> {
    if spec == "self" || spec == "current" {
        return std::env::current_exe().map_err(|e| e.to_string());
    }
    let p = PathBuf::from(spec);
    if p.exists() {
        return Ok(p);
    }
    Err(format!("artifact not found: {spec}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_order_is_counterbalanced() {
        assert_eq!(BLOCK_ORDER[0], ["baseline", "candidate"]);
        assert_eq!(BLOCK_ORDER[1], ["candidate", "baseline"]);
        let base_first = BLOCK_ORDER.iter().filter(|o| o[0] == "baseline").count();
        let cand_first = BLOCK_ORDER.iter().filter(|o| o[0] == "candidate").count();
        assert_eq!(base_first, cand_first);
    }
}
