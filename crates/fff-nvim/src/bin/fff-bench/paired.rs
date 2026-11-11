//! Counterbalanced dual-artifact paired confirmation (METHODOLOGY).

use crate::bootstrap::{self, RatioCi};
use crate::compare::{self, CompareOut};
use crate::cost::{CostSummary, Dist};
use crate::runner::{BenchReport, BlockSpec, StreamError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Counterbalanced block subject order (METHODOLOGY).
/// Doubled ABBA: every adjacent pair cancels linear drift, and the
/// full run is 4-and-4 either way. Stream mode makes 8 blocks cheaper
/// than 4 one-shot blocks (2 corpus opens instead of 16).
const BLOCK_ORDER: &[[&str; 2]] = &[
    ["baseline", "candidate"],
    ["candidate", "baseline"],
    ["candidate", "baseline"],
    ["baseline", "candidate"],
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
    /// Served by a persistent subject process (vs legacy one-shot spawn).
    #[serde(default)]
    pub persistent: bool,
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
    /// stream | oneshot | oneshot-fallback (additive; "" for older outputs).
    #[serde(default)]
    pub subject_mode: String,
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
    /// Sensor packs requested from every block (`--sample`).
    pub sample_packs: String,
}

pub fn run_paired(cfg: &PairedConfig) -> Result<PairedRunOut, String> {
    match run_paired_stream(cfg) {
        Ok(out) => Ok(out),
        Err(e) => {
            eprintln!("paired: persistent-subject stream failed ({e}); one-shot fallback");
            let mut out = run_paired_oneshot(cfg)?;
            out.subject_mode = "oneshot-fallback".into();
            Ok(out)
        }
    }
}

/// Legacy path: one fresh `run` subprocess per block per subject.
pub fn run_paired_oneshot(cfg: &PairedConfig) -> Result<PairedRunOut, String> {
    let mut out = run_blocks(cfg, false, &mut |block, role, output| {
        let _ = block;
        let bin = if role == "baseline" {
            &cfg.baseline_bin
        } else {
            &cfg.candidate_bin
        };
        measure_subject(
            bin,
            &cfg.corpus,
            &cfg.suite,
            cfg.iterations_per_block,
            cfg.warmup,
            output,
            &cfg.load_profile,
        )
    })?;
    out.subject_mode = "oneshot".into();
    Ok(out)
}

/// Persistent path: one `run-stream` child per subject for all blocks.
fn run_paired_stream(cfg: &PairedConfig) -> Result<PairedRunOut, String> {
    let mut base = StreamChild::spawn(&cfg.baseline_bin)?;
    let mut cand = match StreamChild::spawn(&cfg.candidate_bin) {
        Ok(c) => c,
        Err(e) => {
            base.abort();
            return Err(e);
        }
    };
    let result = run_blocks(cfg, true, &mut |block, role, output| {
        let child = if role == "baseline" {
            &mut base
        } else {
            &mut cand
        };
        let report = child.request(&stream_spec_for_block(cfg, block))?;
        let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        fs::write(output, text).map_err(|e| e.to_string())?;
        Ok(report)
    });
    let end_base = base.finish();
    let end_cand = cand.finish();
    let mut out = result?;
    end_base?;
    end_cand?;
    out.subject_mode = "stream".into();
    Ok(out)
}

/// Block spec mirroring the legacy one-shot `run` argv (sample/patch/slim).
pub fn stream_spec_for_block(cfg: &PairedConfig, block: usize) -> BlockSpec {
    BlockSpec {
        suite: cfg.suite.clone(),
        corpus: cfg.corpus.clone(),
        iters: cfg.iterations_per_block,
        warmup: cfg.warmup,
        load_profile: cfg.load_profile.clone(),
        sample: cfg.sample_packs.clone(),
        slim: true,
        patch: "none".into(),
        timeout_secs: 600,
        content_index: false,
        history_dir: None,
        block_index: block,
        repo_id: "default".into(),
    }
}

struct StreamChild {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl StreamChild {
    fn spawn(bin: &Path) -> Result<Self, String> {
        if !bin.exists() {
            return Err(format!("artifact not found: {}", bin.display()));
        }
        let mut child = Command::new(bin)
            .arg("run-stream")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            // Same pin as one-shot spawns; applies to every streamed block.
            .env("FFF_SCAN_THREAD_CAP", "4")
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", bin.display()))?;
        let stdin = child.stdin.take().ok_or("subject stream has no stdin")?;
        let stdout = child.stdout.take().ok_or("subject stream has no stdout")?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn request(&mut self, spec: &BlockSpec) -> Result<BenchReport, String> {
        let mut line = serde_json::to_string(spec).map_err(|e| e.to_string())?;
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .map_err(|e| format!("subject stream write: {e}"))?;
        self.stdin
            .flush()
            .map_err(|e| format!("subject stream write: {e}"))?;
        let mut out = String::new();
        let n = self
            .stdout
            .read_line(&mut out)
            .map_err(|e| format!("subject stream read: {e}"))?;
        if n == 0 {
            return Err("subject stream ended (needs run-stream support)".into());
        }
        if let Ok(report) = serde_json::from_str::<BenchReport>(&out) {
            return Ok(report);
        }
        if let Ok(err) = serde_json::from_str::<StreamError>(&out) {
            return Err(err.error);
        }
        Err("subject stream sent undecodable line".into())
    }

    fn finish(mut self) -> Result<(), String> {
        drop(self.stdin);
        let status = self
            .child
            .wait()
            .map_err(|e| format!("wait subject stream: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("subject stream exited {status}"))
        }
    }

    fn abort(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn run_blocks(
    cfg: &PairedConfig,
    persistent: bool,
    measure: &mut dyn FnMut(usize, &str, &Path) -> Result<BenchReport, String>,
) -> Result<PairedRunOut, String> {
    fs::create_dir_all(&cfg.log_dir).map_err(|e| e.to_string())?;
    let blocks = cfg.blocks.max(1).min(BLOCK_ORDER.len());
    let mut block_log = Vec::new();
    // role -> cell_key -> per-block p95 series for each compared sensor
    let mut base_walls: BTreeMap<String, CellSeries> = BTreeMap::new();
    let mut cand_walls: BTreeMap<String, CellSeries> = BTreeMap::new();
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
            let out = cfg.log_dir.join(format!("block{b}_{role}.json"));
            let report = measure(b, role, &out)?;
            block_log.push(BlockObs {
                block: b,
                role: role.into(),
                run_id: report.metadata.run.run_id.clone(),
                admissible: report.admissible,
                report_path: out.display().to_string(),
                persistent,
            });
            if role == "baseline" {
                ingest(&mut base_walls, &report);
                block_base = Some(report);
            } else {
                ingest(&mut cand_walls, &report);
                block_cand = Some(report);
            }
        }

        // Within-block paired cell means → one (base, cand) observation.
        // Move reports out: last block doubles as aggregate metadata.
        if let (Some(br), Some(cr)) = (block_base, block_cand) {
            if let Some(pair) = block_mean_pair(&br, &cr) {
                block_pairs.push(pair);
            }
            base_meta = Some(br);
            cand_meta = Some(cr);
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
        comparison
            .reason_codes
            .push(format!("paired_block_ci_{cls}"));
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
        comparison
            .reason_codes
            .push("candidate_gates_failed".into());
    }
    if candidate_report
        .stability
        .as_ref()
        .map(|s| s.stability == "unstable")
        .unwrap_or(false)
    {
        comparison.verdict = "unstable".into();
        comparison
            .reason_codes
            .push("between_block_or_host_unstable".into());
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
        subject_mode: String::new(),
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
    // Old artifacts predate --slim: try slim first, retry plain on
    // freshness failure so a stale block file can never pass as data.
    match measure_subject_inner(
        bin, corpus, suite, iterations, warmup, output, load_profile, true,
    ) {
        Ok(r) => Ok(r),
        Err(first) => {
            let retry = measure_subject_inner(
                bin, corpus, suite, iterations, warmup, output, load_profile, false,
            );
            retry.map_err(|_| first)
        }
    }
}

/// Freshness rule: the report file must be newer than the spawn.
/// A subject that errors (unknown flag, crash) without writing must
/// never let a previous run's file pass as this block's data.
fn report_is_fresh(output: &Path, start: std::time::SystemTime) -> bool {
    std::fs::metadata(output)
        .and_then(|m| m.modified())
        .map(|t| t > start)
        .unwrap_or(false)
}

fn measure_subject_inner(
    bin: &Path,
    corpus: &Path,
    suite: &str,
    iterations: usize,
    warmup: usize,
    output: &Path,
    load_profile: &str,
    slim: bool,
) -> Result<BenchReport, String> {
    if !bin.exists() {
        return Err(format!("artifact not found: {}", bin.display()));
    }
    let profile = if load_profile.is_empty() {
        "quiet_v1"
    } else {
        load_profile
    };
    let start = std::time::SystemTime::now();
    let mut cmd = Command::new(bin);
    cmd.arg("run")
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
        .arg(profile);
    if slim {
        cmd.arg("--slim");
    }
    let status = cmd
        .arg("--output")
        .arg(output)
        .arg("--patch")
        .arg("none")
        // Pin thread calibration for both subjects: the bandwidth-knee
        // probe answers 1/2/4 across runs, which would inject per-block
        // thread-count differences into a code comparison. Production
        // still probes; paired runs compare code at fixed caps.
        .env("FFF_SCAN_THREAD_CAP", "4")
        .status()
        .map_err(|e| format!("spawn {}: {e}", bin.display()))?;
    if !status.success() && !report_is_fresh(output, start) {
        return Err(format!(
            "subject {} exited {} without a fresh report",
            bin.display(),
            status
        ));
    }
    if !report_is_fresh(output, start) {
        return Err(format!(
            "subject {} left a stale report {}",
            bin.display(),
            output.display()
        ));
    }
    let s = fs::read_to_string(output).map_err(|e| format!("read report: {e}"))?;
    serde_json::from_str(&s).map_err(|e| format!("parse report: {e}"))
}

/// Per-block p95 series for each compared sensor of one cell.
#[derive(Default, Clone)]
struct CellSeries {
    wall: Vec<f64>,
    cpu_total: Vec<f64>,
    cycles: Vec<f64>,
    faults: Vec<f64>,
}

fn ingest(map: &mut BTreeMap<String, CellSeries>, report: &BenchReport) {
    for c in &report.cells {
        if c.family == "quality" || c.scenario_id.starts_with("gate") {
            continue;
        }
        let e = map.entry(c.cell_key.clone()).or_default();
        e.wall.push(c.cost.wall_us.p95);
        if let Some(d) = c.cost.cpu_total_us.as_ref() {
            e.cpu_total.push(d.p95);
        }
        if let Some(d) = c.cost.cpu_cycles.as_ref() {
            e.cycles.push(d.p95 as f64);
        }
        if let Some(d) = c.cost.page_faults.as_ref() {
            e.faults.push(d.p95 as f64);
        }
    }
}

fn block_mean_pair(base: &BenchReport, cand: &BenchReport) -> Option<(f64, f64)> {
    let index: HashMap<&str, &crate::runner::CellReport> = base
        .cells
        .iter()
        .map(|c| (c.cell_key.as_str(), c))
        .collect();
    let mut bsum = 0.0;
    let mut csum = 0.0;
    let mut n = 0usize;
    for c in &cand.cells {
        if c.family == "quality" {
            continue;
        }
        if let Some(b) = index.get(c.cell_key.as_str()) {
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

/// Single-transient rejection: with few blocks p95 is the max, so one
/// descheduled block defines the cell (seen: +44% while min/median held).
fn block_dist(vs: &[f64]) -> Dist {
    let mut vals = vs.to_vec();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if vals.len() >= 4 {
        let top = vals.len() - 1;
        if vals[top] > vals[top - 1] * 1.25 {
            vals.pop();
        }
    }
    Dist::from_values(vals)
}

fn opt_dist(vs: &[f64]) -> Option<Dist> {
    if vs.is_empty() {
        None
    } else {
        Some(block_dist(vs))
    }
}

fn rewrite_cells_from_blocks(report: &mut BenchReport, series: &BTreeMap<String, CellSeries>) {
    for c in &mut report.cells {
        if let Some(s) = series.get(&c.cell_key) {
            c.cost = CostSummary {
                wall_us: block_dist(&s.wall),
                threads_configured: c.cost.threads_configured,
                primary_span: c.cost.primary_span.clone(),
                qps_derived: c.cost.qps_derived,
                cpu_total_us: opt_dist(&s.cpu_total),
                cpu_cycles: opt_dist(&s.cycles),
                page_faults: opt_dist(&s.faults),
                ..c.cost.clone()
            };
            c.iterations_logged = s.wall.len();
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
    use crate::cost::{CostSummary, Dist, TopHit};
    use crate::identity::*;
    use crate::runner::{CellReport, ReportMetadata};

    #[test]
    fn block_order_is_counterbalanced() {
        assert_eq!(BLOCK_ORDER[0], ["baseline", "candidate"]);
        assert_eq!(BLOCK_ORDER[1], ["candidate", "baseline"]);
        let base_first = BLOCK_ORDER.iter().filter(|o| o[0] == "baseline").count();
        let cand_first = BLOCK_ORDER.iter().filter(|o| o[0] == "candidate").count();
        assert_eq!(base_first, cand_first);
    }

    #[test]
    fn stream_spec_mirrors_oneshot_measure() {
        let cfg = PairedConfig {
            baseline_bin: PathBuf::from("base"),
            candidate_bin: PathBuf::from("cand"),
            corpus: PathBuf::from("/corpus"),
            suite: "agent-confirm".into(),
            blocks: 4,
            iterations_per_block: 12,
            warmup: 3,
            log_dir: PathBuf::from("/logs"),
            seed: PAIRED_SEED,
            load_profile: "quiet_v1".into(),
            sample_packs: "mem,cpu".into(),
        };
        let spec = stream_spec_for_block(&cfg, 2);
        assert_eq!(spec.suite, "agent-confirm");
        assert_eq!(spec.iters, 12);
        assert_eq!(spec.warmup, 3);
        assert_eq!(spec.block_index, 2);
        assert_eq!(spec.sample, "mem,cpu");
        assert_eq!(spec.patch, "none");
        assert!(spec.slim);
        // JSON line round-trips through the stream protocol.
        let line = serde_json::to_string(&spec).unwrap();
        assert!(!line.contains('\n'));
        let back: crate::runner::BlockSpec = serde_json::from_str(&line).unwrap();
        assert_eq!(back.corpus, PathBuf::from("/corpus"));
    }

    #[test]
    fn stream_error_line_is_disjoint_from_report() {
        let err = StreamError {
            block_index: Some(1),
            error: "boom".into(),
        };
        let line = serde_json::to_string(&err).unwrap();
        assert!(serde_json::from_str::<BenchReport>(&line).is_err());
        let back: StreamError = serde_json::from_str(&line).unwrap();
        assert_eq!(back.error, "boom");
    }

    #[test]
    fn paired_envelope_defaults_keep_old_json_readable() {
        let v = serde_json::json!({
            "block": 0, "role": "baseline", "run_id": "r",
            "admissible": true, "report_path": "p",
        });
        let obs: BlockObs = serde_json::from_value(v).unwrap();
        assert!(!obs.persistent);
    }

    fn tcell(key: &str, family: &str, p95: f64) -> CellReport {
        CellReport {
            cell_key: key.into(),
            scenario_id: "profile.search.matrix".into(),
            repo_id: "default".into(),
            axes: serde_json::json!({}),
            kind: "Search".into(),
            family: family.into(),
            tags: vec![],
            state_id: None,
            cost: CostSummary {
                wall_us: Dist {
                    n: 3,
                    min: p95,
                    p50: p95,
                    p90: p95,
                    p95,
                    p99: 0.0,
                    max: p95,
                    mean: p95,
                    stddev: 0.0,
                    sum: p95 * 3.0,
                    mad: 0.0,
                    relative_mad: 0.0,
                },
                threads_configured: 4,
                primary_span: "harness.op_total".into(),
                qps_derived: 1.0,
                ..Default::default()
            },
            last_result: crate::cost::OpResultData {
                total_matched: 3,
                returned: 1,
                files_searched: None,
                top_k: vec![TopHit {
                    path: "f.rs".into(),
                    ..Default::default()
                }],
                rank_target: Some(0),
                ordered_digest: Some(format!("d-for-{key}")),
                error: None,
            },
            rank_target: Some(0),
            iterations_logged: 3,
            throughput_qps: None,
            throughput_qps_dist: None,
            request_wall_us: None,
            operations_per_iteration: None,
            client_concurrency: None,
            arrival_model: None,
            request_errors: None,
            ordered_digest_consistent: true,
            warmup_converged: None,
            warmups_run: None,
            last_metrics: None,
        }
    }

    fn treport(cells: Vec<CellReport>) -> BenchReport {
        BenchReport {
            schema_version: 1,
            admissible: true,
            metadata: ReportMetadata {
                run: RunMeta {
                    run_id: "t".into(),
                    cmdline: "t".into(),
                    started_at_unix: 0,
                    finished_at_unix: Some(1),
                    duration_ms: Some(1),
                    log_dir: None,
                    peak: false,
                    sample_packs: "mem".into(),
                    primary_span: "harness.op_total".into(),
                    suite: "agent-confirm".into(),
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
                    iterations: 3,
                    threads: 4,
                    rng_seed: 1,
                    primary_span: "harness.op_total".into(),
                },
                calibration: None,
            },
            gates: crate::gates::GatesReport {
                passed: true,
                ..Default::default()
            },
            cells,
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
    fn stale_report_never_passes_as_fresh() {
        let dir = std::env::temp_dir().join("fff-paired-fresh-test");
        let _ = std::fs::create_dir_all(&dir);
        let f = dir.join("block.json");
        std::fs::write(&f, "{}").unwrap();
        // File from the past is stale against a fresh spawn instant.
        let past = std::time::SystemTime::now() + std::time::Duration::from_secs(3600);
        assert!(!report_is_fresh(&f, past));
        // A just-written file is fresh.
        std::fs::write(&f, "{}").unwrap();
        let before = std::time::SystemTime::now() - std::time::Duration::from_secs(5);
        assert!(report_is_fresh(&f, before));
        // Missing file is never fresh.
        assert!(!report_is_fresh(&dir.join("nope.json"), before));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn block_mean_pair_joins_by_cell_key() {
        let base = treport(vec![
            tcell("k1", "search", 100.0),
            tcell("k2", "search", 200.0),
            tcell("q1", "quality", 9999.0),
        ]);
        let cand = treport(vec![
            tcell("k2", "search", 250.0),
            tcell("k1", "search", 150.0),
            tcell("zzz", "search", 5.0),
        ]);
        assert_eq!(block_mean_pair(&base, &cand), Some((150.0, 200.0)));
    }

    #[test]
    fn lone_outlier_block_is_trimmed() {
        let mut report = treport(vec![tcell("k1", "search", 0.0)]);
        let mut walls = std::collections::BTreeMap::new();
        walls.insert(
            "k1".to_string(),
            CellSeries {
                wall: vec![615.0, 650.0, 670.0, 995.0],
                ..Default::default()
            },
        );
        rewrite_cells_from_blocks(&mut report, &walls);
        // 995 > 670 * 1.25 trims to [615, 650, 670]: p95 must drop.
        assert!(report.cells[0].cost.wall_us.p95 < 900.0);
        // No outlier: all four kept.
        let mut report2 = treport(vec![tcell("k1", "search", 0.0)]);
        let mut walls2 = std::collections::BTreeMap::new();
        walls2.insert(
            "k1".to_string(),
            CellSeries {
                wall: vec![600.0, 650.0, 670.0, 690.0],
                ..Default::default()
            },
        );
        rewrite_cells_from_blocks(&mut report2, &walls2);
        assert!(report2.cells[0].cost.wall_us.p95 >= 690.0);
    }

    #[test]
    fn slim_output_parses_and_compares() {
        let mut base = treport(vec![tcell("k1", "search", 200.0)]);
        let mut cand = treport(vec![tcell("k1", "search", 100.0)]);
        // Verbose files still parse (backward compat).
        let full: BenchReport =
            serde_json::from_str(&serde_json::to_string(&base).unwrap()).unwrap();
        assert_eq!(full.cells[0].last_result.top_k.len(), 1);
        crate::runner::slim_report(&mut base);
        crate::runner::slim_report(&mut cand);
        let back: BenchReport =
            serde_json::from_str(&serde_json::to_string(&base).unwrap()).unwrap();
        assert!(back.cells[0].last_result.top_k.is_empty());
        assert_eq!(back.cells[0].cost.wall_us.p95, 200.0);
        assert_eq!(
            back.cells[0].last_result.ordered_digest.as_deref(),
            Some("d-for-k1")
        );
        assert!(back.admissible && back.gates.passed);
        let out = compare::compare(&back, &cand);
        assert!(out.comparable);
        assert_eq!(out.verdict, "improve");
    }
}
