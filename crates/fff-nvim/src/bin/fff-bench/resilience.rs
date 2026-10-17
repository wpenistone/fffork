//! resilience-confirm multi-profile execution (+ optional dual-bin pairing).

use crate::compare::CompareOut;
use crate::load_actor;
use crate::paired::{self, PairedConfig, PairedRunOut};
use crate::runner::{self, BenchReport, RunConfig};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceReport {
    pub schema_version: u32,
    pub suite: String,
    pub profiles: Vec<ProfileResult>,
    pub all_admissible: bool,
    pub all_treatments_valid: bool,
    /// Set when dual-bin pairing was used under each profile.
    #[serde(default)]
    pub paired: bool,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    pub load_profile_id: String,
    pub admissible: bool,
    pub treatment_valid: bool,
    pub typical_wall_p95_us: Option<f64>,
    pub calibration_busy_cores: Option<f64>,
    pub report_path: Option<String>,
    pub reason_codes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<BenchReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comparison: Option<CompareOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paired: Option<PairedRunOut>,
}

pub struct ResilienceConfig {
    pub base: RunConfig,
    pub profiles: Vec<String>,
    pub log_dir: PathBuf,
    pub keep_full_reports: bool,
    /// When both set, run counterbalanced dual-bin pairing per load profile.
    pub baseline_bin: Option<PathBuf>,
    pub candidate_bin: Option<PathBuf>,
    pub corpus: Option<PathBuf>,
    pub paired_blocks: usize,
    pub paired_iters: usize,
}

pub fn run_resilience(
    repos: &[(String, PathBuf)],
    cfg: &ResilienceConfig,
) -> Result<ResilienceReport, String> {
    if let (Some(bb), Some(cb)) = (&cfg.baseline_bin, &cfg.candidate_bin) {
        let corpus = cfg
            .corpus
            .clone()
            .or_else(|| repos.first().map(|(_, p)| p.clone()))
            .ok_or("no corpus for paired resilience")?;
        return run_resilience_paired(bb, cb, &corpus, cfg);
    }

    std::fs::create_dir_all(&cfg.log_dir).map_err(|e| e.to_string())?;
    let mut profiles_out = Vec::new();
    let mut all_adm = true;
    let mut all_treat = true;

    for pid in &cfg.profiles {
        if load_actor::resolve_profile(pid).is_none() {
            return Err(format!("unknown load profile: {pid}"));
        }
        let mut run_cfg = clone_run_config(&cfg.base);
        run_cfg.load_profile = pid.clone();
        run_cfg.suite = "resilience-confirm".into();
        run_cfg.mutation_gate = false;
        run_cfg.log_dir = Some(cfg.log_dir.join(pid));

        let report = runner::run_bench_multi(repos, &run_cfg)?;
        let (pr, adm, treat) = summarize_report(pid, report, cfg.keep_full_reports);
        if !adm {
            all_adm = false;
        }
        if !treat {
            all_treat = false;
        }
        profiles_out.push(pr);
    }

    Ok(ResilienceReport {
        schema_version: 1,
        suite: "resilience-confirm".into(),
        profiles: profiles_out,
        all_admissible: all_adm,
        all_treatments_valid: all_treat,
        paired: false,
        summary: format!(
            "resilience profiles={} all_admissible={} all_treatments_valid={} paired=false",
            cfg.profiles.len(),
            all_adm,
            all_treat
        ),
    })
}

/// Dual-bin counterbalanced pairing under each load profile (METHODOLOGY + pack).
pub fn run_resilience_paired(
    baseline_bin: &Path,
    candidate_bin: &Path,
    corpus: &Path,
    cfg: &ResilienceConfig,
) -> Result<ResilienceReport, String> {
    std::fs::create_dir_all(&cfg.log_dir).map_err(|e| e.to_string())?;
    let mut profiles_out = Vec::new();
    let mut all_adm = true;
    let mut all_treat = true;

    for pid in &cfg.profiles {
        if load_actor::resolve_profile(pid).is_none() {
            return Err(format!("unknown load profile: {pid}"));
        }
        let pcfg = PairedConfig {
            baseline_bin: baseline_bin.to_path_buf(),
            candidate_bin: candidate_bin.to_path_buf(),
            corpus: corpus.to_path_buf(),
            suite: "resilience-confirm".into(),
            blocks: cfg.paired_blocks.max(1).min(4),
            iterations_per_block: cfg.paired_iters.max(3),
            warmup: 2,
            log_dir: cfg.log_dir.join(pid),
            seed: paired::PAIRED_SEED ^ identity_hash(pid),
            load_profile: pid.clone(),
            sample_packs: "mem,cpu,cycles".into(),
        };
        let out = paired::run_paired(&pcfg)?;
        let treat_valid = out
            .candidate_report
            .treatment
            .as_ref()
            .map(|t| t.valid)
            .unwrap_or(true)
            && out
                .baseline_report
                .treatment
                .as_ref()
                .map(|t| t.valid)
                .unwrap_or(true);
        // Paired load delivery equivalence: profile id + measured delivery within tolerance.
        let (equiv, mut equiv_reasons) = match (
            &out.baseline_report.treatment,
            &out.candidate_report.treatment,
        ) {
            (Some(a), Some(b)) => load_actor::treatments_delivery_equivalent(a, b),
            (None, None) => (treat_valid, vec![]),
            _ => (false, vec!["treatment_missing_on_one_side".into()]),
        };
        let mut reasons = out.comparison.reason_codes.clone();
        reasons.append(&mut equiv_reasons);
        if !out.candidate_report.admissible {
            all_adm = false;
            reasons.push("candidate_gates_failed".into());
        }
        if !treat_valid || !equiv {
            all_treat = false;
            reasons.push("treatment_invalid_or_inequivalent".into());
        }
        let busy = out
            .candidate_report
            .treatment
            .as_ref()
            .and_then(|t| t.calibration.as_ref())
            .and_then(|c| c.busy_cores);
        let wall = if out.block_wall_pairs.is_empty() {
            None
        } else {
            Some(
                out.block_wall_pairs.iter().map(|(_, c)| *c).sum::<f64>()
                    / out.block_wall_pairs.len() as f64,
            )
        };

        profiles_out.push(ProfileResult {
            load_profile_id: pid.clone(),
            admissible: out.candidate_report.admissible && out.baseline_report.admissible,
            treatment_valid: treat_valid && equiv,
            typical_wall_p95_us: wall,
            calibration_busy_cores: busy,
            report_path: Some(cfg.log_dir.join(pid).display().to_string()),
            reason_codes: reasons,
            report: if cfg.keep_full_reports {
                Some(out.candidate_report.clone())
            } else {
                None
            },
            comparison: Some(out.comparison.clone()),
            paired: Some(out),
        });
    }

    Ok(ResilienceReport {
        schema_version: 1,
        suite: "resilience-confirm".into(),
        profiles: profiles_out,
        all_admissible: all_adm,
        all_treatments_valid: all_treat,
        paired: true,
        summary: format!(
            "resilience paired profiles={} all_admissible={} all_treatments_valid={}",
            cfg.profiles.len(),
            all_adm,
            all_treat
        ),
    })
}

fn summarize_report(pid: &str, report: BenchReport, keep: bool) -> (ProfileResult, bool, bool) {
    let treat_valid = report.treatment.as_ref().map(|t| t.valid).unwrap_or(true);
    let busy = report
        .treatment
        .as_ref()
        .and_then(|t| t.calibration.as_ref())
        .and_then(|c| c.busy_cores);
    let wall = report
        .cells
        .iter()
        .filter(|c| c.family == "search")
        .map(|c| c.cost.wall_us.p95)
        .fold(None, |acc: Option<f64>, v| {
            Some(acc.map(|a| a + v).unwrap_or(v))
        })
        .map(|sum| {
            let n = report
                .cells
                .iter()
                .filter(|c| c.family == "search")
                .count()
                .max(1) as f64;
            sum / n
        });
    let path = report
        .metadata
        .run
        .log_dir
        .clone()
        .map(|d| Path::new(&d).join("report.json").display().to_string());
    let mut reasons = Vec::new();
    if !report.admissible {
        reasons.push("gates_failed".into());
    }
    if !treat_valid {
        reasons.push("treatment_invalid".into());
    }
    let adm = report.admissible;
    (
        ProfileResult {
            load_profile_id: pid.into(),
            admissible: adm,
            treatment_valid: treat_valid,
            typical_wall_p95_us: wall,
            calibration_busy_cores: busy,
            report_path: path,
            reason_codes: reasons,
            report: if keep { Some(report) } else { None },
            comparison: None,
            paired: None,
        },
        adm,
        treat_valid,
    )
}

fn identity_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn clone_run_config(c: &RunConfig) -> RunConfig {
    RunConfig {
        iterations: c.iterations,
        warmup: c.warmup,
        sample_packs: c.sample_packs.clone(),
        timeout_secs: c.timeout_secs,
        suite: c.suite.clone(),
        peak: c.peak,
        log_dir: c.log_dir.clone(),
        require_quiet: c.require_quiet,
        quiet_wait_secs: c.quiet_wait_secs,
        patch_mode: c.patch_mode.clone(),
        history_dir: c.history_dir.clone(),
        history_record: c.history_record,
        primary_span: c.primary_span.clone(),
        adapter: c.adapter.clone(),
        flamegraph: false,
        profile_backend: "none".into(),
        mutation_gate: false,
        converge_warmup: c.converge_warmup,
        load_profile: c.load_profile.clone(),
        network_profile: c.network_profile.clone(),
        slim: c.slim,
    }
}
