//! Timed run loop: exterior snap → adapter.op → snap.

use crate::adapter::FffAdapter;
use crate::cost::{CostSample, CostSummary, IterLog, OpResultData};
use crate::identity::{self, ChangeMeta, NoiseMeta, RepoMeta, RunMeta, StatsMeta};
use crate::noise;
use crate::scenarios::CellKind;
use crate::sensors::Sensors;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellReport {
    pub cell_key: String,
    pub scenario_id: String,
    pub repo_id: String,
    pub axes: serde_json::Value,
    pub kind: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Qualified state identifier (METHODOLOGY state vector).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_id: Option<String>,
    pub cost: CostSummary,
    pub last_result: OpResultData,
    pub rank_target: Option<usize>,
    pub iterations_logged: usize,
    /// Throughput cells: derived queries/sec for the burst.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub throughput_qps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub throughput_qps_dist: Option<crate::cost::Dist>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_wall_us: Option<crate::cost::Dist>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operations_per_iteration: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_concurrency: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrival_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_errors: Option<usize>,
    #[serde(default)]
    pub ordered_digest_consistent: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_converged: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmups_run: Option<usize>,
    /// Typed metrics from the last measured iteration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_metrics: Option<crate::metric::CostMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub schema_version: u32,
    pub admissible: bool,
    pub metadata: ReportMetadata,
    pub gates: crate::gates::GatesReport,
    pub cells: Vec<CellReport>,
    pub noise: NoiseMeta,
    pub change: Option<ChangeMeta>,
    pub score: Option<crate::score_cmd::ScoreOut>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<crate::profile_export::ProfileArtifacts>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator: Option<crate::evaluator::EvaluatorIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stability: Option<crate::stability::StabilitySummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mutation: Option<crate::mutation::MutationReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_coverage: Option<crate::metric::MetricCoverage>,
    /// Optional external agent usage attachment (OPERATIONS; not scored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_usage: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment: Option<crate::load_actor::TreatmentEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<crate::network::NetworkEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub run: RunMeta,
    pub subject: identity::SubjectMeta,
    pub harness: identity::HarnessMeta,
    pub machine: identity::MachineMeta,
    pub repos: Vec<RepoMeta>,
    pub stats: StatsMeta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationMeta>,
}

/// Core thread-calibration actually used for this report's ops.
/// Recorded per report so paired runs can attribute thread-count
/// differences instead of inferring them from wall deltas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMeta {
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub llc_bytes: usize,
    pub match_work_per_thread: usize,
    pub scan_cap: usize,
}

fn default_block_iters() -> usize {
    12
}

fn default_block_warmup() -> usize {
    3
}

fn default_block_load() -> String {
    "quiet_v1".into()
}

fn default_block_sample() -> String {
    "mem,cpu".into()
}

fn default_block_slim() -> bool {
    true
}

fn default_block_patch() -> String {
    "none".into()
}

fn default_block_timeout() -> u64 {
    600
}

fn default_block_repo() -> String {
    "default".into()
}

/// One persistent-subject measurement (run-stream stdin line).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSpec {
    pub suite: String,
    pub corpus: PathBuf,
    #[serde(default = "default_block_iters")]
    pub iters: usize,
    #[serde(default = "default_block_warmup")]
    pub warmup: usize,
    #[serde(default = "default_block_load")]
    pub load_profile: String,
    #[serde(default = "default_block_sample")]
    pub sample: String,
    #[serde(default = "default_block_slim")]
    pub slim: bool,
    #[serde(default = "default_block_patch")]
    pub patch: String,
    #[serde(default = "default_block_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub content_index: bool,
    #[serde(default)]
    pub history_dir: Option<PathBuf>,
    #[serde(default)]
    pub block_index: usize,
    #[serde(default = "default_block_repo")]
    pub repo_id: String,
}

/// Per-block failure line (run-stream stdout); disjoint from BenchReport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamError {
    pub block_index: Option<usize>,
    pub error: String,
}

/// Suites with convergence warmup + mutation gate (mirrors run handler).
pub fn confirmish_suite(suite: &str) -> bool {
    suite.contains("confirm") || suite.contains("peak") || suite == "full"
}

fn stream_default_history_dir() -> PathBuf {
    if let Ok(h) = std::env::var("FFF_BENCH_HISTORY") {
        return PathBuf::from(h);
    }
    PathBuf::from(".fff-bench/history")
}

/// Persistent subject: one open corpus serving many blocks.
pub struct StreamSession {
    open_path: Option<PathBuf>,
    adapter: Option<FffAdapter>,
}

impl StreamSession {
    pub fn new() -> Self {
        Self {
            open_path: None,
            adapter: None,
        }
    }

    pub fn measure(&mut self, spec: &BlockSpec) -> Result<BenchReport, String> {
        let suite = crate::agent::normalize_suite(&spec.suite);
        let canon =
            fff::path_utils::canonicalize(&spec.corpus).unwrap_or_else(|_| spec.corpus.clone());
        let reused = self.open_path.as_ref() == Some(&canon) && self.adapter.is_some();
        if !reused {
            let mut acfg = crate::adapter::AdapterConfig::default();
            acfg.timeout_secs = spec.timeout_secs;
            if spec.content_index {
                acfg.enable_content_indexing = true;
            }
            let adapter = FffAdapter::open_corpus(&canon, acfg.clone())?;
            self.open_path = Some(canon.clone());
            self.adapter = Some(adapter);
        } else if let Some(a) = self.adapter.as_ref() {
            a.flush_block_state();
        }
        let adapter = self.adapter.as_ref().ok_or("stream: no adapter open")?;
        let confirmish = confirmish_suite(&suite);
        let profile = if spec.load_profile.is_empty() {
            "quiet_v1"
        } else {
            spec.load_profile.as_str()
        };
        let mut acfg = crate::adapter::AdapterConfig::default();
        acfg.timeout_secs = spec.timeout_secs;
        if spec.content_index {
            acfg.enable_content_indexing = true;
        }
        let cfg = RunConfig {
            iterations: spec.iters,
            warmup: spec.warmup,
            sample_packs: spec.sample.clone(),
            timeout_secs: spec.timeout_secs,
            suite: suite.clone(),
            peak: false,
            log_dir: None,
            require_quiet: false,
            quiet_wait_secs: 0,
            patch_mode: spec.patch.clone(),
            history_dir: None,
            history_record: false,
            primary_span: "harness.op_total".into(),
            adapter: acfg,
            flamegraph: false,
            profile_backend: "none".into(),
            mutation_gate: confirmish,
            converge_warmup: confirmish,
            load_profile: profile.into(),
            network_profile: "native_local_v1".into(),
            slim: spec.slim,
        };

        let started = crate::identity::now_unix();
        let subject = crate::identity::subject_meta(FffAdapter::id());
        let run_id = crate::identity::make_run_id(&subject);
        noise::wait_quiet(cfg.quiet_wait_secs, cfg.require_quiet)?;
        let net_profile = crate::network::resolve_profile(&cfg.network_profile)
            .ok_or_else(|| format!("unknown network_profile: {}", cfg.network_profile))?;
        let network_ev = crate::network::activate(&net_profile);
        let load_profile = crate::load_actor::resolve_profile(&cfg.load_profile)
            .ok_or_else(|| format!("unknown load_profile: {}", cfg.load_profile))?;
        let treatment = crate::load_actor::Treatment::start(load_profile)?;

        let repo_id = if spec.repo_id.is_empty() {
            "default".to_string()
        } else {
            spec.repo_id.clone()
        };
        let paths = adapter.path_list();
        let repo_metas = vec![crate::identity::RepoMeta {
            id: repo_id.clone(),
            path: canon.display().to_string(),
            role: vec!["gates".into(), "profile".into()],
            optional: false,
            file_count_live: Some(adapter.file_count()),
            path_list_hash: Some(crate::identity::path_list_hash(&paths)),
            git_commit: crate::identity::git_rev(&canon),
            git_dirty: false,
            pin: "path".into(),
        }];
        let cells = crate::scenarios::expand_suite(&suite, &repo_id);
        let repos = vec![(repo_id.clone(), canon.clone())];
        let mut report = measure_with_adapters(
            &repos,
            |id| (id == repo_id).then_some(adapter),
            repo_metas,
            cells,
            &cfg,
            started,
            subject,
            run_id,
            network_ev,
            treatment,
        )?;
        // Mirror the run handler: score against history without recording.
        let hist = spec
            .history_dir
            .clone()
            .unwrap_or_else(stream_default_history_dir);
        let baseline = crate::history::latest_comparable_any(
            &hist,
            &[
                suite.as_str(),
                crate::agent::CONFIRM_SUITE,
                crate::agent::SCREEN_SUITE,
            ],
            Some(report.metadata.run.run_id.as_str()),
        )
        .and_then(|p| {
            std::fs::read_to_string(&p)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        });
        report.score = Some(crate::score_cmd::compute(&report, baseline.as_ref()));
        Ok(report)
    }
}

/// Read block specs (JSON lines) on input, write one report or error
/// object per block on output. Returns blocks measured.
pub fn run_stream_loop<R: BufRead, W: Write>(input: R, mut output: W) -> Result<usize, String> {
    let mut session = StreamSession::new();
    let mut blocks = 0usize;
    for line in input.lines() {
        let line = line.map_err(|e| format!("read block spec: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let spec: BlockSpec =
            serde_json::from_str(&line).map_err(|e| format!("parse block spec: {e}"))?;
        match session.measure(&spec) {
            Ok(report) => {
                serde_json::to_writer(&mut output, &report)
                    .map_err(|e| format!("write report: {e}"))?;
                output
                    .write_all(b"\n")
                    .map_err(|e| format!("write report: {e}"))?;
            }
            Err(e) => {
                let err = StreamError {
                    block_index: Some(spec.block_index),
                    error: e,
                };
                serde_json::to_writer(&mut output, &err)
                    .map_err(|e| format!("write error: {e}"))?;
                output
                    .write_all(b"\n")
                    .map_err(|e| format!("write error: {e}"))?;
            }
        }
        output.flush().map_err(|e| format!("flush report: {e}"))?;
        blocks += 1;
    }
    Ok(blocks)
}

pub struct RunConfig {
    pub iterations: usize,
    pub warmup: usize,
    pub sample_packs: String,
    pub timeout_secs: u64,
    pub suite: String,
    pub peak: bool,
    pub log_dir: Option<PathBuf>,
    pub require_quiet: bool,
    pub quiet_wait_secs: u64,
    pub patch_mode: String,
    pub history_dir: Option<PathBuf>,
    pub history_record: bool,
    pub primary_span: String,
    pub adapter: crate::adapter::AdapterConfig,
    pub flamegraph: bool,
    pub profile_backend: String,
    /// Run mutation/watcher gate on isolated fixture (agent-confirm / diagnose).
    pub mutation_gate: bool,
    /// Use convergence warmup instead of fixed warmup count.
    pub converge_warmup: bool,
    /// Load profile id: quiet_v1 | peer_cpu_light_v1 | developer_composite_v1
    pub load_profile: String,
    /// Network environment profile: native_local_v1 | loopback_fixture_v1 | external_shared_v1
    pub network_profile: String,
    /// Slim output: skip per-iter logs and top_k hit details.
    pub slim: bool,
}

struct TimedCallOutcome {
    wall_us: u64,
    result: OpResultData,
    cost: CostSample,
    qps: Option<f64>,
    request_walls_us: Vec<f64>,
    operations: usize,
    errors: usize,
}

pub fn run_bench(repo_path: &Path, repo_id: &str, cfg: &RunConfig) -> Result<BenchReport, String> {
    run_bench_multi(&[(repo_id.to_string(), repo_path.to_path_buf())], cfg)
}

/// Multi-repo exterior run: one adapter per corpus, cells unioned.
pub fn run_bench_multi(
    repos: &[(String, PathBuf)],
    cfg: &RunConfig,
) -> Result<BenchReport, String> {
    let started = identity::now_unix();
    let subject = identity::subject_meta(FffAdapter::id());
    let run_id = identity::make_run_id(&subject);

    if cfg.require_quiet || cfg.quiet_wait_secs > 0 {
        noise::wait_quiet(cfg.quiet_wait_secs, cfg.require_quiet)?;
    }

    if repos.is_empty() {
        return Err("no repos to benchmark".into());
    }

    // Network/environment identity (Phase 9). Default native_local for FFF.
    let net_profile = crate::network::resolve_profile(&cfg.network_profile)
        .ok_or_else(|| format!("unknown network_profile: {}", cfg.network_profile))?;
    let network_ev = crate::network::activate(&net_profile);
    if network_ev.decision_grade == "informational" {
        // Still run, but mark later for comparability.
    }

    // Controlled interference treatment (Phase 8). Quiet is default.
    let load_profile = crate::load_actor::resolve_profile(&cfg.load_profile)
        .ok_or_else(|| format!("unknown load_profile: {}", cfg.load_profile))?;
    let treatment_session = crate::load_actor::Treatment::start(load_profile)?;

    let mut adapters: Vec<(String, FffAdapter, usize, String)> = Vec::new();
    let mut cells = Vec::new();
    let mut repo_metas = Vec::new();

    for (repo_id, repo_path) in repos {
        let mut acfg = cfg.adapter.clone();
        acfg.timeout_secs = cfg.timeout_secs;
        let adapter = FffAdapter::open_corpus(repo_path, acfg)?;
        let paths = adapter.path_list();
        let path_hash = identity::path_list_hash(&paths);
        let file_count = adapter.file_count();
        repo_metas.push(RepoMeta {
            id: repo_id.clone(),
            path: repo_path.display().to_string(),
            role: vec!["gates".into(), "profile".into()],
            optional: false,
            file_count_live: Some(file_count),
            path_list_hash: Some(path_hash),
            git_commit: identity::git_rev(repo_path),
            // Never run `git status` on the corpus. On sumo_fork that is a
            // 2.7GB object store and blocks the harness for minutes. Path
            // hash + rev is enough identity for search benches.
            git_dirty: false,
            pin: "path".into(),
        });
        cells.extend(crate::scenarios::expand_suite(&cfg.suite, repo_id));
        adapters.push((repo_id.clone(), adapter, file_count, String::new()));
    }

    measure_with_adapters(
        repos,
        |id| {
            adapters
                .iter()
                .find(|(rid, _, _, _)| rid == id)
                .map(|(_, a, _, _)| a)
        },
        repo_metas,
        cells,
        cfg,
        started,
        subject,
        run_id,
        network_ev,
        treatment_session,
    )
}

/// Shared cell loop + gates + report assembly over borrowed adapters.
/// One-shot `run` and persistent `run-stream` blocks run this same
/// function, so verdict/CI/cap/gate semantics cannot drift.
#[allow(clippy::too_many_arguments)]
fn measure_with_adapters<'a, F>(
    repos: &[(String, PathBuf)],
    find_adapter: F,
    repo_metas: Vec<RepoMeta>,
    cells: Vec<crate::scenarios::Cell>,
    cfg: &RunConfig,
    started: u64,
    subject: identity::SubjectMeta,
    run_id: String,
    network_ev: crate::network::NetworkEvidence,
    treatment_session: crate::load_actor::Treatment,
) -> Result<BenchReport, String>
where
    F: Fn(&str) -> Option<&'a FffAdapter>,
{
    let sensors = Sensors::from_packs(&cfg.sample_packs);
    let log_dir = cfg.log_dir.clone().or_else(|| {
        cfg.history_record
            .then(|| PathBuf::from(".fff-bench/runs").join(&run_id))
    });
    if let Some(ref d) = log_dir {
        fs::create_dir_all(d).map_err(|e| e.to_string())?;
        fs::create_dir_all(d.join("scenarios")).ok();
    }

    let mut cell_reports = Vec::new();
    let mut gate_pairs: Vec<(String, OpResultData)> = Vec::new();
    let mut all_walls: Vec<f64> = Vec::new();

    for cell in &cells {
        let repo_path = repos
            .iter()
            .find(|(id, _)| id == &cell.repo_id)
            .map(|(_, p)| p.clone())
            .ok_or_else(|| format!("no path for repo {}", cell.repo_id))?;
        let adapter = find_adapter(&cell.repo_id)
            .ok_or_else(|| format!("no adapter for repo {}", cell.repo_id))?;

        // Cold open: time a fresh open (edge / launch path)
        if cell.kind == CellKind::ColdOpen {
            let mut samples = Vec::new();
            for i in 0..(cfg.warmup + 1).max(1) {
                let snap0 = sensors.snapshot();
                let t0 = Instant::now();
                let mut acfg = cfg.adapter.clone();
                acfg.timeout_secs = cfg.timeout_secs;
                let _fresh = FffAdapter::open_corpus(&repo_path, acfg)?;
                let wall_us = t0.elapsed().as_micros() as u64;
                let snap1 = sensors.snapshot();
                let delta = sensors.delta(&snap0, &snap1);
                let mut cost = CostSample::with_wall(wall_us, 1);
                cost.primary_span = "subject.open".into();
                apply_delta(&mut cost, &delta, &sensors);
                cost.finalize_derived();
                cost.metrics = Some(crate::metric::CostMetrics::from_sample(
                    &cost,
                    &cfg.sample_packs,
                ));
                if i >= cfg.warmup {
                    samples.push(cost);
                    all_walls.push(wall_us as f64);
                }
            }
            let state_id = cell
                .axes
                .get("state_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| Some("open_fresh_index_absent".into()));
            let last_metrics = samples.last().and_then(|s| s.metrics.clone());
            cell_reports.push(CellReport {
                cell_key: cell.cell_key.clone(),
                scenario_id: cell.scenario_id.clone(),
                repo_id: cell.repo_id.clone(),
                axes: cell.axes.clone(),
                kind: format!("{:?}", cell.kind),
                family: cell.family.clone(),
                tags: cell.tags.clone(),
                state_id,
                cost: CostSummary::from_samples(&samples),
                last_result: OpResultData::default(),
                rank_target: None,
                iterations_logged: samples.len(),
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
                last_metrics,
            });
            continue;
        }

        let Some(search) = &cell.search else { continue };
        let spec = search.to_spec();
        let mut samples: Vec<CostSample> = Vec::new();
        let mut last_result = OpResultData::default();
        let mut throughput_samples = Vec::new();
        let mut request_walls = Vec::new();
        let mut request_errors = 0usize;
        let mut result_digests = std::collections::BTreeSet::new();
        let mut operations_per_iteration = None;

        let mut iter_writer = log_dir.as_ref().and_then(|d| {
            if cfg.slim {
                return None;
            }
            let safe = cell
                .cell_key
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect::<String>();
            let path = d.join("scenarios").join(format!("{safe}.jsonl"));
            File::create(path).ok().map(BufWriter::new)
        });

        // Cold-start cells use fixed independent warmups; confirm uses convergence.
        let coldish = cell.family == "cold" || cell.kind == CellKind::ColdOpen;
        let use_converge = cfg.converge_warmup && !coldish;
        let mut warmup_walls: Vec<f64> = Vec::new();
        let mut warmups_done = 0usize;
        let min_w = if use_converge {
            crate::warmup::MIN_WARMUPS.max(cfg.warmup)
        } else {
            cfg.warmup
        };
        let max_w = if use_converge {
            crate::warmup::MAX_WARMUPS
        } else {
            cfg.warmup
        };

        // Phase 1: warmups
        while warmups_done < max_w {
            let outcome = timed_call(adapter, cell, search, &spec, &sensors, cfg);
            warmup_walls.push(outcome.wall_us as f64);
            warmups_done += 1;
            last_result = outcome.result;
            if let Some(digest) = &last_result.ordered_digest {
                result_digests.insert(digest.clone());
            }
            if let Some(w) = iter_writer.as_mut() {
                log_iter(w, cell, warmups_done - 1, true, outcome.cost, &last_result);
            }
            if !use_converge {
                if warmups_done >= cfg.warmup {
                    break;
                }
            } else if !crate::warmup::needs_more(
                &warmup_walls,
                min_w,
                max_w,
                crate::warmup::WINDOW,
                crate::warmup::THRESHOLD,
            ) {
                break;
            }
        }
        let warmup_out = crate::warmup::finalize(
            &warmup_walls,
            max_w,
            crate::warmup::WINDOW,
            crate::warmup::THRESHOLD,
        );

        // Phase 2: measured
        for i in 0..cfg.iterations {
            let outcome = timed_call(adapter, cell, search, &spec, &sensors, cfg);
            samples.push(outcome.cost.clone());
            all_walls.push(outcome.wall_us as f64);
            if let Some(qps) = outcome.qps {
                throughput_samples.push(qps);
            }
            request_walls.extend(outcome.request_walls_us);
            request_errors = request_errors.saturating_add(outcome.errors);
            operations_per_iteration = Some(outcome.operations);
            last_result = outcome.result;
            if let Some(digest) = &last_result.ordered_digest {
                result_digests.insert(digest.clone());
            }
            if let Some(w) = iter_writer.as_mut() {
                log_iter(w, cell, warmups_done + i, false, outcome.cost, &last_result);
            }
        }

        if matches!(cell.kind, CellKind::GateGolden | CellKind::Grep) {
            // Grep goldens also feed gate evaluation via cell_reports.
            if matches!(cell.kind, CellKind::GateGolden) {
                gate_pairs.push((cell.cell_key.clone(), last_result.clone()));
            }
        }

        let state_id = cell
            .axes
            .get("state_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| Some("search_reused_index_warm".into()));

        let last_metrics = samples.last().and_then(|s| s.metrics.clone());
        let throughput_qps_dist = (!throughput_samples.is_empty())
            .then(|| crate::cost::Dist::from_values(throughput_samples));
        let throughput_qps = throughput_qps_dist.as_ref().map(|d| d.p50);
        let request_wall_us =
            (!request_walls.is_empty()).then(|| crate::cost::Dist::from_values(request_walls));
        let is_workload = matches!(
            cell.kind,
            CellKind::ThroughputBurst | CellKind::ThroughputConcurrent | CellKind::GrepPagination
        );
        let mut cost = CostSummary::from_samples(&samples);
        if let Some(qps) = throughput_qps {
            cost.qps_derived = qps;
        }
        cell_reports.push(CellReport {
            cell_key: cell.cell_key.clone(),
            scenario_id: cell.scenario_id.clone(),
            repo_id: cell.repo_id.clone(),
            axes: cell.axes.clone(),
            kind: format!("{:?}", cell.kind),
            family: cell.family.clone(),
            tags: cell.tags.clone(),
            state_id,
            cost,
            last_result: last_result.clone(),
            rank_target: last_result.rank_target,
            iterations_logged: samples.len(),
            throughput_qps,
            throughput_qps_dist,
            request_wall_us,
            operations_per_iteration: is_workload.then_some(operations_per_iteration.unwrap_or(1)),
            client_concurrency: is_workload.then_some(search.clients.max(1)),
            arrival_model: is_workload.then(|| "closed_loop".to_string()),
            request_errors: is_workload.then_some(request_errors),
            ordered_digest_consistent: result_digests.len() <= 1,
            warmup_converged: Some(warmup_out.converged),
            warmups_run: Some(warmup_out.warmups_run),
            last_metrics,
        });
    }

    // Evaluate gates per-repo adapter (goldens + invariants + threads + grep)
    let mut gates = crate::gates::GatesReport {
        passed: true,
        failed: vec![],
        hit_at_1: 1.0,
        hit_at_5: 1.0,
        mrr: 1.0,
        cases: vec![],
        invariants_ok: true,
        entries: vec![],
    };
    for (repo_id, _) in repos {
        let adapter =
            find_adapter(repo_id).ok_or_else(|| format!("no adapter for repo {repo_id}"))?;
        let repo_cells: Vec<_> = cells
            .iter()
            .filter(|c| &c.repo_id == repo_id)
            .cloned()
            .collect();
        let pairs: Vec<_> = gate_pairs
            .iter()
            .filter(|(k, _)| repo_cells.iter().any(|c| &c.cell_key == k))
            .cloned()
            .collect();
        let repo_reports: Vec<_> = cell_reports
            .iter()
            .filter(|c| &c.repo_id == repo_id)
            .cloned()
            .collect();
        let g = crate::gates::evaluate_all(adapter, &repo_cells, &pairs, &repo_reports);
        gates.cases.extend(g.cases);
        gates.failed.extend(g.failed);
        gates.entries.extend(g.entries);
        gates.invariants_ok &= g.invariants_ok;
        gates.passed &= g.passed;
    }
    if !gates.cases.is_empty() {
        let n = gates.cases.len() as f64;
        let hits1 = gates.cases.iter().filter(|c| c.rank == Some(0)).count() as f64;
        let hits5 = gates
            .cases
            .iter()
            .filter(|c| c.rank.map(|r| r < 5).unwrap_or(false))
            .count() as f64;
        let mrr: f64 = gates
            .cases
            .iter()
            .map(|c| c.rank.map(|r| 1.0 / (r as f64 + 1.0)).unwrap_or(0.0))
            .sum::<f64>()
            / n;
        gates.hit_at_1 = hits1 / n;
        gates.hit_at_5 = hits5 / n;
        gates.mrr = mrr;
        if gates.hit_at_5 < 0.80 || gates.mrr < 0.50 {
            if !gates.failed.iter().any(|f| f == "golden_budget") {
                gates.failed.push("golden_budget".into());
            }
            gates.passed = false;
        }
    }
    gates.passed = gates.failed.is_empty();

    // Mutation gate (isolated fixture) for confirm/diagnose suites.
    let mutation = if cfg.mutation_gate {
        match crate::mutation::run_mutation_gate(
            repos
                .first()
                .map(|(_, p)| p.as_path())
                .unwrap_or(Path::new(".")),
        ) {
            Ok(m) => {
                if !m.passed {
                    gates.failed.push("mutation".into());
                    gates.passed = false;
                }
                gates.entries.extend(m.entries.clone());
                Some(m)
            }
            Err(e) => {
                gates.failed.push(format!("mutation_infra:{e}"));
                gates.passed = false;
                None
            }
        }
    } else {
        None
    };

    let treatment = treatment_session.finish();
    let treat_gate = crate::load_actor::validity_gate(&treatment);
    if treat_gate.status == "fail" {
        gates.failed.push("treatment".into());
        gates.passed = false;
    }
    gates.entries.push(treat_gate);
    let net_gate = crate::network::validity_gate(&network_ev);
    if net_gate.status == "fail" {
        gates.failed.push("network".into());
        gates.passed = false;
    }
    gates.entries.push(net_gate);

    // Mark unconverged warmups in stability reasons.
    let unconverged = cell_reports
        .iter()
        .filter(|c| c.warmup_converged == Some(false))
        .count();

    let noise = noise::summarize(&all_walls);
    let mut stability = crate::stability::from_cells(&cell_reports, &noise, cfg.iterations);
    if unconverged > 0 {
        stability.reason_codes.push("warmup_not_converged".into());
        if stability.stability == "clean" {
            stability.stability = "elevated".into();
        }
    }
    if !treatment.valid {
        stability
            .reason_codes
            .push("treatment_delivery_invalid".into());
        stability.stability = "unstable".into();
    }

    let required = if cfg.suite.contains("confirm") || cfg.suite.contains("peak") {
        vec!["wall_us", "cpu_total_us", "rss"]
    } else {
        vec!["wall_us"]
    };
    // Prefer typed metrics from last measured iteration when present.
    let metric_coverage = if let Some(m) = cell_reports.iter().find_map(|c| c.last_metrics.as_ref())
    {
        crate::metric::coverage_from_metrics(&required, m)
    } else {
        let mut available = vec!["wall_us"];
        if cell_reports.iter().any(|c| c.cost.cpu_total_us.is_some()) {
            available.push("cpu_total_us");
        }
        if cell_reports
            .iter()
            .any(|c| c.cost.rss_peak_mb.is_some() || c.cost.rss_mb_end.is_some())
        {
            available.push("rss");
        }
        crate::metric::coverage(&required, &available)
    };

    let finished = identity::now_unix();
    let evaluator = crate::evaluator::current(FffAdapter::id());
    let cell_keys: Vec<String> = cell_reports.iter().map(|c| c.cell_key.clone()).collect();
    let repo_ids: Vec<String> = repos.iter().map(|(id, _)| id.clone()).collect();
    let plan_digest = crate::evaluator::plan_digest(&cfg.suite, &repo_ids, &cell_keys);

    let change = capture_patch(&cfg.patch_mode, log_dir.as_deref(), &subject);

    let profile = log_dir.as_ref().map(|d| {
        crate::profile_export::try_flamegraph(
            d,
            cfg.flamegraph,
            &cfg.profile_backend,
            &["fff-bench", "list", "--suite", "agent-screen"],
        )
    });

    let mut report = BenchReport {
        schema_version: 1,
        admissible: gates.passed,
        metadata: ReportMetadata {
            run: RunMeta {
                run_id: run_id.clone(),
                cmdline: std::env::args().collect::<Vec<_>>().join(" "),
                started_at_unix: started,
                finished_at_unix: Some(finished),
                duration_ms: Some(finished.saturating_sub(started).saturating_mul(1000)),
                log_dir: log_dir.as_ref().map(|p| p.display().to_string()),
                peak: cfg.peak,
                sample_packs: cfg.sample_packs.clone(),
                primary_span: cfg.primary_span.clone(),
                suite: cfg.suite.clone(),
            },
            subject,
            harness: identity::harness_meta(),
            machine: identity::machine_meta(),
            repos: repo_metas,
            stats: StatsMeta {
                warmup: cfg.warmup,
                iterations: cfg.iterations,
                threads: 4,
                rng_seed: 1,
                primary_span: cfg.primary_span.clone(),
            },
            calibration: Some({
                let c = fff::calibration_snapshot();
                CalibrationMeta {
                    logical_cores: c.logical_cores,
                    physical_cores: c.physical_cores,
                    llc_bytes: c.llc_bytes,
                    match_work_per_thread: c.match_work_per_thread,
                    scan_cap: c.scan_cap,
                }
            }),
        },
        gates,
        cells: cell_reports,
        noise: noise.clone(),
        change,
        score: None,
        profile,
        evaluator: Some(evaluator),
        plan_digest: Some(plan_digest),
        stability: Some(stability),
        mutation,
        metric_coverage: Some(metric_coverage),
        agent_usage: None,
        treatment: Some(treatment),
        network: Some(network_ev),
    };

    report.admissible = report.gates.passed;

    if cfg.slim {
        slim_report(&mut report);
    }

    if let Some(ref d) = log_dir {
        let path = d.join("report.json");
        if let Ok(s) = serde_json::to_string_pretty(&report) {
            let _ = fs::write(path, s);
        }
    }
    // History is recorded by main after score is attached (avoid double index).

    Ok(report)
}

fn timed_call(
    adapter: &FffAdapter,
    cell: &crate::scenarios::Cell,
    search: &crate::scenarios::SearchCell,
    spec: &crate::adapter::SearchSpec,
    sensors: &Sensors,
    cfg: &RunConfig,
) -> TimedCallOutcome {
    if cell
        .axes
        .get("cache_state")
        .and_then(|value| value.as_str())
        == Some("cold_query")
    {
        for index in 0..9 {
            let mut eviction = spec.clone();
            eviction.query = format!("__fff_bench_cache_evict_{index}");
            let _ = adapter.call(&eviction);
        }
    }
    let snap0 = sensors.snapshot();
    let t0 = Instant::now();
    let (result, operations, request_walls_us, errors) = match cell.kind {
        CellKind::ThroughputBurst => {
            let mut last = OpResultData::default();
            let mut request_walls_us = Vec::new();
            let mut errors = 0usize;
            let stream = if !search.stream.is_empty() {
                search.stream.clone()
            } else if let Some(arr) = cell.axes.get("stream").and_then(|v| v.as_array()) {
                arr.iter()
                    .filter_map(|q| q.as_str().map(|s| s.to_string()))
                    .collect()
            } else {
                vec![]
            };
            if !stream.is_empty() {
                for step in &stream {
                    let (call, q) = if let Some((c, rest)) = step.split_once(':') {
                        if matches!(
                            c,
                            "file"
                                | "dir"
                                | "mixed"
                                | "grep_plain"
                                | "grep_fuzzy"
                                | "grep_regex"
                                | "grep"
                        ) {
                            (Some(c), rest)
                        } else {
                            (search.call_kind.as_deref(), step.as_str())
                        }
                    } else {
                        (search.call_kind.as_deref(), step.as_str())
                    };
                    let step_spec = crate::adapter::SearchSpec {
                        query: q.into(),
                        limit: search.limit,
                        offset: 0,
                        threads: search.threads,
                        call: parse_call_kind(call.unwrap_or("file")),
                    };
                    let request_started = Instant::now();
                    last = adapter.call(&step_spec);
                    request_walls_us.push(request_started.elapsed().as_secs_f64() * 1_000_000.0);
                    errors += usize::from(last.error.is_some());
                }
                (last, stream.len(), request_walls_us, errors)
            } else {
                let n = search.burst.max(1);
                for _ in 0..n {
                    let request_started = Instant::now();
                    last = adapter.call(spec);
                    request_walls_us.push(request_started.elapsed().as_secs_f64() * 1_000_000.0);
                    errors += usize::from(last.error.is_some());
                }
                (last, n, request_walls_us, errors)
            }
        }
        CellKind::ThroughputConcurrent => {
            let operations = search.burst.max(1);
            let clients = search.clients.max(1).min(operations);
            let barrier = Arc::new(Barrier::new(clients));
            let mut last = OpResultData::default();
            let mut request_walls_us = Vec::with_capacity(operations);
            let mut errors = 0usize;
            std::thread::scope(|scope| {
                let mut workers = Vec::with_capacity(clients);
                for client in 0..clients {
                    let count = operations / clients + usize::from(client < operations % clients);
                    let barrier = barrier.clone();
                    let spec = spec.clone();
                    workers.push(scope.spawn(move || {
                        let mut last = OpResultData::default();
                        let mut walls = Vec::with_capacity(count);
                        let mut errors = 0usize;
                        barrier.wait();
                        for _ in 0..count {
                            let request_started = Instant::now();
                            last = adapter.call(&spec);
                            walls.push(request_started.elapsed().as_secs_f64() * 1_000_000.0);
                            errors += usize::from(last.error.is_some());
                        }
                        (last, walls, errors)
                    }));
                }
                for worker in workers {
                    let (worker_last, mut walls, worker_errors) =
                        worker.join().expect("throughput worker panicked");
                    last = worker_last;
                    request_walls_us.append(&mut walls);
                    errors = errors.saturating_add(worker_errors);
                }
            });
            (last, operations, request_walls_us, errors)
        }
        CellKind::GrepPagination => {
            let mut page_spec = spec.clone();
            page_spec.offset = 0;
            let mut last = OpResultData::default();
            let mut request_walls_us = Vec::new();
            let mut errors = 0usize;
            for _ in 0..search.pages.max(1) {
                let request_started = Instant::now();
                let (result, next_file_offset) = adapter.grep_page(&page_spec);
                request_walls_us.push(request_started.elapsed().as_secs_f64() * 1_000_000.0);
                errors += usize::from(result.error.is_some());
                last = result;
                if next_file_offset == 0 {
                    break;
                }
                page_spec.offset = next_file_offset;
            }
            let operations = request_walls_us.len();
            (last, operations, request_walls_us, errors)
        }
        CellKind::Grep => {
            let result = adapter.call(spec);
            let errors = usize::from(result.error.is_some());
            (result, 1, vec![], errors)
        }
        _ => {
            let result = adapter.call(spec);
            let errors = usize::from(result.error.is_some());
            (result, 1, vec![], errors)
        }
    };
    let wall_us = t0.elapsed().as_micros() as u64;
    let snap1 = sensors.snapshot();
    let delta = sensors.delta(&snap0, &snap1);
    let mut cost = CostSample::with_wall(wall_us, search.threads);
    cost.primary_span = cfg.primary_span.clone();
    apply_delta(&mut cost, &delta, sensors);
    cost.finalize_derived();
    cost.metrics = Some(crate::metric::CostMetrics::from_sample(
        &cost,
        &cfg.sample_packs,
    ));
    let mut result = result;
    if let Some(ref expect) = search.expect_path_contains {
        result.rank_target = adapter.rank_of(&result, expect);
    }
    let qps = if operations > 1 && wall_us > 0 {
        Some(operations as f64 / (wall_us as f64 / 1_000_000.0))
    } else {
        None
    };
    TimedCallOutcome {
        wall_us,
        result,
        cost,
        qps,
        request_walls_us,
        operations,
        errors,
    }
}

fn parse_call_kind(call: &str) -> crate::adapter::CallKind {
    match call {
        "dir" => crate::adapter::CallKind::Dir,
        "mixed" => crate::adapter::CallKind::Mixed,
        "grep_plain" | "grep" => crate::adapter::CallKind::GrepPlain,
        "grep_fuzzy" => crate::adapter::CallKind::GrepFuzzy,
        "grep_regex" => crate::adapter::CallKind::GrepRegex,
        _ => crate::adapter::CallKind::File,
    }
}

fn log_iter(
    w: &mut BufWriter<File>,
    cell: &crate::scenarios::Cell,
    iter: usize,
    warmup: bool,
    cost: CostSample,
    result: &OpResultData,
) {
    let log = IterLog {
        t_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
        scenario_id: cell.scenario_id.clone(),
        repo_id: cell.repo_id.clone(),
        cell: cell.axes.clone(),
        cell_key: cell.cell_key.clone(),
        iter,
        warmup,
        cost,
        result: result.clone(),
    };
    if let Ok(line) = serde_json::to_string(&log) {
        let _ = writeln!(w, "{line}");
    }
}

fn apply_delta(cost: &mut CostSample, delta: &crate::sensors::SensorDelta, sensors: &Sensors) {
    cost.cpu_user_us = delta.cpu_user_us;
    cost.cpu_system_us = delta.cpu_system_us;
    cost.rss_start_mb = delta.rss_start_mb;
    cost.rss_mb = delta.rss_end_mb;
    // SENSOR_SEMANTICS: peak only when OS high-water rose in-interval.
    cost.rss_peak_mb = if delta.rss_peak_is_os_hwm {
        delta.rss_peak_mb
    } else {
        if sensors.mem {
            cost.unavailable
                .insert("rss_peak_mb".into(), "no_interval_os_hwm_rise".into());
        }
        None
    };
    cost.disk_read_bytes = delta.disk_read_bytes;
    cost.disk_write_bytes = delta.disk_write_bytes;
    cost.page_faults = delta.page_faults;
    cost.context_switches = delta.context_switches;
    if let Some(t) = delta.threads_peak {
        cost.threads_observed_peak = Some(t as usize);
    }
    cost.cpu_cycles = delta.cpu_cycles;
    cost.energy_j = delta.energy_j;
    cost.energy_source = delta.energy_source.clone();
    if sensors.cpu_unmeasured(delta) {
        cost.unavailable
            .insert("cpu_total_us".into(), "sub_resolution_or_unmeasured".into());
    }
    if sensors.power && delta.energy_j.is_none() {
        cost.unavailable
            .insert("energy_j".into(), "platform_unmeasured".into());
    }
    if sensors.cycles && delta.cpu_cycles.is_none() {
        cost.unavailable
            .insert("cpu_cycles".into(), "platform_unmeasured".into());
    }
    cost.finalize_derived();
}

fn capture_patch(
    mode: &str,
    log_dir: Option<&Path>,
    subject: &identity::SubjectMeta,
) -> Option<ChangeMeta> {
    if mode == "none" {
        return Some(ChangeMeta {
            kind: "none".into(),
            base_ref: None,
            head_commit: subject.git_commit.clone(),
            dirty: subject.git_dirty,
            patch_path: None,
            patch_sha256: None,
            patch_bytes: None,
            summary: None,
        });
    }

    let cwd = std::env::current_dir().ok()?;
    let diff = if mode.starts_with("file:") {
        let p = mode.trim_start_matches("file:");
        std::fs::read_to_string(p).ok()
    } else {
        identity::git_diff(&cwd, None)
    };

    let Some(diff) = diff else {
        return Some(ChangeMeta {
            kind: if subject.git_dirty {
                "git_diff".into()
            } else {
                "none".into()
            },
            base_ref: Some("HEAD".into()),
            head_commit: subject.git_commit.clone(),
            dirty: subject.git_dirty,
            patch_path: None,
            patch_sha256: None,
            patch_bytes: None,
            summary: Some("empty or unavailable".into()),
        });
    };

    let hash = identity::sha256_hex(diff.as_bytes());
    let mut patch_path = None;
    if let Some(d) = log_dir {
        let p = d.join("change.patch");
        if fs::write(&p, &diff).is_ok() {
            patch_path = Some(p.display().to_string());
        }
    }

    Some(ChangeMeta {
        kind: "git_diff".into(),
        base_ref: Some("HEAD".into()),
        head_commit: subject.git_commit.clone(),
        dirty: subject.git_dirty,
        patch_path,
        patch_sha256: Some(hash),
        patch_bytes: Some(diff.len() as u64),
        summary: Some(format!("{} bytes", diff.len())),
    })
}

pub fn print_table(report: &BenchReport) {
    println!("FFF Bench Report  schema={}", report.schema_version);
    println!(
        "run={}  suite={}  admissible={}",
        report.metadata.run.run_id, report.metadata.run.suite, report.admissible
    );
    println!(
        "subject={}@{:?} dirty={}  host={}",
        report.metadata.subject.adapter_id,
        report
            .metadata
            .subject
            .git_commit
            .as_deref()
            .map(|c| &c[..c.len().min(7)]),
        report.metadata.subject.git_dirty,
        report.metadata.machine.hostname
    );
    println!(
        "gates: passed={} hit@5={:.2} mrr={:.2} failed={:?}",
        report.gates.passed, report.gates.hit_at_5, report.gates.mrr, report.gates.failed
    );
    println!(
        "noise: integrity={} host_cpu={:?}% background={:?}% cv={:?}",
        report.noise.integrity,
        report.noise.host_cpu_busy_pct,
        report.noise.background_cpu_pct,
        report.noise.iter_wall_cv
    );
    if let Some(ref t) = report.treatment {
        println!(
            "load: profile={} valid={} placement={:?} reasons={:?}",
            t.load_profile_id, t.valid, t.placement, t.reason_codes
        );
    }
    if let Some(ref n) = report.network {
        println!(
            "network: profile={} valid={} grade={} reasons={:?}",
            n.profile_id, n.valid, n.decision_grade, n.reason_codes
        );
    }
    if let Some(ref cov) = report.metric_coverage {
        println!(
            "metrics: matched={} present={:?} missing={:?}",
            cov.matched, cov.present, cov.missing
        );
    }
    println!();
    for c in &report.cells {
        if c.scenario_id.starts_with("gate") {
            continue;
        }
        println!(
            "ENTRY [{}] {}  wall p50={:.0} p95={:.0} p99={:.0}  thr_cfg={}  qps={:.1}",
            c.family,
            c.axes,
            c.cost.wall_us.p50,
            c.cost.wall_us.p95,
            c.cost.wall_us.p99,
            c.cost.threads_configured,
            c.cost.qps_derived
        );
        if let Some(tp) = c.throughput_qps {
            println!("       burst_throughput_qps={tp:.1}");
        }
        if let Some(ref cpu) = c.cost.cpu_total_us {
            println!(
                "       cpu_total_us p50={:.0} p95={:.0}  parallelism_p50={:.2}",
                cpu.p50,
                cpu.p95,
                c.cost
                    .cpu_parallelism
                    .as_ref()
                    .map(|d| d.p50)
                    .unwrap_or(0.0)
            );
        }
        if let Some(ref cyc) = c.cost.cpu_cycles {
            println!("       cpu_cycles p50={:.0} p95={:.0}", cyc.p50, cyc.p95);
        }
        if let Some(rss) = c.cost.rss_peak_mb {
            println!("       rss_peak_mb={rss:.1}");
        }
        if let Some(t) = c.cost.threads_observed_peak {
            println!("       threads_observed_peak={t}");
        }
        if let Some(ref cs) = c.cost.context_switches {
            println!("       context_switches p50={:.0}", cs.p50);
        }
        if let Some(r) = c.rank_target {
            println!("       rank_target={r}");
        }
        if let Some(ref m) = c.last_metrics {
            let names = m.measured_names();
            if !names.is_empty() {
                println!("       last_metrics_measured={names:?}");
            }
        }
    }
    // Gate-only suites: still show a one-line wall summary from gate cells.
    if report
        .cells
        .iter()
        .all(|c| c.scenario_id.starts_with("gate") || c.family == "quality")
    {
        let walls: Vec<f64> = report
            .cells
            .iter()
            .filter(|c| c.cost.wall_us.n > 0)
            .map(|c| c.cost.wall_us.p95)
            .collect();
        if !walls.is_empty() {
            let mean = walls.iter().sum::<f64>() / walls.len() as f64;
            println!(
                "GATE_CELLS n={} mean_wall_p95={:.0}µs  (see gates.cases for correctness)",
                walls.len(),
                mean
            );
        }
    }
    if let Some(ref s) = report.score {
        println!();
        println!(
            "SCORE status={} score={:?} delta={:?}  {}",
            s.score_status, s.score, s.score_delta, s.summary_for_agent
        );
        if let Some(w) = s.typical_wall_p95_us {
            println!("  typical_wall_p95={w:.0}µs");
        }
        if let Some(t) = s.typical_throughput_qps {
            println!("  typical_throughput_qps={t:.1}");
        }
        if !s.breakdown.is_empty() {
            println!("  breakdown:");
            for b in &s.breakdown {
                println!(
                    "    family={:12} cells={:3} wall_p95_mean={:8.0}µs weight={:.2} score={:?}",
                    b.family, b.cells, b.wall_p95_mean, b.weight, b.score
                );
            }
        }
    }
    if let Some(ref d) = report.metadata.run.log_dir {
        println!("\nLOGDIR {d}");
    }
}

/// Map score_status to process exit code (aligned with agent::AgentResponse).
pub fn score_exit_code(status: &str) -> u8 {
    match status {
        "ok" | "partial" => 0,
        "inadmissible" | "capped" => 1,
        "incomparable" => 3,
        "noisy" | "unstable" => 4,
        _ => 1,
    }
}

// Slim keeps costs/digests/gates, drops verbose hit details.
pub fn slim_report(report: &mut BenchReport) {
    for c in &mut report.cells {
        c.last_result.top_k.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn block_spec_defaults_match_oneshot_argv() {
        let spec: BlockSpec =
            serde_json::from_str(r#"{"suite":"agent-confirm","corpus":"/tmp/c"}"#).unwrap();
        assert_eq!(spec.iters, 12);
        assert_eq!(spec.warmup, 3);
        assert_eq!(spec.load_profile, "quiet_v1");
        assert_eq!(spec.sample, "mem,cpu");
        assert!(spec.slim);
        assert_eq!(spec.patch, "none");
        assert_eq!(spec.timeout_secs, 600);
        assert_eq!(spec.repo_id, "default");
        assert!(!confirmish_suite("ci-gates"));
        assert!(confirmish_suite("agent-confirm"));
    }

    fn tiny_corpus(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "fff-bench-stream-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["alpha.rs", "beta.rs", "gamma.md"] {
            std::fs::write(dir.join(name), format!("content of {name}\n")).unwrap();
        }
        dir
    }

    #[test]
    fn stream_loop_reuses_corpus_across_blocks() {
        let dir = tiny_corpus("reuse");
        let hist = std::env::temp_dir().join(format!(
            "fff-bench-stream-hist-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let input = format!(
            "{}\n{}\n",
            serde_json::json!({
                "suite": "ci-gates", "corpus": dir, "iters": 1,
                "warmup": 0, "block_index": 0, "history_dir": hist,
            }),
            serde_json::json!({
                "suite": "ci-gates", "corpus": dir, "iters": 1,
                "warmup": 0, "block_index": 1, "history_dir": hist,
            }),
        );
        let mut out = Vec::new();
        let blocks = run_stream_loop(Cursor::new(input), &mut out).unwrap();
        assert_eq!(blocks, 2);
        let lines: Vec<&str> = std::str::from_utf8(&out)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 2);
        let a: BenchReport = serde_json::from_str(lines[0]).unwrap();
        let b: BenchReport = serde_json::from_str(lines[1]).unwrap();
        assert!(!a.cells.is_empty());
        let ka: Vec<&str> = a.cells.iter().map(|c| c.cell_key.as_str()).collect();
        let kb: Vec<&str> = b.cells.iter().map(|c| c.cell_key.as_str()).collect();
        assert_eq!(ka, kb);
        assert_eq!(a.metadata.run.suite, "ci-gates");
        assert_ne!(a.metadata.run.run_id, b.metadata.run.run_id);
        // Slim parity with one-shot children: hit details dropped, costs kept.
        assert!(a.cells.iter().all(|c| c.last_result.top_k.is_empty()));
        assert!(a.cells.iter().all(|c| c.cost.wall_us.n > 0));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&hist);
    }

    #[test]
    fn stream_loop_reports_bad_corpus_per_block() {
        let input = serde_json::json!({
            "suite": "ci-gates",
            "corpus": "/nonexistent-fff-bench-corpus",
            "iters": 1, "warmup": 0, "block_index": 7,
        })
        .to_string()
            + "\n";
        let mut out = Vec::new();
        let blocks = run_stream_loop(Cursor::new(input), &mut out).unwrap();
        assert_eq!(blocks, 1);
        let err: StreamError = serde_json::from_str(std::str::from_utf8(&out).unwrap()).unwrap();
        assert_eq!(err.block_index, Some(7));
        assert!(!err.error.is_empty());
    }
}
