//! fff-bench — exterior, pluggable benchmark harness (doc/benchmark).
//!
//! Agent loop: baseline once → check(candidate) → evidence (AGENT_GUIDE / OPERATIONS).

mod adapter;
mod agent;
mod bootstrap;
mod compare;
mod cost;
mod evaluator;
mod fixture;
mod gates;
mod history;
mod identity;
mod inspect_cmd;
mod load_actor;
mod manifest;
mod metric;
mod mutation;
mod network;
mod noise;
mod paired;
mod profile_export;
mod resilience;
mod runner;
mod scenarios;
mod score_cmd;
mod sensors;
mod stability;
mod warmup;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "fff-bench")]
#[command(about = "FFF exterior benchmark harness — gates, cost, agent check")]
#[command(long_about = "\
Exterior benchmark for FFF search/grep (gates + wall/CPU/mem/I/O cost).

Agent loop (JSON):
  fff-bench baseline --path . --json
  fff-bench check --baseline auto --path . --json

Measure once:
  fff-bench run --path . --suite agent-screen -n 20 --sample mem,cpu --json

Build: cargo build -p fff-nvim --bin fff-bench --release
Docs:  doc/benchmark/AGENT_GUIDE.md")]
#[command(after_help = "\
SENSOR PACKS (--sample, comma-separated or 'all'):
  mem     RSS start/end/peak, page faults (+ context switches on Linux)
  cpu     user/system/total CPU µs, cpu_parallelism
  io      process disk read/write bytes
  cycles  process cycle counter (Windows QueryProcessCycleTime; else often unavailable)
  power   package energy joules when RAPL readable (Linux)

SUITES:
  agent-screen      fast screen + gates
  agent-confirm     decision-grade confirm plan
  ci-gates          gates only
  public-v1         public fixture pack
  resilience-confirm  load-profile matrix
  peak-diagnose     wide diagnostic cells

LOAD (--load-profile): quiet_v1 | peer_cpu_light_v1 | peer_cpu_moderate_v1
  | peer_memory_capacity_v1 | peer_storage_v1 | peer_event_v1 | developer_composite_v1

NETWORK (--network-profile): native_local_v1 | loopback_fixture_v1
  | loopback_delay_20ms_v1 | external_shared_v1

ENV: FFF_BENCH_REPO  FFF_BENCH_HISTORY  FFF_BENCH_CACHE")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Scaffold a minimal evaluation pack (fff-bench.json)
    Init {
        #[arg(long, default_value = ".", help = "Directory for the pack")]
        dir: PathBuf,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// Validate schemas, manifests, or reports (read-only)
    Validate {
        #[arg(long, help = "Path to fff-bench.json manifest")]
        manifest: Option<PathBuf>,
        #[arg(long, help = "Path to a report.json")]
        report: Option<PathBuf>,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// Detect semantic benchmark hazards (read-only)
    Lint {
        #[arg(long, default_value = "agent-confirm", help = "Suite whose cells are linted")]
        suite: String,
        #[arg(long, default_value = "default", help = "Corpus id used when expanding cells")]
        repo_id: String,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// Qualify clocks, snapshot overhead, and sensor pack availability
    Calibrate {
        #[arg(long, help = "JSON output (includes sensor_packs probe)")]
        json: bool,
    },
    /// Validate environment, sensors, corpora (read-only)
    Doctor {
        #[arg(long, help = "Corpus root (default: cwd or FFF_BENCH_REPO)")]
        path: Option<PathBuf>,
        #[arg(long, help = "Manifest path")]
        manifest: Option<PathBuf>,
        #[arg(long, help = "Fail when host CPU is busy")]
        require_quiet: bool,
        #[arg(long, help = "JSON agent-style response")]
        json: bool,
    },
    /// Inspect the exact path set indexed by this subject (read-only)
    Corpus {
        #[arg(short, long, help = "Corpus root")]
        path: Option<PathBuf>,
        #[arg(short, long, help = "Write the corpus snapshot as JSON")]
        output: Option<PathBuf>,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// Setup corpora from path and/or --manifest (git_clone with --fetch)
    Setup {
        #[arg(long, help = "Corpus path to verify")]
        path: Option<PathBuf>,
        #[arg(long, help = "Manifest with corpus sources")]
        manifest: Option<PathBuf>,
        #[arg(long, help = "Fetch git_clone corpus sources")]
        fetch: bool,
        #[arg(long, help = "Materialize public-v1 fixture under .fff-bench/cache")]
        fixture: bool,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// List scenarios / expanded cells
    List {
        #[arg(long, default_value = "agent-screen", help = "Suite id")]
        suite: String,
        #[arg(long, default_value = "default", help = "Corpus id in cell keys")]
        repo_id: String,
        #[arg(long, help = "JSON output")]
        json: bool,
        #[arg(long, help = "Expand full cell list instead of suite summary")]
        expanded: bool,
        #[arg(long, help = "Manifest path")]
        manifest: Option<PathBuf>,
    },
    /// Resolve expanded cells and plan digest (optionally freeze to disk)
    Plan {
        #[arg(long, default_value = "agent-confirm", help = "Suite id")]
        suite: String,
        #[arg(long, default_value = "default", help = "Corpus id")]
        repo_id: String,
        #[arg(long, help = "Write frozen plan under .fff-bench or --output")]
        freeze: bool,
        #[arg(short, long, help = "Plan output path")]
        output: Option<PathBuf>,
        #[arg(long, help = "JSON output")]
        json: bool,
        #[arg(long, help = "Manifest path")]
        manifest: Option<PathBuf>,
    },
    /// Measure and record a baseline (one-time; in-process FFF subject)
    Baseline {
        #[arg(long, help = "Provenance label for the baseline (not an executable path)")]
        subject: Option<String>,
        #[arg(short, long, help = "Corpus root")]
        path: Option<PathBuf>,
        #[arg(long, help = "Manifest path")]
        manifest: Option<PathBuf>,
        #[arg(long, default_value = "default", help = "Corpus id")]
        repo_id: String,
        #[arg(long, help = "Suite (default: pack confirm suite / agent-confirm)")]
        suite: Option<String>,
        #[arg(long, help = "History dir (default: .fff-bench/history)")]
        history_dir: Option<PathBuf>,
        #[arg(long, help = "Per-run log directory")]
        log_dir: Option<PathBuf>,
        #[arg(short, long, help = "Write report.json here")]
        output: Option<PathBuf>,
        #[arg(long, help = "Require quiet host before measuring")]
        require_quiet: bool,
        #[arg(long, help = "JSON agent response")]
        json: bool,
        #[arg(long, help = "Enable content indexing in the adapter")]
        content_index: bool,
    },
    /// Normal agent path: readiness → screen → (optional) confirm
    Check {
        #[arg(
            long,
            default_value = "auto",
            help = "History baseline: auto | run_id | path to report | dual-bin path"
        )]
        baseline: String,
        #[arg(long, help = "Candidate label, or path to candidate fff-bench binary")]
        candidate: Option<String>,
        #[arg(long, help = "Baseline fff-bench binary (paired dual-bin confirm)")]
        baseline_bin: Option<PathBuf>,
        #[arg(long, help = "Candidate fff-bench binary (paired dual-bin confirm)")]
        candidate_bin: Option<PathBuf>,
        #[arg(
            long,
            default_value = "auto",
            help = "auto | screen | confirm — confirm only if screen is promising when auto"
        )]
        mode: String,
        #[arg(short, long, help = "Corpus root")]
        path: Option<PathBuf>,
        #[arg(long, help = "Manifest path")]
        manifest: Option<PathBuf>,
        #[arg(long, default_value = "default", help = "Corpus id")]
        repo_id: String,
        #[arg(long, help = "History dir")]
        history_dir: Option<PathBuf>,
        #[arg(long, help = "Log directory for this check")]
        log_dir: Option<PathBuf>,
        #[arg(short, long, help = "Write final report here")]
        output: Option<PathBuf>,
        #[arg(long, help = "Require quiet host")]
        require_quiet: bool,
        #[arg(long, help = "JSON agent response (preferred for agents)")]
        json: bool,
        #[arg(long, help = "Enable content indexing")]
        content_index: bool,
        #[arg(long, default_value_t = 4, help = "Paired dual-bin blocks")]
        blocks: usize,
        #[arg(long, default_value_t = 12, help = "Iterations per dual-bin block")]
        iters_per_block: usize,
        #[arg(long, help = "Optional agent_usage JSON attachment (not scored)")]
        agent_usage: Option<PathBuf>,
    },
    /// Paired confirmation: measure candidate and compare to baseline
    CompareRun {
        #[arg(long, default_value = "auto", help = "History baseline selector or dual-bin path")]
        baseline: String,
        #[arg(long, help = "Candidate label or binary path")]
        candidate: Option<String>,
        #[arg(long, help = "Baseline executable for counterbalanced blocks")]
        baseline_bin: Option<PathBuf>,
        #[arg(long, help = "Candidate executable for counterbalanced blocks")]
        candidate_bin: Option<PathBuf>,
        #[arg(
            long,
            default_value = "quiet_v1",
            help = "Load profile; use all-resilience with suite resilience-confirm"
        )]
        load_profile: String,
        #[arg(long, default_value = "agent-confirm", help = "Suite id")]
        suite: String,
        #[arg(short, long, help = "Corpus root")]
        path: Option<PathBuf>,
        #[arg(long, help = "Manifest path")]
        manifest: Option<PathBuf>,
        #[arg(long, default_value = "default", help = "Corpus id")]
        repo_id: String,
        #[arg(long, help = "History dir")]
        history_dir: Option<PathBuf>,
        #[arg(long, help = "Log directory")]
        log_dir: Option<PathBuf>,
        #[arg(short, long, help = "Write report here")]
        output: Option<PathBuf>,
        #[arg(long, help = "Record into history")]
        history_record: bool,
        #[arg(long, help = "Require quiet host")]
        require_quiet: bool,
        #[arg(long, help = "JSON output")]
        json: bool,
        #[arg(long, help = "Enable content indexing")]
        content_index: bool,
        #[arg(long, default_value_t = 4, help = "Counterbalanced blocks (dual-bin)")]
        blocks: usize,
        #[arg(long, default_value_t = 12, help = "Iters per block (dual-bin)")]
        iters_per_block: usize,
    },
    /// Run gates + cost profiles for one subject measurement
    Run {
        #[arg(short, long, help = "Corpus root (default: cwd / FFF_BENCH_REPO)")]
        path: Option<PathBuf>,
        #[arg(long, default_value = "default", help = "Corpus id")]
        repo_id: String,
        #[arg(
            long,
            help = "Suite override (default: manifest suite or agent-screen): agent-screen | agent-confirm | ci-gates | public-v1 | peak-diagnose | resilience-confirm"
        )]
        suite: Option<String>,
        #[arg(short = 'n', long, default_value_t = 50, help = "Measured iterations per cell")]
        iterations: usize,
        #[arg(long, default_value_t = 3, help = "Warmup iterations discarded")]
        warmup: usize,
        #[arg(
            long,
            default_value = "all",
            help = "Sensor packs: all | mem,cpu,io,cycles,power"
        )]
        sample: String,
        #[arg(long, default_value_t = 120, help = "Per-op timeout seconds")]
        timeout_secs: u64,
        #[arg(long, help = "Alias for peak-diagnose suite sizing")]
        peak: bool,
        #[arg(short, long, help = "Write report.json here")]
        output: Option<PathBuf>,
        #[arg(long, help = "Scenario logs directory")]
        log_dir: Option<PathBuf>,
        #[arg(long, help = "Record report into history")]
        history_record: bool,
        #[arg(long, help = "History directory")]
        history_dir: Option<PathBuf>,
        #[arg(long, default_value = "auto", help = "Patch capture: auto | none")]
        patch: String,
        #[arg(long, help = "Require quiet host before run")]
        require_quiet: bool,
        #[arg(long, default_value_t = 0, help = "Seconds to wait for quiet host")]
        quiet_wait_secs: u64,
        #[arg(long, default_value = "harness.op_total", help = "Primary span label")]
        primary_span: String,
        #[arg(long, help = "JSON report on stdout")]
        json: bool,
        #[arg(long, help = "Exit non-zero if any gate fails")]
        require_gates: bool,
        #[arg(long, help = "Manifest path")]
        manifest: Option<PathBuf>,
        #[arg(long, help = "Attempt exterior flamegraph (needs samply/perf)")]
        flamegraph: bool,
        #[arg(long, default_value = "auto", help = "Profile backend: auto | samply | perf | none")]
        profile_backend: String,
        #[arg(long, help = "Enable content indexing")]
        content_index: bool,
        #[arg(
            long,
            default_value = "quiet_v1",
            help = "Controlled load: quiet_v1 | peer_cpu_light_v1 | developer_composite_v1 | …"
        )]
        load_profile: String,
        #[arg(
            long,
            default_value = "native_local_v1",
            help = "Network identity: native_local_v1 | loopback_fixture_v1 | external_shared_v1"
        )]
        network_profile: String,
    },
    /// Gates only (same as run with suite=ci-gates)
    Gates {
        #[arg(short, long, help = "Corpus root")]
        path: Option<PathBuf>,
        #[arg(long, help = "Manifest path")]
        manifest: Option<PathBuf>,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// Compute agent score for a report
    Score {
        #[arg(long, help = "Report path")]
        report: PathBuf,
        #[arg(long, help = "Baseline report path or history run_id")]
        baseline: Option<String>,
        #[arg(long, help = "History dir")]
        history_dir: Option<PathBuf>,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// Compare two completed reports
    Compare {
        #[arg(long, help = "Baseline report path or history run_id")]
        baseline: String,
        #[arg(long, help = "Current report path")]
        current: PathBuf,
        #[arg(long, help = "History dir")]
        history_dir: Option<PathBuf>,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// List recorded history runs
    History {
        #[arg(long, help = "History dir (default: .fff-bench/history)")]
        history_dir: Option<PathBuf>,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// Explain verdict precedence from a report (and optional baseline)
    Explain {
        #[arg(long, help = "Report path")]
        report: PathBuf,
        #[arg(long, help = "Baseline report path or run_id")]
        baseline: Option<String>,
        #[arg(long, help = "History dir")]
        history_dir: Option<PathBuf>,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// Semantic diff between two reports
    Diff {
        #[arg(long, help = "Baseline report path")]
        baseline: PathBuf,
        #[arg(long, help = "Current report path")]
        current: PathBuf,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
    /// Export derived CSV from a report
    Export {
        #[arg(long, help = "Report path")]
        report: PathBuf,
        #[arg(long, default_value = "csv", help = "Export format (csv)")]
        format: String,
        #[arg(short, long, help = "Output path")]
        output: Option<PathBuf>,
        #[arg(long, help = "JSON status")]
        json: bool,
    },
    /// Re-run a frozen plan or report's suite (linked by plan digest)
    Replay {
        #[arg(long, help = "Prior report to replay suite from")]
        report: Option<PathBuf>,
        #[arg(long, help = "Frozen plan.json")]
        plan: Option<PathBuf>,
        #[arg(short, long, help = "Corpus root")]
        path: Option<PathBuf>,
        #[arg(long, help = "History dir")]
        history_dir: Option<PathBuf>,
        #[arg(long, help = "JSON output")]
        json: bool,
    },
}

fn default_history_dir() -> PathBuf {
    if let Ok(h) = std::env::var("FFF_BENCH_HISTORY") {
        return PathBuf::from(h);
    }
    PathBuf::from(".fff-bench/history")
}

fn resolve_run_suite(requested: Option<&str>, manifest_suite: Option<&str>) -> String {
    agent::normalize_suite(requested.or(manifest_suite).unwrap_or(agent::SCREEN_SUITE))
}

fn resolve_path(path: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = path {
        if p.exists() {
            return Ok(fff::path_utils::canonicalize(&p).unwrap_or(p));
        }
        return Err(format!("path not found: {}", p.display()));
    }
    if let Ok(env) = std::env::var("FFF_BENCH_REPO") {
        let p = PathBuf::from(env);
        if p.exists() {
            return Ok(fff::path_utils::canonicalize(&p).unwrap_or(p));
        }
    }
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    if cwd.join("Cargo.toml").exists() {
        return Ok(cwd);
    }
    Err("No corpus. Pass --path or set FFF_BENCH_REPO".into())
}

fn discover_manifest() -> Option<PathBuf> {
    for name in [
        "fff-bench.json",
        "bench-manifest.json",
        "benchmark/pack.json",
        "doc/benchmark/pack.json",
    ] {
        let p = PathBuf::from(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn load_report(path: &PathBuf) -> Result<runner::BenchReport, String> {
    let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&s).map_err(|e| e.to_string())
}

fn resolve_baseline(
    spec: &str,
    history_dir: Option<&PathBuf>,
    suite: &str,
    exclude_run_id: Option<&str>,
) -> Result<runner::BenchReport, String> {
    if spec == "auto" {
        let hist = history_dir
            .cloned()
            .unwrap_or_else(default_history_dir);
        // Prefer exact suite, then confirm/screen aliases used as baselines.
        let p = history::latest_comparable_any(
            &hist,
            &[
                suite,
                agent::CONFIRM_SUITE,
                agent::SCREEN_SUITE,
                "agent-loop",
            ],
            exclude_run_id,
        )
        .ok_or_else(|| "no comparable baseline in history".to_string())?;
        return load_report(&p);
    }
    let p = PathBuf::from(spec);
    if p.exists() {
        return load_report(&p);
    }
    let hist = history_dir
        .cloned()
        .unwrap_or_else(default_history_dir);
    let p = hist.join("runs").join(spec).join("report.json");
    if p.exists() {
        return load_report(&p);
    }
    Err(format!("baseline not found: {spec}"))
}

fn resolve_repos(
    path: Option<PathBuf>,
    repo_id: &str,
    manifest_path: Option<&PathBuf>,
) -> Result<(Vec<(String, PathBuf)>, Option<manifest::Manifest>), String> {
    let discovered = manifest_path.cloned().or_else(discover_manifest);
    if let Some(mp) = discovered {
        let m = manifest::Manifest::load(&mp)?;
        let cache = m.cache_dir_resolved();
        let mut out = Vec::new();
        for e in &m.repos {
            match manifest::resolve_repo_path(e, &cache) {
                Ok(p) => out.push((e.id.clone(), p)),
                Err(err) => {
                    if e.optional {
                        eprintln!("skip optional {}: {err}", e.id);
                    } else {
                        return Err(format!("{}: {err}", e.id));
                    }
                }
            }
        }
        if out.is_empty() {
            return Err("no repos resolved from manifest".into());
        }
        return Ok((out, Some(m)));
    }
    let path = resolve_path(path)?;
    Ok((vec![(repo_id.to_string(), path)], None))
}

fn emit_agent(resp: &agent::AgentResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(resp).unwrap());
    } else {
        eprintln!(
            "status={} verdict={:?} next={} {}",
            resp.status, resp.verdict, resp.next_action, resp.summary
        );
        if let Some(ref cmd) = resp.next_command {
            eprintln!("next_command: {cmd}");
        }
    }
}

fn build_plan(suite: &str, repo_id: &str) -> serde_json::Value {
    let suite = agent::normalize_suite(suite);
    let cells = scenarios::expand_suite(&suite, repo_id);
    let plan_body = serde_json::json!({
        "schema_version": "1.0",
        "suite": suite,
        "repo_id": repo_id,
        "cell_count": cells.len(),
        "cells": cells.iter().map(|c| serde_json::json!({
            "cell_key": c.cell_key,
            "scenario_id": c.scenario_id,
            "family": c.family,
            "axes": c.axes,
            "kind": format!("{:?}", c.kind),
        })).collect::<Vec<_>>(),
    });
    let digest = identity::sha256_hex(serde_json::to_string(&plan_body).unwrap().as_bytes());
    serde_json::json!({
        "schema_version": "1.0",
        "plan_id": format!("{suite}/{repo_id}"),
        "plan_digest": format!("sha256:{digest}"),
        "evaluator_digest": format!("sha256:{}", identity::harness_meta().version),
        "suite": suite,
        "seed": 1,
        "cell_count": cells.len(),
        "cells": plan_body["cells"],
        "measurement": {
            "primary_span": "harness.op_total",
            "required_packs": ["cpu", "mem"],
        },
    })
}

fn load_agent_usage(path: Option<&PathBuf>) -> Option<serde_json::Value> {
    let p = path?;
    let s = std::fs::read_to_string(p).ok()?;
    serde_json::from_str(&s).ok()
}

fn ensure_public_fixture_repos(
    suite: &str,
    repos: &[(String, PathBuf)],
) -> Result<Vec<(String, PathBuf)>, String> {
    if suite != "public-v1" && suite != "fixture-gates" {
        return Ok(repos.to_vec());
    }
    let cache = if let Ok(c) = std::env::var("FFF_BENCH_CACHE") {
        PathBuf::from(c)
    } else {
        PathBuf::from(".fff-bench/cache")
    };
    let (fx, _) = fixture::materialize_public_v1(&cache)?;
    Ok(vec![("public-v1".into(), fx)])
}

fn run_measured(
    repos: &[(String, PathBuf)],
    suite: &str,
    warmup: usize,
    iterations: usize,
    adapter_cfg: adapter::AdapterConfig,
    history_dir: Option<PathBuf>,
    history_record: bool,
    log_dir: Option<PathBuf>,
    require_quiet: bool,
    sample_packs: &str,
    peak: bool,
) -> Result<runner::BenchReport, String> {
    run_measured_with_load(
        repos,
        suite,
        warmup,
        iterations,
        adapter_cfg,
        history_dir,
        history_record,
        log_dir,
        require_quiet,
        sample_packs,
        peak,
        "quiet_v1",
    )
}

fn run_measured_with_load(
    repos: &[(String, PathBuf)],
    suite: &str,
    warmup: usize,
    iterations: usize,
    adapter_cfg: adapter::AdapterConfig,
    history_dir: Option<PathBuf>,
    history_record: bool,
    log_dir: Option<PathBuf>,
    require_quiet: bool,
    sample_packs: &str,
    peak: bool,
    load_profile: &str,
) -> Result<runner::BenchReport, String> {
    let suite = agent::normalize_suite(suite);
    let repos = ensure_public_fixture_repos(&suite, repos)?;
    let confirmish = suite.contains("confirm") || suite.contains("peak") || suite == "full";
    let cfg = runner::RunConfig {
        iterations,
        warmup,
        sample_packs: sample_packs.into(),
        timeout_secs: adapter_cfg.timeout_secs,
        suite: suite.clone(),
        peak,
        log_dir,
        require_quiet,
        quiet_wait_secs: if require_quiet { 5 } else { 0 },
        patch_mode: "none".into(),
        history_dir: history_dir.clone(),
        history_record,
        primary_span: "harness.op_total".into(),
        adapter: adapter_cfg,
        flamegraph: false,
        profile_backend: "none".into(),
        mutation_gate: confirmish,
        converge_warmup: confirmish,
        load_profile: load_profile.into(),
        network_profile: "native_local_v1".into(),
    };
    let mut report = runner::run_bench_multi(&repos, &cfg)?;
    let hist = history_dir.unwrap_or_else(default_history_dir);
    let baseline = history::latest_comparable_any(
        &hist,
        &[&suite, agent::CONFIRM_SUITE, agent::SCREEN_SUITE],
        Some(report.metadata.run.run_id.as_str()),
    )
    .and_then(|p| load_report(&p).ok());
    report.score = Some(score_cmd::compute(&report, baseline.as_ref()));
    if history_record {
        let _ = history::record(
            &hist,
            &report,
            report
                .metadata
                .run
                .log_dir
                .as_ref()
                .map(PathBuf::from)
                .as_deref(),
        );
    }
    if let Some(ref d) = report.metadata.run.log_dir {
        let p = PathBuf::from(d).join("report.json");
        if let Ok(s) = serde_json::to_string_pretty(&report) {
            let _ = std::fs::write(p, s);
        }
    }
    Ok(report)
}

fn main() -> ExitCode {
    // Debug builds on Windows use a small default stack; clap + FilePicker need headroom.
    std::thread::Builder::new()
        .name("fff-bench-main".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(real_main)
        .expect("spawn main thread")
        .join()
        .unwrap_or(ExitCode::from(2))
}

fn real_main() -> ExitCode {
    // Child load-actor workers re-exec this binary with FFF_BENCH_LOAD_WORKER set.
    if let Ok(kind) = std::env::var("FFF_BENCH_LOAD_WORKER") {
        return load_actor::run_worker(&kind);
    }
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Init { dir, json } => match inspect_cmd::init_pack(&dir) {
            Ok(p) => {
                if json {
                    println!(
                        "{}",
                        serde_json::json!({
                            "status": "ok",
                            "manifest": p.display().to_string(),
                            "evaluator": inspect_cmd::current_evaluator(),
                        })
                    );
                } else {
                    eprintln!("wrote {}", p.display());
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("init error: {e}");
                ExitCode::from(2)
            }
        },
        Commands::Validate {
            manifest,
            report,
            json,
        } => {
            let out = if let Some(m) = manifest {
                inspect_cmd::validate_manifest(&m)
            } else if let Some(r) = report {
                inspect_cmd::validate_report(&r)
            } else {
                inspect_cmd::ValidateOut {
                    ok: false,
                    status: "invalid".into(),
                    issues: vec![inspect_cmd::Issue {
                        code: "input_required".into(),
                        severity: "error".into(),
                        message: "pass --manifest or --report".into(),
                        path: None,
                    }],
                    summary: "nothing to validate".into(),
                }
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&out).unwrap());
            } else {
                eprintln!("{} {}", out.status, out.summary);
                for i in &out.issues {
                    eprintln!("  [{}] {}: {}", i.severity, i.code, i.message);
                }
            }
            if out.ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            }
        }
        Commands::Lint {
            suite,
            repo_id,
            json,
        } => {
            let suite = agent::normalize_suite(&suite);
            let out = inspect_cmd::lint_suite(&suite, &repo_id);
            if json {
                println!("{}", serde_json::to_string_pretty(&out).unwrap());
            } else {
                eprintln!("{}", out.summary);
            }
            if out.ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            }
        }
        Commands::Calibrate { json } => {
            let out = inspect_cmd::calibrate();
            if json {
                println!("{}", serde_json::to_string_pretty(&out).unwrap());
            } else {
                eprintln!("{}", out.summary);
            }
            ExitCode::SUCCESS
        }
        Commands::Doctor {
            path,
            manifest: manifest_path,
            require_quiet,
            json,
        } => {
            let mut msgs = Vec::new();
            let corpus: Option<String>;
            let path_res = if manifest_path.is_some() || discover_manifest().is_some() {
                resolve_repos(path, "default", manifest_path.as_ref()).map(|(r, _)| r[0].1.clone())
            } else {
                resolve_path(path)
            };
            match path_res {
                Ok(p) => {
                    corpus = Some(p.display().to_string());
                    msgs.push(format!("corpus_ok={}", p.display()));
                }
                Err(e) => {
                    let resp = agent::doctor_setup(
                        &e,
                        "fff-bench setup --path <corpus> --json  # or --manifest <file>",
                    );
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }
            }
            if require_quiet {
                if let Err(e) = noise::wait_quiet(5, true) {
                    let resp = agent::doctor_noisy(
                        noise::host_cpu_busy_pct().unwrap_or(100.0),
                    );
                    let mut resp = resp;
                    resp.summary = e;
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }
                msgs.push("quiet_ok=true".into());
            }
            let busy = noise::host_cpu_busy_pct();
            if let Some(b) = busy {
                if b > 60.0 && require_quiet {
                    let resp = agent::doctor_noisy(b);
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }
            }
            let sensor_packs = sensors::probe_packs();
            let avail: Vec<&str> = sensor_packs
                .iter()
                .filter(|p| p.available)
                .map(|p| p.pack.as_str())
                .collect();
            msgs.push(format!("sensors=[{}]", avail.join(",")));
            let summary = format!(
                "ready corpus={:?} host_cpu_busy_pct={busy:?} sensors=[{}] adapter=fff",
                corpus,
                avail.join(",")
            );
            let mut resp = agent::doctor_ready(&summary);
            if json {
                // Attach diagnostic fields agents may use.
                let mut v = serde_json::to_value(&resp).unwrap();
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("corpus".into(), serde_json::json!(corpus));
                    obj.insert("host_cpu_busy_pct".into(), serde_json::json!(busy));
                    obj.insert("messages".into(), serde_json::json!(msgs));
                    obj.insert("adapter".into(), serde_json::json!("fff"));
                    obj.insert("sensor_packs".into(), serde_json::json!(sensor_packs));
                    obj.insert("ok".into(), serde_json::json!(true));
                    obj.insert("ready".into(), serde_json::json!(true));
                }
                println!("{}", serde_json::to_string_pretty(&v).unwrap());
            } else {
                emit_agent(&resp, false);
            }
            // silence unused mut warning path
            let _ = &mut resp;
            ExitCode::SUCCESS
        }
        Commands::Corpus { path, output, json } => {
            let root = match resolve_path(path) {
                Ok(path) => path,
                Err(e) => {
                    eprintln!("corpus error: {e}");
                    return ExitCode::from(2);
                }
            };
            let adapter = match adapter::FffAdapter::open_corpus(
                &root,
                adapter::AdapterConfig::default(),
            ) {
                Ok(adapter) => adapter,
                Err(e) => {
                    eprintln!("corpus error: {e}");
                    return ExitCode::from(2);
                }
            };
            let mut paths = adapter.path_list();
            paths.sort_unstable();
            paths.dedup();
            let snapshot = serde_json::json!({
                "schema_version": "1.0",
                "adapter": adapter::FffAdapter::id(),
                "root": root.display().to_string(),
                "indexed_path_count": paths.len(),
                "path_list_hash": identity::path_list_hash(&paths),
                "paths": paths,
            });
            if let Some(path) = output.as_ref() {
                if let Some(parent) = path.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        eprintln!("corpus output error: {e}");
                        return ExitCode::from(2);
                    }
                }
                let body = serde_json::to_string_pretty(&snapshot).unwrap();
                if let Err(e) = std::fs::write(path, body) {
                    eprintln!("corpus output error: {e}");
                    return ExitCode::from(2);
                }
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&snapshot).unwrap());
            } else {
                eprintln!(
                    "indexed_paths={} hash={} root={}{}",
                    snapshot["indexed_path_count"],
                    snapshot["path_list_hash"].as_str().unwrap_or("unknown"),
                    root.display(),
                    output
                        .as_ref()
                        .map(|p| format!(" output={}", p.display()))
                        .unwrap_or_default()
                );
                if output.is_none() {
                    for path in snapshot["paths"].as_array().unwrap() {
                        println!("{}", path.as_str().unwrap_or_default());
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Commands::Setup {
            path,
            manifest: manifest_path,
            fetch,
            fixture,
            json,
        } => {
            let mut fixture_info = None;
            if fixture {
                let cache = if let Ok(c) = std::env::var("FFF_BENCH_CACHE") {
                    PathBuf::from(c)
                } else {
                    PathBuf::from(".fff-bench/cache")
                };
                match fixture::materialize_public_v1(&cache) {
                    Ok((p, dig)) => {
                        fixture_info = Some(serde_json::json!({
                            "fixture_id": fixture::FIXTURE_ID,
                            "path": p.display().to_string(),
                            "semantic_digest": dig,
                            "query_bank": fixture::QUERY_BANK_VERSION,
                        }));
                    }
                    Err(e) => {
                        if json {
                            println!(
                                "{}",
                                serde_json::json!({ "ok": false, "error": e })
                            );
                        } else {
                            eprintln!("fixture error: {e}");
                        }
                        return ExitCode::from(2);
                    }
                }
            }
            if let Some(mp) = manifest_path {
                match manifest::Manifest::load(&mp) {
                    Ok(m) => {
                        let res = manifest::setup_manifest(&m, fetch);
                        if json {
                            let mut v = serde_json::to_value(&res).unwrap();
                            if let Some(fi) = fixture_info {
                                v.as_object_mut().unwrap().insert("fixture".into(), fi);
                            }
                            println!("{}", serde_json::to_string_pretty(&v).unwrap());
                        } else {
                            eprintln!("setup ok={} cache={}", res.ok, res.cache_dir);
                            for r in &res.repos {
                                eprintln!(
                                    "  {} ready={} action={} path={:?}",
                                    r.id, r.ready, r.action, r.path
                                );
                            }
                        }
                        if res.ok {
                            ExitCode::SUCCESS
                        } else {
                            ExitCode::from(2)
                        }
                    }
                    Err(e) => {
                        eprintln!("setup error: {e}");
                        ExitCode::from(2)
                    }
                }
            } else {
                match resolve_path(path) {
                    Ok(p) => {
                        if json {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "ok": true,
                                    "ready": true,
                                    "corpus": p.display().to_string(),
                                    "action": "verified_path",
                                    "fixture": fixture_info,
                                })
                            );
                        } else {
                            eprintln!("setup ok corpus={}", p.display());
                            if let Some(fi) = fixture_info {
                                eprintln!("fixture {fi}");
                            }
                        }
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        // fixture-only setup is ok
                        if fixture_info.is_some() {
                            if json {
                                println!(
                                    "{}",
                                    serde_json::json!({
                                        "ok": true,
                                        "ready": true,
                                        "fixture": fixture_info,
                                    })
                                );
                            } else {
                                eprintln!("setup ok fixture-only");
                            }
                            return ExitCode::SUCCESS;
                        }
                        if json {
                            println!(
                                "{}",
                                serde_json::json!({ "ok": false, "ready": false, "error": e })
                            );
                        } else {
                            eprintln!("setup error: {e}");
                        }
                        ExitCode::from(2)
                    }
                }
            }
        }
        Commands::List {
            suite,
            repo_id,
            json,
            expanded,
            manifest: _manifest,
        } => {
            let suite = agent::normalize_suite(&suite);
            let cells = scenarios::expand_suite(&suite, &repo_id);
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "suite": suite,
                        "repo_id": repo_id,
                        "cell_count": cells.len(),
                        "catalog": scenarios::list_scenarios(),
                        "load_profiles": load_actor::list_profiles(),
                        "network_profiles": network::list_profiles(),
                        "workflow": {
                            "screen_suite": agent::SCREEN_SUITE,
                            "confirm_suite": agent::CONFIRM_SUITE,
                            "diagnose_suite": agent::DIAGNOSE_SUITE,
                        },
                        "cells": if expanded { serde_json::to_value(&cells).unwrap() } else { serde_json::Value::Null },
                    })
                );
            } else {
                println!("suite={suite} cells={}", cells.len());
                let mut by_family: std::collections::BTreeMap<String, usize> =
                    std::collections::BTreeMap::new();
                for c in &cells {
                    *by_family.entry(c.family.clone()).or_default() += 1;
                }
                println!("families: {by_family:?}");
                for c in cells.iter().take(40) {
                    println!(
                        "  [{}] {} {:?}",
                        c.family,
                        c.cell_key.replace('\0', "|"),
                        c.kind
                    );
                }
                if cells.len() > 40 {
                    println!("  ... {} more", cells.len() - 40);
                }
                println!("catalog:");
                for (id, desc) in scenarios::list_scenarios() {
                    println!("  {id:40} {desc}");
                }
            }
            ExitCode::SUCCESS
        }
        Commands::Plan {
            suite,
            repo_id,
            freeze,
            output,
            json,
            manifest: _manifest,
        } => {
            let plan = build_plan(&suite, &repo_id);
            if freeze {
                let out = output.unwrap_or_else(|| PathBuf::from(".fff-bench/plan.json"));
                if let Some(parent) = out.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&out, serde_json::to_string_pretty(&plan).unwrap());
                if json {
                    println!(
                        "{}",
                        serde_json::json!({
                            "status": "frozen",
                            "plan_digest": plan.get("plan_digest"),
                            "path": out.display().to_string(),
                            "cell_count": plan.get("cell_count"),
                        })
                    );
                } else {
                    eprintln!(
                        "plan frozen digest={} path={}",
                        plan.get("plan_digest").and_then(|v| v.as_str()).unwrap_or("?"),
                        out.display()
                    );
                }
            } else if json {
                println!("{}", serde_json::to_string_pretty(&plan).unwrap());
            } else {
                println!(
                    "suite={} cells={} digest={}",
                    plan.get("suite").and_then(|v| v.as_str()).unwrap_or("?"),
                    plan.get("cell_count").and_then(|v| v.as_u64()).unwrap_or(0),
                    plan.get("plan_digest").and_then(|v| v.as_str()).unwrap_or("?"),
                );
            }
            ExitCode::SUCCESS
        }
        Commands::Baseline {
            subject,
            path,
            manifest: manifest_path,
            repo_id,
            suite,
            history_dir,
            log_dir,
            output,
            require_quiet,
            json,
            content_index,
        } => {
            let hist = history_dir.unwrap_or_else(default_history_dir);
            let (repos, m) = match resolve_repos(path, &repo_id, manifest_path.as_ref()) {
                Ok(r) => r,
                Err(e) => {
                    let resp = agent::doctor_infra(&e);
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }
            };
            let suite = suite
                .or_else(|| m.as_ref().map(|x| x.workflow.confirm_suite.clone()))
                .unwrap_or_else(|| agent::CONFIRM_SUITE.into());
            let mut adapter_cfg = m
                .as_ref()
                .map(|x| adapter::AdapterConfig::from(&x.adapter))
                .unwrap_or_default();
            if content_index {
                adapter_cfg.enable_content_indexing = true;
            }
            let (warmup, iterations) = agent::confirm_iters();
            let log = log_dir.or_else(|| Some(PathBuf::from(".fff-bench/runs").join("baseline")));
            match run_measured(
                &repos,
                &suite,
                warmup,
                iterations,
                adapter_cfg,
                Some(hist.clone()),
                true,
                log,
                require_quiet,
                "mem,cpu",
                false,
            ) {
                Ok(report) => {
                    if let Some(ref o) = output {
                        let _ = std::fs::write(o, serde_json::to_string_pretty(&report).unwrap());
                    }
                    let _ = subject; // provenance label for future artifact identity
                    let resp = agent::AgentResponse {
                        status: "baseline_recorded".into(),
                        evidence_grade: Some("confirm".into()),
                        verdict: None,
                        reason_codes: if report.admissible {
                            vec!["baseline_ok".into(), "gates_passed".into()]
                        } else {
                            vec!["baseline_gates_failed".into()]
                        },
                        next_action: if report.admissible {
                            "run_check".into()
                        } else {
                            "inspect_gates".into()
                        },
                        next_command: if report.admissible {
                            Some(
                                "fff-bench check --baseline auto --candidate <artifact-ref> --json"
                                    .into(),
                            )
                        } else {
                            None
                        },
                        agent_recommendations: vec![],
                        summary: format!(
                            "baseline run_id={} suite={} admissible={}",
                            report.metadata.run.run_id,
                            report.metadata.run.suite,
                            report.admissible
                        ),
                        baseline_run_id: Some(report.metadata.run.run_id.clone()),
                        candidate_run_id: None,
                        screen_run_id: None,
                        report_path: report.metadata.run.log_dir.clone(),
                        score: report.score.clone(),
                        comparison: None,
                        plan: None,
                    };
                    emit_agent(&resp, json);
                    if report.admissible {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(e) => {
                    let resp = agent::doctor_infra(&e);
                    emit_agent(&resp, json);
                    resp.exit_code()
                }
            }
        }
        Commands::Check {
            baseline,
            candidate,
            baseline_bin,
            candidate_bin,
            mode,
            path,
            manifest: manifest_path,
            repo_id,
            history_dir,
            log_dir,
            output,
            require_quiet,
            json,
            content_index,
            blocks,
            iters_per_block,
            agent_usage,
        } => {
            let hist = history_dir.unwrap_or_else(default_history_dir);
            let (repos, m) = match resolve_repos(path, &repo_id, manifest_path.as_ref()) {
                Ok(r) => r,
                Err(e) => {
                    let resp = agent::doctor_setup(
                        &e,
                        "fff-bench setup --path <corpus> --json",
                    );
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }
            };
            let (screen_suite, confirm_suite, _) = m
                .as_ref()
                .map(|x| {
                    (
                        x.workflow.screen_suite.clone(),
                        x.workflow.confirm_suite.clone(),
                        x.workflow.diagnose_suite.clone(),
                    )
                })
                .unwrap_or_else(agent::default_workflow);

            let usage = load_agent_usage(agent_usage.as_ref());

            // Dual-bin artifacts for decision-grade paired confirm.
            let cand_bin = candidate_bin.clone().or_else(|| {
                candidate.as_ref().and_then(|c| {
                    let p = PathBuf::from(c);
                    (p.exists() && p.is_file()).then_some(p)
                })
            });
            let base_bin = baseline_bin.clone().or_else(|| {
                if baseline != "auto" {
                    let p = PathBuf::from(&baseline);
                    if p.exists() && p.is_file() {
                        return Some(p);
                    }
                }
                None
            });
            let dual = base_bin.clone().zip(cand_bin.clone());

            // History baseline required unless dual-bin confirm-only skips screen history.
            let base_report = if dual.is_some() && mode.to_lowercase() == "confirm" {
                None
            } else {
                match resolve_baseline(&baseline, Some(&hist), &confirm_suite, None) {
                    Ok(b) => Some(b),
                    Err(_) if dual.is_some() => None,
                    Err(_) => {
                        let resp = agent::baseline_required();
                        emit_agent(&resp, json);
                        return resp.exit_code();
                    }
                }
            };

            let mut adapter_cfg = m
                .as_ref()
                .map(|x| adapter::AdapterConfig::from(&x.adapter))
                .unwrap_or_default();
            if content_index {
                adapter_cfg.enable_content_indexing = true;
            }

            let mode = mode.to_lowercase();
            let force_confirm = mode == "confirm";
            let auto_confirm = mode == "auto";

            let screen_log = log_dir
                .clone()
                .map(|d| d.join("screen"))
                .or_else(|| Some(PathBuf::from(".fff-bench/runs").join("screen")));

            let (sw, sn) = agent::screen_iters();
            let screen_report = if force_confirm {
                None
            } else {
                match run_measured(
                    &repos,
                    &screen_suite,
                    sw,
                    sn,
                    adapter_cfg.clone(),
                    Some(hist.clone()),
                    true,
                    screen_log,
                    require_quiet,
                    "mem,cpu",
                    false,
                ) {
                    Ok(mut r) => {
                        r.agent_usage = usage.clone();
                        Some(r)
                    }
                    Err(e) => {
                        let resp = agent::doctor_infra(&e);
                        emit_agent(&resp, json);
                        return resp.exit_code();
                    }
                }
            };

            if let Some(ref sr) = screen_report {
                if !sr.admissible {
                    let mut resp = agent::from_report_gates_only(sr, "screen");
                    if let Some(ref b) = base_report {
                        resp.baseline_run_id = Some(b.metadata.run.run_id.clone());
                    }
                    resp.screen_run_id = Some(sr.metadata.run.run_id.clone());
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }
                if sr.noise.integrity == "noisy" {
                    let mut resp = agent::from_report_gates_only(sr, "screen");
                    if let Some(ref b) = base_report {
                        resp.baseline_run_id = Some(b.metadata.run.run_id.clone());
                    }
                    resp.screen_run_id = Some(sr.metadata.run.run_id.clone());
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }

                if let Some(ref b) = base_report {
                    let screen_cmp = compare::compare_screen(b, sr);
                    if mode == "screen"
                        || matches!(
                            screen_cmp.verdict.as_str(),
                            "regress"
                                | "inadmissible"
                                | "unstable"
                                | "incomparable"
                                | "screen_no_signal"
                        )
                    {
                        let mut resp = agent::from_compare(
                            &screen_cmp,
                            "screen",
                            Some(b.metadata.run.run_id.clone()),
                            Some(sr.metadata.run.run_id.clone()),
                            sr.metadata.run.log_dir.clone(),
                            sr.score.clone(),
                        );
                        resp.screen_run_id = Some(sr.metadata.run.run_id.clone());
                        if let Some(ref o) = output {
                            let _ = std::fs::write(o, serde_json::to_string_pretty(sr).unwrap());
                        }
                        emit_agent(&resp, json);
                        return resp.exit_code();
                    }
                    if !auto_confirm && !force_confirm {
                        let mut resp = agent::from_compare(
                            &screen_cmp,
                            "screen",
                            Some(b.metadata.run.run_id.clone()),
                            Some(sr.metadata.run.run_id.clone()),
                            sr.metadata.run.log_dir.clone(),
                            sr.score.clone(),
                        );
                        resp.screen_run_id = Some(sr.metadata.run.run_id.clone());
                        emit_agent(&resp, json);
                        return resp.exit_code();
                    }
                } else if mode == "screen" {
                    // Dual-bin screen without history: gates-only pass
                    let mut resp = agent::from_report_gates_only(sr, "screen");
                    resp.status = "promising".into();
                    resp.next_action = "run_confirmation".into();
                    resp.screen_run_id = Some(sr.metadata.run.run_id.clone());
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }
            }

            // --- confirm (paired dual-bin preferred) ---
            let confirm_log = log_dir
                .clone()
                .map(|d| d.join("confirm"))
                .or_else(|| Some(PathBuf::from(".fff-bench/runs").join("confirm")));
            let corpus = repos[0].1.clone();

            if let Some((bb, cb)) = dual {
                let cfg = paired::PairedConfig {
                    baseline_bin: bb,
                    candidate_bin: cb,
                    corpus,
                    suite: confirm_suite.clone(),
                    blocks,
                    iterations_per_block: iters_per_block,
                    warmup: 3,
                    log_dir: confirm_log.clone().unwrap(),
                    seed: paired::PAIRED_SEED,
                    load_profile: "quiet_v1".into(),
                };
                match paired::run_paired(&cfg) {
                    Ok(mut out) => {
                        out.candidate_report.agent_usage = usage;
                        let _ = history::record(
                            &hist,
                            &out.candidate_report,
                            confirm_log.as_deref(),
                        );
                        if let Some(ref o) = output {
                            let _ =
                                std::fs::write(o, serde_json::to_string_pretty(&out).unwrap());
                        }
                        let mut resp = agent::from_compare(
                            &out.comparison,
                            "confirm",
                            Some(out.baseline_report.metadata.run.run_id.clone()),
                            Some(out.candidate_report.metadata.run.run_id.clone()),
                            confirm_log.map(|p| p.display().to_string()),
                            out.candidate_report.score.clone(),
                        );
                        if let Some(ref sr) = screen_report {
                            resp.screen_run_id = Some(sr.metadata.run.run_id.clone());
                        }
                        resp.summary = format!(
                            "paired {} blocks; {}",
                            out.blocks, out.comparison.summary_for_agent
                        );
                        if json {
                            let mut v = serde_json::to_value(&resp).unwrap();
                            if let Some(obj) = v.as_object_mut() {
                                obj.insert(
                                    "pairing".into(),
                                    serde_json::json!({
                                        "mode": out.pairing,
                                        "blocks": out.blocks,
                                        "wall_ratio_ci95": out.wall_ratio_ci95,
                                        "baseline_artifact": out.baseline_artifact,
                                        "candidate_artifact": out.candidate_artifact,
                                    }),
                                );
                            }
                            println!("{}", serde_json::to_string_pretty(&v).unwrap());
                        } else {
                            emit_agent(&resp, false);
                        }
                        return resp.exit_code();
                    }
                    Err(e) => {
                        let resp = agent::doctor_infra(&e);
                        emit_agent(&resp, json);
                        return resp.exit_code();
                    }
                }
            }

            let Some(base_report) = base_report else {
                let resp = agent::baseline_required();
                emit_agent(&resp, json);
                return resp.exit_code();
            };

            let (cw, cn) = agent::confirm_iters();
            match run_measured(
                &repos,
                &confirm_suite,
                cw,
                cn,
                adapter_cfg,
                Some(hist.clone()),
                true,
                confirm_log,
                require_quiet,
                "mem,cpu",
                false,
            ) {
                Ok(mut cr) => {
                    cr.agent_usage = usage;
                    if let Some(ref o) = output {
                        let _ = std::fs::write(o, serde_json::to_string_pretty(&cr).unwrap());
                    }
                    let mut cmp = compare::compare(&base_report, &cr);
                    cmp.reason_codes
                        .push("history_baseline_not_paired_blocks".into());
                    let mut resp = agent::from_compare(
                        &cmp,
                        "confirm",
                        Some(base_report.metadata.run.run_id.clone()),
                        Some(cr.metadata.run.run_id.clone()),
                        cr.metadata.run.log_dir.clone(),
                        cr.score.clone(),
                    );
                    if let Some(ref sr) = screen_report {
                        resp.screen_run_id = Some(sr.metadata.run.run_id.clone());
                    }
                    emit_agent(&resp, json);
                    resp.exit_code()
                }
                Err(e) => {
                    let resp = agent::doctor_infra(&e);
                    emit_agent(&resp, json);
                    resp.exit_code()
                }
            }
        }
        Commands::CompareRun {
            baseline,
            candidate,
            baseline_bin,
            candidate_bin,
            load_profile,
            suite,
            path,
            manifest: manifest_path,
            repo_id,
            history_dir,
            log_dir,
            output,
            history_record,
            require_quiet,
            json,
            content_index,
            blocks,
            iters_per_block,
        } => {
            let hist = history_dir.unwrap_or_else(default_history_dir);
            let suite = agent::normalize_suite(&suite);
            let (repos, m) = match resolve_repos(path, &repo_id, manifest_path.as_ref()) {
                Ok(r) => r,
                Err(e) => {
                    let resp = agent::doctor_infra(&e);
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }
            };
            let corpus = repos[0].1.clone();
            let log = log_dir.unwrap_or_else(|| PathBuf::from(".fff-bench/runs").join("compare-run"));

            // Dual-artifact counterbalanced pairing when both bins provided.
            // baseline_bin + candidate_bin (or candidate as path to bin).
            let cand_bin = candidate_bin.or_else(|| {
                candidate
                    .as_ref()
                    .and_then(|c| {
                        let p = PathBuf::from(c);
                        if p.exists() && p.is_file() {
                            Some(p)
                        } else {
                            None
                        }
                    })
            });
            let base_bin = baseline_bin.or_else(|| {
                if baseline != "auto" {
                    let p = PathBuf::from(&baseline);
                    if p.exists() && p.is_file() {
                        return Some(p);
                    }
                }
                None
            });

            // Paired resilience: dual-bin × load profile(s).
            let is_resilience = suite == agent::RESILIENCE_SUITE || suite == "resilience";
            if is_resilience {
                if let (Some(bb), Some(cb)) = (base_bin.clone(), cand_bin.clone()) {
                    let profiles: Vec<String> = if matches!(
                        load_profile.as_str(),
                        "all" | "all-resilience" | "resilience" | ""
                    ) {
                        load_actor::resilience_profiles()
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect()
                    } else {
                        vec![load_profile.clone()]
                    };
                    let base = runner::RunConfig {
                        iterations: iters_per_block,
                        warmup: 2,
                        sample_packs: "mem,cpu".into(),
                        timeout_secs: 120,
                        suite: agent::RESILIENCE_SUITE.into(),
                        peak: false,
                        log_dir: None,
                        require_quiet: false,
                        quiet_wait_secs: 0,
                        patch_mode: "none".into(),
                        history_dir: None,
                        history_record: false,
                        primary_span: "harness.op_total".into(),
                        adapter: adapter::AdapterConfig::default(),
                        flamegraph: false,
                        profile_backend: "none".into(),
                        mutation_gate: false,
                        converge_warmup: false,
                        load_profile: "quiet_v1".into(),
                        network_profile: "native_local_v1".into(),
                    };
                    let rcfg = resilience::ResilienceConfig {
                        base,
                        profiles,
                        log_dir: log.clone(),
                        keep_full_reports: json,
                        baseline_bin: Some(bb),
                        candidate_bin: Some(cb),
                        corpus: Some(corpus.clone()),
                        paired_blocks: blocks.min(4).max(1),
                        paired_iters: iters_per_block.max(3),
                    };
                    match resilience::run_resilience(&repos, &rcfg) {
                        Ok(rep) => {
                            if let Some(ref o) = output {
                                let _ =
                                    std::fs::write(o, serde_json::to_string_pretty(&rep).unwrap());
                            }
                            if json {
                                println!("{}", serde_json::to_string_pretty(&rep).unwrap());
                            } else {
                                eprintln!("{}", rep.summary);
                                for p in &rep.profiles {
                                    eprintln!(
                                        "  {} adm={} treat={} verdict={:?}",
                                        p.load_profile_id,
                                        p.admissible,
                                        p.treatment_valid,
                                        p.comparison.as_ref().map(|c| &c.verdict)
                                    );
                                }
                            }
                            return if rep.all_admissible && rep.all_treatments_valid {
                                ExitCode::SUCCESS
                            } else {
                                ExitCode::from(1)
                            };
                        }
                        Err(e) => {
                            let resp = agent::doctor_infra(&e);
                            emit_agent(&resp, json);
                            return resp.exit_code();
                        }
                    }
                }
            }

            if let (Some(bb), Some(cb)) = (base_bin, cand_bin) {
                let cfg = paired::PairedConfig {
                    baseline_bin: bb,
                    candidate_bin: cb,
                    corpus,
                    suite: suite.clone(),
                    blocks,
                    iterations_per_block: iters_per_block,
                    warmup: 3,
                    log_dir: log.clone(),
                    seed: paired::PAIRED_SEED,
                    load_profile: load_profile.clone(),
                };
                match paired::run_paired(&cfg) {
                    Ok(out) => {
                        if history_record {
                            let _ = history::record(
                                &hist,
                                &out.candidate_report,
                                Some(&log),
                            );
                        }
                        if let Some(ref o) = output {
                            let _ = std::fs::write(o, serde_json::to_string_pretty(&out).unwrap());
                        }
                        let mut resp = agent::from_compare(
                            &out.comparison,
                            "confirm",
                            Some(out.baseline_report.metadata.run.run_id.clone()),
                            Some(out.candidate_report.metadata.run.run_id.clone()),
                            Some(log.display().to_string()),
                            out.candidate_report.score.clone(),
                        );
                        resp.summary = format!(
                            "paired {} blocks; {}",
                            out.blocks, out.comparison.summary_for_agent
                        );
                        if json {
                            // Include full paired envelope for agents.
                            let mut v = serde_json::to_value(&resp).unwrap();
                            if let Some(obj) = v.as_object_mut() {
                                obj.insert(
                                    "pairing".into(),
                                    serde_json::json!({
                                        "mode": out.pairing,
                                        "blocks": out.blocks,
                                        "block_log": out.block_log,
                                        "wall_ratio_ci95": out.wall_ratio_ci95,
                                        "baseline_artifact": out.baseline_artifact,
                                        "candidate_artifact": out.candidate_artifact,
                                    }),
                                );
                            }
                            println!("{}", serde_json::to_string_pretty(&v).unwrap());
                        } else {
                            emit_agent(&resp, false);
                        }
                        return resp.exit_code();
                    }
                    Err(e) => {
                        let resp = agent::doctor_infra(&e);
                        emit_agent(&resp, json);
                        return resp.exit_code();
                    }
                }
            }

            // History baseline + single-process candidate (informational pairing).
            let base_report = match resolve_baseline(&baseline, Some(&hist), &suite, None) {
                Ok(b) => b,
                Err(_) => {
                    let resp = agent::baseline_required();
                    emit_agent(&resp, json);
                    return resp.exit_code();
                }
            };
            let mut adapter_cfg = m
                .as_ref()
                .map(|x| adapter::AdapterConfig::from(&x.adapter))
                .unwrap_or_default();
            if content_index {
                adapter_cfg.enable_content_indexing = true;
            }
            let _ = candidate;
            let (cw, cn) = agent::confirm_iters();
            let _ = history_record;
            match run_measured(
                &repos,
                &suite,
                cw,
                cn,
                adapter_cfg,
                Some(hist),
                true,
                Some(log),
                require_quiet,
                "mem,cpu",
                false,
            ) {
                Ok(cr) => {
                    if let Some(ref o) = output {
                        let _ = std::fs::write(o, serde_json::to_string_pretty(&cr).unwrap());
                    }
                    let mut cmp = compare::compare(&base_report, &cr);
                    cmp.reason_codes
                        .push("history_baseline_not_paired_blocks".into());
                    let resp = agent::from_compare(
                        &cmp,
                        "confirm",
                        Some(base_report.metadata.run.run_id.clone()),
                        Some(cr.metadata.run.run_id.clone()),
                        cr.metadata.run.log_dir.clone(),
                        cr.score.clone(),
                    );
                    emit_agent(&resp, json);
                    resp.exit_code()
                }
                Err(e) => {
                    let resp = agent::doctor_infra(&e);
                    emit_agent(&resp, json);
                    resp.exit_code()
                }
            }
        }
        Commands::Run {
            path,
            repo_id,
            suite,
            iterations,
            warmup,
            sample,
            timeout_secs,
            peak,
            output,
            log_dir,
            history_record,
            history_dir,
            patch,
            require_quiet,
            quiet_wait_secs,
            primary_span,
            json,
            require_gates,
            manifest: manifest_path,
            flamegraph,
            profile_backend,
            content_index,
            load_profile,
            network_profile,
        } => {
            let n = if peak { iterations.max(50) } else { iterations };
            let mut adapter_cfg = adapter::AdapterConfig::default();
            adapter_cfg.enable_content_indexing = content_index;
            adapter_cfg.timeout_secs = timeout_secs;

            let (repos, m) = match resolve_repos(path, &repo_id, manifest_path.as_ref()) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            if let Some(ref m) = m {
                adapter_cfg = adapter::AdapterConfig::from(&m.adapter);
                adapter_cfg.timeout_secs = timeout_secs;
                if content_index {
                    adapter_cfg.enable_content_indexing = true;
                }
            }

            let sample = if sample == "all" {
                "mem,cpu,io,cycles,power".to_string()
            } else {
                sample
            };

            let suite = resolve_run_suite(
                suite.as_deref(),
                m.as_ref().and_then(|m| m.suite.as_deref()),
            );

            let confirmish =
                suite.contains("confirm") || suite.contains("peak") || suite == "full";
            let is_resilience = suite == agent::RESILIENCE_SUITE || suite == "resilience";
            // Top-level resilience-confirm with default quiet_v1 means sweep all profiles.
            // Explicit profile ids (including quiet_v1 from a parent paired subprocess) stay single.
            // Use --load-profile all-resilience to force a multi sweep when needed.
            let load_profile = if is_resilience && load_profile == "quiet_v1" {
                // Detect subprocess single-shot: --output is set for paired measures.
                if output.is_some() {
                    load_profile
                } else {
                    "all-resilience".into()
                }
            } else {
                load_profile
            };
            let multi_load = matches!(
                load_profile.as_str(),
                "all" | "all-resilience" | "resilience"
            );

            let repos = match ensure_public_fixture_repos(&suite, &repos) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };

            // Multi-profile resilience sweep.
            if is_resilience && multi_load {
                let base = runner::RunConfig {
                    iterations: n.min(15),
                    warmup: warmup.min(3),
                    sample_packs: sample.clone(),
                    timeout_secs,
                    suite: agent::RESILIENCE_SUITE.into(),
                    peak: false,
                    log_dir: None,
                    require_quiet: false,
                    quiet_wait_secs: 0,
                    patch_mode: patch,
                    history_dir: history_dir.clone(),
                    history_record: false,
                    primary_span,
                    adapter: adapter_cfg,
                    flamegraph: false,
                    profile_backend: "none".into(),
                    mutation_gate: false,
                    converge_warmup: false,
                    load_profile: "quiet_v1".into(),
                    network_profile: "native_local_v1".into(),
                };
                let rcfg = resilience::ResilienceConfig {
                    base,
                    profiles: load_actor::resilience_profiles()
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect(),
                    log_dir: log_dir
                        .unwrap_or_else(|| PathBuf::from(".fff-bench/runs/resilience")),
                    keep_full_reports: json,
                    baseline_bin: None,
                    candidate_bin: None,
                    corpus: repos.first().map(|(_, p)| p.clone()),
                    paired_blocks: 2,
                    paired_iters: n.min(8).max(3),
                };
                match resilience::run_resilience(&repos, &rcfg) {
                    Ok(rep) => {
                        if let Some(ref o) = output {
                            let _ = std::fs::write(o, serde_json::to_string_pretty(&rep).unwrap());
                        }
                        if json {
                            println!("{}", serde_json::to_string_pretty(&rep).unwrap());
                        } else {
                            eprintln!("{}", rep.summary);
                            for p in &rep.profiles {
                                eprintln!(
                                    "  {} adm={} treat={} wall_p95={:?} busy={:?}",
                                    p.load_profile_id,
                                    p.admissible,
                                    p.treatment_valid,
                                    p.typical_wall_p95_us,
                                    p.calibration_busy_cores
                                );
                            }
                        }
                        return if rep.all_admissible && rep.all_treatments_valid {
                            ExitCode::SUCCESS
                        } else {
                            ExitCode::from(1)
                        };
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            }

            let cfg = runner::RunConfig {
                iterations: n,
                warmup,
                sample_packs: sample,
                timeout_secs,
                suite: suite.clone(),
                peak,
                log_dir,
                require_quiet,
                quiet_wait_secs,
                patch_mode: patch,
                history_dir: history_dir.clone(),
                history_record,
                primary_span,
                adapter: adapter_cfg,
                flamegraph,
                profile_backend,
                mutation_gate: confirmish || peak,
                converge_warmup: confirmish || peak,
                load_profile,
                network_profile,
            };
            match runner::run_bench_multi(&repos, &cfg) {
                Ok(mut report) => {
                    let hist = history_dir.clone().unwrap_or_else(default_history_dir);
                    let baseline = history::latest_comparable_any(
                        &hist,
                        &[&suite, agent::CONFIRM_SUITE, agent::SCREEN_SUITE],
                        Some(report.metadata.run.run_id.as_str()),
                    )
                    .and_then(|p| load_report(&p).ok());
                    report.score = Some(score_cmd::compute(&report, baseline.as_ref()));
                    if history_record {
                        let _ = history::record(
                            &hist,
                            &report,
                            report
                                .metadata
                                .run
                                .log_dir
                                .as_ref()
                                .map(PathBuf::from)
                                .as_deref(),
                        );
                    }

                    if let Some(ref o) = output {
                        if let Ok(s) = serde_json::to_string_pretty(&report) {
                            let _ = std::fs::write(o, s);
                        }
                    }
                    if let Some(ref d) = report.metadata.run.log_dir {
                        let p = PathBuf::from(d).join("report.json");
                        if let Ok(s) = serde_json::to_string_pretty(&report) {
                            let _ = std::fs::write(p, s);
                        }
                    }
                    if json {
                        println!("{}", serde_json::to_string_pretty(&report).unwrap());
                    } else {
                        runner::print_table(&report);
                    }

                    if require_gates && !report.admissible {
                        return ExitCode::from(1);
                    }
                    if let Some(ref s) = report.score {
                        let code = runner::score_exit_code(&s.score_status);
                        if code != 0 {
                            return ExitCode::from(code);
                        }
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(2)
                }
            }
        }
        Commands::Gates {
            path,
            manifest: manifest_path,
            json,
        } => {
            let repos: Vec<(String, PathBuf)> = if let Some(mp) = manifest_path {
                match manifest::Manifest::load(&mp) {
                    Ok(m) => {
                        let cache = m.cache_dir_resolved();
                        m.repos
                            .iter()
                            .filter_map(|e| {
                                manifest::resolve_repo_path(e, &cache)
                                    .ok()
                                    .map(|p| (e.id.clone(), p))
                            })
                            .collect()
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            } else {
                match resolve_path(path) {
                    Ok(p) => vec![("default".into(), p)],
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            };
            let cfg = runner::RunConfig {
                iterations: 3,
                warmup: 1,
                sample_packs: "mem,cpu".into(),
                timeout_secs: 120,
                suite: "ci-gates".into(),
                peak: false,
                log_dir: None,
                require_quiet: false,
                quiet_wait_secs: 0,
                patch_mode: "none".into(),
                history_dir: None,
                history_record: false,
                primary_span: "harness.op_total".into(),
                adapter: adapter::AdapterConfig::default(),
                flamegraph: false,
                profile_backend: "none".into(),
                mutation_gate: false,
                converge_warmup: false,
                load_profile: "quiet_v1".into(),
                network_profile: "native_local_v1".into(),
            };
            match runner::run_bench_multi(&repos, &cfg) {
                Ok(report) => {
                    if json {
                        println!("{}", serde_json::to_string_pretty(&report.gates).unwrap());
                    } else {
                        println!(
                            "gates passed={} hit@5={:.2} mrr={:.2} failed={:?}",
                            report.gates.passed,
                            report.gates.hit_at_5,
                            report.gates.mrr,
                            report.gates.failed
                        );
                        for c in &report.gates.cases {
                            println!(
                                "  {} pass={} rank={:?} top1={:?}",
                                c.id, c.passed, c.rank, c.top1
                            );
                        }
                    }
                    if report.gates.passed {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(2)
                }
            }
        }
        Commands::Score {
            report,
            baseline,
            history_dir,
            json,
        } => {
            let report = match load_report(&report) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let base = baseline.as_ref().and_then(|b| {
                resolve_baseline(
                    b,
                    history_dir.as_ref(),
                    &report.metadata.run.suite,
                    Some(report.metadata.run.run_id.as_str()),
                )
                .ok()
            });
            let score = score_cmd::compute(&report, base.as_ref());
            if json {
                println!("{}", serde_json::to_string_pretty(&score).unwrap());
            } else {
                println!(
                    "status={} score={:?} delta={:?}\n{}",
                    score.score_status, score.score, score.score_delta, score.summary_for_agent
                );
            }
            ExitCode::from(runner::score_exit_code(&score.score_status))
        }
        Commands::Compare {
            baseline,
            current,
            history_dir,
            json,
        } => {
            let cur = match load_report(&current) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let base = match resolve_baseline(
                &baseline,
                history_dir.as_ref(),
                &cur.metadata.run.suite,
                Some(cur.metadata.run.run_id.as_str()),
            ) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let out = compare::compare(&base, &cur);
            if json {
                println!("{}", serde_json::to_string_pretty(&out).unwrap());
            } else {
                println!("{}", out.summary_for_agent);
                for d in &out.cell_deltas {
                    println!(
                        "  wall_rel={:?} cpu_rel={:?} qreg={}",
                        d.wall_us_p95_rel, d.cpu_total_us_p95_rel, d.quality_regressed
                    );
                }
            }
            match out.verdict.as_str() {
                "improve" | "mixed" => ExitCode::SUCCESS,
                "incomparable" => ExitCode::from(3),
                _ => ExitCode::from(1),
            }
        }
        Commands::History { history_dir, json } => {
            let hist = history_dir.unwrap_or_else(default_history_dir);
            let lines = history::list(&hist);
            if json {
                println!("{}", serde_json::to_string_pretty(&lines).unwrap());
            } else {
                for l in lines {
                    println!(
                        "{} suite={} adm={} score={:?} patch={:?}",
                        l.run_id, l.suite, l.admissible, l.score, l.patch_sha256
                    );
                }
            }
            ExitCode::SUCCESS
        }
        Commands::Explain {
            report,
            baseline,
            history_dir,
            json,
        } => {
            let report = match load_report(&report) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let cmp = baseline.as_ref().and_then(|b| {
                resolve_baseline(
                    b,
                    history_dir.as_ref(),
                    &report.metadata.run.suite,
                    Some(report.metadata.run.run_id.as_str()),
                )
                .ok()
                .map(|base| compare::compare(&base, &report))
            });
            let out = inspect_cmd::explain_report(&report, cmp.as_ref());
            if json {
                println!("{}", serde_json::to_string_pretty(&out).unwrap());
            } else {
                eprintln!("{}", out.summary);
                for l in &out.layers {
                    eprintln!("  [{}] {} — {}", l.status, l.layer, l.detail);
                }
            }
            ExitCode::SUCCESS
        }
        Commands::Diff {
            baseline,
            current,
            json,
        } => {
            let a = match load_report(&baseline) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let b = match load_report(&current) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let out = inspect_cmd::diff_reports(&a, &b);
            if json {
                println!("{}", serde_json::to_string_pretty(&out).unwrap());
            } else {
                eprintln!("{}", out.summary);
                for d in out.differences.iter().take(40) {
                    eprintln!("  [{}] {}", d.kind, d.field);
                }
            }
            ExitCode::SUCCESS
        }
        Commands::Export {
            report,
            format,
            output,
            json,
        } => {
            let report = match load_report(&report) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            if format != "csv" {
                eprintln!("only format=csv is supported currently");
                return ExitCode::from(2);
            }
            let csv = inspect_cmd::export_csv(&report);
            if let Some(o) = output {
                if let Err(e) = std::fs::write(&o, &csv) {
                    eprintln!("write error: {e}");
                    return ExitCode::from(2);
                }
                if json {
                    println!(
                        "{}",
                        serde_json::json!({
                            "status": "ok",
                            "format": "csv",
                            "path": o.display().to_string(),
                            "rows": report.cells.len(),
                        })
                    );
                } else {
                    eprintln!("wrote {} ({} rows)", o.display(), report.cells.len());
                }
            } else if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "ok",
                        "format": "csv",
                        "csv": csv,
                    })
                );
            } else {
                print!("{csv}");
            }
            ExitCode::SUCCESS
        }
        Commands::Replay {
            report,
            plan,
            path,
            history_dir,
            json,
        } => {
            let (suite, plan_digest_expect) = if let Some(rpath) = report {
                match load_report(&rpath) {
                    Ok(r) => (
                        r.metadata.run.suite.clone(),
                        r.plan_digest.clone(),
                    ),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            } else if let Some(ppath) = plan {
                match std::fs::read_to_string(&ppath)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                {
                    Some(v) => (
                        v.get("suite")
                            .and_then(|s| s.as_str())
                            .unwrap_or(agent::CONFIRM_SUITE)
                            .to_string(),
                        v.get("plan_digest")
                            .and_then(|s| s.as_str())
                            .map(|s| s.to_string()),
                    ),
                    None => {
                        eprintln!("error: cannot parse plan");
                        return ExitCode::from(2);
                    }
                }
            } else {
                eprintln!("error: pass --report or --plan");
                return ExitCode::from(2);
            };
            let (repos, _) = match resolve_repos(path, "default", None) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let hist = history_dir.unwrap_or_else(default_history_dir);
            let (warmup, iterations) = if suite.contains("screen") {
                agent::screen_iters()
            } else {
                agent::confirm_iters()
            };
            match run_measured(
                &repos,
                &suite,
                warmup,
                iterations,
                adapter::AdapterConfig::default(),
                Some(hist),
                true,
                Some(PathBuf::from(".fff-bench/runs").join("replay")),
                false,
                "mem,cpu",
                false,
            ) {
                Ok(r) => {
                    let digest_match = match (&plan_digest_expect, &r.plan_digest) {
                        (Some(a), Some(b)) => a == b,
                        _ => true,
                    };
                    if json {
                        println!(
                            "{}",
                            serde_json::json!({
                                "status": if digest_match { "complete" } else { "plan_drift" },
                                "run_id": r.metadata.run.run_id,
                                "suite": r.metadata.run.suite,
                                "plan_digest": r.plan_digest,
                                "expected_plan_digest": plan_digest_expect,
                                "admissible": r.admissible,
                                "score": r.score,
                            })
                        );
                    } else {
                        runner::print_table(&r);
                    }
                    if r.admissible {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(2)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_cli_preserves_whether_suite_was_explicit() {
        let defaulted = Cli::try_parse_from(["fff-bench", "run"]).unwrap();
        let explicit =
            Cli::try_parse_from(["fff-bench", "run", "--suite", "agent-confirm"]).unwrap();

        let Commands::Run {
            suite: defaulted_suite,
            ..
        } = defaulted.cmd
        else {
            panic!("expected run command");
        };
        let Commands::Run {
            suite: explicit_suite,
            ..
        } = explicit.cmd
        else {
            panic!("expected run command");
        };

        assert_eq!(defaulted_suite, None);
        assert_eq!(explicit_suite.as_deref(), Some("agent-confirm"));
    }

    #[test]
    fn explicit_run_suite_wins_over_manifest_suite() {
        assert_eq!(
            resolve_run_suite(Some("agent-confirm"), Some("agent-screen")),
            "agent-confirm"
        );
        assert_eq!(
            resolve_run_suite(None, Some("agent-confirm")),
            "agent-confirm"
        );
        assert_eq!(resolve_run_suite(None, None), "agent-screen");
    }
}
