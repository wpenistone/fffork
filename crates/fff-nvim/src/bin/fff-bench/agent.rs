//! Agent-facing response contract (AGENT_GUIDE / OPERATIONS).

use crate::compare::CompareOut;
use crate::runner::BenchReport;
use crate::score_cmd::ScoreOut;
use serde::{Deserialize, Serialize};
use std::process::ExitCode;

/// Default pack workflow roles (FFF_REFERENCE_PACK).
pub const SCREEN_SUITE: &str = "agent-screen";
pub const CONFIRM_SUITE: &str = "agent-confirm";
pub const DIAGNOSE_SUITE: &str = "peak-diagnose";
pub const RESILIENCE_SUITE: &str = "resilience-confirm";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub code: String,
    pub priority: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence_grade: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict: Option<String>,
    #[serde(default)]
    pub reason_codes: Vec<String>,
    pub next_action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_recommendations: Vec<Recommendation>,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screen_run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<ScoreOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comparison: Option<CompareOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<serde_json::Value>,
}

impl AgentResponse {
    pub fn exit_code(&self) -> ExitCode {
        match self.status.as_str() {
            "ready" | "complete" | "baseline_recorded" | "promising" => ExitCode::SUCCESS,
            "inadmissible" | "regress" | "capped" => ExitCode::from(1),
            "infra_error" | "setup_required" | "manifest_required" | "baseline_required" => {
                ExitCode::from(2)
            }
            "incomparable" => ExitCode::from(3),
            "unstable" | "noisy" => ExitCode::from(4),
            // screen-only / no confirmed improvement
            "screen_no_signal"
            | "equivalent"
            | "mixed"
            | "screen_complete"
            | "confirm_complete" => ExitCode::from(5),
            _ => {
                // Fall back on verdict when status is generic complete-ish
                match self.verdict.as_deref() {
                    Some("improve") => ExitCode::SUCCESS,
                    Some("inadmissible") | Some("regress") => ExitCode::from(1),
                    Some("incomparable") => ExitCode::from(3),
                    Some("unstable") => ExitCode::from(4),
                    Some("equivalent") | Some("mixed") => ExitCode::from(5),
                    _ => ExitCode::from(2),
                }
            }
        }
    }
}

pub fn doctor_ready(summary: &str) -> AgentResponse {
    AgentResponse {
        status: "ready".into(),
        evidence_grade: None,
        verdict: None,
        reason_codes: vec!["environment_ready".into()],
        next_action: "continue".into(),
        next_command: None,
        agent_recommendations: vec![],
        summary: summary.into(),
        baseline_run_id: None,
        candidate_run_id: None,
        screen_run_id: None,
        report_path: None,
        score: None,
        comparison: None,
        plan: None,
    }
}

pub fn doctor_setup(reason: &str, setup_cmd: &str) -> AgentResponse {
    AgentResponse {
        status: "setup_required".into(),
        evidence_grade: None,
        verdict: None,
        reason_codes: vec!["setup_required".into()],
        next_action: "run_setup".into(),
        next_command: Some(setup_cmd.into()),
        agent_recommendations: vec![],
        summary: reason.into(),
        baseline_run_id: None,
        candidate_run_id: None,
        screen_run_id: None,
        report_path: None,
        score: None,
        comparison: None,
        plan: None,
    }
}

pub fn doctor_noisy(pct: f64) -> AgentResponse {
    AgentResponse {
        status: "noisy".into(),
        evidence_grade: None,
        verdict: None,
        reason_codes: vec!["host_noise_background_cpu".into()],
        next_action: "retry_after_quiet".into(),
        next_command: None,
        agent_recommendations: vec![Recommendation {
            code: "rerun_after_quiet".into(),
            priority: "required".into(),
            reason_code: Some("host_noise_background_cpu".into()),
            document: Some("benchmark/METHODOLOGY.md".into()),
            message: format!("Background CPU ~{pct:.0}% exceeds quiet threshold."),
        }],
        summary: format!("Host is noisy ({pct:.0}% CPU). Quiet the machine and retry."),
        baseline_run_id: None,
        candidate_run_id: None,
        screen_run_id: None,
        report_path: None,
        score: None,
        comparison: None,
        plan: None,
    }
}

pub fn doctor_infra(reason: &str) -> AgentResponse {
    AgentResponse {
        status: "infra_error".into(),
        evidence_grade: None,
        verdict: None,
        reason_codes: vec!["infra_error".into()],
        next_action: "repair_environment".into(),
        next_command: None,
        agent_recommendations: vec![],
        summary: reason.into(),
        baseline_run_id: None,
        candidate_run_id: None,
        screen_run_id: None,
        report_path: None,
        score: None,
        comparison: None,
        plan: None,
    }
}

pub fn baseline_required() -> AgentResponse {
    AgentResponse {
        status: "baseline_required".into(),
        evidence_grade: None,
        verdict: None,
        reason_codes: vec!["baseline_required".into()],
        next_action: "measure_baseline".into(),
        next_command: Some(
            "fff-bench baseline --subject <baseline-artifact-ref> --json".into(),
        ),
        agent_recommendations: vec![],
        summary: "No compatible baseline in history. Measure one before check.".into(),
        baseline_run_id: None,
        candidate_run_id: None,
        screen_run_id: None,
        report_path: None,
        score: None,
        comparison: None,
        plan: None,
    }
}

pub fn from_compare(
    cmp: &CompareOut,
    evidence_grade: &str,
    baseline_id: Option<String>,
    candidate_id: Option<String>,
    report_path: Option<String>,
    score: Option<ScoreOut>,
) -> AgentResponse {
    let (status, next_action, next_command) = match cmp.verdict.as_str() {
        "improve" => (
            "improve".into(),
            "benchmark_complete".into(),
            None,
        ),
        "equivalent" => (
            "equivalent".into(),
            "benchmark_complete".into(),
            None,
        ),
        "mixed" => ("mixed".into(), "benchmark_complete".into(), None),
        "regress" => ("regress".into(), "benchmark_complete".into(), None),
        "inadmissible" => (
            "inadmissible".into(),
            "inspect_gates".into(),
            None,
        ),
        "incomparable" => (
            "incomparable".into(),
            "repair_comparability".into(),
            None,
        ),
        "unstable" => (
            "unstable".into(),
            "retry_once_quiet".into(),
            None,
        ),
        "promising" => (
            "promising".into(),
            "run_confirmation".into(),
            Some(
                "fff-bench check --mode confirm --baseline auto --candidate <artifact-ref> --json"
                    .into(),
            ),
        ),
        "screen_no_signal" => (
            "screen_no_signal".into(),
            "screen_complete".into(),
            None,
        ),
        other => (other.into(), "inspect".into(), None),
    };

    AgentResponse {
        status,
        evidence_grade: Some(evidence_grade.into()),
        verdict: Some(cmp.verdict.clone()),
        reason_codes: cmp.reason_codes.clone(),
        next_action,
        next_command,
        agent_recommendations: vec![],
        summary: cmp.summary_for_agent.clone(),
        baseline_run_id: baseline_id,
        candidate_run_id: candidate_id,
        screen_run_id: None,
        report_path,
        score,
        comparison: Some(cmp.clone()),
        plan: None,
    }
}

pub fn from_report_gates_only(report: &BenchReport, grade: &str) -> AgentResponse {
    if !report.admissible {
        return AgentResponse {
            status: "inadmissible".into(),
            evidence_grade: Some(grade.into()),
            verdict: Some("inadmissible".into()),
            reason_codes: vec!["gates_failed".into()],
            next_action: "inspect_gates".into(),
            next_command: None,
            agent_recommendations: vec![],
            summary: format!("gates failed: {:?}", report.gates.failed),
            baseline_run_id: None,
            candidate_run_id: Some(report.metadata.run.run_id.clone()),
            screen_run_id: None,
            report_path: report.metadata.run.log_dir.clone(),
            score: report.score.clone(),
            comparison: None,
            plan: None,
        };
    }
    if report.noise.integrity == "noisy" {
        return AgentResponse {
            status: "unstable".into(),
            evidence_grade: Some(grade.into()),
            verdict: Some("unstable".into()),
            reason_codes: vec!["host_noise".into()],
            next_action: "retry_once_quiet".into(),
            next_command: None,
            agent_recommendations: vec![Recommendation {
                code: "rerun_after_quiet".into(),
                priority: "required".into(),
                reason_code: Some("host_noise".into()),
                document: Some("benchmark/METHODOLOGY.md".into()),
                message: "Host noise invalidated automatic acceptance.".into(),
            }],
            summary: "Measurement unstable due to host noise.".into(),
            baseline_run_id: None,
            candidate_run_id: Some(report.metadata.run.run_id.clone()),
            screen_run_id: None,
            report_path: report.metadata.run.log_dir.clone(),
            score: report.score.clone(),
            comparison: None,
            plan: None,
        };
    }
    AgentResponse {
        status: "complete".into(),
        evidence_grade: Some(grade.into()),
        verdict: None,
        reason_codes: vec!["gates_passed".into()],
        next_action: "benchmark_complete".into(),
        next_command: None,
        agent_recommendations: vec![],
        summary: format!("suite={} admissible=true", report.metadata.run.suite),
        baseline_run_id: None,
        candidate_run_id: Some(report.metadata.run.run_id.clone()),
        screen_run_id: None,
        report_path: report.metadata.run.log_dir.clone(),
        score: report.score.clone(),
        comparison: None,
        plan: None,
    }
}

/// Screen sample policy (METHODOLOGY / FFF_REFERENCE_PACK).
pub fn screen_iters() -> (usize, usize) {
    (3, 10) // warmup, measured
}

/// Confirm sample policy (decision-grade; lighter than full diagnose).
pub fn confirm_iters() -> (usize, usize) {
    (5, 30)
}

/// Normalize suite aliases to canonical pack names.
pub fn normalize_suite(suite: &str) -> String {
    match suite {
        "agent-loop" | "screen" => SCREEN_SUITE.into(),
        "confirm" => CONFIRM_SUITE.into(),
        "peak" | "peak-frontier" | "diagnose" => DIAGNOSE_SUITE.into(),
        "resilience" => RESILIENCE_SUITE.into(),
        "gates" => "ci-gates".into(),
        other => other.into(),
    }
}

pub fn default_workflow() -> (String, String, String) {
    (
        SCREEN_SUITE.into(),
        CONFIRM_SUITE.into(),
        DIAGNOSE_SUITE.into(),
    )
}
