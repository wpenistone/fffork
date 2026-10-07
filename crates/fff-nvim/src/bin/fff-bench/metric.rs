//! Typed metric samples (SENSOR_SEMANTICS / Phase 6).

use crate::cost::CostSample;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSample {
    pub value: Option<f64>,
    pub unit: String,
    /// measured | unavailable | denied | not_requested | coarse
    pub status: String,
    pub scope: String,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution_unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
}

impl MetricSample {
    pub fn measured(value: f64, unit: &str, scope: &str, source: &str) -> Self {
        Self {
            value: Some(value),
            unit: unit.into(),
            status: "measured".into(),
            scope: scope.into(),
            source: source.into(),
            quality: Some("metered".into()),
            resolution: None,
            resolution_unit: None,
            reason_code: None,
        }
    }

    pub fn measured_quality(
        value: f64,
        unit: &str,
        scope: &str,
        source: &str,
        quality: &str,
        reason: Option<&str>,
    ) -> Self {
        Self {
            value: Some(value),
            unit: unit.into(),
            status: "measured".into(),
            scope: scope.into(),
            source: source.into(),
            quality: Some(quality.into()),
            resolution: None,
            resolution_unit: None,
            reason_code: reason.map(|s| s.into()),
        }
    }

    pub fn unavailable(unit: &str, scope: &str, source: &str, reason: &str) -> Self {
        Self {
            value: None,
            unit: unit.into(),
            status: "unavailable".into(),
            scope: scope.into(),
            source: source.into(),
            quality: None,
            resolution: None,
            resolution_unit: None,
            reason_code: Some(reason.into()),
        }
    }

    pub fn not_requested(unit: &str, scope: &str) -> Self {
        Self {
            value: None,
            unit: unit.into(),
            status: "not_requested".into(),
            scope: scope.into(),
            source: "none".into(),
            quality: None,
            resolution: None,
            resolution_unit: None,
            reason_code: Some("pack_not_requested".into()),
        }
    }
}

/// Schema-stable cost metrics object for one iteration (REPORT_SCHEMA cost.metrics).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostMetrics {
    pub wall_us: Option<MetricSample>,
    pub cpu_user_us: Option<MetricSample>,
    pub cpu_system_us: Option<MetricSample>,
    pub cpu_total_us: Option<MetricSample>,
    pub cpu_cycles: Option<MetricSample>,
    pub rss_start_mb: Option<MetricSample>,
    pub rss_end_mb: Option<MetricSample>,
    pub rss_peak_mb: Option<MetricSample>,
    pub page_faults_minor: Option<MetricSample>,
    pub context_switches: Option<MetricSample>,
    pub threads_observed_peak: Option<MetricSample>,
    pub process_io_read_bytes: Option<MetricSample>,
    pub process_io_write_bytes: Option<MetricSample>,
    pub energy_j: Option<MetricSample>,
}

impl CostMetrics {
    /// Build typed metrics from a finalized exterior CostSample + pack intent.
    pub fn from_sample(s: &CostSample, packs: &str) -> Self {
        let want_cpu = pack_has(packs, "cpu");
        let want_mem = pack_has(packs, "mem");
        let want_io = pack_has(packs, "io");
        let want_cycles = pack_has(packs, "cycles");
        let want_power = pack_has(packs, "power") || pack_has(packs, "energy");

        let wall = MetricSample::measured(s.wall_us as f64, "us", "operation", "harness_monotonic");

        let cpu_user = match s.cpu_user_us {
            Some(v) => Some(MetricSample::measured(
                v as f64,
                "us",
                "process",
                "os_process_times",
            )),
            None if want_cpu => Some(MetricSample::unavailable(
                "us",
                "process",
                "os_process_times",
                s.unavailable
                    .get("cpu_total_us")
                    .or_else(|| s.unavailable.get("cpu_user_us"))
                    .map(|x| x.as_str())
                    .unwrap_or("unmeasured"),
            )),
            None => Some(MetricSample::not_requested("us", "process")),
        };

        let cpu_system = match s.cpu_system_us {
            Some(v) => Some(MetricSample::measured(
                v as f64,
                "us",
                "process",
                "os_process_times",
            )),
            None if want_cpu => Some(MetricSample::unavailable(
                "us",
                "process",
                "os_process_times",
                "unmeasured",
            )),
            None => Some(MetricSample::not_requested("us", "process")),
        };

        let cpu_total = match s.cpu_total_us {
            Some(v) => Some(MetricSample::measured(
                v as f64,
                "us",
                "process",
                "os_process_times",
            )),
            None if want_cpu => Some(MetricSample::unavailable(
                "us",
                "process",
                "os_process_times",
                s.unavailable
                    .get("cpu_total_us")
                    .map(|x| x.as_str())
                    .unwrap_or("unmeasured"),
            )),
            None => Some(MetricSample::not_requested("us", "process")),
        };

        let cpu_cycles = match s.cpu_cycles {
            Some(v) => Some(MetricSample::measured(
                v as f64,
                "cycles",
                "process",
                "os_cycle_counter",
            )),
            None if want_cycles => Some(MetricSample::unavailable(
                "cycles",
                "process",
                "os_cycle_counter",
                s.unavailable
                    .get("cpu_cycles")
                    .map(|x| x.as_str())
                    .unwrap_or("unmeasured"),
            )),
            None => Some(MetricSample::not_requested("cycles", "process")),
        };

        let rss_start = match s.rss_start_mb {
            Some(v) if want_mem => Some(MetricSample::measured(
                v,
                "mb",
                "process",
                "os_process_memory",
            )),
            None if want_mem => Some(MetricSample::unavailable(
                "mb",
                "process",
                "os_process_memory",
                "unmeasured",
            )),
            _ => Some(MetricSample::not_requested("mb", "process")),
        };

        let rss_end = match s.rss_mb {
            Some(v) if want_mem => Some(MetricSample::measured_quality(
                v,
                "mb",
                "process",
                "os_process_memory",
                "end_sample",
                Some("not_peak_rss"),
            )),
            None if want_mem => Some(MetricSample::unavailable(
                "mb",
                "process",
                "os_process_memory",
                "unmeasured",
            )),
            _ => Some(MetricSample::not_requested("mb", "process")),
        };

        let rss_peak = match s.rss_peak_mb {
            Some(v) if want_mem => {
                let reason = s.unavailable.get("rss_peak_mb").map(|x| x.as_str());
                let quality = if reason.is_some() {
                    "approx_max_start_end"
                } else {
                    "metered"
                };
                Some(MetricSample::measured_quality(
                    v,
                    "mb",
                    "process",
                    "os_process_memory",
                    quality,
                    reason,
                ))
            }
            None if want_mem => Some(MetricSample::unavailable(
                "mb",
                "process",
                "os_process_memory",
                "unmeasured",
            )),
            _ => Some(MetricSample::not_requested("mb", "process")),
        };

        let faults = match s.page_faults {
            Some(v) if want_mem => Some(MetricSample::measured(
                v as f64,
                "count",
                "process",
                "os_process_faults",
            )),
            None if want_mem => Some(MetricSample::unavailable(
                "count",
                "process",
                "os_process_faults",
                "unmeasured",
            )),
            _ => Some(MetricSample::not_requested("count", "process")),
        };

        let ctxt = match s.context_switches {
            Some(v) if want_mem => Some(MetricSample::measured(
                v as f64,
                "count",
                "process",
                "os_context_switches",
            )),
            None if want_mem => Some(MetricSample::unavailable(
                "count",
                "process",
                "os_context_switches",
                "platform_unmeasured",
            )),
            _ => Some(MetricSample::not_requested("count", "process")),
        };

        let threads = match s.threads_observed_peak {
            Some(v) if want_cpu || want_mem => Some(MetricSample::measured(
                v as f64,
                "count",
                "process",
                "os_thread_count",
            )),
            None if want_cpu || want_mem => Some(MetricSample::unavailable(
                "count",
                "process",
                "os_thread_count",
                "unmeasured",
            )),
            _ => Some(MetricSample::not_requested("count", "process")),
        };

        let io_r = match s.disk_read_bytes {
            Some(v) if want_io => Some(MetricSample::measured(
                v as f64,
                "bytes",
                "process",
                "os_process_io",
            )),
            None if want_io => Some(MetricSample::unavailable(
                "bytes",
                "process",
                "os_process_io",
                "unmeasured",
            )),
            _ => Some(MetricSample::not_requested("bytes", "process")),
        };

        let io_w = match s.disk_write_bytes {
            Some(v) if want_io => Some(MetricSample::measured(
                v as f64,
                "bytes",
                "process",
                "os_process_io",
            )),
            None if want_io => Some(MetricSample::unavailable(
                "bytes",
                "process",
                "os_process_io",
                "unmeasured",
            )),
            _ => Some(MetricSample::not_requested("bytes", "process")),
        };

        let energy = match s.energy_j {
            Some(v) if want_power => Some(MetricSample::measured(
                v,
                "j",
                "process",
                s.energy_source.as_deref().unwrap_or("platform_energy"),
            )),
            None if want_power => Some(MetricSample::unavailable(
                "j",
                "process",
                "platform_energy",
                s.unavailable
                    .get("energy_j")
                    .map(|x| x.as_str())
                    .unwrap_or("unmeasured"),
            )),
            _ => Some(MetricSample::not_requested("j", "process")),
        };

        Self {
            wall_us: Some(wall),
            cpu_user_us: cpu_user,
            cpu_system_us: cpu_system,
            cpu_total_us: cpu_total,
            cpu_cycles,
            rss_start_mb: rss_start,
            rss_end_mb: rss_end,
            rss_peak_mb: rss_peak,
            page_faults_minor: faults,
            context_switches: ctxt,
            threads_observed_peak: threads,
            process_io_read_bytes: io_r,
            process_io_write_bytes: io_w,
            energy_j: energy,
        }
    }

    /// Names with status == measured.
    pub fn measured_names(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        let check = |m: &Option<MetricSample>, name: &'static str, acc: &mut Vec<&'static str>| {
            if m.as_ref().map(|x| x.status == "measured").unwrap_or(false) {
                acc.push(name);
            }
        };
        check(&self.wall_us, "wall_us", &mut out);
        check(&self.cpu_total_us, "cpu_total_us", &mut out);
        check(&self.rss_end_mb, "rss", &mut out);
        check(&self.rss_peak_mb, "rss_peak_mb", &mut out);
        check(&self.context_switches, "context_switches", &mut out);
        check(
            &self.threads_observed_peak,
            "threads_observed_peak",
            &mut out,
        );
        check(&self.cpu_cycles, "cpu_cycles", &mut out);
        check(&self.energy_j, "energy_j", &mut out);
        out
    }
}

fn pack_has(packs: &str, name: &str) -> bool {
    packs == "all"
        || packs.is_empty()
        || packs
            .split(',')
            .any(|p| p.trim() == name || p.trim() == "all")
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricCoverage {
    pub required: Vec<String>,
    pub present: Vec<String>,
    pub missing: Vec<String>,
    pub matched: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub by_status: BTreeMap<String, usize>,
}

pub fn coverage(required: &[&str], available: &[&str]) -> MetricCoverage {
    let present: Vec<String> = available.iter().map(|s| (*s).into()).collect();
    let missing: Vec<String> = required
        .iter()
        .filter(|r| !available.iter().any(|a| a == *r))
        .map(|s| (*s).into())
        .collect();
    MetricCoverage {
        required: required.iter().map(|s| (*s).into()).collect(),
        present,
        missing: missing.clone(),
        matched: missing.is_empty(),
        by_status: BTreeMap::new(),
    }
}

pub fn coverage_from_metrics(required: &[&str], metrics: &CostMetrics) -> MetricCoverage {
    let present = metrics.measured_names();
    let mut cov = coverage(required, &present);
    // Count statuses for audit.
    let mut by = BTreeMap::new();
    for m in [
        &metrics.wall_us,
        &metrics.cpu_total_us,
        &metrics.rss_end_mb,
        &metrics.rss_peak_mb,
        &metrics.cpu_cycles,
        &metrics.energy_j,
    ]
    .into_iter()
    .flatten()
    {
        *by.entry(m.status.clone()).or_insert(0usize) += 1;
    }
    cov.by_status = by;
    cov
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_detects_missing() {
        let c = coverage(&["wall_us", "cpu_total_us"], &["wall_us"]);
        assert!(!c.matched);
        assert_eq!(c.missing, vec!["cpu_total_us".to_string()]);
    }

    #[test]
    fn from_sample_marks_wall_measured() {
        let mut s = CostSample::with_wall(42, 4);
        s.finalize_derived();
        let m = CostMetrics::from_sample(&s, "mem,cpu");
        assert_eq!(m.wall_us.as_ref().unwrap().status, "measured");
        assert_eq!(m.wall_us.as_ref().unwrap().value, Some(42.0));
        // zero cpu becomes unavailable after finalize
        assert_eq!(m.cpu_total_us.as_ref().unwrap().status, "unavailable");
    }

    #[test]
    fn not_requested_when_pack_off() {
        let s = CostSample::with_wall(10, 1);
        let m = CostMetrics::from_sample(&s, "mem");
        assert_eq!(m.cpu_total_us.as_ref().unwrap().status, "not_requested");
    }
}
