//! Canonical cost vector + distributions (BENCHMARK_SPEC).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Dist {
    pub n: usize,
    pub min: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
    pub stddev: f64,
    pub sum: f64,
    /// Median absolute deviation (METHODOLOGY primary dispersion).
    #[serde(default)]
    pub mad: f64,
    /// mad / p50 when p50 > 0.
    #[serde(default)]
    pub relative_mad: f64,
}

impl Dist {
    pub fn from_values(mut v: Vec<f64>) -> Self {
        if v.is_empty() {
            return Self::default();
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        let sum: f64 = v.iter().sum();
        let mean = sum / n as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let p50 = percentile(&v, 50.0);
        let mut abs_dev: Vec<f64> = v.iter().map(|x| (x - p50).abs()).collect();
        abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = percentile(&abs_dev, 50.0);
        let relative_mad = if p50 > 0.0 { mad / p50 } else { 0.0 };
        // METHODOLOGY: p99 is not decision-grade when n < 200 → store 0 (null-for-decision).
        let p99 = if n >= 200 {
            percentile(&v, 99.0)
        } else {
            0.0
        };
        Self {
            n,
            min: v[0],
            p50,
            p90: percentile(&v, 90.0),
            p95: percentile(&v, 95.0),
            p99,
            max: v[n - 1],
            mean,
            stddev: var.sqrt(),
            sum,
            mad,
            relative_mad,
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostSample {
    pub wall_us: u64,
    pub cpu_user_us: Option<u64>,
    pub cpu_system_us: Option<u64>,
    pub cpu_total_us: Option<u64>,
    pub cpu_cycles: Option<u64>,
    pub cpu_parallelism: Option<f64>,
    pub threads_configured: usize,
    pub threads_observed_peak: Option<usize>,
    pub rss_start_mb: Option<f64>,
    pub rss_mb: Option<f64>,
    pub rss_peak_mb: Option<f64>,
    pub rss_delta_mb: Option<f64>,
    pub page_faults: Option<u64>,
    pub context_switches: Option<u64>,
    pub disk_read_bytes: Option<u64>,
    pub disk_write_bytes: Option<u64>,
    pub energy_j: Option<f64>,
    pub joules_per_op: Option<f64>,
    pub energy_source: Option<String>,
    pub primary_span: String,
    pub integrity: String,
    /// Reasons for null exterior fields (honest sensors).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub unavailable: std::collections::BTreeMap<String, String>,
    /// Typed metric samples for this iteration (SENSOR_SEMANTICS).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<crate::metric::CostMetrics>,
}

impl CostSample {
    pub fn with_wall(wall_us: u64, threads: usize) -> Self {
        Self {
            wall_us,
            threads_configured: threads,
            primary_span: "harness.op_total".into(),
            integrity: "unknown".into(),
            ..Default::default()
        }
    }

    pub fn finalize_derived(&mut self) {
        let total = match (self.cpu_user_us, self.cpu_system_us) {
            (Some(u), Some(s)) => Some(u.saturating_add(s)),
            (Some(u), None) => Some(u),
            (None, Some(s)) => Some(s),
            (None, None) => None,
        };
        match total {
            Some(0) => {
                self.cpu_user_us = None;
                self.cpu_system_us = None;
                self.cpu_total_us = None;
                self.unavailable.insert(
                    "cpu_total_us".into(),
                    "sub_resolution_or_zero_delta".into(),
                );
            }
            Some(t) => self.cpu_total_us = Some(t),
            None => {}
        }
        if let Some(cpu) = self.cpu_total_us {
            if self.wall_us > 0 {
                self.cpu_parallelism = Some(cpu as f64 / self.wall_us as f64);
            }
        }
        if let (Some(s), Some(e)) = (self.rss_start_mb, self.rss_mb) {
            self.rss_delta_mb = Some(e - s);
        }
        if let Some(e) = self.energy_j {
            if e > 0.0 {
                self.joules_per_op = Some(e);
            } else {
                self.energy_j = None;
                self.joules_per_op = None;
                self.unavailable
                    .insert("energy_j".into(), "zero_or_unavailable".into());
            }
        }
        if self.cpu_cycles == Some(0) {
            self.cpu_cycles = None;
            self.unavailable
                .insert("cpu_cycles".into(), "sub_resolution_or_zero_delta".into());
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSummary {
    pub wall_us: Dist,
    pub cpu_total_us: Option<Dist>,
    pub cpu_cycles: Option<Dist>,
    pub cpu_parallelism: Option<Dist>,
    pub rss_peak_mb: Option<f64>,
    pub rss_mb_end: Option<Dist>,
    pub disk_read_bytes: Option<Dist>,
    pub energy_j: Option<Dist>,
    /// Peak OS thread count observed across samples.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads_observed_peak: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_switches: Option<Dist>,
    pub threads_configured: usize,
    pub primary_span: String,
    pub qps_derived: f64,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub unavailable: std::collections::BTreeMap<String, String>,
}

impl Default for CostSummary {
    fn default() -> Self {
        Self {
            wall_us: Dist::default(),
            cpu_total_us: None,
            cpu_cycles: None,
            cpu_parallelism: None,
            rss_peak_mb: None,
            rss_mb_end: None,
            disk_read_bytes: None,
            energy_j: None,
            threads_observed_peak: None,
            context_switches: None,
            threads_configured: 0,
            primary_span: "harness.op_total".into(),
            qps_derived: 0.0,
            unavailable: Default::default(),
        }
    }
}

impl CostSummary {
    pub fn from_samples(samples: &[CostSample]) -> Self {
        let walls: Vec<f64> = samples.iter().map(|s| s.wall_us as f64).collect();
        let wall = Dist::from_values(walls);
        let sum_s = wall.sum / 1_000_000.0;
        let qps = if sum_s > 0.0 {
            samples.len() as f64 / sum_s
        } else {
            0.0
        };

        // Only positive CPU samples count as measured evidence.
        let cpu: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.cpu_total_us.filter(|v| *v > 0).map(|v| v as f64))
            .collect();
        let par: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.cpu_parallelism.filter(|p| *p > 0.0))
            .collect();
        let rss_end: Vec<f64> = samples.iter().filter_map(|s| s.rss_mb).collect();
        let disk: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.disk_read_bytes.map(|v| v as f64))
            .collect();
        let energy: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.energy_j.filter(|e| *e > 0.0))
            .collect();
        let cycles: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.cpu_cycles.filter(|c| *c > 0).map(|c| c as f64))
            .collect();
        let peak = samples
            .iter()
            .filter_map(|s| s.rss_peak_mb)
            .fold(0.0_f64, f64::max);
        let thr_peak = samples
            .iter()
            .filter_map(|s| s.threads_observed_peak)
            .max();
        let ctxt: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.context_switches.map(|v| v as f64))
            .collect();

        let mut unavailable = std::collections::BTreeMap::new();
        if cpu.is_empty()
            && samples
                .iter()
                .any(|s| s.unavailable.contains_key("cpu_total_us"))
        {
            unavailable.insert(
                "cpu_total_us".into(),
                "sub_resolution_or_unmeasured".into(),
            );
        }
        if cycles.is_empty()
            && samples
                .iter()
                .any(|s| s.unavailable.contains_key("cpu_cycles"))
        {
            unavailable.insert("cpu_cycles".into(), "unmeasured".into());
        }
        if energy.is_empty()
            && samples
                .iter()
                .any(|s| s.unavailable.contains_key("energy_j"))
        {
            unavailable.insert("energy_j".into(), "unmeasured".into());
        }

        Self {
            wall_us: wall,
            cpu_total_us: if cpu.is_empty() {
                None
            } else {
                Some(Dist::from_values(cpu))
            },
            cpu_cycles: if cycles.is_empty() {
                None
            } else {
                Some(Dist::from_values(cycles))
            },
            cpu_parallelism: if par.is_empty() {
                None
            } else {
                Some(Dist::from_values(par))
            },
            rss_peak_mb: if peak > 0.0 { Some(peak) } else { None },
            rss_mb_end: if rss_end.is_empty() {
                None
            } else {
                Some(Dist::from_values(rss_end))
            },
            disk_read_bytes: if disk.is_empty() {
                None
            } else {
                Some(Dist::from_values(disk))
            },
            energy_j: if energy.is_empty() {
                None
            } else {
                Some(Dist::from_values(energy))
            },
            threads_observed_peak: thr_peak,
            context_switches: if ctxt.is_empty() {
                None
            } else {
                Some(Dist::from_values(ctxt))
            },
            threads_configured: samples.first().map(|s| s.threads_configured).unwrap_or(0),
            primary_span: samples
                .first()
                .map(|s| s.primary_span.clone())
                .unwrap_or_else(|| "harness.op_total".into()),
            qps_derived: qps,
            unavailable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p99_zero_when_n_below_200() {
        let d = Dist::from_values((1..=50).map(|i| i as f64).collect());
        assert_eq!(d.n, 50);
        assert_eq!(d.p99, 0.0);
        assert!(d.p95 > 0.0);
    }

    #[test]
    fn p99_set_when_n_at_least_200() {
        let d = Dist::from_values((1..=200).map(|i| i as f64).collect());
        assert_eq!(d.n, 200);
        assert!(d.p99 > 0.0);
    }

    #[test]
    fn percentile_p95_nearest_rank() {
        let d = Dist::from_values((1..=100).map(|i| i as f64).collect());
        assert_eq!(d.n, 100);
        assert!((d.p50 - 50.0).abs() < 1.0 || (d.p50 - 51.0).abs() < 1.0);
        assert!(d.p95 >= 95.0);
        assert_eq!(d.min, 1.0);
        assert_eq!(d.max, 100.0);
    }

    #[test]
    fn zero_cpu_sample_not_in_summary_dist() {
        let mut s = CostSample::with_wall(100, 4);
        s.cpu_user_us = Some(0);
        s.cpu_system_us = Some(0);
        s.finalize_derived();
        assert!(s.cpu_total_us.is_none());
        assert!(s.unavailable.contains_key("cpu_total_us"));

        let mut s2 = CostSample::with_wall(200, 4);
        s2.cpu_user_us = Some(1000);
        s2.cpu_system_us = Some(0);
        s2.finalize_derived();
        // only user set - finalize handles (Some(u), None)
        assert_eq!(s2.cpu_total_us, Some(1000));

        let sum = CostSummary::from_samples(&[s, s2.clone()]);
        assert!(sum.cpu_total_us.is_some());
        assert!(sum.cpu_total_us.as_ref().unwrap().p50 > 0.0);
    }

    #[test]
    fn score_composition_matches_core_formula() {
        let s = ScoreBreakdown {
            total: 100 + 10 + 5 - 3 + 20 + 0 + 7 + 2,
            base_score: 100,
            filename_bonus: 20,
            special_filename_bonus: 20, // informational only
            frecency_boost: 10,
            git_status_boost: 5,
            git_recency_boost: 0,
            evidence_bonus: 0,
            distance_penalty: -3,
            current_file_penalty: 0,
            combo_match_boost: 7,
            path_alignment_bonus: 2,
            exact_match: true,
            match_type: "exact_filename".into(),
        };
        assert!(score_composition_ok(&s));
        assert_eq!(score_components_sum(&s), s.total);
    }

    #[test]
    fn all_zero_cpu_summary_is_none() {
        let mut samples = Vec::new();
        for w in [50u64, 60, 70] {
            let mut s = CostSample::with_wall(w, 1);
            s.cpu_user_us = Some(0);
            s.cpu_system_us = Some(0);
            s.finalize_derived();
            samples.push(s);
        }
        let sum = CostSummary::from_samples(&samples);
        assert!(sum.cpu_total_us.is_none());
        assert!(sum.wall_us.p50 > 0.0);
        assert!(sum.unavailable.contains_key("cpu_total_us"));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScoreBreakdown {
    pub total: i32,
    pub base_score: i32,
    pub filename_bonus: i32,
    pub special_filename_bonus: i32,
    pub frecency_boost: i32,
    pub git_status_boost: i32,
    #[serde(default)]
    pub git_recency_boost: i32,
    #[serde(default)]
    pub evidence_bonus: i32,
    pub distance_penalty: i32,
    pub current_file_penalty: i32,
    pub combo_match_boost: i32,
    pub path_alignment_bonus: i32,
    pub exact_match: bool,
    pub match_type: String,
}

impl From<&fff::Score> for ScoreBreakdown {
    fn from(s: &fff::Score) -> Self {
        Self {
            total: s.total,
            base_score: s.base_score,
            filename_bonus: s.filename_bonus,
            special_filename_bonus: s.special_filename_bonus,
            frecency_boost: s.frecency_boost,
            git_status_boost: s.git_status_boost,
            git_recency_boost: s.git_recency_boost,
            evidence_bonus: s.evidence_bonus,
            distance_penalty: s.distance_penalty,
            current_file_penalty: s.current_file_penalty,
            combo_match_boost: s.combo_match_boost,
            path_alignment_bonus: s.path_alignment_bonus,
            exact_match: s.exact_match,
            match_type: s.match_type.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TopHit {
    pub path: String,
    pub score: ScoreBreakdown,
    /// Grep line number when known (1-indexed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

/// Reconstruct total from public components (matches fff-core score composition).
/// Note: special_filename_bonus is informational and already included in filename_bonus.
pub fn score_components_sum(s: &ScoreBreakdown) -> i32 {
    s.base_score
        .saturating_add(s.frecency_boost)
        .saturating_add(s.git_status_boost)
        .saturating_add(s.git_recency_boost)
        .saturating_add(s.distance_penalty)
        .saturating_add(s.filename_bonus)
        .saturating_add(s.current_file_penalty)
        .saturating_add(s.combo_match_boost)
        .saturating_add(s.path_alignment_bonus)
        .saturating_add(s.evidence_bonus)
}

/// Composition check when breakdown is populated.
pub fn score_composition_ok(s: &ScoreBreakdown) -> bool {
    // Grep-only or totals without component breakdown.
    if s.base_score == 0
        && s.filename_bonus == 0
        && s.frecency_boost == 0
        && s.distance_penalty == 0
        && s.combo_match_boost == 0
    {
        return true;
    }
    let recon = score_components_sum(s);
    recon == s.total
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpResultData {
    pub total_matched: usize,
    pub returned: usize,
    pub files_searched: Option<usize>,
    pub top_k: Vec<TopHit>,
    pub rank_target: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordered_digest: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterLog {
    pub t_unix_ms: u128,
    pub scenario_id: String,
    pub repo_id: String,
    pub cell: serde_json::Value,
    pub cell_key: String,
    pub iter: usize,
    pub warmup: bool,
    pub cost: CostSample,
    pub result: OpResultData,
}
