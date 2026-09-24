//! Stability rollup (METHODOLOGY vocabulary).

use crate::cost::Dist;
use crate::identity::NoiseMeta;
use crate::runner::CellReport;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StabilitySummary {
    /// clean | elevated | unstable
    pub stability: String,
    /// low | elevated | high
    pub within_cell_dispersion: String,
    /// low | elevated | high | unknown
    pub between_block_drift: String,
    /// clean | noisy | throttled | unknown
    pub host_noise: String,
    /// low | high
    pub tail_uncertainty: String,
    #[serde(default)]
    pub reason_codes: Vec<String>,
    /// Max relative MAD across cost cells.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_relative_mad: Option<f64>,
    /// Max relative block-to-block median drift when blocks exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_block_drift: Option<f64>,
    /// Cells with relative_mad > 0.20.
    #[serde(default)]
    pub high_dispersion_cells: usize,
}

pub fn classify_relative_mad(rmad: f64) -> &'static str {
    if rmad <= 0.10 {
        "low"
    } else if rmad <= 0.20 {
        "elevated"
    } else {
        "high"
    }
}

/// METHODOLOGY: max relative median drift across consecutive blocks.
/// ≤5% low, ≤10% elevated, >10% high.
pub fn classify_block_drift(block_medians: &[f64]) -> (&'static str, Option<f64>) {
    if block_medians.len() < 2 {
        return ("unknown", None);
    }
    let mut max_rel = 0.0_f64;
    for w in block_medians.windows(2) {
        let mid = (w[0] + w[1]) / 2.0;
        if mid > 1e-9 {
            max_rel = max_rel.max((w[1] - w[0]).abs() / mid);
        }
    }
    let label = if max_rel <= 0.05 {
        "low"
    } else if max_rel <= 0.10 {
        "elevated"
    } else {
        "high"
    };
    (label, Some(max_rel))
}

pub fn from_cells(cells: &[CellReport], noise: &NoiseMeta, iterations: usize) -> StabilitySummary {
    from_cells_with_drift(cells, noise, iterations, None)
}

pub fn from_cells_with_drift(
    cells: &[CellReport],
    noise: &NoiseMeta,
    iterations: usize,
    block_medians: Option<&[f64]>,
) -> StabilitySummary {
    let mut max_rmad = 0.0_f64;
    let mut high = 0usize;
    let mut elevated = 0usize;
    for c in cells {
        if c.family == "quality" || c.scenario_id.starts_with("gate") {
            continue;
        }
        let rmad = c.cost.wall_us.relative_mad;
        if rmad > max_rmad {
            max_rmad = rmad;
        }
        match classify_relative_mad(rmad) {
            "high" => high += 1,
            "elevated" => elevated += 1,
            _ => {}
        }
    }

    let within = if high > 0 {
        "high"
    } else if elevated > 0 || max_rmad > 0.10 {
        "elevated"
    } else {
        "low"
    };

    let host = match noise.integrity.as_str() {
        "noisy" => "noisy",
        "clean" => "clean",
        _ => "unknown",
    };

    let (drift, max_block_drift) = match block_medians {
        Some(m) => classify_block_drift(m),
        None => ("unknown", None),
    };

    let tail = if iterations < 50 { "high" } else { "low" };

    let mut reasons = Vec::new();
    if within == "high" {
        reasons.push("within_cell_dispersion_high".into());
    } else if within == "elevated" {
        reasons.push("within_cell_dispersion_elevated".into());
    }
    if host == "noisy" {
        reasons.push("host_noise".into());
    }
    if tail == "high" {
        reasons.push("tail_uncertainty_high".into());
    }
    if drift == "high" {
        reasons.push("between_block_drift_high".into());
    } else if drift == "elevated" {
        reasons.push("between_block_drift_elevated".into());
    }

    // METHODOLOGY: high block drift or high within-cell or host noise → unstable.
    let stability = if within == "high" || host == "noisy" || drift == "high" {
        "unstable"
    } else if within == "elevated" || tail == "high" || drift == "elevated" {
        "elevated"
    } else {
        "clean"
    };

    StabilitySummary {
        stability: stability.into(),
        within_cell_dispersion: within.into(),
        between_block_drift: drift.into(),
        host_noise: host.into(),
        tail_uncertainty: tail.into(),
        reason_codes: reasons,
        max_relative_mad: if max_rmad > 0.0 { Some(max_rmad) } else { None },
        max_block_drift,
        high_dispersion_cells: high,
    }
}

pub fn wall_mad_label(d: &Dist) -> String {
    classify_relative_mad(d.relative_mad).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mad_bands() {
        assert_eq!(classify_relative_mad(0.05), "low");
        assert_eq!(classify_relative_mad(0.15), "elevated");
        assert_eq!(classify_relative_mad(0.25), "high");
    }

    #[test]
    fn block_drift_bands() {
        assert_eq!(classify_block_drift(&[100.0]).0, "unknown");
        assert_eq!(classify_block_drift(&[100.0, 103.0]).0, "low");
        assert_eq!(classify_block_drift(&[100.0, 108.0]).0, "elevated");
        let (lab, max) = classify_block_drift(&[100.0, 120.0]);
        assert_eq!(lab, "high");
        assert!(max.unwrap() > 0.10);
    }

    #[test]
    fn high_block_drift_marks_unstable() {
        let noise = NoiseMeta {
            integrity: "clean".into(),
            host_cpu_busy_pct: Some(5.0),
            background_cpu_pct: None,
            iter_wall_cv: Some(0.05),
            policy: "allow".into(),
        };
        let s = from_cells_with_drift(&[], &noise, 50, Some(&[100.0, 130.0]));
        assert_eq!(s.between_block_drift, "high");
        assert_eq!(s.stability, "unstable");
        assert!(s
            .reason_codes
            .iter()
            .any(|r| r.contains("between_block_drift")));
    }
}
