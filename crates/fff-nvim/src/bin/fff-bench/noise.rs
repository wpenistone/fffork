//! Noisy-neighbor detection.

use crate::identity::NoiseMeta;
use crate::sensors::Sensors;
use sysinfo::{CpuRefreshKind, RefreshKind, System};

pub fn wait_quiet(wait_secs: u64, require: bool) -> Result<(), String> {
    if wait_secs == 0 && !require {
        return Ok(());
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(wait_secs.max(1));
    loop {
        let busy = host_cpu_busy_pct().unwrap_or(0.0);
        if busy < 15.0 {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            if require {
                return Err(format!(
                    "host busy ({busy:.0}% CPU); refuse --require-quiet (close other apps or wait)"
                ));
            }
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

pub fn host_cpu_busy_pct() -> Option<f64> {
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
    // first refresh is zero; wait and re-sample
    std::thread::sleep(std::time::Duration::from_millis(200));
    sys.refresh_cpu_usage();
    let cpus = sys.cpus();
    if cpus.is_empty() {
        return None;
    }
    let sum: f32 = cpus.iter().map(|c| c.cpu_usage()).sum();
    Some((sum / cpus.len() as f32) as f64)
}

/// This process's CPU share of one host core-equivalent, as % of one logical CPU
/// averaged over a short window (0–100×n_cpus scale, reported as % of machine).
pub fn process_cpu_busy_pct() -> Option<f64> {
    let sensors = Sensors::from_packs("cpu");
    let a = sensors.snapshot();
    let t0 = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(50));
    // Touch some CPU so the window is not empty when host is idle.
    let mut x = 1u64;
    for _ in 0..50_000 {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    }
    let _ = x;
    let b = sensors.snapshot();
    let d = sensors.delta(&a, &b);
    let wall_us = t0.elapsed().as_micros() as f64;
    if wall_us < 1.0 {
        return None;
    }
    let cpu_us = match (d.cpu_user_us, d.cpu_system_us) {
        (Some(u), Some(s)) => u.saturating_add(s) as f64,
        (Some(u), None) => u as f64,
        (None, Some(s)) => s as f64,
        _ => return None,
    };
    let ncpus = std::thread::available_parallelism()
        .map(|n| n.get() as f64)
        .unwrap_or(1.0)
        .max(1.0);
    // busy_cores / ncpus * 100 → machine utilization percent from this process.
    Some((cpu_us / wall_us / ncpus * 100.0).clamp(0.0, 100.0))
}

pub fn summarize(walls: &[f64]) -> NoiseMeta {
    let cv = coeff_var(walls);
    let host = host_cpu_busy_pct();
    let proc = process_cpu_busy_pct();
    let background = match (host, proc) {
        (Some(h), Some(p)) => Some((h - p).max(0.0)),
        (Some(h), None) => Some(h),
        _ => None,
    };
    let mut integrity = "clean".to_string();
    let mut policy = "allow".to_string();

    // Wall CV alone is informational on sub-ms microbenchmarks (high natural
    // variance). Score-blocking "noisy" requires clear host contention.
    if let Some(h) = host {
        if h > 60.0 {
            integrity = "noisy".into();
            policy = "no_auto_win".into();
        } else if h > 40.0 {
            if let Some(cv) = cv {
                if cv > 0.6 {
                    integrity = "noisy".into();
                    policy = "no_auto_win".into();
                }
            }
        }
    }
    // High background (host busy beyond this process) also marks noisy.
    if integrity == "clean" {
        if let Some(bg) = background {
            if bg > 45.0 {
                integrity = "noisy".into();
                policy = "no_auto_win".into();
            }
        }
    }
    if integrity == "clean" && host.is_none() && cv.is_none() {
        integrity = "unknown".into();
        policy = "allow".into();
    }

    NoiseMeta {
        integrity,
        host_cpu_busy_pct: host,
        background_cpu_pct: background,
        iter_wall_cv: cv,
        policy,
    }
}

fn coeff_var(v: &[f64]) -> Option<f64> {
    if v.len() < 5 {
        return None;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    if mean <= 0.0 {
        return None;
    }
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    Some(var.sqrt() / mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_cpu_returns_value() {
        let h = host_cpu_busy_pct();
        assert!(h.is_some());
        assert!(h.unwrap() >= 0.0);
    }

    #[test]
    fn summarize_sets_background_when_host_known() {
        let walls: Vec<f64> = (0..10).map(|i| 1000.0 + i as f64).collect();
        let n = summarize(&walls);
        // Host sample should succeed on supported platforms.
        if n.host_cpu_busy_pct.is_some() {
            assert!(n.background_cpu_pct.is_some());
        }
    }
}
