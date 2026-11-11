//! Convergence-based warmup (METHODOLOGY).

/// Defaults from METHODOLOGY.
pub const MIN_WARMUPS: usize = 3;
pub const WINDOW: usize = 3;
pub const THRESHOLD: f64 = 0.05;
pub const MAX_WARMUPS: usize = 15;

#[derive(Debug, Clone)]
pub struct WarmupOutcome {
    pub warmups_run: usize,
    pub converged: bool,
    pub last_rel_diff: Option<f64>,
}

/// Decide whether more warmups are needed given measured warmup walls so far.
/// `walls` is the list of warmup observation walls (already excluding measured).
pub fn needs_more(walls: &[f64], min: usize, max: usize, window: usize, thr: f64) -> bool {
    if walls.len() < min {
        return true;
    }
    if walls.len() >= max {
        return false;
    }
    if walls.len() < window * 2 {
        return true;
    }
    let n = walls.len();
    let w1 = &walls[n - 2 * window..n - window];
    let w2 = &walls[n - window..];
    let m1 = median(w1);
    let m2 = median(w2);
    if m1 <= 0.0 && m2 <= 0.0 {
        return false;
    }
    let base = m1.max(m2).max(1.0);
    let rel = (m2 - m1).abs() / base;
    rel > thr
}

pub fn relative_diff(walls: &[f64], window: usize) -> Option<f64> {
    if walls.len() < window * 2 {
        return None;
    }
    let n = walls.len();
    let m1 = median(&walls[n - 2 * window..n - window]);
    let m2 = median(&walls[n - window..]);
    let base = m1.max(m2).max(1.0);
    Some((m2 - m1).abs() / base)
}

pub fn finalize(walls: &[f64], max: usize, window: usize, thr: f64) -> WarmupOutcome {
    let last = relative_diff(walls, window);
    let converged = match last {
        Some(r) => r <= thr,
        None => walls.len() >= MIN_WARMUPS,
    };
    // If we hit max without converging, mark not converged.
    let converged = if walls.len() >= max {
        last.map(|r| r <= thr).unwrap_or(false)
    } else {
        converged
    };
    WarmupOutcome {
        warmups_run: walls.len(),
        converged,
        last_rel_diff: last,
    }
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[mid - 1] + s[mid]) / 2.0
    } else {
        s[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needs_min_warmups() {
        assert!(needs_more(&[1.0, 1.0], 3, 15, 3, 0.05));
    }

    #[test]
    fn converges_when_stable() {
        let walls = vec![100.0; 12];
        assert!(!needs_more(&walls, 3, 15, 3, 0.05));
        let o = finalize(&walls, 15, 3, 0.05);
        assert!(o.converged);
    }

    #[test]
    fn continues_when_drifting() {
        let mut walls = vec![100.0; 6];
        walls.extend([200.0, 200.0, 200.0]);
        assert!(needs_more(&walls, 3, 15, 3, 0.05));
    }
}
