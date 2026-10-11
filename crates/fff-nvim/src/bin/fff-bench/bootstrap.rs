//! Deterministic bootstrap CIs for paired cost ratios (METHODOLOGY).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatioCi {
    pub low: f64,
    pub high: f64,
    pub point: f64,
    pub n_pairs: usize,
    pub n_resamples: usize,
    pub seed: u64,
}

/// SplitMix64 generator for deterministic bootstrap (seed recorded in plan).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((self.next() as u128 * n as u128) >> 64) as usize
    }
}

/// Bootstrap 95% CI for geometric-mean candidate/baseline ratio of paired positive samples.
/// Each pair is (baseline_value, candidate_value). Ratio < 1 means candidate faster (lower cost).
pub fn ratio_ci95(pairs: &[(f64, f64)], seed: u64, n_resamples: usize) -> Option<RatioCi> {
    let pairs: Vec<(f64, f64)> = pairs
        .iter()
        .copied()
        .filter(|(b, c)| *b > 0.0 && *c > 0.0)
        .collect();
    if pairs.is_empty() {
        return None;
    }
    let n = pairs.len();
    let point = geo_mean_ratio(&pairs);
    let mut rng = Lcg(seed | 1);
    let mut ratios = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut sample = Vec::with_capacity(n);
        for _ in 0..n {
            sample.push(pairs[rng.index(n)]);
        }
        ratios.push(geo_mean_ratio(&sample));
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_i = ((n_resamples as f64) * 0.025).floor() as usize;
    let hi_i = ((n_resamples as f64) * 0.975).floor() as usize;
    let hi_i = hi_i.min(n_resamples.saturating_sub(1));
    Some(RatioCi {
        low: ratios[lo_i],
        high: ratios[hi_i],
        point,
        n_pairs: n,
        n_resamples,
        seed,
    })
}

fn geo_mean_ratio(pairs: &[(f64, f64)]) -> f64 {
    let mut sum_log = 0.0;
    let mut n = 0usize;
    for (b, c) in pairs {
        if *b > 0.0 && *c > 0.0 {
            sum_log += (c / b).ln();
            n += 1;
        }
    }
    if n == 0 {
        return 1.0;
    }
    (sum_log / n as f64).exp()
}

/// Classify using CI and materiality δ (lower-is-better).
pub fn classify_ratio(ci: &RatioCi, delta: f64) -> &'static str {
    // improve: entire CI below 1-δ
    if ci.high < 1.0 - delta {
        return "improve";
    }
    // regress: entire CI above 1+δ
    if ci.low > 1.0 + delta {
        return "regress";
    }
    // equivalent: interval within [1-δ, 1+δ]
    if ci.low >= 1.0 - delta && ci.high <= 1.0 + delta {
        return "equivalent";
    }
    "mixed"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faster_candidate_ci_below_one() {
        let pairs: Vec<_> = (0..20).map(|_| (200.0, 100.0)).collect();
        let ci = ratio_ci95(&pairs, 42, 2000).unwrap();
        assert!(ci.high < 0.6, "ci={:?}", ci);
        assert_eq!(classify_ratio(&ci, 0.05), "improve");
    }

    #[test]
    fn parity_is_equivalent() {
        let pairs: Vec<_> = (0..30)
            .map(|i| (100.0, 100.0 + (i % 3) as f64 - 1.0))
            .collect();
        let ci = ratio_ci95(&pairs, 7, 2000).unwrap();
        let c = classify_ratio(&ci, 0.10);
        assert!(c == "equivalent" || c == "mixed", "got {c} ci={:?}", ci);
    }

    #[test]
    fn heterogeneous_pairs_produce_a_nonzero_interval() {
        let pairs = vec![
            (100.0, 110.0),
            (100.0, 120.0),
            (100.0, 130.0),
            (100.0, 140.0),
        ];
        let ci = ratio_ci95(&pairs, 4195872001, 10_000).unwrap();
        assert!(ci.high > ci.low, "ci={ci:?}");
    }
}
