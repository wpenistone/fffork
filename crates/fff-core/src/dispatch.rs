//! Picks serial or parallel per call from measured cost.
//! Estimates total work, divides by what one fork/join must amortize,
//! and caps by pool width. Same rule at every call site.

use std::sync::LazyLock;

// Thread count and chunk size for one parallel loop.
pub(crate) struct Plan {
    pub threads: usize,
    pub chunk: usize,
}

fn fork_join_measured() -> u64 {
    static NS: LazyLock<u64> = LazyLock::new(|| {
        if let Some(v) = std::env::var("FFF_FORK_JOIN_NS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &u64| *v > 0)
        {
            return v;
        }
        // Time an empty fan-out on the bg pool: exactly the cost a
        // parallel loop must amortize. Median of 5 rejects a bad run.
        let workers = crate::parallelism::effective_core_count().max(1);
        let mut reps = [0u64; 5];
        for r in reps.iter_mut() {
            let t0 = std::time::Instant::now();
            crate::parallelism::run_on_bg_pool(|| {
                use rayon::prelude::*;
                (0..workers).into_par_iter().for_each(|i| {
                    std::hint::black_box(i);
                });
            });
            *r = t0.elapsed().as_nanos() as u64;
        }
        reps.sort_unstable();
        let m = reps[2];
        // Scheduler-physical guard: outside this the probe was
        // descheduled; fall back to the historical measured value.
        if (1_000..10_000_000).contains(&m) {
            m
        } else {
            25_000
        }
    });
    *NS
}

pub(crate) fn fork_join_ns() -> u64 {
    fork_join_measured()
}

// Bytes of path work that amortize one fork/join on this box.
pub(crate) fn bytes_to_amortize() -> usize {
    static B: LazyLock<usize> = LazyLock::new(|| {
        if let Some(v) = std::env::var("FFF_MATCH_WORK_PER_THREAD")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &usize| *v > 0)
        {
            return v;
        }
        let m = crate::machine::machine();
        let b = (m.read_bytes_per_ns() * fork_join_ns() as f64) as usize;
        let lo = m.cache_line.max(1);
        let hi = if m.llc_bytes > 0 { m.llc_bytes } else { 8 << 20 };
        b.clamp(lo, hi)
    });
    *B
}

// Byte-equivalents of one gate file: DP cost over SIMD lanes scaled
// by strictness, plus the memory walk itself. Shared by the thread
// estimator and the group sizer so they cannot disagree.
pub(crate) fn match_bytes_per_file(
    needle_len: usize,
    max_haystack_len: usize,
    max_typos: Option<u16>,
) -> usize {
    let nl = needle_len.max(1) as f64;
    let strict = match max_typos {
        None => 1.0,
        Some(t) => (nl + 1.0 - (t as f64).min(nl)) / (nl + 1.0),
    };
    // 16 is the SIMD lane width the DP batches over; 512 caps the
    // estimate so a few outlier paths cannot demand the whole pool.
    let chunks = max_haystack_len.min(512).div_ceil(16).max(1) as f64;
    (nl * chunks * strict).max(1.0) as usize
}

// Plan a file-count loop. `bytes_per_file` is a conservative upper
// bound (max path len scaled by query cost); over-estimate saturates
// the cap, under-estimate serializes. Both directions cost at most
// one fork/join of waste, never correctness.
pub(crate) fn plan_files(n: usize, bytes_per_file: usize, cap: usize) -> Plan {
    let cap = cap.max(1);
    if n == 0 {
        return Plan { threads: 1, chunk: 1 };
    }
    let work = (n as u128) * (bytes_per_file.max(1) as u128);
    let amort = bytes_to_amortize().max(1) as u128;
    let threads = work.div_ceil(amort).min(cap as u128).max(1) as usize;
    Plan {
        threads,
        chunk: n.div_ceil(threads).max(1),
    }
}

// Plan a match-count loop (top-k, scoring). `bytes_per_match` covers
// the score struct plus the path bytes each item touches.
pub(crate) fn plan_matches(n: usize, bytes_per_match: usize, cap: usize) -> Plan {
    plan_files(n, bytes_per_match, cap)
}

// Match count whose metadata working set fills one private L2.
// Keeps TLS scratch resident; falls back when topology is unknown.
pub(crate) fn sw_metadata_chunk() -> usize {
    static C: LazyLock<usize> = LazyLock::new(|| {
        let l2 = crate::machine::machine().l2_bytes;
        // Trim (8 B) + scores/ecols/order/perm outputs per match.
        let per_match = std::mem::size_of::<crate::matcher::Trim>()
            + std::mem::size_of::<u16>() * 4
            + std::mem::size_of::<u32>();
        if l2 == 0 {
            return 32 * 1024;
        }
        (l2 / per_match.max(1)).max(1024)
    });
    *C
}

// Probe length whose cost is ~one fork/join (negligible). Floor is
// sampling statistics (SE <= 9% on a proportion), not corpus tuning.
pub(crate) fn probe_len(n: usize, avg_bytes: usize) -> usize {
    let p = bytes_to_amortize().div_ceil(avg_bytes.max(1)).max(32);
    p.min(n)
}

// Mean path bytes over a strided sample (same statistics floor as
// the probe). Scales per-match work estimates without walking 227k.
pub(crate) fn sampled_path_avg(files: &crate::matcher::FileItems<'_>) -> usize {
    let n = files.len();
    if n == 0 {
        return 1;
    }
    let m = n.min(32);
    let stride = (n / m).max(1);
    let (mut sum, mut count) = (0usize, 0usize);
    for i in (0..n).step_by(stride) {
        sum += files.index(i).path.byte_len as usize;
        count += 1;
        if count >= m {
            break;
        }
    }
    sum.div_ceil(count.max(1)).max(1)
}

// Inner fan-out decision for fixed-size SIMD batches. Amdahl
// break-even: parallel wins iff saved time exceeds dispatch.
pub(crate) fn parallel_batches(n_batches: usize, ns_per_batch: f64, threads: usize) -> bool {
    if n_batches < 2 || ns_per_batch <= 0.0 || threads < 2 {
        return false;
    }
    let t = threads as f64;
    (n_batches as f64) * ns_per_batch * (1.0 - 1.0 / t) > fork_join_ns() as f64
}

// Measured cost of one SW batch per lane class; seeds refine per slice,
// so seed error only moves the serial/parallel boundary slightly.
static BATCH_NS_U8: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(3_000);
static BATCH_NS_U16: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(6_000);

pub(crate) fn batch_ns_estimate(u8_class: bool) -> f64 {
    let cell = if u8_class {
        &BATCH_NS_U8
    } else {
        &BATCH_NS_U16
    };
    cell.load(std::sync::atomic::Ordering::Relaxed) as f64
}

// Fold one slice's mean batch cost into the estimate (EWMA, /8).
// Benign races: concurrent slices may drop samples, never corrupt.
pub(crate) fn observe_batch_ns(u8_class: bool, slice_ns: f64, n_batches: usize) {
    if n_batches == 0 || slice_ns <= 0.0 {
        return;
    }
    let sample = (slice_ns / n_batches as f64) as u64;
    if sample == 0 {
        return;
    }
    let cell = if u8_class {
        &BATCH_NS_U8
    } else {
        &BATCH_NS_U16
    };
    let _ = cell.fetch_update(
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
        |ema| {
            let ema = ema.max(1) as i64;
            Some((ema + (sample as i64 - ema) / 8).max(1) as u64)
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_work_stays_serial() {
        let p = plan_files(10, 64, 16);
        assert_eq!(p.threads, 1);
        assert_eq!(p.chunk, 10);
    }

    #[test]
    fn huge_work_saturates_cap() {
        let p = plan_files(1_000_000, 1024, 8);
        assert_eq!(p.threads, 8);
        assert_eq!(p.chunk, 125_000);
    }

    #[test]
    fn probe_bounded_both_sides() {
        assert_eq!(probe_len(10, 64), 10);
        let p = probe_len(1_000_000, 64);
        assert!(p >= 32 && p < 1_000_000);
    }

    #[test]
    fn batches_need_break_even() {
        assert!(!parallel_batches(1, 1e6, 8));
        assert!(!parallel_batches(4, 100.0, 8));
        assert!(parallel_batches(10_000, 10_000.0, 8));
    }

    #[test]
    fn batch_estimate_tracks_observations() {
        use std::sync::atomic::Ordering;
        let save = BATCH_NS_U8.load(Ordering::Relaxed);
        observe_batch_ns(true, 80_000.0, 10);
        let mid = batch_ns_estimate(true);
        assert!(mid > 3_000.0 && mid < 8_000.0, "ema {mid}");
        for _ in 0..200 {
            observe_batch_ns(true, 80_000.0, 10);
        }
        let near = batch_ns_estimate(true);
        assert!((near - 8_000.0).abs() < 200.0, "ema {near}");
        BATCH_NS_U8.store(save, Ordering::Relaxed);
    }
}
