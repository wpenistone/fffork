//! Machine calibration.
//!
//! The tuning constants this crate used to hardcode (`MATCH_WORK_PER_THREAD`,
//! the `.min(8)` scan cap) were measured on one 20-core desktop and shipped as
//! literals, which is wrong everywhere else: a 4-core laptop, a 32-core server
//! and an M-series Mac all want different numbers, and the *shape* of the
//! curve differs too.
//!
//! Derive them here, once per process, from CPUID (cores, cache line, LLC) plus
//! a short memory-throughput micro-benchmark - how much work one thread should
//! own depends on how fast this machine does the work, not on what a desktop
//! did last year.
//!
//! Overrides exist for benchmarking and tests: `FFF_MATCH_WORK_PER_THREAD`,
//! `FFF_SCAN_THREAD_CAP`, `FFF_FORK_JOIN_NS`.

use std::sync::LazyLock;

pub(crate) struct Machine {
    pub cores: usize,
    /// Data cache line, bytes (64 on essentially every current CPU).
    pub cache_line: usize,
    /// Last-level data cache, bytes. 0 when unknown.
    pub llc_bytes: usize,
    /// Measured single-thread sequential read throughput, bytes/ns.
    pub read_bytes_per_ns: f64,
}

impl Machine {
    /// Bytes of path data one match thread should own.
    ///
    /// A thread only pays for itself once it has enough work to amortize the
    /// fork/join. `FORK_JOIN_NS` is rayon's real dispatch cost, not a tuned
    /// corpus constant, and the throughput term scales with the machine: a box
    /// that chews path bytes twice as fast needs twice the bytes per thread.
    pub fn match_work_per_thread(&self) -> usize {
        let bytes = (self.read_bytes_per_ns * fork_join_ns() as f64) as usize;
        // Clamped so a paused VM or a bad probe cannot pick 0 or saturate.
        bytes.clamp(8 * 1024, 8 * 1024 * 1024)
    }

    /// Upper bound on threads for an internal scan.
    ///
    /// Was a literal `.min(8)`, which under-uses a 32-core server and
    /// over-uses a phone. Cores alone are the wrong bound: the gate is memory
    /// bound, so the bound is the point where extra threads stop adding
    /// bandwidth. Measured on a 20-core desktop, p50 for a 226k-file query
    /// is 32700us at 1 thread, 10214us at 2, then flat-to-worse (11457us at
    /// 6, 10691us at 11) - so the knee, not the core count, is the limit.
    pub fn scan_thread_cap(&self) -> usize {
        let by_cores = self.cores.saturating_sub(1).max(1);
        let by_bandwidth = self.saturation_threads();
        let by_cache = if self.llc_bytes == 0 {
            by_cores
        } else {
            // ~2 MB working set per active thread; past that they evict each other.
            (self.llc_bytes / (2 * 1024 * 1024)).max(1)
        };
        by_bandwidth.min(by_cache).min(by_cores).clamp(1, 64)
    }

    /// Smallest thread count reaching ~90% of the best observed read
    /// bandwidth: the knee of the memory-scaling curve. Threads past it
    /// contend for the same DRAM and only add fork/join and cache pressure.
    /// Measured once, so it adapts to the box instead of assuming the core
    /// count is the limit.
    pub fn saturation_threads(&self) -> usize {
        static KNEE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *KNEE.get_or_init(|| {
            let probe = self.llc_bytes.saturating_mul(2).clamp(32 << 20, 256 << 20);
            let mut peak = 0.0f64;
            for t in [2usize, 4, 8, 16] {
                if t > self.cores {
                    break;
                }
                let b = parallel_read_bw(probe, t);
                if b > peak {
                    peak = b;
                }
                if b >= peak * 0.9 {
                    return t;
                }
            }
            self.cores.clamp(1, 16)
        })
    }
}

/// rayon's fork/join dispatch cost. Universal, unlike a tuned corpus size.
const FORK_JOIN_NS: usize = 25_000;

fn fork_join_ns() -> usize {
    static NS: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("FFF_FORK_JOIN_NS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(FORK_JOIN_NS)
    });
    *NS
}

static MACHINE: LazyLock<Machine> = LazyLock::new(Machine::probe);

pub(crate) fn machine() -> &'static Machine {
    &MACHINE
}

pub(crate) fn match_work_per_thread() -> usize {
    static W: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("FFF_MATCH_WORK_PER_THREAD")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or_else(|| machine().match_work_per_thread())
    });
    *W
}

/// Aggregate read bandwidth in bytes/ns with `threads` threads over a shared
/// buffer, used only to locate the memory-scaling knee.
fn parallel_read_bw(n_bytes: usize, threads: usize) -> f64 {
    let mut buf = vec![0u8; n_bytes];
    let mut x: u64 = 0x243f_6a88_85a3_08d3;
    for (i, b) in buf.iter_mut().enumerate() {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (x >> 33) as u8 ^ (i as u8);
    }
    let t0 = std::time::Instant::now();
    let mut total = 0u64;
    std::thread::scope(|s| {
        let per = n_bytes / threads;
        for t in 0..threads {
            let slice = &buf[t * per..(t + 1) * per];
            let h = s.spawn(move || {
                let mut acc: u64 = 0;
                for i in (0..slice.len()).step_by(64) {
                    acc = acc.wrapping_add(slice[i] as u64);
                }
                acc
            });
            total += h.join().unwrap_or(0);
        }
    });
    std::hint::black_box(total);
    n_bytes as f64 / t0.elapsed().as_nanos().max(1) as f64
}

pub(crate) fn scan_thread_cap() -> usize {
    static C: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("FFF_SCAN_THREAD_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v >= 1)
            .unwrap_or_else(|| machine().scan_thread_cap())
    });
    *C
}

impl Machine {
    fn probe() -> Self {
        // Core count comes from `parallelism`, which already prefers Apple
        // performance cores; re-deriving it here with `available_parallelism`
        // would size every cap in this module off E-cores on asymmetric chips.
        let cores = crate::parallelism::effective_core_count();
        let (cache_line, llc_bytes) = cache_geometry();
        // The probe must not fit in LLC, or it measures cache bandwidth and
        // over-estimates how fast this box walks a real (multi-MB) index.
        let probe_bytes = llc_bytes.saturating_mul(2).clamp(32 << 20, 512 << 20);
        let read_bytes_per_ns = measure_read_throughput(probe_bytes);
        Machine {
            cores,
            cache_line,
            llc_bytes,
            read_bytes_per_ns,
        }
    }
}

/// Sequential read throughput in bytes/ns over a buffer sized to sit outside
/// LLC. Takes the best of three to skip scheduler noise.
fn measure_read_throughput(n_bytes: usize) -> f64 {
    const N: usize = 32 * 1024 * 1024;
    let n = n_bytes.clamp(N, 512 * 1024 * 1024);
    let mut buf = vec![0u8; n];
    let mut x: u64 = 0x243f_6a88_85a3_08d3;
    for (i, b) in buf.iter_mut().enumerate() {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (x >> 33) as u8 ^ (i as u8);
    }
    let mut best = 0.0f64;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let mut acc: u64 = 0;
        // Stride by 64 so every access is a distinct cache line.
        for i in (0..n).step_by(64) {
            acc = acc.wrapping_add(buf[i] as u64);
        }
        let ns = t0.elapsed().as_nanos() as f64;
        std::hint::black_box(acc);
        let bpn = n as f64 / ns.max(1.0);
        if bpn > best {
            best = bpn;
        }
    }
    best.clamp(0.05, 64.0)
}

#[cfg(target_arch = "x86_64")]
fn cache_geometry() -> (usize, usize) {
    use core::arch::x86_64::__cpuid_count;
    let max_leaf = unsafe { __cpuid_count(0x8000_0000, 0) }.eax;
    if max_leaf < 0x8000_0004 {
        return (64, 0);
    }
    // CPUID leaf 4: deterministic cache parameters.
    let mut line = 64usize;
    let mut llc = 0usize;
    let mut sub = 0u32;
    loop {
        let r = unsafe { __cpuid_count(4, sub) };
        let ty = r.eax & 0x1f;
        if ty == 0 {
            break;
        }
        let level = (r.eax >> 5) & 0x7;
        let line_size = ((r.ebx & 0xfff) + 1) as usize;
        let partitions = (((r.ebx >> 12) & 0x3ff) + 1) as usize;
        let ways = (((r.ebx >> 22) & 0x3ff) + 1) as usize;
        let sets = (r.ecx + 1) as usize;
        // bit1 set means data cache.
        if ty & 0b10 != 0 {
            line = line_size;
            if level == 3 {
                llc = ways * partitions * line_size * sets;
            }
        }
        sub += 1;
        if sub > 16 {
            break;
        }
    }
    (line, llc)
}

#[cfg(not(target_arch = "x86_64"))]
fn cache_geometry() -> (usize, usize) {
    #[cfg(target_arch = "aarch64")]
    let line = {
        let ctr: usize;
        unsafe {
            std::arch::asm!("mrs {}, CTR_EL0", out(reg) ctr, options(nostack));
        }
        4usize << (ctr & 0xf)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let line = 64usize;
    (line, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_produces_usable_values() {
        let m = machine();
        assert!(m.cores >= 1);
        assert!(
            (16..=256).contains(&m.cache_line),
            "cache_line={}",
            m.cache_line
        );
        assert!(m.read_bytes_per_ns > 0.0);
        let w = m.match_work_per_thread();
        assert!((8 * 1024..=8 * 1024 * 1024).contains(&w), "work={w}");
        assert!((1..=64).contains(&m.scan_thread_cap()));
    }

    #[test]
    fn cap_never_exceeds_cores() {
        let m = Machine {
            cores: 4,
            cache_line: 64,
            llc_bytes: 64 * 1024 * 1024,
            read_bytes_per_ns: 10.0,
        };
        assert!(m.scan_thread_cap() <= 4);
    }
}
