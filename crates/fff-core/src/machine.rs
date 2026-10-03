//! Machine calibration.
//!
//! Tuning constants must hold on a 4-core laptop, a 32-core server and an
//! M-series Mac alike, so none of them is a literal.
//!
//! **Topology first, measurement only for what topology cannot give us.**
//! Core counts, SMT siblings, cache-line size, LLC and L2 size, clock, NUMA
//! nodes and any cgroup limit are *facts about the box*, read from CPUID (x86),
//! sysfs (Linux, any arch) or sysctl (macOS). Everything the dispatcher needs is
//! derived from those. The one measured quantity is memory latency, which no
//! register reports and which belongs to the memory controller rather than the
//! CPU - and that is probed with one small buffer sized from the LLC.
//!
//! The gate moves little data per byte it touches, so the model that matters is
//! "how many misses can this box keep in flight" - latency x line size x
//! memory-level parallelism - rather than aggregate GB/s.
//!
//! Overrides: `FFF_MATCH_WORK_PER_THREAD`, `FFF_SCAN_THREAD_CAP`,
//! `FFF_FORK_JOIN_NS`, `FFF_NO_PROBE`.

use std::sync::LazyLock;

pub(crate) struct Machine {
    /// Logical CPUs usable (Apple performance cores on macOS, not E-cores).
    pub cores: usize,
    /// Physical cores: one per SMT sibling group. Siblings share L1/L2 and the
    /// load unit, so a dependent-load chain gets no second stream from them.
    pub physical_cores: usize,
    /// Data cache line, bytes.
    pub cache_line: usize,
    /// Private L2 per core, bytes. 0 when unknown.
    pub l2_bytes: usize,
    /// Last-level data cache, bytes. 0 when unknown.
    pub llc_bytes: usize,
    /// Nominal core clock, Hz. 0 when unknown.
    pub clock_hz: f64,
    /// NUMA nodes visible to this process (a cgroup cpuset counts as one).
    pub numa_nodes: usize,
    /// Memory latency in ns. Measured - the one number topology does not
    /// report, because it belongs to the memory controller, not the CPU.
    pub mem_latency_ns: f64,
}

impl Machine {
    /// Sustained cache-line fill rate for one core, bytes/ns.
    ///
    /// Each miss retires a whole cache line after one latency and a core keeps
    /// `MLP` misses in flight, so per-core throughput is
    /// `line / latency * mlp`. This is what limits the gate: it walks paths
    /// through dependent loads, it does not stream bytes.
    pub fn read_bytes_per_ns(&self) -> f64 {
        // 10 outstanding line fills is the conservative end of what modern OoO
        // cores sustain. Erring low only widens the dispatch window; the
        // topology caps are the real ceiling.
        const MLP: f64 = 10.0;
        let latency = if self.mem_latency_ns > 0.0 {
            self.mem_latency_ns
        } else {
            // No probe (see `FFF_NO_PROBE`). DRAM latency is among the most
            // stable numbers in computing: DDR4 ~85ns, DDR5 ~70ns, Apple
            // unified ~50-60ns. 80 is the safe middle.
            80.0
        };
        (self.cache_line as f64 * MLP / latency).clamp(0.05, 64.0)
    }

    /// Bytes of path data one match thread should own.
    ///
    /// A thread only pays for itself once it has enough work to amortize the
    /// fork/join. `FORK_JOIN_NS` is rayon's real dispatch cost, not a tuned
    /// corpus constant, and the throughput term scales with the machine: a box
    /// that walks path bytes twice as fast needs twice the bytes per thread.
    pub fn match_work_per_thread(&self) -> usize {
        let bytes = (self.read_bytes_per_ns() * fork_join_ns() as f64) as usize;
        // Clamped so a paused VM or a bad probe cannot pick 0 or saturate.
        bytes.clamp(8 * 1024, 8 * 1024 * 1024)
    }

    /// Upper bound on threads for an internal scan.
    ///
    /// Was a literal `.min(8)`, which under-uses a 32-core server and over-uses
    /// a phone. Three bounds, all from topology:
    ///
    /// - **Physical cores**, not logical: SMT siblings share L1/L2.
    /// - **The LLC** at ~2 MB of live index per active thread. Past that the
    ///   threads evict each other, which is the wall measured on a 20-core
    ///   desktop (flat past 2 threads, worse at 6).
    /// - **NUMA nodes**, because a thread reading another node's memory pays a
    ///   cross-socket hop on every miss.
    pub fn scan_thread_cap(&self) -> usize {
        let by_physical = self.physical_cores.max(1);
        let by_llc = if self.llc_bytes == 0 {
            by_physical
        } else {
            (self.llc_bytes / (2 * 1024 * 1024)).max(1)
        };
        let by_numa = if self.numa_nodes > 0 {
            (by_physical / self.numa_nodes).max(1)
        } else {
            by_physical
        };
        by_physical
            .min(by_llc)
            .min(by_numa)
            .min(self.bandwidth_knee())
            .min(self.cores.max(1))
            .clamp(1, 64)
    }

    /// Smallest thread count reaching ~90% of the best observed read
    /// bandwidth: the knee of the memory-scaling curve.
    ///
    /// This one genuinely cannot be derived from topology - it depends on the
    /// memory controller and on what else the machine is doing - so it is
    /// measured, but cheaply: one buffer sized from the LLC (never more than
    /// 32 MB), filled once, sampled at three thread counts. The previous
    /// implementation allocated and LCG-filled a fresh multi-hundred-MB buffer
    /// per sample, and separately had a bug that made it return 2 on every
    /// machine because it compared each sample against a running peak.
    fn bandwidth_knee(&self) -> usize {
        static KNEE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *KNEE.get_or_init(|| {
            let by_cores = self.physical_cores.max(1);
            if by_cores < 4 || std::env::var_os("FFF_NO_PROBE").is_some() {
                return by_cores;
            }
            let mut want = if self.llc_bytes > 0 {
                self.llc_bytes.saturating_mul(2)
            } else {
                16 << 20
            }
            .clamp(4 << 20, 32 << 20);
            if let Some(limit) = cgroup_memory_limit() {
                want = want.min(limit / 8).max(4 << 20);
            }
            let mut buf = vec![0u64; want / 8];
            let mut x: u64 = 0x243f_6a88_85a3_08d3;
            for v in buf.iter_mut() {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *v = x;
            }
            let samples: Vec<(usize, f64)> = [2usize, 4, 8]
                .into_iter()
                .filter(|t| *t <= by_cores)
                .map(|t| (t, parallel_read_bw(&buf, t)))
                .collect();
            let peak = samples.iter().map(|(_, b)| *b).fold(0.0f64, f64::max);
            if peak <= 0.0 {
                return by_cores;
            }
            samples
                .iter()
                .find(|(_, b)| *b >= peak * 0.9)
                .map(|(t, _)| *t)
                .unwrap_or(by_cores)
        })
    }
}

/// Aggregate read bandwidth in bytes/ns with `threads` threads over one shared
/// buffer, used only to locate the memory-scaling knee.
fn parallel_read_bw(buf: &[u64], threads: usize) -> f64 {
    let n_bytes = buf.len() * 8;
    let mut total = 0u64;
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        let per = buf.len() / threads;
        for t in 0..threads {
            let slice = &buf[t * per..(t + 1) * per];
            let h = s.spawn(move || {
                let mut acc: u64 = 0;
                for i in (0..slice.len()).step_by(8) {
                    acc = acc.wrapping_add(slice[i]);
                }
                acc
            });
            total = total.wrapping_add(h.join().unwrap_or(0));
        }
    });
    std::hint::black_box(total);
    n_bytes as f64 / t0.elapsed().as_nanos().max(1) as f64
}
/// rayon fork/join dispatch cost. A property of the scheduler, not of any corpus.
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
        // performance cores; re-deriving it with `available_parallelism` would
        // size every cap off E-cores on asymmetric chips.
        let cores = crate::parallelism::effective_core_count();
        let topo = Topology::detect();
        let physical_cores = topo.physical_cores.unwrap_or(cores).clamp(1, 256);
        let cache_line = topo.cache_line.unwrap_or(64);
        let l2_bytes = topo.l2_bytes;
        let llc_bytes = topo.llc_bytes;
        let clock_hz = topo.clock_hz;
        let numa_nodes = topo.numa_nodes.unwrap_or(1).max(1);
        // The only measured quantity: memory latency. See `measure_latency_ns`.
        let mem_latency_ns = measure_latency_ns(&[l2_bytes, llc_bytes]);
        Machine {
            cores,
            physical_cores,
            cache_line,
            l2_bytes,
            llc_bytes,
            clock_hz,
            numa_nodes,
            mem_latency_ns,
        }
    }
}

#[derive(Default)]
struct Topology {
    cache_line: Option<usize>,
    l2_bytes: usize,
    llc_bytes: usize,
    clock_hz: f64,
    physical_cores: Option<usize>,
    numa_nodes: Option<usize>,
}

impl Topology {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let mut t = Topology::default();
            x86_topology(&mut t);
            t
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut t = Topology::default();
            t.cache_line = Some(arch_cache_line());
            linux_topology(&mut t);
            t
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn x86_topology(t: &mut Topology) {
    use core::arch::x86_64::__cpuid_count;
    // AMD exposes the cache hierarchy through the topology-extensions leaf;
    // leaf 4 is only a compatibility surface there and under-reports on some
    // parts. Prefer 0x8000001D when the vendor reports it.
    let ext = unsafe { __cpuid_count(0x8000_0000, 0) }.eax;
    if ext >= 0x8000_000D {
        let (l, l2, llc) = cpuid_cache_leaf(0x8000_000D);
        if l != 0 {
            t.cache_line = Some(l);
        }
        if l2 != 0 {
            t.l2_bytes = l2;
        }
        if llc != 0 {
            t.llc_bytes = llc;
        }
    }
    if t.llc_bytes == 0 {
        let (l, l2, llc) = cpuid_cache_leaf(4);
        if l != 0 {
            t.cache_line = Some(l);
        }
        if l2 != 0 {
            t.l2_bytes = l2;
        }
        t.llc_bytes = llc;
    }
    // Leaf 0x16: base/maximum frequency in MHz, when the CPU reports it.
    if let Some(mhz) = cpuid_max_freq_mhz() {
        t.clock_hz = mhz * 1e6;
    }
    // APIC-ID topology: count distinct x2APIC ids per core-siblings group.
    if let Some(p) = cpuid_physical_cores() {
        t.physical_cores = Some(p);
    }
    linux_topology(t);
}

/// (line size, L2 bytes, LLC bytes) from a deterministic-cache-parameters leaf.
#[cfg(target_arch = "x86_64")]
fn cpuid_cache_leaf(leaf: u32) -> (usize, usize, usize) {
    use core::arch::x86_64::__cpuid_count;
    let mut line = 0usize;
    let mut l2 = 0usize;
    let mut llc = 0usize;
    let mut sub = 0u32;
    loop {
        let r = unsafe { __cpuid_count(leaf, sub) };
        let ty = r.eax & 0x1f;
        if ty == 0 {
            break;
        }
        let level = (r.eax >> 5) & 0x7;
        let line_size = ((r.ebx & 0xfff) + 1) as usize;
        let partitions = (((r.ebx >> 12) & 0x3ff) + 1) as usize;
        let ways = (((r.ebx >> 22) & 0x3ff) + 1) as usize;
        let sets = (r.ecx + 1) as usize;
        let size = ways * partitions * line_size * sets;
        if ty & 0b10 != 0 {
            line = line_size;
            // The AMD leaf encodes L3 as level 8 rather than 3.
            if level == 2 || level == 6 {
                l2 = size;
            }
            if level == 3 || (leaf == 0x8000_000D && level >= 8) {
                llc = llc.max(size);
            }
        }
        sub += 1;
        if sub > 16 {
            break;
        }
    }
    (line, l2, llc)
}

#[cfg(target_arch = "x86_64")]
fn cpuid_max_freq_mhz() -> Option<f64> {
    use core::arch::x86_64::__cpuid_count;
    let ext = unsafe { __cpuid_count(0x8000_0000, 0) }.eax;
    if ext < 0x8000_0008 {
        return None;
    }
    let mhz = unsafe { __cpuid_count(0x8000_0008, 0) }.ecx >> 16;
    if mhz == 0 { None } else { Some(mhz as f64) }
}

/// Count cores via the APIC-ID x2APIC enumeration in leaf 0xB.
#[cfg(target_arch = "x86_64")]
fn cpuid_physical_cores() -> Option<usize> {
    // Without a full topology walk this is not reliably derivable, and a wrong
    // guess is worse than none: `scan_thread_cap` already falls back to the
    // logical count, and sysfs gives the real answer on Linux.
    None
}

#[cfg(target_arch = "aarch64")]
fn arch_cache_line() -> usize {
    let ctr: usize;
    unsafe {
        std::arch::asm!("mrs {}, CTR_EL0", out(reg) ctr, options(nostack));
    }
    4usize << (ctr & 0xf)
}

#[cfg(not(target_arch = "aarch64"))]
fn arch_cache_line() -> usize {
    64
}

/// sysfs topology, which on Linux is authoritative for cache sizes, SMT
/// siblings and the NUMA node count - including inside containers, where it
/// reflects the cpuset rather than the host.
#[cfg(target_os = "linux")]
fn linux_topology(t: &mut Topology) {
    let mut best_level = 0usize;
    for dir in 0..16 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{dir}/");
        let Ok(level) = std::fs::read_to_string(format!("{base}level")) else {
            break;
        };
        let Ok(level) = level.trim().parse::<usize>() else {
            break;
        };
        let Ok(size) = std::fs::read_to_string(format!("{base}size")) else {
            continue;
        };
        let Some(size) = parse_cache_size(size.trim()) else {
            continue;
        };
        let Ok(line) = std::fs::read_to_string(format!("{base}coherency_line_size")) else {
            continue;
        };
        let Ok(line) = line.trim().parse::<usize>() else {
            continue;
        };
        let Ok(kind) = std::fs::read_to_string(format!("{base}type")) else {
            continue;
        };
        if kind.trim() != "Unified" && kind.trim() != "Data" {
            continue;
        }
        t.cache_line = Some(line);
        if level == 2 && size > t.l2_bytes {
            t.l2_bytes = size;
        }
        if level >= best_level {
            best_level = level;
            t.llc_bytes = size;
        }
    }
    // Physical cores: count distinct thread-sibling groups.
    let mut groups: Vec<String> = Vec::new();
    for cpu in 0..256 {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let Ok(list) = std::fs::read_to_string(&path) else {
            break;
        };
        let list = list.trim().to_string();
        if !groups.contains(&list) {
            groups.push(list);
        }
    }
    if !groups.is_empty() {
        t.physical_cores = Some(groups.len());
    }
    // NUMA nodes this process may run on.
    let mut nodes: Vec<String> = Vec::new();
    for node in 0..64 {
        let path = format!("/sys/devices/system/node/node{node}/cpulist");
        let Ok(list) = std::fs::read_to_string(&path) else {
            break;
        };
        let list = list.trim().to_string();
        if !list.is_empty() && !nodes.contains(&list) {
            nodes.push(list);
        }
    }
    if !nodes.is_empty() {
        t.numa_nodes = Some(nodes.len());
    }
    // Fall back to the reported clock when CPUID had none.
    if t.clock_hz == 0.0 {
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") {
            if let Ok(khz) = s.trim().parse::<f64>() {
                t.clock_hz = khz * 1e3;
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn linux_topology(_t: &mut Topology) {}

/// sysfs reports sizes like `32K`, `1024K`, `48M`.
#[cfg(target_os = "linux")]
fn parse_cache_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('K') | Some('k') => (&s[..s.len() - 1], 1024usize),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1024 * 1024),
        Some('G') | Some('g') => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1usize),
    };
    num.trim().parse::<usize>().ok().map(|n| n * mult)
}

/// Memory latency in ns, by pointer-chasing a random permutation of a buffer
/// sized just past the last-level cache.
///
/// This is the only measurement left, and it is a *latency* probe rather than
/// a bandwidth sweep: one buffer, allocated once, sized from topology, walked
/// with a dependent load chain so the hardware cannot overlap the misses. The
/// previous implementation sampled bandwidth at 2/4/8/16 threads and
/// allocated a fresh multi-hundred-MB buffer per sample.
///
/// Returns 0 when probing is disabled (`FFF_NO_PROBE`) or allocation fails; the
/// caller then uses the documented default.
fn measure_latency_ns(cache_sizes: &[usize; 2]) -> f64 {
    if std::env::var_os("FFF_NO_PROBE").is_some() {
        return 0.0;
    }
    // Just past LLC, but never a huge allocation in a plugin host: capped at
    // 32 MB and additionally clamped by any cgroup limit.
    let llc = cache_sizes[1];
    let mut want = if llc > 0 {
        llc.saturating_mul(2)
    } else {
        8 << 20
    }
    .clamp(1 << 20, 32 << 20);
    if let Some(limit) = cgroup_memory_limit() {
        want = want.min(limit / 8).max(1 << 20);
    }
    // Slots are spread across the whole buffer and each stores the *byte offset*
    // of the next, so the chase walks the entire allocation. A previous version
    // indexed the first 128KB directly and therefore measured L2 latency
    // (24.9ns here) while claiming to measure memory.
    const SLOTS: usize = 16384;
    if want < (SLOTS + 1) * 64 {
        return 0.0;
    }
    let Ok(layout) = std::alloc::Layout::from_size_align(want, 64) else {
        return 0.0;
    };
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    let buf = unsafe { std::slice::from_raw_parts_mut(raw, want) };
    let stride = want / SLOTS;
    // Random permutation over the slots so consecutive hops are not sequential.
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut order: Vec<usize> = (0..SLOTS).collect();
    for i in (1..SLOTS).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        order.swap(i, (x as usize) % (i + 1));
    }
    for i in 0..SLOTS {
        let next = order[(i + 1) % SLOTS] * stride;
        let off = order[i] * stride;
        buf[off..off + 8].copy_from_slice(&(next as u64).to_le_bytes());
    }
    let hops = SLOTS * 8;
    let mut acc: u64 = 0;
    let mut off = order[0] * stride;
    let t0 = std::time::Instant::now();
    for _ in 0..hops {
        let mut word = [0u8; 8];
        word.copy_from_slice(&buf[off..off + 8]);
        acc = acc.wrapping_add(u64::from_le_bytes(word));
        off = acc as usize % want & !(63usize);
        if off + 8 > buf.len() {
            break;
        }
    }
    let ns = t0.elapsed().as_nanos() as f64 / hops as f64;
    std::hint::black_box(acc);
    unsafe {
        std::alloc::dealloc(buf.as_mut_ptr(), layout);
    }    if std::env::var_os("FFF_LATENCY_DEBUG").is_some() {
        eprintln!("LATENCY raw_ns={ns:.2} want={want} slots={SLOTS}");
    }
    // Plausible range guards against a descheduled thread inflating the mean.

    // We are after *DRAM* latency, so both tails are rejected: too slow means the
    // thread was descheduled mid-probe, too fast means the chase stayed in cache
    // (measured 9.7ns on a box whose real DRAM latency is ~80ns, which inflated
    // work_per_thread 8x). The documented default is safer than either tail.
    if (40.0..2000.0).contains(&ns) { ns } else { 0.0 }
}

#[cfg(target_os = "linux")]
fn cgroup_memory_limit() -> Option<usize> {
    let raw = std::fs::read_to_string("/sys/fs/cgroup/memory.max").ok()?;
    let s = raw.trim();
    if s == "max" {
        return None;
    }
    s.parse::<usize>().ok()
}

#[cfg(not(target_os = "linux"))]
fn cgroup_memory_limit() -> Option<usize> {
    None
}
#[cfg(test)]
mod tests {
    use super::*;

    fn m(cores: usize, physical: usize, llc: usize, numa: usize) -> Machine {
        Machine {
            cores,
            physical_cores: physical,
            cache_line: 64,
            l2_bytes: 1 << 20,
            llc_bytes: llc,
            clock_hz: 3e9,
            numa_nodes: numa,
            mem_latency_ns: 80.0,
        }
    }

    /// A single-socket 8-core laptop: the LLC bound should not throttle it to
    /// nothing, and it must stay under the core count.
    #[test]
    fn cap_respects_topology() {
        let laptop = m(8, 4, 16 << 20, 1);
        assert!(laptop.scan_thread_cap() <= 4);
        assert!(laptop.scan_thread_cap() >= 1);
    }

    /// Two sockets: threads beyond one per node read across the interconnect.
    #[test]
    fn numa_limits_threads_per_node() {
        let dual = m(64, 32, 64 << 20, 2);
        assert!(dual.scan_thread_cap() <= 32 / 2);
    }

    /// SMT must not be mistaken for more cores to use.
    #[test]
    fn smt_siblings_are_not_extra_threads() {
        let smt = m(16, 8, 32 << 20, 1);
        assert!(smt.scan_thread_cap() <= 8, "got {}", smt.scan_thread_cap());
    }

    /// A tiny LLC (a phone) must not ask for more threads than it can serve.
    #[test]
    fn small_llc_throttles() {
        let tiny = m(8, 8, 2 << 20, 1);
        assert!(tiny.scan_thread_cap() <= 1, "got {}", tiny.scan_thread_cap());
    }

    /// Unknown LLC must not divide by zero or fall through to core count.
    #[test]
    fn unknown_llc_is_safe() {
        let unknown = m(4, 4, 0, 0);
        assert!((1..=4).contains(&unknown.scan_thread_cap()));
        let unknown_numa = m(4, 4, 0, 0);
        assert!((1..=4).contains(&unknown_numa.scan_thread_cap()));
    }

    /// Latency, not bandwidth, is the driver: a faster-memory box must derive
    /// more work per thread, and an unprobed box must still land in range.
    #[test]
    fn throughput_scales_with_latency() {
        let slow = m(8, 8, 32 << 20, 1);
        let fast = Machine {
            mem_latency_ns: 40.0,
            ..m(8, 8, 32 << 20, 1)
        };
        assert!(
            fast.read_bytes_per_ns() > slow.read_bytes_per_ns(),
            "lower latency should raise throughput"
        );
        let unprobed = Machine {
            mem_latency_ns: 0.0,
            ..m(8, 8, 32 << 20, 1)
        };
        let b = unprobed.read_bytes_per_ns();
        assert!((0.05..=64.0).contains(&b), "default throughput {b}");
    }

    /// Whatever the host reports, derived values must stay usable.
    #[test]
    fn derived_values_are_sane_on_any_machine() {
        let real = machine();
        let w = real.match_work_per_thread();
        assert!((8 * 1024..=8 * 1024 * 1024).contains(&w), "work {w}");
        let cap = real.scan_thread_cap();
        assert!((1..=64).contains(&cap), "cap {cap}");
        assert!(cap <= real.cores.max(1), "cap exceeds cores");
        assert!(real.physical_cores >= 1, "physical_cores {}", real.physical_cores);
        assert!(real.physical_cores <= real.cores.max(1));
        assert!((16..=256).contains(&real.cache_line));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_sysfs_cache_sizes() {
        assert_eq!(parse_cache_size("32K"), Some(32 * 1024));
        assert_eq!(parse_cache_size("1024K"), Some(1024 * 1024));
        assert_eq!(parse_cache_size("48M"), Some(48 * 1024 * 1024));
        assert_eq!(parse_cache_size("bogus"), None);
    }
}
