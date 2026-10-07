//! Dedicated rayon pools. The global pool spans every logical core, which
//! oversubscribes asymmetric chips (Apple P+E): E-cores are ~2× slower and
//! `open()` contends on a per-VFS lock past P-core count, so a larger pool is
//! slower on file-heavy work.

use std::sync::LazyLock;

/// Override for background pool size (`FFF_BG_THREADS`).
fn bg_threads_override() -> Option<usize> {
    std::env::var("FFF_BG_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n >= 1)
}

/// Override for search pool size (`FFF_SEARCH_THREADS`).
fn search_threads_override() -> Option<usize> {
    std::env::var("FFF_SEARCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n >= 1)
}

fn pin_qos() {
    #[cfg(target_os = "macos")]
    unsafe {
        let _ = libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INITIATED, 0);
    }
}

/// Dedicated thread pool for background work (scan, warmup, bigram build).
pub static BACKGROUND_THREAD_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let total = effective_core_count();
    // Background work is mostly syscall-bound; halving parallelism leaves
    // cores for search/UI at negligible throughput cost.
    let bg_threads = bg_threads_override().unwrap_or_else(|| (total / 2).max(2));
    rayon::ThreadPoolBuilder::new()
        .num_threads(bg_threads)
        .thread_name(|i| format!("fff-bg-{i}"))
        .start_handler(|_| pin_qos())
        .build()
        .expect("failed to create background rayon pool")
});

/// Cores worth actually using for parallel work.
///
/// On Apple silicon the E-cores are ~2x slower and `open()` contends on a
/// per-VFS lock past P-core count, so prefer the performance cores: on a 12P+4E
/// M4 Max, grep runs 16t=6.2s vs 13t=4.9s. On Linux the cgroup CPU
/// quota/cpuset caps this further so containers do not size pools off the host.
///
/// This is the single source of truth for "how many threads can this machine
/// usefully run" - the pools below and `crate::machine` all derive from it, so
/// they cannot disagree about core count.
pub fn effective_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        let mut count: libc::c_int = 0;
        let mut size = std::mem::size_of::<libc::c_int>();
        let name = c"hw.perflevel0.physicalcpu";
        let ok = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut count as *mut _ as *mut libc::c_void,
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if ok == 0 && count > 0 {
            return count as usize;
        }
    }
    let base = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    #[cfg(target_os = "linux")]
    {
        if let Some(cap) = cgroup_cpu_count() {
            return base.min(cap).max(1);
        }
    }
    base
}

/// Linux cgroup CPU cap: min of quota and cpuset size. `None` when uncapped.
#[cfg(target_os = "linux")]
pub(crate) fn cgroup_cpu_count() -> Option<usize> {
    let mut cap: Option<usize> = None;
    let tighten = |cap: &mut Option<usize>, n: usize| {
        if n >= 1 {
            *cap = Some(cap.unwrap_or(usize::MAX).min(n));
        }
    };
    // v2 `cpu.max`: `$MAX $PERIOD` or `max $PERIOD`.
    if let Ok(raw) = std::fs::read_to_string("/sys/fs/cgroup/cpu.max") {
        let mut it = raw.split_whitespace();
        if let (Some(max), Some(period)) = (it.next(), it.next())
            && max != "max"
            && let (Ok(m), Ok(p)) = (max.parse::<f64>(), period.parse::<f64>())
            && p > 0.0
        {
            tighten(&mut cap, (m / p).ceil() as usize);
        }
    }
    // v1 quota/period pair.
    if let (Ok(q), Ok(p)) = (
        std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ) && let (Ok(qf), Ok(pf)) = (q.trim().parse::<f64>(), p.trim().parse::<f64>())
        && qf > 0.0
        && pf > 0.0
    {
        tighten(&mut cap, (qf / pf).ceil() as usize);
    }
    // Cpuset size (v2, then v1).
    for path in [
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
    ] {
        if let Ok(raw) = std::fs::read_to_string(path)
            && let Some(n) = parse_cpuset_len(raw.trim())
        {
            tighten(&mut cap, n);
        }
    }
    cap.filter(|n| *n >= 1)
}

/// Count CPUs in a cpuset list like `0-3,8,10-11`.
#[cfg(target_os = "linux")]
fn parse_cpuset_len(s: &str) -> Option<usize> {
    cpuset_cpus_inner(s).map(|v| v.len())
}

/// Allowed CPUs from the cpuset files, for pinning probe threads inside the
/// container instead of `t % host_ncpu` (which can address forbidden cores).
#[cfg(target_os = "linux")]
pub(crate) fn cpuset_cpus() -> Option<Vec<usize>> {
    [
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
    ]
    .iter()
    .find_map(|p| {
        std::fs::read_to_string(p)
            .ok()
            .and_then(|raw| cpuset_cpus_inner(raw.trim()))
    })
}

#[cfg(any(test, target_os = "linux"))]
fn cpuset_cpus_inner(s: &str) -> Option<Vec<usize>> {
    if s.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a.trim().parse().ok()?;
            let b: usize = b.trim().parse().ok()?;
            if b < a {
                return None;
            }
            out.extend(a..=b);
        } else {
            out.push(part.parse::<usize>().ok()?);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Pool for grep content search: full effective cores, QoS-pinned on macOS.
/// Grep is I/O-latency-bound (open+read per file), so unlike the
/// LLC-capped scan pool it wants width to overlap stalls; `scan_thread_cap`
/// stays with the background pool. `FFF_SEARCH_THREADS` forces a size.
pub static SEARCH_THREAD_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let threads = search_threads_override().unwrap_or_else(|| effective_core_count().max(1));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .thread_name(|i| format!("fff-search-{i}"))
        .start_handler(|_| pin_qos())
        .build()
        .expect("failed to create search rayon pool")
});

/// Threads the search pool was built with. For non-owned call sites (grep
/// chunking) that must not read ambient `current_num_threads`.
pub fn search_pool_threads() -> usize {
    SEARCH_THREAD_POOL.current_num_threads().max(1)
}

/// Threads the background pool was built with.
pub fn bg_pool_threads() -> usize {
    BACKGROUND_THREAD_POOL.current_num_threads().max(1)
}

/// Default `max_threads` when the caller passes 0: effective cores, not the
/// ambient pool size (which collapses under nested installs).
pub fn default_max_threads() -> usize {
    effective_core_count().max(1)
}

/// Run `f` on `BACKGROUND_THREAD_POOL` unless we are already inside a rayon
/// worker. Nested `install()` from a pool worker deadlocks on large scans
/// (the spawn task occupies a worker while `install` waits for workers).
///
/// Covers `par_sort` too: the walk/indexable sorts must go through here, never
/// a bare `BACKGROUND_THREAD_POOL.install`, so a sort issued from a worker
/// degrades to serial instead of deadlocking.
pub fn run_on_bg_pool<R>(f: impl FnOnce() -> R + Send) -> R
where
    R: Send,
{
    if rayon::current_thread_index().is_some() {
        f()
    } else {
        BACKGROUND_THREAD_POOL.install(f)
    }
}

/// Same guard for `SEARCH_THREAD_POOL`. Every grep entry (`grep`, `multi_grep`)
/// must route through here: an unconditional `install` deadlocks when grep is
/// invoked from a pool worker, and inherits the wrong pool otherwise.
pub fn run_on_search_pool<R>(f: impl FnOnce() -> R + Send) -> R
where
    R: Send,
{
    if rayon::current_thread_index().is_some() {
        f()
    } else {
        SEARCH_THREAD_POOL.install(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guards_run_outside_pools() {
        assert!(run_on_bg_pool(|| 1 + 1) == 2);
        assert!(run_on_search_pool(|| 2 + 2) == 4);
    }

    #[test]
    fn pool_sizes_sane() {
        assert!(search_pool_threads() >= 1);
        assert!(search_pool_threads() <= effective_core_count().max(1));
        assert!(bg_pool_threads() >= 1);
        assert!(default_max_threads() >= 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_cpusets() {
        assert_eq!(parse_cpuset_len("0-3"), Some(4));
        assert_eq!(parse_cpuset_len("0-3,8,10-11"), Some(7));
        assert_eq!(parse_cpuset_len(""), None);
        assert_eq!(parse_cpuset_len("bogus"), None);
    }

    #[test]
    fn expands_cpuset_lists() {
        assert_eq!(
            cpuset_cpus_inner("0-3,8,10-11"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(cpuset_cpus_inner("5"), Some(vec![5]));
        assert_eq!(cpuset_cpus_inner(""), None);
        assert_eq!(cpuset_cpus_inner("bogus"), None);
        assert_eq!(cpuset_cpus_inner("3-0"), None);
    }
}
