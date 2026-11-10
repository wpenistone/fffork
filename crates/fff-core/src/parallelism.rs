use std::sync::LazyLock;

fn bg_threads_override() -> Option<usize> {
    std::env::var("FFF_BG_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n >= 1)
}

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
    // Syscall-bound work: half the cores leaves room for search/UI.
    let bg_threads = bg_threads_override().unwrap_or_else(|| (total / 2).max(2));
    rayon::ThreadPoolBuilder::new()
        .num_threads(bg_threads)
        .thread_name(|i| format!("fff-bg-{i}"))
        .start_handler(|_| pin_qos())
        .build()
        .expect("failed to create background rayon pool")
});

/// Process-stable core count for pool size and match partitioning.
/// Apple performance cores, otherwise cgroup-capped parallelism.
pub fn effective_core_count() -> usize {
    static N: LazyLock<usize> = LazyLock::new(detect_effective_cores);
    *N
}

fn detect_effective_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        // Unknown topology falls back inside; level 0 alone undercounts.
        return non_efficiency_core_count();
    }
    #[cfg(not(target_os = "macos"))]
    {
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
}

// Linux cgroup CPU cap: min of quota and cpuset size. `None` when uncapped.
#[cfg(target_os = "linux")]
pub(crate) fn cgroup_cpu_count() -> Option<usize> {
    static C: LazyLock<Option<usize>> = LazyLock::new(read_cgroup_cpu_count);
    *C
}

#[cfg(target_os = "linux")]
fn read_cgroup_cpu_count() -> Option<usize> {
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
    if let (Ok(q), Ok(p)) = (
        std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ) && let (Ok(qf), Ok(pf)) = (q.trim().parse::<f64>(), p.trim().parse::<f64>())
        && qf > 0.0
        && pf > 0.0
    {
        tighten(&mut cap, (qf / pf).ceil() as usize);
    }
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

// Count CPUs in a cpuset list like `0-3,8,10-11`.
#[cfg(target_os = "linux")]
fn parse_cpuset_len(s: &str) -> Option<usize> {
    cpuset_cpus_inner(s).map(|v| v.len())
}

// Container-local CPUs for pinning; host indexing can address forbidden cores.
#[cfg(target_os = "linux")]
pub(crate) fn cpuset_cpus() -> Option<Vec<usize>> {
    static C: LazyLock<Option<Vec<usize>>> = LazyLock::new(read_cpuset_cpus);
    C.clone()
}

#[cfg(target_os = "linux")]
fn read_cpuset_cpus() -> Option<Vec<usize>> {
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

/// Grep pool: full effective cores, since I/O latency wants width to overlap stalls.
/// `FFF_SEARCH_THREADS` forces a size.
pub static SEARCH_THREAD_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let threads = search_threads_override().unwrap_or_else(|| effective_core_count().max(1));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .thread_name(|i| format!("fff-search-{i}"))
        .start_handler(|_| {
            pin_qos();
            maybe_pin_search_thread();
        })
        .build()
        .expect("failed to create search rayon pool")
});

/// Search pool size, for non-owned call sites that must not read ambient threads.
pub fn search_pool_threads() -> usize {
    SEARCH_THREAD_POOL.current_num_threads().max(1)
}

/// Threads the background pool was built with. Test helper.
#[cfg(test)]
pub fn bg_pool_threads() -> usize {
    BACKGROUND_THREAD_POOL.current_num_threads().max(1)
}

// In-flight searches; background indexing yields while nonzero.
static QUERY_ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn query_active() -> bool {
    QUERY_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) != 0
}

pub(crate) struct QueryActiveGuard;

impl QueryActiveGuard {
    pub(crate) fn new() -> Self {
        QUERY_ACTIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}

impl Drop for QueryActiveGuard {
    fn drop(&mut self) {
        QUERY_ACTIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

// Per-picker search epoch; group loops abandon their slice when it moves.
// Never aborts other pickers; one load per group.
pub(crate) fn begin_search(counter: &std::sync::atomic::AtomicU64) -> u64 {
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

// Bails a slice when a newer query moves the epoch; disabled costs one bool check.
#[derive(Copy, Clone)]
pub(crate) struct AbortToken<'a> {
    epoch: u64,
    enabled: bool,
    counter: Option<&'a std::sync::atomic::AtomicU64>,
}

impl<'a> AbortToken<'a> {
    #[allow(dead_code)]
    pub(crate) fn disabled() -> Self {
        Self {
            epoch: 0,
            enabled: false,
            counter: None,
        }
    }

    pub(crate) fn live(counter: &'a std::sync::atomic::AtomicU64) -> Self {
        Self {
            epoch: begin_search(counter),
            enabled: cancel_enabled(),
            counter: Some(counter),
        }
    }

    pub(crate) fn aborted(&self) -> bool {
        self.enabled
            && self
                .counter
                .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed) != self.epoch)
    }
}

/// Runs on `BACKGROUND_THREAD_POOL`, or inline inside a rayon worker where
/// nested `install()` deadlocks. Sorts must route here too, degrading to serial.
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

/// Same guard for `SEARCH_THREAD_POOL`; every grep entry must route through here.
/// Unconditional `install` deadlocks from a pool worker and inherits the wrong pool.
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

// One join and at most `width` workers. Results stay in job order.
// `width` caps concurrency; the caller owns the index partition.
pub(crate) fn map_search_width<T, F>(n: usize, width: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    if n == 0 {
        return Vec::new();
    }
    let width = width.max(1).min(n);
    if width == 1 {
        return (0..n).map(f).collect();
    }
    run_on_search_pool(|| {
        let f = &f;
        let mut bags: Vec<Vec<T>> = (0..width).map(|_| Vec::new()).collect();
        rayon::scope(|s| {
            for (w, bag) in bags.iter_mut().enumerate() {
                s.spawn(move |_| {
                    let mut i = w;
                    while i < n {
                        bag.push(f(i));
                        i += width;
                    }
                });
            }
        });
        let mut iters = bags.into_iter().map(Vec::into_iter).collect::<Vec<_>>();
        let mut out = Vec::with_capacity(n);
        loop {
            let mut progressed = false;
            for it in &mut iters {
                if let Some(v) = it.next() {
                    out.push(v);
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        out
    })
}

#[cfg(target_os = "macos")]
fn non_efficiency_core_count() -> usize {
    let detected = sysctl_count(c"hw.nperflevels").and_then(|levels| {
        count_non_efficiency_cores((0..levels).map(|level| {
            let name = std::ffi::CString::new(format!("hw.perflevel{level}.name")).ok()?;
            let count = std::ffi::CString::new(format!("hw.perflevel{level}.physicalcpu")).ok()?;
            Some((sysctl_name(&name)?, sysctl_count(&count)?))
        }))
    });

    // Unknown topology falls back to all logical cores, never to a single thread.
    detected
        .or_else(|| sysctl_count(c"hw.perflevel0.physicalcpu").filter(|&count| count > 0))
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        })
}

#[cfg(any(target_os = "macos", test))]
fn count_non_efficiency_cores(
    levels: impl IntoIterator<Item = Option<(String, usize)>>,
) -> Option<usize> {
    let count = levels.into_iter().try_fold(0usize, |total, level| {
        let (name, count) = level?;
        if name.eq_ignore_ascii_case("Efficiency") {
            Some(total)
        } else if ["Super", "Performance", "Standard"]
            .iter()
            .any(|known| name.eq_ignore_ascii_case(known))
        {
            total.checked_add(count)
        } else {
            None
        }
    })?;
    (count > 0).then_some(count)
}

#[cfg(target_os = "macos")]
fn sysctl_count(name: &std::ffi::CStr) -> Option<usize> {
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of_val(&value);
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || size != std::mem::size_of_val(&value) {
        return None;
    }
    usize::try_from(value).ok()
}

#[cfg(target_os = "macos")]
fn sysctl_name(name: &std::ffi::CStr) -> Option<String> {
    let mut value = [0u8; 64];
    let mut size = value.len();
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || size > value.len() {
        return None;
    }
    std::ffi::CStr::from_bytes_with_nul(&value[..size])
        .ok()?
        .to_str()
        .ok()
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
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
    fn abort_token_fires_only_on_newer_search() {
        let counter = std::sync::atomic::AtomicU64::new(0);
        let t = AbortToken::disabled();
        let _ = begin_search(&counter);
        assert!(!t.aborted());
        let live = AbortToken::live(&counter);
        assert!(!live.aborted());
        let _ = begin_search(&counter);
        assert!(live.aborted());
        assert!(!AbortToken::disabled().aborted());
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

    #[test]
    fn includes_super_and_performance_cores() {
        assert_eq!(count(&[("Super", 6), ("Performance", 12)]), Some(18));
        assert_eq!(count(&[("Performance", 12), ("Super", 6)]), Some(18));
    }

    #[test]
    fn excludes_efficiency_cores_in_every_position() {
        assert_eq!(count(&[("Performance", 12), ("Efficiency", 4)]), Some(12));
        assert_eq!(count(&[("Efficiency", 4), ("Performance", 4)]), Some(4));
        assert_eq!(
            count(&[("Super", 6), ("Efficiency", 4), ("Performance", 12)]),
            Some(18)
        );
        assert_eq!(count(&[("Performance", 8), ("efficiency", 4)]), Some(8));
    }

    #[test]
    fn accepts_standard_cores() {
        assert_eq!(count(&[("Performance", 8)]), Some(8));
        assert_eq!(count(&[("Standard", 8)]), Some(8));
    }

    #[test]
    fn unknown_core_types_fall_back() {
        assert_eq!(count(&[("Faster", 6), ("Performance", 12)]), None);
        assert_eq!(count(&[("Super", 6), ("PowerSaving", 4)]), None);
    }

    #[test]
    fn incomplete_or_unusable_topology_falls_back() {
        assert_eq!(count(&[]), None);
        assert_eq!(count(&[("Efficiency", 4)]), None);
        assert_eq!(count(&[("Performance", 0)]), None);
        assert_eq!(
            count_non_efficiency_cores([Some(("Super".into(), 6)), None]),
            None
        );
        assert_eq!(count(&[("Super", usize::MAX), ("Performance", 1)]), None);
    }

    fn count(levels: &[(&str, usize)]) -> Option<usize> {
        count_non_efficiency_cores(
            levels
                .iter()
                .map(|&(name, count)| Some((name.to_owned(), count))),
        )
    }

    #[test]
    fn parses_affinity_masks() {
        assert_eq!(parse_affinity_mask("0xff"), Some(0xff));
        assert_eq!(parse_affinity_mask("0XFFF"), Some(0xfff));
        assert_eq!(parse_affinity_mask("255"), Some(255));
        assert_eq!(parse_affinity_mask(""), None);
        assert_eq!(parse_affinity_mask("bogus"), None);
        assert_eq!(parse_affinity_mask("0x"), None);
    }
}

// Default `max_threads` (caller passed 0): effective cores, collapsing under nesting.
#[cfg(test)]
pub fn default_max_threads() -> usize {
    effective_core_count().max(1)
}

// One parse of `FFF_AFFINITY`. Unset or zero leaves workers unpinned.
fn affinity_mask() -> Option<usize> {
    static M: LazyLock<Option<usize>> = LazyLock::new(|| {
        let raw = std::env::var("FFF_AFFINITY").ok()?;
        #[cfg(windows)]
        {
            static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            LOGGED.get_or_init(|| {
                eprintln!("AFFINITY probe active (mask from FFF_AFFINITY)");
            });
        }
        parse_affinity_mask(raw.trim())
    });
    *M
}

// Opt-in search-thread affinity (`FFF_AFFINITY=mask`, hex or decimal).
pub(crate) fn maybe_pin_search_thread() {
    let Some(mask) = affinity_mask() else {
        return;
    };
    #[cfg(windows)]
    // SAFETY: pseudo-handle valid, mask nonzero; failure ignored (best-effort).
    unsafe {
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
    #[cfg(not(windows))]
    let _ = mask;
}

// One env read. `FFF_NO_CANCEL` disables abort for the whole process.
fn cancel_enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FFF_NO_CANCEL").is_none());
    *ON
}

fn parse_affinity_mask(s: &str) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        if hex.is_empty() {
            return None;
        }
        usize::from_str_radix(hex, 16).ok().filter(|m| *m != 0)
    } else {
        s.parse::<usize>().ok().filter(|m| *m != 0)
    }
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentThread() -> usize;
    fn SetThreadAffinityMask(h: usize, mask: usize) -> usize;
}
