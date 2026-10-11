//! Process-level sensors (exterior measurement).

use std::time::Instant;

#[derive(Debug, Clone)]
pub struct SensorSnap {
    pub at: Instant,
    pub cpu_user_us: Option<u64>,
    pub cpu_system_us: Option<u64>,
    /// Current resident set / working set (MB).
    pub rss_mb: Option<f64>,
    /// OS process high-water mark when available (MB). Lifetime of process, not interval.
    pub rss_peak_mb: Option<f64>,
    pub disk_read_bytes: Option<u64>,
    pub disk_write_bytes: Option<u64>,
    pub page_faults: Option<u64>,
    pub context_switches: Option<u64>,
    /// Live OS thread count for this process.
    pub thread_count: Option<u64>,
    pub cpu_cycles: Option<u64>,
    pub energy_uj: Option<u64>,
}

impl Default for SensorSnap {
    fn default() -> Self {
        Self {
            at: Instant::now(),
            cpu_user_us: None,
            cpu_system_us: None,
            rss_mb: None,
            rss_peak_mb: None,
            disk_read_bytes: None,
            disk_write_bytes: None,
            page_faults: None,
            context_switches: None,
            thread_count: None,
            cpu_cycles: None,
            energy_uj: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SensorDelta {
    pub cpu_user_us: Option<u64>,
    pub cpu_system_us: Option<u64>,
    pub rss_start_mb: Option<f64>,
    pub rss_end_mb: Option<f64>,
    /// Peak RSS for the interval when observable.
    pub rss_peak_mb: Option<f64>,
    /// true when peak came from OS high-water rise (PeakWorkingSetSize / VmHWM).
    pub rss_peak_is_os_hwm: bool,
    pub disk_read_bytes: Option<u64>,
    pub disk_write_bytes: Option<u64>,
    pub page_faults: Option<u64>,
    pub context_switches: Option<u64>,
    /// Max observed thread count across the two snapshots.
    pub threads_peak: Option<u64>,
    pub cpu_cycles: Option<u64>,
    pub energy_uj: Option<u64>,
    pub energy_j: Option<f64>,
    pub energy_source: Option<String>,
}

/// What one sensor pack can actually deliver on this host.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PackProbe {
    pub pack: String,
    pub available: bool,
    pub fields: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

pub struct Sensors {
    pub mem: bool,
    pub cpu: bool,
    pub io: bool,
    pub cycles: bool,
    pub power: bool,
}

impl Sensors {
    pub fn from_packs(packs: &str) -> Self {
        let all = packs == "all" || packs.is_empty();
        let list: Vec<&str> = packs.split(',').map(str::trim).collect();
        let has = |p: &str| all || list.iter().any(|x| *x == p || *x == "all");
        Self {
            mem: has("mem"),
            cpu: has("cpu"),
            io: has("io"),
            cycles: has("cycles"),
            power: has("power") || has("energy"),
        }
    }

    pub fn snapshot(&self) -> SensorSnap {
        let mut snap = SensorSnap {
            at: Instant::now(),
            ..Default::default()
        };
        #[cfg(windows)]
        windows_snap(&mut snap, self);
        #[cfg(unix)]
        unix_snap(&mut snap, self);
        if self.power {
            snap.energy_uj = read_energy_uj();
        }
        if self.cycles && snap.cpu_cycles.is_none() {
            snap.cpu_cycles = read_process_cycles();
        }
        snap
    }

    pub fn delta(&self, a: &SensorSnap, b: &SensorSnap) -> SensorDelta {
        let energy_uj = sub_opt(a.energy_uj, b.energy_uj);
        let (rss_peak_mb, rss_peak_is_os_hwm) = peak_for_interval(a, b);
        SensorDelta {
            cpu_user_us: sub_opt_nonzero(a.cpu_user_us, b.cpu_user_us),
            cpu_system_us: sub_opt_nonzero(a.cpu_system_us, b.cpu_system_us),
            rss_start_mb: a.rss_mb,
            rss_end_mb: b.rss_mb,
            rss_peak_mb,
            rss_peak_is_os_hwm,
            disk_read_bytes: sub_opt(a.disk_read_bytes, b.disk_read_bytes),
            disk_write_bytes: sub_opt(a.disk_write_bytes, b.disk_write_bytes),
            page_faults: sub_opt(a.page_faults, b.page_faults),
            context_switches: sub_opt(a.context_switches, b.context_switches),
            threads_peak: match (a.thread_count, b.thread_count) {
                (Some(x), Some(y)) => Some(x.max(y)),
                (None, Some(y)) | (Some(y), None) => Some(y),
                _ => None,
            },
            cpu_cycles: sub_opt_nonzero(a.cpu_cycles, b.cpu_cycles),
            energy_uj,
            energy_j: energy_uj.map(|u| u as f64 / 1_000_000.0),
            energy_source: if energy_uj.is_some() {
                Some(energy_source_label())
            } else if self.power {
                Some("unavailable".into())
            } else {
                None
            },
        }
    }

    pub fn cpu_unmeasured(&self, delta: &SensorDelta) -> bool {
        self.cpu && delta.cpu_user_us.is_none() && delta.cpu_system_us.is_none()
    }
}

/// Probe each pack on this host. Availability = OS can read the counter (not that a
/// short probe always produces a non-zero delta — Windows process times are coarse).
pub fn probe_packs() -> Vec<PackProbe> {
    let s = Sensors::from_packs("all");
    let a = s.snapshot();
    let mut x = 1u64;
    for _ in 0..2_000_000 {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    }
    let _ = x;
    std::thread::sleep(std::time::Duration::from_millis(20));
    let b = s.snapshot();
    let d = s.delta(&a, &b);

    vec![
        PackProbe {
            pack: "cpu".into(),
            available: b.cpu_user_us.is_some() || b.cpu_system_us.is_some(),
            fields: filter_present(&[
                ("cpu_user_us", b.cpu_user_us.is_some()),
                ("cpu_system_us", b.cpu_system_us.is_some()),
                (
                    "cpu_delta_nonzero",
                    d.cpu_user_us.is_some() || d.cpu_system_us.is_some(),
                ),
            ]),
            note: Some(
                "GetProcessTimes / proc stat (µs); short ops may be below timer resolution".into(),
            ),
        },
        PackProbe {
            pack: "mem".into(),
            available: b.rss_mb.is_some(),
            fields: filter_present(&[
                ("rss_mb", b.rss_mb.is_some()),
                ("rss_peak_mb", b.rss_peak_mb.is_some()),
                ("page_faults", b.page_faults.is_some()),
                ("context_switches", b.context_switches.is_some()),
                ("thread_count", b.thread_count.is_some()),
            ]),
            note: Some("working set + OS high-water (PeakWorkingSetSize/VmHWM)".into()),
        },
        PackProbe {
            pack: "io".into(),
            available: b.disk_read_bytes.is_some() || b.disk_write_bytes.is_some(),
            fields: filter_present(&[
                ("disk_read_bytes", b.disk_read_bytes.is_some()),
                ("disk_write_bytes", b.disk_write_bytes.is_some()),
            ]),
            note: Some("process I/O counters when OS exposes them".into()),
        },
        PackProbe {
            pack: "cycles".into(),
            available: b.cpu_cycles.is_some(),
            fields: filter_present(&[
                ("cpu_cycles", b.cpu_cycles.is_some()),
                ("cpu_cycles_delta_nonzero", d.cpu_cycles.is_some()),
            ]),
            note: Some(
                if cfg!(windows) {
                    "QueryProcessCycleTime"
                } else {
                    "unavailable without perf (not process-isolated here)"
                }
                .into(),
            ),
        },
        PackProbe {
            pack: "power".into(),
            available: b.energy_uj.is_some(),
            fields: filter_present(&[("energy_uj", b.energy_uj.is_some())]),
            note: Some("Linux RAPL package energy when readable; else unavailable".into()),
        },
    ]
}

fn filter_present(items: &[(&str, bool)]) -> Vec<String> {
    items
        .iter()
        .filter(|(_, ok)| *ok)
        .map(|(n, _)| (*n).into())
        .collect()
}

/// Prefer OS high-water only when it rose during the interval (true interval peak).
/// SENSOR_SEMANTICS: do not copy end RSS as peak; return None when unobserved.
fn peak_for_interval(a: &SensorSnap, b: &SensorSnap) -> (Option<f64>, bool) {
    match (a.rss_peak_mb, b.rss_peak_mb) {
        (Some(ps), Some(pe)) if pe > ps + 0.01 => (Some(pe), true),
        // First snapshot in process: HWM present at end is process-lifetime, not op peak.
        (None, Some(_pe)) => (None, false),
        _ => (None, false),
    }
}

fn sub_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(y.saturating_sub(x)),
        _ => None,
    }
}

pub fn sub_opt_nonzero(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match sub_opt(a, b) {
        Some(0) => None,
        other => other,
    }
}

fn energy_source_label() -> String {
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/sys/class/powercap/intel-rapl").exists() {
            return "rapl_pkg".into();
        }
    }
    "platform".into()
}

/// Package energy (µJ) via Linux RAPL powercap when present.
fn read_energy_uj() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let candidates = [
            "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
            "/sys/class/powercap/intel-rapl:0/energy_uj",
        ];
        for c in candidates {
            if let Ok(s) = std::fs::read_to_string(c) {
                if let Ok(v) = s.trim().parse::<u64>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn read_process_cycles() -> Option<u64> {
    #[cfg(windows)]
    {
        windows_process_cycles()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_cpu_delta_is_none_not_fake_zero() {
        assert_eq!(sub_opt_nonzero(Some(100), Some(100)), None);
        assert_eq!(sub_opt_nonzero(Some(100), Some(250)), Some(150));
        assert_eq!(sub_opt_nonzero(None, Some(1)), None);
    }

    #[test]
    fn packs_parse_all_includes_power_cycles() {
        let s = Sensors::from_packs("all");
        assert!(s.mem && s.cpu && s.io && s.cycles && s.power);
        let s2 = Sensors::from_packs("mem,power");
        assert!(s2.mem && s2.power && !s2.cpu);
    }

    #[test]
    fn energy_delta_to_joules() {
        let a = SensorSnap {
            energy_uj: Some(1_000_000),
            ..Default::default()
        };
        let b = SensorSnap {
            energy_uj: Some(1_500_000),
            ..Default::default()
        };
        let s = Sensors::from_packs("power");
        let d = s.delta(&a, &b);
        assert_eq!(d.energy_uj, Some(500_000));
        assert!((d.energy_j.unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn peak_from_os_hwm_rise() {
        let a = SensorSnap {
            rss_mb: Some(10.0),
            rss_peak_mb: Some(12.0),
            ..Default::default()
        };
        let b = SensorSnap {
            rss_mb: Some(14.0),
            rss_peak_mb: Some(20.0),
            ..Default::default()
        };
        let d = Sensors::from_packs("mem").delta(&a, &b);
        assert_eq!(d.rss_peak_mb, Some(20.0));
        assert!(d.rss_peak_is_os_hwm);
    }

    #[test]
    fn peak_unavailable_without_hwm_rise() {
        let a = SensorSnap {
            rss_mb: Some(10.0),
            rss_peak_mb: Some(30.0),
            ..Default::default()
        };
        let b = SensorSnap {
            rss_mb: Some(12.0),
            rss_peak_mb: Some(30.0),
            ..Default::default()
        };
        let d = Sensors::from_packs("mem").delta(&a, &b);
        // SENSOR_SEMANTICS: no interval HWM rise → peak unavailable, not end RSS.
        assert!(d.rss_peak_mb.is_none());
        assert!(!d.rss_peak_is_os_hwm);
    }

    #[test]
    fn mem_snapshot_has_rss_on_supported_os() {
        let s = Sensors::from_packs("mem");
        let snap = s.snapshot();
        #[cfg(any(windows, target_os = "linux"))]
        {
            assert!(snap.rss_mb.is_some());
            assert!(snap.rss_peak_mb.is_some());
            assert!(snap.rss_peak_mb.unwrap() + 0.01 >= snap.rss_mb.unwrap());
        }
    }

    #[test]
    fn probe_packs_lists_all_packs() {
        let p = probe_packs();
        let names: Vec<_> = p.iter().map(|x| x.pack.as_str()).collect();
        assert_eq!(names, ["cpu", "mem", "io", "cycles", "power"]);
        #[cfg(any(windows, target_os = "linux"))]
        {
            assert!(p.iter().any(|x| x.pack == "mem" && x.available));
            assert!(p.iter().any(|x| x.pack == "cpu" && x.available));
        }
        #[cfg(windows)]
        {
            assert!(p.iter().any(|x| x.pack == "cycles" && x.available));
        }
    }
}

#[cfg(unix)]
fn unix_snap(snap: &mut SensorSnap, s: &Sensors) {
    if s.cpu {
        if let Some((u, k)) = read_proc_stat_cpu() {
            snap.cpu_user_us = Some(u);
            snap.cpu_system_us = Some(k);
        }
    }
    if s.mem {
        if let Some((rss, peak)) = read_proc_rss_and_peak_mb() {
            snap.rss_mb = Some(rss);
            snap.rss_peak_mb = Some(peak);
        }
        if let Some(pf) = read_proc_faults() {
            snap.page_faults = Some(pf);
        }
        #[cfg(target_os = "linux")]
        {
            snap.context_switches = read_proc_context_switches();
            snap.thread_count = read_proc_thread_count();
        }
    }
    if s.io {
        if let Some((r, w)) = read_proc_io() {
            snap.disk_read_bytes = Some(r);
            snap.disk_write_bytes = Some(w);
        }
    }
}

#[cfg(target_os = "linux")]
fn read_proc_stat_cpu() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rparen = s.rfind(')')?;
    let rest = s.get(rparen + 2)?;
    let parts: Vec<&str> = rest.split_whitespace().collect();
    let utime: u64 = parts.get(11)?.parse().ok()?;
    let stime: u64 = parts.get(12)?.parse().ok()?;
    let ticks = 100u64;
    Some((utime * 1_000_000 / ticks, stime * 1_000_000 / ticks))
}

#[cfg(all(unix, not(target_os = "linux")))]
fn read_proc_stat_cpu() -> Option<(u64, u64)> {
    None
}

/// (VmRSS, VmHWM) in MB. HWM is process lifetime peak resident.
#[cfg(target_os = "linux")]
fn read_proc_rss_and_peak_mb() -> Option<(f64, f64)> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut rss = None;
    let mut peak = None;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            let kb: f64 = v.split_whitespace().next()?.parse().ok()?;
            rss = Some(kb / 1024.0);
        } else if let Some(v) = line.strip_prefix("VmHWM:") {
            let kb: f64 = v.split_whitespace().next()?.parse().ok()?;
            peak = Some(kb / 1024.0);
        }
    }
    Some((rss?, peak.unwrap_or(rss?)))
}

/// voluntary + nonvoluntary context switches from /proc/self/status.
#[cfg(target_os = "linux")]
fn read_proc_context_switches() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut vol = None;
    let mut non = None;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("voluntary_ctxt_switches:") {
            vol = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("nonvoluntary_ctxt_switches:") {
            non = v.trim().parse().ok();
        }
    }
    match (vol, non) {
        (Some(a), Some(b)) => Some(a.saturating_add(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn read_proc_thread_count() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("Threads:") {
            return v.trim().parse().ok();
        }
    }
    None
}

#[cfg(all(unix, not(target_os = "linux")))]
fn read_proc_rss_and_peak_mb() -> Option<(f64, f64)> {
    None
}

#[cfg(target_os = "linux")]
fn read_proc_faults() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rparen = s.rfind(')')?;
    let rest = s.get(rparen + 2)?;
    let parts: Vec<&str> = rest.split_whitespace().collect();
    let min: u64 = parts.get(7)?.parse().ok()?;
    let maj: u64 = parts.get(9)?.parse().ok()?;
    Some(min + maj)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn read_proc_faults() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn read_proc_io() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/io").ok()?;
    let mut r = None;
    let mut w = None;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("read_bytes: ") {
            r = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("write_bytes: ") {
            w = v.trim().parse().ok();
        }
    }
    Some((r?, w.unwrap_or(0)))
}

#[cfg(all(unix, not(target_os = "linux")))]
fn read_proc_io() -> Option<(u64, u64)> {
    None
}

#[cfg(windows)]
fn windows_process_cycles() -> Option<u64> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn QueryProcessCycleTime(h: *mut core::ffi::c_void, cycles: *mut u64) -> i32;
    }
    unsafe {
        let mut c = 0u64;
        if QueryProcessCycleTime(GetCurrentProcess(), &mut c) != 0 {
            Some(c)
        } else {
            None
        }
    }
}

#[cfg(windows)]
fn windows_snap(snap: &mut SensorSnap, s: &Sensors) {
    use std::mem::{size_of, zeroed};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn GetProcessTimes(
            h: *mut core::ffi::c_void,
            creation: *mut u64,
            exit: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
        fn GetProcessIoCounters(h: *mut core::ffi::c_void, io: *mut IoCounters) -> i32;
    }

    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            h: *mut core::ffi::c_void,
            info: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[repr(C)]
    struct IoCounters {
        read_op_count: u64,
        write_op_count: u64,
        other_op_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    unsafe {
        let h = GetCurrentProcess();
        if s.cpu {
            let mut c = 0u64;
            let mut e = 0u64;
            let mut k = 0u64;
            let mut u = 0u64;
            if GetProcessTimes(h, &mut c, &mut e, &mut k, &mut u) != 0 {
                snap.cpu_user_us = Some(u / 10);
                snap.cpu_system_us = Some(k / 10);
            }
        }
        if s.mem {
            let mut info: ProcessMemoryCounters = zeroed();
            info.cb = size_of::<ProcessMemoryCounters>() as u32;
            if GetProcessMemoryInfo(h, &mut info, info.cb) != 0 {
                snap.rss_mb = Some(info.working_set_size as f64 / (1024.0 * 1024.0));
                snap.rss_peak_mb = Some(info.peak_working_set_size as f64 / (1024.0 * 1024.0));
                snap.page_faults = Some(info.page_fault_count as u64);
            }
            snap.thread_count = windows_thread_count();
        }
        if s.io {
            let mut io: IoCounters = zeroed();
            if GetProcessIoCounters(h, &mut io) != 0 {
                snap.disk_read_bytes = Some(io.read_transfer_count);
                snap.disk_write_bytes = Some(io.write_transfer_count);
            }
        }
    }
}

#[cfg(windows)]
fn windows_thread_count() -> Option<u64> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcessId() -> u32;
        fn CreateToolhelp32Snapshot(flags: u32, pid: u32) -> *mut core::ffi::c_void;
        fn Thread32First(h: *mut core::ffi::c_void, te: *mut ThreadEntry32) -> i32;
        fn Thread32Next(h: *mut core::ffi::c_void, te: *mut ThreadEntry32) -> i32;
        fn CloseHandle(h: *mut core::ffi::c_void) -> i32;
    }
    const TH32CS_SNAPTHREAD: u32 = 0x0000_0004;
    const INVALID: isize = -1;

    #[repr(C)]
    struct ThreadEntry32 {
        dw_size: u32,
        cnt_usage: u32,
        th32_thread_id: u32,
        th32_owner_process_id: u32,
        tp_base_pri: i32,
        tp_delta_pri: i32,
        dw_flags: u32,
    }

    unsafe {
        let pid = GetCurrentProcessId();
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap as isize == INVALID || snap.is_null() {
            return None;
        }
        let mut te = ThreadEntry32 {
            dw_size: std::mem::size_of::<ThreadEntry32>() as u32,
            cnt_usage: 0,
            th32_thread_id: 0,
            th32_owner_process_id: 0,
            tp_base_pri: 0,
            tp_delta_pri: 0,
            dw_flags: 0,
        };
        let mut n = 0u64;
        if Thread32First(snap, &mut te) != 0 {
            loop {
                if te.th32_owner_process_id == pid {
                    n += 1;
                }
                te.dw_size = std::mem::size_of::<ThreadEntry32>() as u32;
                if Thread32Next(snap, &mut te) == 0 {
                    break;
                }
            }
        }
        CloseHandle(snap);
        (n > 0).then_some(n)
    }
}
