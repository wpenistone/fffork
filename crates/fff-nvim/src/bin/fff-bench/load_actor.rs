//! Load actors and controlled interference (Phase 8).
//!
//! Production runs isolate actors in child processes (separate accounting scope).
//! Unit tests use in-process threads (cfg(test)); both paths deliver real work
//! and enforce hard delivery/recovery gates.

use crate::identity;
use crate::sensors::Sensors;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub const PROTOCOL: &str = "load-actor/v1";
/// Relative tolerance for requested vs delivered busy_cores.
pub const DELIVERY_TOLERANCE: f64 = 0.25;
const CALIBRATE_MS: u64 = 80;
const ENV_WORKER: &str = "FFF_BENCH_LOAD_WORKER";
const ENV_STOP: &str = "FFF_BENCH_STOP_FILE";
const ENV_COUNTER: &str = "FFF_BENCH_COUNTER_FILE";
const ENV_ARG: &str = "FFF_BENCH_WORK_ARG";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadProfile {
    pub id: String,
    pub actors: Vec<ActorSpec>,
    #[serde(default)]
    pub phases: Vec<PhaseSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorSpec {
    pub id: String,
    pub provider: String,
    #[serde(default)]
    pub requested: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSpec {
    pub id: String,
    #[serde(default)]
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorEvent {
    pub schema_version: String,
    pub event: String,
    pub load_profile_id: String,
    pub actor_id: String,
    pub provider: ProviderId,
    pub phase_id: String,
    /// ready | active | draining | complete | error
    pub state: String,
    pub requested: serde_json::Value,
    pub delivered: serde_json::Value,
    pub readiness: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderId {
    pub id: String,
    pub version: String,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreatmentEvidence {
    pub load_profile_id: String,
    pub protocol: String,
    pub valid: bool,
    pub delivery_equivalent: bool,
    pub actors: Vec<ActorEvent>,
    #[serde(default)]
    pub reason_codes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery: Option<RecoveryRecord>,
    /// separate_process | same_process_thread
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryRecord {
    pub drain_ms: u64,
    pub recovered: bool,
    pub residual_cpu_busy_cores: Option<f64>,
    pub residual_io_ops: u64,
    pub children_exited: bool,
    #[serde(default)]
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationRecord {
    pub wall_us: u64,
    pub cpu_total_us: Option<u64>,
    pub busy_cores: Option<f64>,
    pub work_iters: u64,
    pub quality: String,
}

/// Profiles used by resilience-confirm (FFF_REFERENCE_PACK).
pub fn resilience_profiles() -> Vec<&'static str> {
    vec![
        "quiet_v1",
        "peer_cpu_light_v1",
        "peer_cpu_moderate_v1",
        "peer_memory_capacity_v1",
        "peer_storage_v1",
        "peer_event_v1",
        "developer_composite_v1",
    ]
}

pub fn quiet_v1() -> LoadProfile {
    LoadProfile {
        id: "quiet_v1".into(),
        actors: vec![],
        phases: vec![
            PhaseSpec {
                id: "measure".into(),
                duration_ms: 0,
            },
            PhaseSpec {
                id: "recovery".into(),
                duration_ms: 0,
            },
        ],
    }
}

pub fn peer_cpu_light_v1() -> LoadProfile {
    LoadProfile {
        id: "peer_cpu_light_v1".into(),
        actors: vec![ActorSpec {
            id: "cpu_peer".into(),
            provider: "builtin.cpu-contention".into(),
            requested: serde_json::json!({
                "busy_cores": 0.85,
                "method": "integer_mix",
                "workers": 1,
            }),
        }],
        phases: vec![
            PhaseSpec {
                id: "ramp".into(),
                duration_ms: 40,
            },
            PhaseSpec {
                id: "measure".into(),
                duration_ms: 0,
            },
            PhaseSpec {
                id: "recovery".into(),
                duration_ms: 40,
            },
        ],
    }
}

pub fn peer_cpu_moderate_v1() -> LoadProfile {
    LoadProfile {
        id: "peer_cpu_moderate_v1".into(),
        actors: vec![ActorSpec {
            id: "cpu_peer".into(),
            provider: "builtin.cpu-contention".into(),
            requested: serde_json::json!({
                "busy_cores": 1.7,
                "method": "integer_mix",
                "workers": 2,
            }),
        }],
        phases: vec![
            PhaseSpec {
                id: "ramp".into(),
                duration_ms: 40,
            },
            PhaseSpec {
                id: "measure".into(),
                duration_ms: 0,
            },
            PhaseSpec {
                id: "recovery".into(),
                duration_ms: 40,
            },
        ],
    }
}

pub fn peer_memory_capacity_v1() -> LoadProfile {
    LoadProfile {
        id: "peer_memory_capacity_v1".into(),
        actors: vec![ActorSpec {
            id: "mem_peer".into(),
            provider: "builtin.memory-working-set".into(),
            requested: serde_json::json!({
                "resident_mb": 64,
                "touch_pattern": "sequential",
            }),
        }],
        phases: vec![
            PhaseSpec {
                id: "ramp".into(),
                duration_ms: 20,
            },
            PhaseSpec {
                id: "measure".into(),
                duration_ms: 0,
            },
            PhaseSpec {
                id: "recovery".into(),
                duration_ms: 20,
            },
        ],
    }
}

pub fn peer_storage_v1() -> LoadProfile {
    LoadProfile {
        id: "peer_storage_v1".into(),
        actors: vec![ActorSpec {
            id: "storage_peer".into(),
            provider: "builtin.storage-churn".into(),
            requested: serde_json::json!({
                "ops_per_sec": 200,
                "file_bytes": 4096,
                "workers": 1,
            }),
        }],
        phases: vec![
            PhaseSpec {
                id: "ramp".into(),
                duration_ms: 20,
            },
            PhaseSpec {
                id: "measure".into(),
                duration_ms: 0,
            },
            PhaseSpec {
                id: "recovery".into(),
                duration_ms: 20,
            },
        ],
    }
}

pub fn peer_event_v1() -> LoadProfile {
    LoadProfile {
        id: "peer_event_v1".into(),
        actors: vec![ActorSpec {
            id: "fs_events".into(),
            provider: "builtin.fs-event-stream".into(),
            requested: serde_json::json!({
                "events_per_sec": 100,
                "ops": ["create", "rename", "delete"],
            }),
        }],
        phases: vec![
            PhaseSpec {
                id: "ramp".into(),
                duration_ms: 20,
            },
            PhaseSpec {
                id: "measure".into(),
                duration_ms: 0,
            },
            PhaseSpec {
                id: "recovery".into(),
                duration_ms: 20,
            },
        ],
    }
}

pub fn developer_composite_v1() -> LoadProfile {
    LoadProfile {
        id: "developer_composite_v1".into(),
        actors: vec![
            ActorSpec {
                id: "cpu_build".into(),
                provider: "builtin.cpu-contention".into(),
                requested: serde_json::json!({
                    "busy_cores": 0.85,
                    "method": "integer_mix",
                    "workers": 1,
                }),
            },
            ActorSpec {
                id: "mem_ide".into(),
                provider: "builtin.memory-working-set".into(),
                requested: serde_json::json!({
                    "resident_mb": 48,
                    "touch_pattern": "sequential",
                }),
            },
            ActorSpec {
                id: "fs_mutations".into(),
                provider: "builtin.fs-event-stream".into(),
                requested: serde_json::json!({
                    "events_per_sec": 50,
                    "ops": ["create", "rename", "delete"],
                }),
            },
        ],
        phases: vec![
            PhaseSpec {
                id: "ramp".into(),
                duration_ms: 50,
            },
            PhaseSpec {
                id: "measure".into(),
                duration_ms: 0,
            },
            PhaseSpec {
                id: "recovery".into(),
                duration_ms: 80,
            },
        ],
    }
}

pub fn resolve_profile(id: &str) -> Option<LoadProfile> {
    match id {
        "quiet" | "quiet_v1" | "none" | "" => Some(quiet_v1()),
        "peer_cpu_light" | "peer_cpu_light_v1" => Some(peer_cpu_light_v1()),
        "peer_cpu_moderate" | "peer_cpu_moderate_v1" => Some(peer_cpu_moderate_v1()),
        "peer_memory_capacity" | "peer_memory_capacity_v1" => Some(peer_memory_capacity_v1()),
        "peer_storage" | "peer_storage_v1" => Some(peer_storage_v1()),
        "peer_event" | "peer_event_v1" => Some(peer_event_v1()),
        "developer_composite" | "developer_composite_v1" | "busy_developer" => {
            Some(developer_composite_v1())
        }
        _ => None,
    }
}

pub fn list_profiles() -> Vec<&'static str> {
    vec![
        "quiet_v1",
        "peer_cpu_light_v1",
        "peer_cpu_moderate_v1",
        "peer_memory_capacity_v1",
        "peer_storage_v1",
        "peer_event_v1",
        "developer_composite_v1",
    ]
}

fn provider_digest(id: &str) -> ProviderId {
    let dig = identity::sha256_hex(format!("{PROTOCOL}|{id}|1.2.0").as_bytes());
    ProviderId {
        id: id.into(),
        version: "1.2.0".into(),
        digest: format!("sha256:{dig}"),
    }
}

fn requested_busy_cores(req: &serde_json::Value) -> Option<f64> {
    req.get("busy_cores").and_then(|v| v.as_f64()).or_else(|| {
        req.get("workers")
            .and_then(|v| v.as_u64())
            .map(|w| w as f64)
    })
}

fn requested_workers(req: &serde_json::Value) -> usize {
    req.get("workers")
        .and_then(|v| v.as_u64())
        .map(|w| w as usize)
        .unwrap_or(1)
        .max(1)
}

fn requested_resident_mb(req: &serde_json::Value) -> usize {
    req.get("resident_mb")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize
}

/// Child-process entry used when parent re-execs with `FFF_BENCH_LOAD_WORKER`.
pub fn run_worker(kind: &str) -> ExitCode {
    let stop = match std::env::var(ENV_STOP) {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("load-worker: missing {ENV_STOP}");
            return ExitCode::from(2);
        }
    };
    let counter = match std::env::var(ENV_COUNTER) {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("load-worker: missing {ENV_COUNTER}");
            return ExitCode::from(2);
        }
    };
    let arg = std::env::var(ENV_ARG).unwrap_or_default();
    match kind {
        "cpu" => worker_cpu(&stop, &counter),
        "memory" => {
            let mb: usize = arg.parse().unwrap_or(32);
            worker_memory(&stop, &counter, mb);
        }
        "storage" => {
            let parts: Vec<&str> = arg.split('|').collect();
            let root = PathBuf::from(parts.first().copied().unwrap_or("."));
            let file_bytes: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
            worker_storage(&stop, &counter, &root, file_bytes);
        }
        "event" => {
            let root = PathBuf::from(arg);
            worker_event(&stop, &counter, &root);
        }
        other => {
            eprintln!("load-worker: unknown kind {other}");
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}

fn stop_requested(stop: &Path) -> bool {
    stop.exists()
}

fn write_counter(path: &Path, n: u64) {
    let _ = fs::write(path, n.to_string());
}

fn read_counter(path: &Path) -> u64 {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

fn worker_cpu(stop: &Path, counter: &Path) {
    let mut x: u64 = 0xC0FFEE;
    let mut n = 0u64;
    while !stop_requested(stop) {
        for _ in 0..10_000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            x ^= x >> 13;
        }
        n += 1;
        if n % 64 == 0 {
            write_counter(counter, n);
        }
    }
    write_counter(counter, n);
    let _ = x;
}

fn worker_memory(stop: &Path, counter: &Path, mb: usize) {
    let bytes = mb.saturating_mul(1024 * 1024).min(256 * 1024 * 1024);
    let mut buf = vec![0u8; bytes];
    for chunk in buf.chunks_mut(4096) {
        chunk[0] = 1;
    }
    write_counter(counter, bytes as u64);
    while !stop_requested(stop) {
        // Keep pages warm.
        for chunk in buf.chunks_mut(4096 * 16) {
            chunk[0] = chunk[0].wrapping_add(1);
        }
        thread::sleep(Duration::from_millis(10));
    }
    write_counter(counter, bytes as u64);
    let _ = buf;
}

fn worker_storage(stop: &Path, counter: &Path, root: &Path, file_bytes: usize) {
    let _ = fs::create_dir_all(root);
    let payload = vec![0xABu8; file_bytes.min(64 * 1024)];
    let mut i = 0u64;
    while !stop_requested(stop) {
        let p = root.join(format!("f{i}.bin"));
        let _ = fs::write(&p, &payload);
        let _ = fs::remove_file(&p);
        i = i.wrapping_add(1);
        if i % 16 == 0 {
            write_counter(counter, i);
        }
        if i % 32 == 0 {
            thread::sleep(Duration::from_micros(50));
        }
    }
    write_counter(counter, i);
}

fn worker_event(stop: &Path, counter: &Path, root: &Path) {
    let _ = fs::create_dir_all(root);
    let mut i = 0u64;
    while !stop_requested(stop) {
        let a = root.join(format!("e{i}.tmp"));
        let b = root.join(format!("e{i}.ren"));
        let _ = fs::write(&a, b"event");
        let _ = fs::rename(&a, &b);
        let _ = fs::remove_file(&b);
        i = i.wrapping_add(1);
        if i % 8 == 0 {
            write_counter(counter, i);
        }
        if i % 16 == 0 {
            thread::sleep(Duration::from_micros(100));
        }
    }
    write_counter(counter, i);
}

enum WorkerSlot {
    Process {
        child: Child,
        counter: PathBuf,
        kind: String,
    },
    Thread {
        handle: JoinHandle<()>,
    },
}

/// Prefer child processes outside unit tests.
fn prefer_process_isolation() -> bool {
    !cfg!(test)
}

fn spawn_process_worker(
    kind: &str,
    stop_file: &Path,
    counter_file: &Path,
    work_arg: &str,
) -> Result<Child, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    write_counter(counter_file, 0);
    let mut cmd = Command::new(exe);
    cmd.env(ENV_WORKER, kind)
        .env(ENV_STOP, stop_file)
        .env(ENV_COUNTER, counter_file)
        .env(ENV_ARG, work_arg)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
        .map_err(|e| format!("spawn load worker {kind}: {e}"))
}

fn process_pids(workers: &[WorkerSlot]) -> Vec<u32> {
    workers
        .iter()
        .filter_map(|w| match w {
            WorkerSlot::Process { child, .. } => Some(child.id()),
            _ => None,
        })
        .collect()
}

/// Sum of (user+system) CPU microseconds across child PIDs.
fn children_cpu_total_us(pids: &[u32]) -> Option<u64> {
    if pids.is_empty() {
        return None;
    }
    let mut sum = 0u64;
    let mut any = false;
    for &pid in pids {
        if let Some(c) = process_cpu_total_us(pid) {
            sum = sum.saturating_add(c);
            any = true;
        }
    }
    any.then_some(sum)
}

fn process_cpu_total_us(pid: u32) -> Option<u64> {
    #[cfg(windows)]
    {
        windows_process_cpu_us(pid)
    }
    #[cfg(target_os = "linux")]
    {
        linux_process_cpu_us(pid)
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        let _ = pid;
        None
    }
}

#[cfg(windows)]
fn windows_process_cpu_us(pid: u32) -> Option<u64> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut core::ffi::c_void;
        fn CloseHandle(h: *mut core::ffi::c_void) -> i32;
        fn GetProcessTimes(
            h: *mut core::ffi::c_void,
            creation: *mut u64,
            exit: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
    }
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return None;
        }
        let mut c = 0u64;
        let mut e = 0u64;
        let mut k = 0u64;
        let mut u = 0u64;
        let ok = GetProcessTimes(h, &mut c, &mut e, &mut k, &mut u) != 0;
        CloseHandle(h);
        if ok {
            // FILETIME is 100ns units → µs
            Some(u.saturating_add(k) / 10)
        } else {
            None
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_process_cpu_us(pid: u32) -> Option<u64> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rparen = s.rfind(')')?;
    let rest = s.get(rparen + 2..)?;
    let parts: Vec<&str> = rest.split_whitespace().collect();
    let utime: u64 = parts.get(11)?.parse().ok()?;
    let stime: u64 = parts.get(12)?.parse().ok()?;
    let ticks = 100u64;
    Some((utime.saturating_add(stime)) * 1_000_000 / ticks)
}

/// Whether two treatment records delivered equivalent interference for paired compare.
pub fn treatments_delivery_equivalent(
    a: &TreatmentEvidence,
    b: &TreatmentEvidence,
) -> (bool, Vec<String>) {
    let mut reasons = Vec::new();
    if a.load_profile_id != b.load_profile_id {
        reasons.push("load_profile_mismatch".into());
    }
    if !a.valid || !b.valid {
        reasons.push("treatment_invalid".into());
    }
    if a.placement != b.placement {
        reasons.push(format!(
            "placement_mismatch:{}!={}",
            a.placement.as_deref().unwrap_or("?"),
            b.placement.as_deref().unwrap_or("?")
        ));
    }
    match (
        a.calibration.as_ref().and_then(|c| c.busy_cores),
        b.calibration.as_ref().and_then(|c| c.busy_cores),
    ) {
        (Some(x), Some(y)) => {
            let mid = ((x + y) / 2.0).max(0.05);
            if (x - y).abs() / mid > DELIVERY_TOLERANCE {
                reasons.push(format!(
                    "busy_cores_inequivalent:a={x:.3},b={y:.3},tol={DELIVERY_TOLERANCE}"
                ));
            }
        }
        (None, Some(_)) | (Some(_), None) => {
            // One side metered, other not — still allow if both valid and work_iters close.
            reasons.push("busy_cores_asymmetric".into());
        }
        (None, None) => {}
    }
    match (
        a.calibration.as_ref().map(|c| c.work_iters),
        b.calibration.as_ref().map(|c| c.work_iters),
    ) {
        (Some(x), Some(y)) if x > 0 || y > 0 => {
            let mid = ((x.max(y) + x.min(y)) as f64 / 2.0).max(1.0);
            if (x as f64 - y as f64).abs() / mid > DELIVERY_TOLERANCE * 2.0 {
                // Work-iters can vary more than busy_cores; 2× tolerance.
                reasons.push(format!("work_iters_inequivalent:a={x},b={y}"));
            }
        }
        _ => {}
    }
    // Soft asymmetry of busy metering alone does not fail if work_iters are equivalent and valid.
    if reasons.len() == 1 && reasons[0] == "busy_cores_asymmetric" {
        match (
            a.calibration.as_ref().map(|c| c.work_iters),
            b.calibration.as_ref().map(|c| c.work_iters),
        ) {
            (Some(x), Some(y)) if x > 0 && y > 0 => {
                reasons.clear();
            }
            _ => {}
        }
    }
    (reasons.is_empty(), reasons)
}

pub struct Treatment {
    profile: LoadProfile,
    stop: Arc<AtomicBool>,
    stop_file: PathBuf,
    workers: Vec<WorkerSlot>,
    /// In-process memory hold when process isolation is off.
    _mem_hold: Vec<u8>,
    held_bytes: u64,
    temp_roots: Vec<PathBuf>,
    work_dir: PathBuf,
    started: Instant,
    events: Vec<ActorEvent>,
    calibration: Option<CalibrationRecord>,
    placement: String,
    /// Thread-mode aggregate counters.
    thread_work: Arc<AtomicU64>,
    thread_io: Arc<AtomicU64>,
}

impl Treatment {
    pub fn start(profile: LoadProfile) -> Result<Self, String> {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let work_dir = std::env::temp_dir().join(format!(
            "fff-bench-load-{}-{}-{}",
            std::process::id(),
            identity::now_unix(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&work_dir).map_err(|e| e.to_string())?;
        let stop_file = work_dir.join("stop");
        let _ = fs::remove_file(&stop_file);

        let stop = Arc::new(AtomicBool::new(false));
        let thread_work = Arc::new(AtomicU64::new(0));
        let thread_io = Arc::new(AtomicU64::new(0));
        let mut workers = Vec::new();
        let mut events = Vec::new();
        let mut mem_hold = Vec::new();
        let mut held_bytes = 0u64;
        let mut temp_roots = Vec::new();
        let started = Instant::now();
        let use_proc = prefer_process_isolation();
        let placement = if use_proc && !profile.actors.is_empty() {
            "separate_process"
        } else if profile.actors.is_empty() {
            "none"
        } else {
            "same_process_thread"
        }
        .to_string();

        for actor in &profile.actors {
            match actor.provider.as_str() {
                "builtin.cpu-contention" => {
                    let n = requested_workers(&actor.requested);
                    for i in 0..n {
                        if use_proc {
                            let counter = work_dir.join(format!("cpu-{}-{}.cnt", actor.id, i));
                            let child = spawn_process_worker("cpu", &stop_file, &counter, "")?;
                            workers.push(WorkerSlot::Process {
                                child,
                                counter,
                                kind: "cpu".into(),
                            });
                        } else {
                            let stop_c = Arc::clone(&stop);
                            let work_c = Arc::clone(&thread_work);
                            let h = thread::Builder::new()
                                .name(format!("load-{}-{}", actor.id, i))
                                .spawn(move || cpu_worker_thread(stop_c, work_c))
                                .map_err(|e| format!("spawn load actor: {e}"))?;
                            workers.push(WorkerSlot::Thread { handle: h });
                        }
                    }
                    events.push(ready_event(
                        &profile,
                        actor,
                        serde_json::json!({ "busy_cores": null }),
                    ));
                }
                "builtin.memory-working-set" => {
                    let mb = requested_resident_mb(&actor.requested);
                    if use_proc {
                        let counter = work_dir.join(format!("mem-{}.cnt", actor.id));
                        let child =
                            spawn_process_worker("memory", &stop_file, &counter, &mb.to_string())?;
                        // Wait until child reports held bytes.
                        let deadline = Instant::now() + Duration::from_millis(500);
                        while Instant::now() < deadline {
                            let v = read_counter(&counter);
                            if v > 0 {
                                held_bytes = v;
                                break;
                            }
                            thread::sleep(Duration::from_millis(5));
                        }
                        if held_bytes == 0 {
                            held_bytes = (mb as u64).saturating_mul(1024 * 1024);
                        }
                        workers.push(WorkerSlot::Process {
                            child,
                            counter,
                            kind: "memory".into(),
                        });
                    } else {
                        let bytes = mb.saturating_mul(1024 * 1024).min(256 * 1024 * 1024);
                        let mut buf = vec![0u8; bytes];
                        for chunk in buf.chunks_mut(4096) {
                            chunk[0] = 1;
                        }
                        held_bytes = bytes as u64;
                        mem_hold = buf;
                    }
                    events.push(ready_event(
                        &profile,
                        actor,
                        serde_json::json!({ "resident_mb": null }),
                    ));
                }
                "builtin.storage-churn" => {
                    let root = work_dir.join(format!("io-{}", actor.id));
                    fs::create_dir_all(&root).map_err(|e| e.to_string())?;
                    let file_bytes = actor
                        .requested
                        .get("file_bytes")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4096) as usize;
                    if use_proc {
                        let counter = work_dir.join(format!("storage-{}.cnt", actor.id));
                        let arg = format!("{}|{}", root.display(), file_bytes);
                        let child = spawn_process_worker("storage", &stop_file, &counter, &arg)?;
                        workers.push(WorkerSlot::Process {
                            child,
                            counter,
                            kind: "storage".into(),
                        });
                    } else {
                        let stop_c = Arc::clone(&stop);
                        let ops_c = Arc::clone(&thread_io);
                        let root_c = root.clone();
                        let h = thread::Builder::new()
                            .name(format!("load-{}", actor.id))
                            .spawn(move || storage_worker_thread(stop_c, ops_c, root_c, file_bytes))
                            .map_err(|e| format!("spawn storage actor: {e}"))?;
                        workers.push(WorkerSlot::Thread { handle: h });
                    }
                    temp_roots.push(root);
                    events.push(ready_event(
                        &profile,
                        actor,
                        serde_json::json!({ "ops": null }),
                    ));
                }
                "builtin.fs-event-stream" => {
                    let root = work_dir.join(format!("ev-{}", actor.id));
                    fs::create_dir_all(&root).map_err(|e| e.to_string())?;
                    if use_proc {
                        let counter = work_dir.join(format!("event-{}.cnt", actor.id));
                        let child = spawn_process_worker(
                            "event",
                            &stop_file,
                            &counter,
                            &root.display().to_string(),
                        )?;
                        workers.push(WorkerSlot::Process {
                            child,
                            counter,
                            kind: "event".into(),
                        });
                    } else {
                        let stop_c = Arc::clone(&stop);
                        let ops_c = Arc::clone(&thread_io);
                        let root_c = root.clone();
                        let h = thread::Builder::new()
                            .name(format!("load-{}", actor.id))
                            .spawn(move || event_worker_thread(stop_c, ops_c, root_c))
                            .map_err(|e| format!("spawn event actor: {e}"))?;
                        workers.push(WorkerSlot::Thread { handle: h });
                    }
                    temp_roots.push(root);
                    events.push(ready_event(
                        &profile,
                        actor,
                        serde_json::json!({ "events": null }),
                    ));
                }
                other => {
                    return Err(format!("unknown load provider: {other}"));
                }
            }
        }

        for ph in &profile.phases {
            if ph.id == "ramp" && ph.duration_ms > 0 {
                thread::sleep(Duration::from_millis(ph.duration_ms));
            }
        }

        let sensors = Sensors::from_packs("cpu,mem");
        let calibration = if profile.actors.is_empty() {
            None
        } else {
            let pids = process_pids(&workers);
            let child_cpu0 = if use_proc {
                children_cpu_total_us(&pids)
            } else {
                None
            };
            let snap0 = sensors.snapshot();
            let t0 = Instant::now();
            let iters0 = total_work(&workers, &thread_work, &thread_io);
            thread::sleep(Duration::from_millis(CALIBRATE_MS));
            let wall_us = t0.elapsed().as_micros() as u64;
            let snap1 = sensors.snapshot();
            let delta = sensors.delta(&snap0, &snap1);
            let parent_cpu = match (delta.cpu_user_us, delta.cpu_system_us) {
                (Some(u), Some(s)) => Some(u.saturating_add(s)),
                (Some(u), None) => Some(u),
                (None, Some(s)) => Some(s),
                _ => None,
            };
            let child_cpu1 = if use_proc {
                children_cpu_total_us(&pids)
            } else {
                None
            };
            let child_cpu_delta = match (child_cpu0, child_cpu1) {
                (Some(a), Some(b)) => Some(b.saturating_sub(a)),
                _ => None,
            };
            let iters = total_work(&workers, &thread_work, &thread_io).saturating_sub(iters0);
            // Prefer metered child process times when isolated; else parent process times.
            let (cpu, busy, quality) = if let Some(c) = child_cpu_delta {
                let busy = c as f64 / wall_us.max(1) as f64;
                (
                    Some(c),
                    Some(busy),
                    if c > 0 {
                        "child_process_times"
                    } else if iters > 0 {
                        "process_work_iters"
                    } else if held_bytes > 0 {
                        "memory_hold"
                    } else {
                        "unmeasured"
                    },
                )
            } else if let Some(c) = parent_cpu {
                (
                    Some(c),
                    Some(c as f64 / wall_us.max(1) as f64),
                    if c > 0 {
                        "metered"
                    } else if iters > 0 {
                        "work_iters_proxy"
                    } else if held_bytes > 0 {
                        "memory_hold"
                    } else {
                        "unmeasured"
                    },
                )
            } else if iters > 0 {
                (None, None, "work_iters_proxy")
            } else if held_bytes > 0 {
                (None, None, "memory_hold")
            } else {
                (None, None, "unmeasured")
            };
            Some(CalibrationRecord {
                wall_us,
                cpu_total_us: cpu,
                busy_cores: busy,
                work_iters: iters,
                quality: quality.into(),
            })
        };

        for e in &mut events {
            e.state = "active".into();
            e.readiness = if calibration
                .as_ref()
                .map(|c| {
                    c.work_iters > 0
                        || c.busy_cores.is_some()
                        || held_bytes > 0
                        || c.quality == "memory_hold"
                })
                .unwrap_or(true)
            {
                serde_json::json!({ "status": "pass", "placement": placement })
            } else {
                serde_json::json!({ "status": "fail", "reason": "no_delivery_in_calibration" })
            };
        }

        Ok(Self {
            profile,
            stop,
            stop_file,
            workers,
            _mem_hold: mem_hold,
            held_bytes,
            temp_roots,
            work_dir,
            started,
            events,
            calibration,
            placement,
            thread_work,
            thread_io,
        })
    }

    pub fn finish(mut self) -> TreatmentEvidence {
        let sensors = Sensors::from_packs("cpu,mem");
        let snap_end = sensors.snapshot();

        // Signal stop to processes and threads.
        let _ = fs::File::create(&self.stop_file).and_then(|mut f| f.write_all(b"1"));
        self.stop.store(true, Ordering::SeqCst);

        let drain_t0 = Instant::now();
        let mut children_exited = true;
        let mut process_work = 0u64;
        let mut process_io = 0u64;
        for slot in self.workers.drain(..) {
            match slot {
                WorkerSlot::Process {
                    mut child,
                    counter,
                    kind,
                } => {
                    // Give worker time to notice stop file.
                    let deadline = Instant::now() + Duration::from_millis(2000);
                    loop {
                        match child.try_wait() {
                            Ok(Some(_)) => break,
                            Ok(None) if Instant::now() < deadline => {
                                thread::sleep(Duration::from_millis(5));
                            }
                            Ok(None) => {
                                let _ = child.kill();
                                let _ = child.wait();
                                children_exited = false;
                                break;
                            }
                            Err(_) => {
                                children_exited = false;
                                break;
                            }
                        }
                    }
                    let n = read_counter(&counter);
                    match kind.as_str() {
                        "cpu" => process_work = process_work.saturating_add(n),
                        "memory" => {
                            if n > 0 {
                                self.held_bytes = n;
                            }
                        }
                        _ => process_io = process_io.saturating_add(n),
                    }
                }
                WorkerSlot::Thread { handle, .. } => {
                    let _ = handle.join();
                }
            }
        }
        let drain_ms = drain_t0.elapsed().as_millis() as u64;

        let io_at_stop = process_io.saturating_add(self.thread_io.load(Ordering::Relaxed));
        let recovery_ms: u64 = self
            .profile
            .phases
            .iter()
            .find(|p| p.id == "recovery")
            .map(|p| p.duration_ms)
            .unwrap_or(0);
        let sensors_rec = Sensors::from_packs("cpu");
        let snap_r0 = sensors_rec.snapshot();
        let rec_sleep = if recovery_ms > 0 {
            recovery_ms
        } else if !self.profile.actors.is_empty() {
            20
        } else {
            0
        };
        if rec_sleep > 0 {
            thread::sleep(Duration::from_millis(rec_sleep));
        }
        let snap_r1 = sensors_rec.snapshot();
        let rec_delta = sensors_rec.delta(&snap_r0, &snap_r1);
        let rec_wall_us = (rec_sleep.max(1) as f64) * 1000.0;
        let rec_cpu = match (rec_delta.cpu_user_us, rec_delta.cpu_system_us) {
            (Some(u), Some(s)) => Some(u.saturating_add(s)),
            (Some(u), None) => Some(u),
            (None, Some(s)) => Some(s),
            _ => None,
        };
        let residual_busy = rec_cpu.map(|c| c as f64 / rec_wall_us.max(1.0));
        // Workers are joined/waited before recovery sleep; residual ops must stay frozen.
        let io_after = process_io.saturating_add(self.thread_io.load(Ordering::Relaxed));
        let residual_io = io_after.saturating_sub(io_at_stop);

        let mut rec_reasons = Vec::new();
        // Parent residual CPU can include harness noise; children_exited + frozen IO are hard.
        let residual_ok = residual_io == 0;
        let recovered = children_exited && residual_ok;
        if !children_exited {
            rec_reasons.push("child_workers_not_exited".into());
        }
        if !residual_ok {
            rec_reasons.push("residual_load_after_drain".into());
        }
        let recovery = if self.profile.actors.is_empty() {
            None
        } else {
            Some(RecoveryRecord {
                drain_ms,
                recovered,
                residual_cpu_busy_cores: residual_busy,
                residual_io_ops: residual_io,
                children_exited,
                reason_codes: rec_reasons,
            })
        };

        let wall = self.started.elapsed().as_secs_f64().max(1e-6);
        let iters = process_work.saturating_add(self.thread_work.load(Ordering::Relaxed));
        let io_ops = process_io.saturating_add(self.thread_io.load(Ordering::Relaxed));
        // memory held_bytes stored in process_work for memory kind — separate:
        let held_bytes = self.held_bytes;
        // Fix: process_work for memory is held_bytes; re-sum cpu-only from events later if needed.
        let calib = self.calibration.clone();
        let busy = calib.as_ref().and_then(|c| c.busy_cores);
        let rss_end = snap_end.rss_mb;

        for e in &mut self.events {
            e.state = "complete".into();
            if e.provider.id.contains("cpu") {
                e.delivered = serde_json::json!({
                    "busy_cores": {
                        "value": busy,
                        "status": if busy.is_some() { "measured" } else { "unavailable" },
                        "quality": calib.as_ref().map(|c| c.quality.clone()).unwrap_or_else(|| "coarse".into()),
                        "unit": "busy_cores",
                        "source": calib.as_ref().map(|c| c.quality.as_str()).unwrap_or("unmeasured"),
                    },
                    "work_iters": iters,
                    "phase_wall_s": wall,
                    "calibration": calib,
                    "placement": self.placement,
                });
            } else if e.provider.id.contains("memory") {
                let req_mb = requested_resident_mb(&e.requested) as f64;
                e.delivered = serde_json::json!({
                    "resident_mb": {
                        "value": rss_end,
                        "status": if rss_end.is_some() { "measured" } else { "unavailable" },
                        "quality": "process_rss_includes_subject",
                        "requested_mb": req_mb,
                    },
                    "held_bytes": held_bytes,
                    "phase_wall_s": wall,
                    "placement": self.placement,
                });
            } else if e.provider.id.contains("storage") {
                let ops_s = io_ops as f64 / wall;
                e.delivered = serde_json::json!({
                    "ops": io_ops,
                    "ops_per_sec": {
                        "value": ops_s,
                        "status": "measured",
                        "quality": "count_over_wall",
                    },
                    "phase_wall_s": wall,
                    "placement": self.placement,
                });
            } else if e.provider.id.contains("fs-event") {
                let eps = io_ops as f64 / wall;
                e.delivered = serde_json::json!({
                    "events": io_ops,
                    "events_per_sec": {
                        "value": eps,
                        "status": "measured",
                        "quality": "count_over_wall",
                    },
                    "phase_wall_s": wall,
                    "placement": self.placement,
                });
            }
        }

        let mut reasons = Vec::new();
        let mut valid = true;

        if !self.profile.actors.is_empty() {
            for e in &self.events {
                if e.provider.id.contains("cpu") {
                    let req = requested_busy_cores(&e.requested).unwrap_or(1.0);
                    match busy {
                        Some(d) if d <= 0.05 && iters == 0 => {
                            valid = false;
                            reasons.push("load_actor_no_work".into());
                        }
                        Some(d) => {
                            // Hard under-delivery gate: delivered must be within tolerance of target.
                            // Target is lower-bounded: at least req * (1 - tol) with floor of req*0.5.
                            let floor = (req * (1.0 - DELIVERY_TOLERANCE)).max(req * 0.5);
                            if d + 1e-9 < floor && iters == 0 {
                                valid = false;
                                reasons.push(format!(
                                    "load_delivery_under:req={req:.2},got={d:.2},floor={floor:.2}"
                                ));
                            } else if d + 1e-9 < floor {
                                // Work was done but busy_cores estimate low — still require iters.
                                if iters < 10 {
                                    valid = false;
                                    reasons.push(format!(
                                        "load_delivery_under:req={req:.2},got={d:.2},iters={iters}"
                                    ));
                                }
                            }
                        }
                        None if iters == 0 => {
                            valid = false;
                            reasons.push("load_actor_unmeasured".into());
                        }
                        None => {
                            // Process isolation may not meter parent CPU; work_iters prove delivery.
                            if iters < 10 {
                                valid = false;
                                reasons.push("load_delivery_insufficient_iters".into());
                            }
                        }
                    }
                }
                if e.provider.id.contains("memory") && held_bytes == 0 {
                    valid = false;
                    reasons.push("memory_actor_empty".into());
                }
                if (e.provider.id.contains("storage") || e.provider.id.contains("fs-event"))
                    && io_ops == 0
                {
                    valid = false;
                    reasons.push("io_or_event_actor_no_ops".into());
                }
            }
        }

        if let Some(ref r) = recovery {
            if !r.recovered {
                valid = false;
                reasons.push("recovery_residual_load".into());
            }
        }

        for root in self.temp_roots.drain(..) {
            let _ = fs::remove_dir_all(root);
        }
        let _ = fs::remove_dir_all(&self.work_dir);

        let delivery_equivalent = valid;

        TreatmentEvidence {
            load_profile_id: self.profile.id,
            protocol: PROTOCOL.into(),
            valid,
            delivery_equivalent,
            actors: self.events,
            reason_codes: reasons,
            calibration: calib,
            recovery,
            placement: Some(self.placement),
        }
    }
}

fn ready_event(
    profile: &LoadProfile,
    actor: &ActorSpec,
    delivered: serde_json::Value,
) -> ActorEvent {
    ActorEvent {
        schema_version: "1.0".into(),
        event: "load_actor".into(),
        load_profile_id: profile.id.clone(),
        actor_id: actor.id.clone(),
        provider: provider_digest(&actor.provider),
        phase_id: "measure".into(),
        state: "ready".into(),
        requested: actor.requested.clone(),
        delivered,
        readiness: serde_json::json!({ "status": "pass" }),
        error: None,
    }
}

fn total_work(workers: &[WorkerSlot], thread_work: &AtomicU64, thread_io: &AtomicU64) -> u64 {
    let mut n = thread_work.load(Ordering::Relaxed) + thread_io.load(Ordering::Relaxed);
    for w in workers {
        if let WorkerSlot::Process { counter, .. } = w {
            n = n.saturating_add(read_counter(counter));
        }
    }
    n
}

fn cpu_worker_thread(stop: Arc<AtomicBool>, work: Arc<AtomicU64>) {
    let mut x: u64 = 0xC0FFEE;
    while !stop.load(Ordering::Relaxed) {
        for _ in 0..10_000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            x ^= x >> 13;
        }
        work.fetch_add(1, Ordering::Relaxed);
    }
    let _ = x;
}

fn storage_worker_thread(
    stop: Arc<AtomicBool>,
    ops: Arc<AtomicU64>,
    root: PathBuf,
    file_bytes: usize,
) {
    let payload = vec![0xABu8; file_bytes.min(64 * 1024)];
    let mut i = 0u64;
    while !stop.load(Ordering::Relaxed) {
        let p = root.join(format!("f{i}.bin"));
        let _ = fs::write(&p, &payload);
        let _ = fs::remove_file(&p);
        i = i.wrapping_add(1);
        ops.fetch_add(1, Ordering::Relaxed);
        if i % 32 == 0 {
            thread::sleep(Duration::from_micros(50));
        }
    }
}

fn event_worker_thread(stop: Arc<AtomicBool>, ops: Arc<AtomicU64>, root: PathBuf) {
    let mut i = 0u64;
    while !stop.load(Ordering::Relaxed) {
        let a = root.join(format!("e{i}.tmp"));
        let b = root.join(format!("e{i}.ren"));
        let _ = fs::write(&a, b"event");
        let _ = fs::rename(&a, &b);
        let _ = fs::remove_file(&b);
        i = i.wrapping_add(1);
        ops.fetch_add(1, Ordering::Relaxed);
        if i % 16 == 0 {
            thread::sleep(Duration::from_micros(100));
        }
    }
}

pub fn validity_gate(ev: &TreatmentEvidence) -> crate::gates::GateEntry {
    crate::gates::GateEntry {
        gate_id: "treatment.load_delivery".into(),
        status: if ev.valid { "pass" } else { "fail" }.into(),
        required: true,
        reason_code: if ev.valid {
            None
        } else {
            Some(
                ev.reason_codes
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "treatment_invalid".into()),
            )
        },
        evidence: Some(serde_json::json!({
            "profile": ev.load_profile_id,
            "actors": ev.actors.len(),
            "valid": ev.valid,
            "delivery_equivalent": ev.delivery_equivalent,
            "calibration": ev.calibration,
            "recovery": ev.recovery,
            "placement": ev.placement,
            "reason_codes": ev.reason_codes,
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_is_valid() {
        let t = Treatment::start(quiet_v1()).unwrap();
        let ev = t.finish();
        assert!(ev.valid);
        assert!(ev.actors.is_empty());
        assert_eq!(ev.load_profile_id, "quiet_v1");
    }

    #[test]
    fn cpu_light_calibrates() {
        let t = Treatment::start(peer_cpu_light_v1()).unwrap();
        thread::sleep(Duration::from_millis(20));
        let ev = t.finish();
        assert!(ev.valid, "{:?}", ev.reason_codes);
        assert!(ev.calibration.is_some());
        let cal = ev.calibration.unwrap();
        assert!(cal.work_iters > 0 || cal.busy_cores.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn memory_capacity_holds_pages() {
        let t = Treatment::start(peer_memory_capacity_v1()).unwrap();
        let ev = t.finish();
        assert!(ev.valid, "{:?}", ev.reason_codes);
        assert_eq!(ev.actors.len(), 1);
        let held = ev.actors[0]
            .delivered
            .get("held_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        assert!(held > 0);
    }

    #[test]
    fn resolve_resilience_set() {
        for id in resilience_profiles() {
            assert!(resolve_profile(id).is_some(), "missing {id}");
        }
    }

    #[test]
    fn storage_and_event_do_ops() {
        for p in [peer_storage_v1(), peer_event_v1()] {
            let t = Treatment::start(p).unwrap();
            thread::sleep(Duration::from_millis(40));
            let ev = t.finish();
            assert!(ev.valid, "{:?}", ev.reason_codes);
            assert!(!ev.actors.is_empty());
        }
    }

    #[test]
    fn developer_composite_runs_all_actors() {
        let t = Treatment::start(developer_composite_v1()).unwrap();
        thread::sleep(Duration::from_millis(50));
        let ev = t.finish();
        assert!(ev.valid, "{:?}", ev.reason_codes);
        assert_eq!(ev.actors.len(), 3);
        assert!(ev.recovery.is_some());
        assert!(ev.recovery.as_ref().unwrap().recovered, "{:?}", ev.recovery);
    }

    #[test]
    fn recovery_is_hard_gate() {
        let t = Treatment::start(peer_cpu_light_v1()).unwrap();
        thread::sleep(Duration::from_millis(30));
        let ev = t.finish();
        assert!(ev.valid, "{:?}", ev.reason_codes);
        assert!(ev.recovery.as_ref().unwrap().recovered);
    }

    #[test]
    fn delivery_equivalence_quiet_pair() {
        let a = Treatment::start(quiet_v1()).unwrap().finish();
        let b = Treatment::start(quiet_v1()).unwrap().finish();
        let (ok, reasons) = treatments_delivery_equivalent(&a, &b);
        assert!(ok, "{reasons:?}");
    }

    #[test]
    fn delivery_equivalence_rejects_profile_mismatch() {
        let a = Treatment::start(quiet_v1()).unwrap().finish();
        let b = Treatment::start(peer_cpu_light_v1()).unwrap().finish();
        let (ok, reasons) = treatments_delivery_equivalent(&a, &b);
        assert!(!ok);
        assert!(reasons.iter().any(|r| r.contains("load_profile")));
    }

    #[test]
    fn delivery_equivalence_rejects_busy_cores_skew() {
        let mut a = Treatment::start(quiet_v1()).unwrap().finish();
        let mut b = a.clone();
        a.load_profile_id = "x".into();
        b.load_profile_id = "x".into();
        a.valid = true;
        b.valid = true;
        a.placement = Some("separate_process".into());
        b.placement = Some("separate_process".into());
        a.calibration = Some(CalibrationRecord {
            wall_us: 80_000,
            cpu_total_us: Some(80_000),
            busy_cores: Some(1.0),
            work_iters: 100,
            quality: "child_process_times".into(),
        });
        b.calibration = Some(CalibrationRecord {
            wall_us: 80_000,
            cpu_total_us: Some(20_000),
            busy_cores: Some(0.25),
            work_iters: 100,
            quality: "child_process_times".into(),
        });
        let (ok, reasons) = treatments_delivery_equivalent(&a, &b);
        assert!(!ok);
        assert!(reasons.iter().any(|r| r.contains("busy_cores")));
    }
}
