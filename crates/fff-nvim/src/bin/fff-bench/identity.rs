//! Run / subject / machine identity.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub run_id: String,
    pub cmdline: String,
    pub started_at_unix: u64,
    pub finished_at_unix: Option<u64>,
    pub duration_ms: Option<u64>,
    pub log_dir: Option<String>,
    pub peak: bool,
    pub sample_packs: String,
    pub primary_span: String,
    pub suite: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectMeta {
    pub adapter_id: String,
    pub git_commit: Option<String>,
    pub git_dirty: bool,
    pub version: String,
    pub cargo_profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessMeta {
    pub name: String,
    pub git_commit: Option<String>,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineMeta {
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_ram_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoMeta {
    pub id: String,
    pub path: String,
    pub role: Vec<String>,
    pub optional: bool,
    pub file_count_live: Option<usize>,
    pub path_list_hash: Option<String>,
    pub git_commit: Option<String>,
    pub git_dirty: bool,
    pub pin: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsMeta {
    pub warmup: usize,
    pub iterations: usize,
    pub threads: usize,
    pub rng_seed: u64,
    pub primary_span: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeMeta {
    pub kind: String,
    pub base_ref: Option<String>,
    pub head_commit: Option<String>,
    pub dirty: bool,
    pub patch_path: Option<String>,
    pub patch_sha256: Option<String>,
    pub patch_bytes: Option<u64>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseMeta {
    pub integrity: String,
    pub host_cpu_busy_pct: Option<f64>,
    pub background_cpu_pct: Option<f64>,
    pub iter_wall_cv: Option<f64>,
    pub policy: String,
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn git_rev(path: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["-C", &path.to_string_lossy(), "rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        // fallback: .git/HEAD
        let head = std::fs::read_to_string(path.join(".git/HEAD")).ok()?;
        if let Some(sha) = head.strip_prefix("ref: ") {
            let ref_path = path.join(".git").join(sha.trim());
            return std::fs::read_to_string(ref_path)
                .ok()
                .map(|s| s.trim().to_string());
        }
        return Some(head.trim().to_string());
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn git_dirty(path: &Path) -> bool {
    let out = Command::new("git")
        .args(["-C", &path.to_string_lossy(), "status", "--porcelain"])
        .output();
    match out {
        Ok(o) if o.status.success() => !o.stdout.is_empty(),
        _ => false,
    }
}

pub fn git_diff(path: &Path, base: Option<&str>) -> Option<String> {
    let mut args = vec!["-C".to_string(), path.to_string_lossy().to_string()];
    if let Some(b) = base {
        args.push("diff".into());
        args.push(b.into());
    } else {
        args.push("diff".into());
        args.push("HEAD".into());
    }
    let out = Command::new("git").args(&args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

pub fn hostname() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".into())
}

pub fn machine_meta() -> MachineMeta {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total_ram_mb = sys.total_memory() / (1024 * 1024);

    MachineMeta {
        hostname: hostname(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpu_count: num_cpus::get(),
        total_ram_mb,
    }
}

pub fn subject_meta(adapter_id: &str) -> SubjectMeta {
    let cwd = std::env::current_dir().unwrap_or_default();
    SubjectMeta {
        adapter_id: adapter_id.into(),
        git_commit: git_rev(&cwd),
        git_dirty: git_dirty(&cwd),
        version: env!("CARGO_PKG_VERSION").into(),
        cargo_profile: if cfg!(debug_assertions) {
            "debug".into()
        } else {
            "release".into()
        },
    }
}

pub fn harness_meta() -> HarnessMeta {
    let cwd = std::env::current_dir().unwrap_or_default();
    HarnessMeta {
        name: "fff-bench".into(),
        git_commit: git_rev(&cwd),
        version: env!("CARGO_PKG_VERSION").into(),
    }
}

pub fn path_list_hash(paths: &[String]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut sorted: Vec<String> = paths.iter().map(|p| p.replace('\\', "/")).collect();
    sorted.sort_unstable();
    sorted.dedup();
    for p in sorted {
        hasher.update(p.as_bytes());
        hasher.update([0]);
    }
    hex::encode(hasher.finalize())
}

pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(data))
}

pub fn make_run_id(subject: &SubjectMeta) -> String {
    let short = subject
        .git_commit
        .as_deref()
        .map(|c| &c[..c.len().min(7)])
        .unwrap_or("unknown");
    format!(
        "{}-{}-{}",
        short,
        now_unix(),
        hostname().chars().take(12).collect::<String>()
    )
}
