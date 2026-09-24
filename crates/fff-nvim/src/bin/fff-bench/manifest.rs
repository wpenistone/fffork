//! Bench manifest: multi-repo sources, roles, cache, suite defaults.

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    /// Default suite when not overridden by CLI (legacy).
    #[serde(default)]
    pub suite: Option<String>,
    /// Pack workflow roles for `check` (OPERATIONS / FFF_REFERENCE_PACK).
    #[serde(default)]
    pub workflow: WorkflowManifest,
    #[serde(default)]
    pub cache_dir: Option<String>,
    /// Corpora entries (canonical name). `repos` is accepted as an alias.
    #[serde(default, alias = "corpora")]
    pub repos: Vec<RepoEntry>,
    #[serde(default)]
    pub adapter: AdapterManifest,
    #[serde(default)]
    pub sample_packs: Option<String>,
    #[serde(default)]
    pub history: Option<HistoryManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowManifest {
    #[serde(default = "default_screen_suite")]
    pub screen_suite: String,
    #[serde(default = "default_confirm_suite")]
    pub confirm_suite: String,
    #[serde(default = "default_diagnose_suite")]
    pub diagnose_suite: String,
}

impl Default for WorkflowManifest {
    fn default() -> Self {
        Self {
            screen_suite: default_screen_suite(),
            confirm_suite: default_confirm_suite(),
            diagnose_suite: default_diagnose_suite(),
        }
    }
}

fn default_screen_suite() -> String {
    "agent-screen".into()
}
fn default_confirm_suite() -> String {
    "agent-confirm".into()
}
fn default_diagnose_suite() -> String {
    "peak-diagnose".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistoryManifest {
    #[serde(default)]
    pub directory: Option<String>,
    #[serde(default = "default_true")]
    pub record: bool,
}

fn default_schema() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdapterManifest {
    /// Build content index (slower open, richer filter).
    #[serde(default)]
    pub enable_content_indexing: bool,
    #[serde(default)]
    pub enable_mmap_cache: bool,
    #[serde(default = "default_true")]
    pub score_breakdown: bool,
    #[serde(default = "default_top_k")]
    pub top_k_log: usize,
}

fn default_true() -> bool {
    true
}
fn default_top_k() -> usize {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoEntry {
    pub id: String,
    #[serde(default)]
    pub role: Vec<String>,
    #[serde(default)]
    pub optional: bool,
    pub source: RepoSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RepoSource {
    Path {
        path: String,
    },
    Fixture {
        fixture_id: String,
        #[serde(default)]
        path: Option<String>,
    },
    GitClone {
        url: String,
        commit: String,
        #[serde(default)]
        dest: Option<String>,
        #[serde(default)]
        depth: Option<u32>,
    },
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = fs::read_to_string(path).map_err(|e| format!("read manifest: {e}"))?;
        let expanded = expand_env(&raw);
        serde_json::from_str(&expanded).map_err(|e| format!("parse manifest: {e}"))
    }

    pub fn cache_dir_resolved(&self) -> PathBuf {
        if let Some(ref c) = self.cache_dir {
            return PathBuf::from(expand_env(c));
        }
        if let Ok(c) = env::var("FFF_BENCH_CACHE") {
            return PathBuf::from(c);
        }
        dirs_cache().join("fff-bench").join("corpora")
    }

    pub fn repos_for_role(&self, role: &str) -> Vec<&RepoEntry> {
        self.repos
            .iter()
            .filter(|r| r.role.is_empty() || r.role.iter().any(|x| x == role || x == "all"))
            .collect()
    }
}

fn dirs_cache() -> PathBuf {
    env::var("LOCALAPPDATA")
        .or_else(|_| env::var("XDG_CACHE_HOME"))
        .or_else(|_| env::var("HOME").map(|h| format!("{h}/.cache")))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".fff-bench-cache"))
}

/// Expand `${VAR}` and `$VAR` in strings.
pub fn expand_env(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() {
            if chars[i + 1] == '{' {
                if let Some(end) = chars[i + 2..].iter().position(|c| *c == '}') {
                    let name: String = chars[i + 2..i + 2 + end].iter().collect();
                    if let Ok(v) = env::var(&name) {
                        out.push_str(&v);
                    } else {
                        out.push_str(&format!("${{{name}}}"));
                    }
                    i += 3 + end;
                    continue;
                }
            } else {
                let start = i + 1;
                let mut j = start;
                while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                    j += 1;
                }
                if j > start {
                    let name: String = chars[start..j].iter().collect();
                    if let Ok(v) = env::var(&name) {
                        out.push_str(&v);
                    } else {
                        out.push('$');
                        out.push_str(&name);
                    }
                    i = j;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct SetupResult {
    pub ok: bool,
    pub repos: Vec<SetupRepoStatus>,
    pub cache_dir: String,
    pub messages: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SetupRepoStatus {
    pub id: String,
    pub ready: bool,
    pub path: Option<String>,
    pub git_commit: Option<String>,
    pub action: String,
    pub error: Option<String>,
}

pub fn setup_manifest(m: &Manifest, fetch: bool) -> SetupResult {
    let cache = m.cache_dir_resolved();
    let mut messages = vec![format!("cache_dir={}", cache.display())];
    let mut repos = Vec::new();
    let mut ok = true;

    for entry in &m.repos {
        match setup_one(entry, &cache, fetch) {
            Ok(st) => {
                if !st.ready && !entry.optional {
                    ok = false;
                }
                if !st.ready && entry.optional {
                    messages.push(format!("optional repo {} skipped: {:?}", entry.id, st.error));
                }
                repos.push(st);
            }
            Err(e) => {
                if entry.optional {
                    messages.push(format!("optional {}: {e}", entry.id));
                    repos.push(SetupRepoStatus {
                        id: entry.id.clone(),
                        ready: false,
                        path: None,
                        git_commit: None,
                        action: "skipped_optional".into(),
                        error: Some(e),
                    });
                } else {
                    ok = false;
                    repos.push(SetupRepoStatus {
                        id: entry.id.clone(),
                        ready: false,
                        path: None,
                        git_commit: None,
                        action: "failed".into(),
                        error: Some(e),
                    });
                }
            }
        }
    }

    SetupResult {
        ok,
        repos,
        cache_dir: cache.display().to_string(),
        messages,
    }
}

fn setup_one(entry: &RepoEntry, cache: &Path, fetch: bool) -> Result<SetupRepoStatus, String> {
    match &entry.source {
        RepoSource::Path { path } => {
            let p = PathBuf::from(expand_env(path));
            if !p.exists() {
                return Ok(SetupRepoStatus {
                    id: entry.id.clone(),
                    ready: false,
                    path: Some(p.display().to_string()),
                    git_commit: None,
                    action: "missing_path".into(),
                    error: Some("path does not exist".into()),
                });
            }
            Ok(SetupRepoStatus {
                id: entry.id.clone(),
                ready: true,
                path: Some(p.display().to_string()),
                git_commit: crate::identity::git_rev(&p),
                action: "verified_path".into(),
                error: None,
            })
        }
        RepoSource::Fixture { fixture_id, path } => {
            let p = if let Some(path) = path {
                PathBuf::from(expand_env(path))
            } else {
                // Built-in: workspace itself as public fixture for self-host
                std::env::current_dir().map_err(|e| e.to_string())?
            };
            if !p.exists() {
                return Err(format!("fixture {fixture_id} path missing: {}", p.display()));
            }
            Ok(SetupRepoStatus {
                id: entry.id.clone(),
                ready: true,
                path: Some(p.display().to_string()),
                git_commit: crate::identity::git_rev(&p),
                action: format!("fixture:{fixture_id}"),
                error: None,
            })
        }
        RepoSource::GitClone {
            url,
            commit,
            dest,
            depth,
        } => {
            let dest_path = if let Some(d) = dest {
                PathBuf::from(expand_env(d))
            } else {
                cache.join(&entry.id)
            };
            if dest_path.exists()
                && dest_path.join(".git").exists()
                && crate::identity::git_rev(&dest_path).as_deref() == Some(commit.as_str())
            {
                return Ok(SetupRepoStatus {
                    id: entry.id.clone(),
                    ready: true,
                    path: Some(dest_path.display().to_string()),
                    git_commit: Some(commit.clone()),
                    action: "already_pinned".into(),
                    error: None,
                });
            }
            if !fetch {
                return Ok(SetupRepoStatus {
                    id: entry.id.clone(),
                    ready: dest_path.exists(),
                    path: Some(dest_path.display().to_string()),
                    git_commit: crate::identity::git_rev(&dest_path),
                    action: if dest_path.exists() {
                        "exists_wrong_pin_use_fetch".into()
                    } else {
                        "missing_use_fetch".into()
                    },
                    error: if dest_path.exists() {
                        Some(format!(
                            "want commit {commit}, have {:?}",
                            crate::identity::git_rev(&dest_path)
                        ))
                    } else {
                        Some("clone missing; pass --fetch".into())
                    },
                });
            }
            fs::create_dir_all(cache).map_err(|e| e.to_string())?;
            if dest_path.exists() {
                // fetch + checkout
                run_git(&dest_path, &["fetch", "--all"])?;
                run_git(&dest_path, &["checkout", "--detach", commit])?;
            } else {
                let mut args = vec!["clone".to_string()];
                if let Some(d) = depth {
                    args.push("--depth".into());
                    args.push(d.to_string());
                }
                // shallow may not include commit; prefer full if pin needed
                if depth.is_some() {
                    // filter for commit
                    args.push("--filter=blob:none".into());
                }
                args.push(url.clone());
                args.push(dest_path.display().to_string());
                let st = Command::new("git")
                    .args(&args)
                    .status()
                    .map_err(|e| format!("git clone: {e}"))?;
                if !st.success() {
                    return Err(format!("git clone failed for {}", entry.id));
                }
                // ensure commit
                let _ = run_git(&dest_path, &["fetch", "--depth", "1", "origin", commit]);
                run_git(&dest_path, &["checkout", "--detach", commit]).or_else(|_| {
                    // unshallow and retry
                    let _ = run_git(&dest_path, &["fetch", "--unshallow"]);
                    run_git(&dest_path, &["fetch", "origin", commit])?;
                    run_git(&dest_path, &["checkout", "--detach", commit])
                })?;
            }
            let have = crate::identity::git_rev(&dest_path);
            if have.as_deref() != Some(commit.as_str()) {
                // allow prefix match
                let ok = have
                    .as_ref()
                    .map(|h| h.starts_with(commit) || commit.starts_with(&h[..h.len().min(7)]))
                    .unwrap_or(false);
                if !ok {
                    return Err(format!(
                        "pin mismatch after checkout: want {commit}, have {have:?}"
                    ));
                }
            }
            Ok(SetupRepoStatus {
                id: entry.id.clone(),
                ready: true,
                path: Some(dest_path.display().to_string()),
                git_commit: have,
                action: "cloned_or_updated".into(),
                error: None,
            })
        }
    }
}

fn run_git(cwd: &Path, args: &[&str]) -> Result<(), String> {
    let st = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .status()
        .map_err(|e| format!("git {:?}: {e}", args))?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("git {:?} failed", args))
    }
}

/// Resolve concrete path for a repo entry (no fetch).
pub fn resolve_repo_path(entry: &RepoEntry, cache: &Path) -> Result<PathBuf, String> {
    match &entry.source {
        RepoSource::Path { path } => {
            let p = PathBuf::from(expand_env(path));
            if p.exists() {
                Ok(p)
            } else {
                Err(format!("{} missing", p.display()))
            }
        }
        RepoSource::Fixture { path, .. } => {
            if let Some(path) = path {
                Ok(PathBuf::from(expand_env(path)))
            } else {
                std::env::current_dir().map_err(|e| e.to_string())
            }
        }
        RepoSource::GitClone { dest, .. } => {
            let p = if let Some(d) = dest {
                PathBuf::from(expand_env(d))
            } else {
                cache.join(&entry.id)
            };
            if p.exists() {
                Ok(p)
            } else {
                Err(format!("git clone dest missing: {}", p.display()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_env_braces() {
        // SAFETY: test-only env mutation
        unsafe { env::set_var("FFF_TEST_X", "hello") };
        assert_eq!(expand_env("pre-${FFF_TEST_X}-post"), "pre-hello-post");
        unsafe { env::remove_var("FFF_TEST_X") };
    }

    #[test]
    fn parse_path_manifest() {
        let j = r#"{
          "schema_version": 1,
          "suite": "agent-loop",
          "repos": [
            {"id": "local", "role": ["gates","profile"], "source": {"kind": "path", "path": "."}}
          ],
          "adapter": {"enable_content_indexing": false, "top_k_log": 5}
        }"#;
        let m: Manifest = serde_json::from_str(j).unwrap();
        assert_eq!(m.repos.len(), 1);
        assert_eq!(m.adapter.top_k_log, 5);
    }
}
