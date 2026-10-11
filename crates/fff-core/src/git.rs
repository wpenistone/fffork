use crate::error::Result;
use ahash::AHashMap;
use git2::{Repository, Status, StatusOptions};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

/// Process-wide libgit2 tuning for read-heavy local use. By default libgit2
/// never caches trees over 4KB and additionally verifies every object. This roughly costs 150% of
/// its valuable runtime
pub(crate) fn tune_libgit2_for_local_reads() {
    static TUNE: std::sync::Once = std::sync::Once::new();
    TUNE.call_once(|| {
        // Same tradeoff cargo makes: local objects are trusted, skip hashing.
        git2::opts::strict_hash_verification(false);
        // SAFETY: plain process-global size limit; total memory stays bounded
        // by libgit2's 256MB odb cache.
        let _ =
            unsafe { git2::opts::set_cache_object_limit(git2::ObjectType::Tree, 8 * 1024 * 1024) };
    });
}

pub(crate) fn default_status_options() -> StatusOptions {
    let mut opts = StatusOptions::new();
    opts.include_untracked(true)
        .recurse_untracked_dirs(true)
        .include_unmodified(true)
        .exclude_submodules(true);
    opts
}

/// Status options for the initial scan / rescan.
///
/// Skips `include_unmodified` because every `FileItem` starts with
/// `git_status: None` (== clean), so a missing cache entry already means
/// "clean" — no need to ask libgit2 to enumerate every tracked path.
/// Saves seconds on huge dirty trees (e.g. chromium with 400k+ entries).
pub(crate) fn initial_scan_status_options() -> StatusOptions {
    let mut opts = StatusOptions::new();
    opts.include_untracked(true)
        .recurse_untracked_dirs(true)
        .exclude_submodules(true);
    opts
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GitStatusCache {
    map: AHashMap<PathBuf, Status>,
    // Workdir prefix when keys are stored relative (`FFF_GIT_RELATIVE_KEYS`).
    // Empty means keys are absolute (default).
    prefix: PathBuf,
}

impl IntoIterator for GitStatusCache {
    type Item = (PathBuf, Status);
    type IntoIter = <AHashMap<PathBuf, Status> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        // Absolute keys and relative keys both hash by content; std's
        // IntoIter hides which map type backs us.
        let full: AHashMap<PathBuf, Status> = if self.prefix.as_os_str().is_empty() {
            self.map
        } else {
            self.map
                .into_iter()
                .map(|(rel, st)| (self.prefix.join(rel), st))
                .collect()
        };
        full.into_iter()
    }
}

/// `FFF_GIT_RELATIVE_KEYS=1` stores keys relative to the workdir: one short
/// relative path per entry instead of repeating the absolute prefix (~40B
/// saved/entry on deep roots). Lookup strips the prefix first.
fn relative_keys_enabled() -> bool {
    matches!(
        std::env::var("FFF_GIT_RELATIVE_KEYS").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Max pathspec entries per libgit2 call (`FFF_GIT_PATHSPEC_BATCH`).
fn pathspec_batch_cap() -> usize {
    std::env::var("FFF_GIT_PATHSPEC_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(500)
}

impl GitStatusCache {
    pub fn statuses_len(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub fn lookup_status(&self, full_path: &Path) -> Option<Status> {
        if self.prefix.as_os_str().is_empty() {
            return self.map.get(full_path).copied();
        }
        // Relative keys: strip the workdir prefix, fall back to the full
        // key so mixed callers still hit.
        if let Ok(rel) = full_path.strip_prefix(&self.prefix) {
            if let Some(st) = self.map.get(rel) {
                return Some(*st);
            }
        }
        self.map.get(full_path).copied()
    }

    /// Sorted entries for a linear merge against the sorted file index.
    /// Byte order matches `Path::cmp`, so the merge is exact on case-variant
    /// names. Public API; no in-tree consumer yet.
    #[allow(dead_code)]
    pub fn into_sorted_vec(self) -> Vec<(PathBuf, Status)> {
        let mut v: Vec<(PathBuf, Status)> = self.into_iter().collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    #[tracing::instrument(skip(repo, status_options))]
    pub(crate) fn read_status(
        repo: &Repository,
        status_options: &mut StatusOptions,
    ) -> Result<Self> {
        let statuses = repo.statuses(Some(status_options))?;
        let Some(repo_path) = repo.workdir() else {
            return Ok(Self::default()); // repo is bare
        };

        let repo_path = crate::path_utils::normalize(repo_path.to_path_buf());

        // Single pass: strip the prefix during insert when relative keys are
        // on, instead of building a full map and re-hashing it. Entries
        // outside the workdir only exist in the default absolute shape.
        let relative = relative_keys_enabled();
        let mut entries = AHashMap::with_capacity(statuses.len());
        for entry in &statuses {
            if let Ok(entry_path) = entry.path() {
                // libgit2 returns entry paths with forward slashes on every platform
                // fff stores native paths - meaning we have forward slash issue on windows
                let full_path = crate::path_utils::normalize(repo_path.join(entry_path));
                if relative {
                    if let Ok(rel) = full_path.strip_prefix(&repo_path) {
                        entries.insert(rel.to_path_buf(), entry.status());
                    }
                } else {
                    entries.insert(full_path, entry.status());
                }
            }
        }

        Ok(Self {
            map: entries,
            prefix: if relative { repo_path } else { PathBuf::new() },
        })
    }

    /// Dirty-only refresh: same cheap options as the initial scan (no
    /// `include_unmodified`). Missing entries already mean "clean" because
    /// every `FileItem` starts with `git_status: None`, so enumerating all
    /// tracked paths is pure overhead on huge trees.
    /// Test-only helper: production refresh goes through the status worker.
    #[cfg(test)]
    pub(crate) fn read_dirty_status(repo: &Repository) -> Result<Self> {
        let mut opts = initial_scan_status_options();
        Self::read_status(repo, &mut opts)
    }

    pub fn read_git_status(
        git_workdir: Option<&Path>,
        status_options: &mut StatusOptions,
    ) -> Option<Self> {
        let git_workdir = git_workdir.as_ref()?;
        let repository = Repository::open(git_workdir).ok()?;

        let status = Self::read_status(&repository, status_options);

        match status {
            Ok(status) => Some(status),
            Err(e) => {
                tracing::error!(?e, "Failed to read git status");

                None
            }
        }
    }

    #[tracing::instrument(skip(repo), fields(paths_count = paths.len()), level = tracing::Level::DEBUG)]
    pub fn git_status_for_paths<TPath: AsRef<Path> + Debug>(
        repo: &Repository,
        paths: &[TPath],
    ) -> Result<Self> {
        if paths.is_empty() {
            return Ok(Self::default());
        }

        let Some(workdir) = repo.workdir() else {
            return Ok(Self::default());
        };
        let workdir = crate::path_utils::normalize(workdir.to_path_buf());

        // git pathspec is pretty slow and requires to walk the whole directory
        // so for a single file which is the most general use case we query directly the file
        if paths.len() == 1 {
            let full_path = paths[0].as_ref();
            let relative_path = full_path.strip_prefix(&workdir)?;
            let status = repo.status_file(relative_path)?;

            let mut map = AHashMap::with_capacity(1);
            // Store relative when the flag is on so the shape matches bulk.
            if relative_keys_enabled() {
                let rel = full_path.strip_prefix(&workdir)?.to_path_buf();
                map.insert(rel, status);
                return Ok(Self {
                    map,
                    prefix: workdir,
                });
            }
            map.insert(full_path.to_path_buf(), status);
            return Ok(Self {
                map,
                prefix: PathBuf::new(),
            });
        }

        // Dirty-only options: `include_unmodified` would enumerate every
        // tracked path even with a pathspec, turning a targeted refresh into
        // a full walk. Chunked so one huge watcher burst cannot build an
        // unbounded pathspec.
        let cap = pathspec_batch_cap();
        let mut merged = AHashMap::with_capacity(paths.len());
        let mut merged_prefix = PathBuf::new();
        for chunk in paths.chunks(cap) {
            let mut status_options = initial_scan_status_options();
            for path in chunk {
                status_options.pathspec(path.as_ref().strip_prefix(&workdir)?);
            }
            let part = Self::read_status(repo, &mut status_options)?;
            if !part.prefix.as_os_str().is_empty() {
                merged_prefix = part.prefix.clone();
            }
            merged.extend(part.map);
        }
        Ok(Self {
            map: merged,
            prefix: merged_prefix,
        })
    }
}

const MODIFIED_STATUS_MASK: Status = Status::from_bits_truncate(
    Status::WT_MODIFIED.bits()
        | Status::INDEX_MODIFIED.bits()
        | Status::WT_NEW.bits()
        | Status::INDEX_NEW.bits()
        | Status::WT_RENAMED.bits(),
);

#[inline(always)]
pub fn is_modified_status(status: Status) -> bool {
    status.intersects(MODIFIED_STATUS_MASK)
}

pub fn format_git_status_opt(status: Option<Status>) -> Option<&'static str> {
    match status {
        None => Some("clean"),
        Some(status) => {
            if status.contains(Status::WT_NEW) {
                Some("untracked")
            } else if status.contains(Status::WT_MODIFIED) {
                Some("modified")
            } else if status.contains(Status::WT_DELETED) {
                Some("deleted")
            } else if status.contains(Status::WT_RENAMED) {
                Some("renamed")
            } else if status.contains(Status::INDEX_NEW) {
                Some("staged_new")
            } else if status.contains(Status::INDEX_MODIFIED) {
                Some("staged_modified")
            } else if status.contains(Status::INDEX_DELETED) {
                Some("staged_deleted")
            } else if status.contains(Status::IGNORED) {
                Some("ignored")
            // `Status::CURRENT` is 0, so `contains` is true for every status:
            // testing it here labelled conflicted/typechanged files "clean".
            } else if status.is_empty() {
                Some("clean")
            } else {
                None
            }
        }
    }
}

pub fn format_git_status(status: Option<Status>) -> &'static str {
    format_git_status_opt(status).unwrap_or("unknown")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;
    use tempfile::TempDir;

    fn git(dir: &Path, args: &[&str]) {
        let out = Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@t")
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?} failed");
    }

    #[test]
    fn unclassified_status_is_not_reported_as_clean() {
        // A merge-conflicted file must not claim to be clean.
        assert_eq!(format_git_status_opt(Some(Status::CONFLICTED)), None);
        assert_eq!(format_git_status(Some(Status::CONFLICTED)), "unknown");
        assert_eq!(format_git_status(Some(Status::WT_TYPECHANGE)), "unknown");

        // Every already-classified state keeps its label.
        assert_eq!(format_git_status_opt(None), Some("clean"));
        assert_eq!(format_git_status_opt(Some(Status::empty())), Some("clean"));
        assert_eq!(format_git_status_opt(Some(Status::CURRENT)), Some("clean"));
        assert_eq!(
            format_git_status_opt(Some(Status::WT_NEW)),
            Some("untracked")
        );
        assert_eq!(
            format_git_status_opt(Some(Status::WT_MODIFIED)),
            Some("modified")
        );
        assert_eq!(
            format_git_status_opt(Some(Status::INDEX_MODIFIED)),
            Some("staged_modified")
        );
        assert_eq!(
            format_git_status_opt(Some(Status::IGNORED)),
            Some("ignored")
        );
    }

    /// Regression: on case-insensitive filesystems libgit2 returns
    /// statuses in a case-insensitive order. Our previous sorted-`Vec` +
    /// `binary_search_by(Path::cmp)` lookup silently missed entries
    /// because `Path::cmp` is byte-wise.
    ///
    /// This test uses deliberately mixed-case filenames so the two
    /// orderings disagree, then checks every lookup succeeds.
    #[test]
    fn lookup_is_case_exact_regardless_of_libgit2_sort_order() {
        let tmp = TempDir::new().unwrap();
        // `std::fs::canonicalize` on Windows adds a `\\?\` UNC prefix that
        // libgit2's workdir string lacks. Use dunce so both sides match.
        let base = crate::path_utils::canonicalize(tmp.path()).unwrap();

        // Mixed-case names that sort differently under byte-wise vs
        // case-insensitive comparators.
        let names = [
            "README.md",
            "a_lower.rs",
            "Z_upper.rs",
            "mixed_Case.txt",
            "nested/Inner_File.rs",
        ];
        for n in &names {
            let p = base.join(n);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, format!("// {n}\n")).unwrap();
        }

        git(&base, &["init", "-b", "main"]);
        git(&base, &["add", "-A"]);
        git(&base, &["commit", "-m", "seed", "--no-gpg-sign"]);

        // Modify every file so they all end up in the status output as
        // WT_MODIFIED — guarantees a non-trivial map we have to look up.
        for n in &names {
            let p = base.join(n);
            fs::write(&p, format!("// {n}\n// edit\n")).unwrap();
        }

        let repo = Repository::open(&base).unwrap();
        let paths: Vec<PathBuf> = names.iter().map(|n| base.join(n)).collect();
        let cache = GitStatusCache::git_status_for_paths(&repo, &paths).unwrap();

        for (n, abs) in names.iter().zip(paths.iter()) {
            let status = cache.lookup_status(abs);
            assert!(
                status.is_some(),
                "lookup for {n} returned None; cache holds {} entries",
                cache.statuses_len(),
            );
            assert!(
                status.unwrap().contains(Status::WT_MODIFIED),
                "expected WT_MODIFIED for {n}, got {:?}",
                status
            );
        }
    }

    #[test]
    fn relative_keys_lookup_matches_absolute() {
        unsafe {
            std::env::set_var("FFF_GIT_RELATIVE_KEYS", "1");
        }
        let tmp = TempDir::new().unwrap();
        let base = crate::path_utils::canonicalize(tmp.path()).unwrap();
        fs::write(base.join("a.rs"), "// a\n").unwrap();
        fs::write(base.join("b.rs"), "// b\n").unwrap();
        git(&base, &["init", "-b", "main"]);
        git(&base, &["add", "-A"]);
        git(&base, &["commit", "-m", "seed", "--no-gpg-sign"]);
        fs::write(base.join("a.rs"), "// a\n// edit\n").unwrap();

        let repo = Repository::open(&base).unwrap();
        let cache = GitStatusCache::read_dirty_status(&repo).unwrap();
        let abs = base.join("a.rs");
        assert!(cache.lookup_status(&abs).is_some());
        // into_iter re-expands to absolute paths for index application.
        let mut found = false;
        for (p, _) in cache {
            if p == abs {
                found = true;
            }
        }
        assert!(found, "relative cache must re-expand to absolute paths");
        unsafe {
            std::env::remove_var("FFF_GIT_RELATIVE_KEYS");
        }
    }

    #[test]
    fn dirty_status_skips_clean_files() {
        let tmp = TempDir::new().unwrap();
        let base = crate::path_utils::canonicalize(tmp.path()).unwrap();
        fs::write(base.join("clean.rs"), "// clean\n").unwrap();
        fs::write(base.join("dirty.rs"), "// dirty\n").unwrap();
        git(&base, &["init", "-b", "main"]);
        git(&base, &["add", "-A"]);
        git(&base, &["commit", "-m", "seed", "--no-gpg-sign"]);
        fs::write(base.join("dirty.rs"), "// dirty\n// edit\n").unwrap();

        let repo = Repository::open(&base).unwrap();
        let cache = GitStatusCache::read_dirty_status(&repo).unwrap();
        assert!(cache.lookup_status(&base.join("dirty.rs")).is_some());
        assert!(cache.lookup_status(&base.join("clean.rs")).is_none());
        // Sorted vec stays byte-ordered for linear merges.
        let v = cache.into_sorted_vec();
        let mut sorted = v.clone();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        assert!(v.iter().map(|(p, _)| p).eq(sorted.iter().map(|(p, _)| p)));
    }
}
