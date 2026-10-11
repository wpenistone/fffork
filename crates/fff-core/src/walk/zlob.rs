use crate::file_picker::is_known_binary_extension_basename;
use crate::ignore::IGNORED_DIRS;
use crate::types::FileItem;
use crate::walk::{WalkIgnoreRules, WalkOutput};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use zlob::walk::{WalkBuilder, WalkFlags, WalkMetadata, WalkState};

const PROGRESS_FLUSH: usize = 512;

#[inline]
fn is_git_rel(rel: &[u8]) -> bool {
    if rel == b".git" {
        return true;
    }
    if rel.starts_with(b".git/") {
        return true;
    }
    if rel.len() >= 5 {
        if memchr::memmem::find(rel, b"/.git/").is_some() {
            return true;
        }
        if rel.ends_with(b"/.git") {
            return true;
        }
    }
    false
}

#[tracing::instrument(skip_all, name = "zlob walker", level = "info")]
pub(crate) fn walk_collect_files(
    base_path: &Path,
    is_git_repo: bool,
    follow_symlinks: bool,
    threads: usize,
    synced_files_count: &Arc<AtomicUsize>,
) -> crate::Result<WalkOutput> {
    // gitignore on; skip hidden on non-git roots (so `~/` doesn't recurse into
    // ~/.cache, ~/.config, etc.); optionally follow symlinks.
    let mut flags = WalkFlags::GITIGNORE;
    if !is_git_repo {
        flags |= WalkFlags::SKIP_HIDDEN;
    }
    if follow_symlinks {
        flags |= WalkFlags::FOLLOW_SYMLINKS;
    }

    let mut builder = WalkBuilder::new(base_path)
        .map_err(|e| crate::Error::WalkFailed(format!("WalkBuilder::new: {e:?}")))?;
    builder
        .options(flags)
        .threads(threads)
        // Bulk-fetch the only metadata FileItem needs; zlob never stats more.
        .metadata(WalkMetadata::SIZE | WalkMetadata::MTIME);

    if !is_git_repo
        && !IGNORED_DIRS.is_empty()
        && let Err(e) = builder.extra_ignore(IGNORED_DIRS)
    {
        // Interior NUL in one of the extra_ignore patterns would fail
        // here — treat as if no extras were supplied rather than
        // aborting the whole walk.
        tracing::warn!(?e, "zlob extra_ignore rejected; walking without it");
    }

    // Per-worker shards indexed by `worker_id`: the lock is never
    // contended across workers, and progress flushes every 512 files.
    let workers = builder.max_workers().max(1).min(256);
    let shards: Vec<Mutex<(Vec<(FileItem, String)>, Vec<String>, usize)>> = (0..workers)
        .map(|_| Mutex::new((Vec::with_capacity(1024), Vec::with_capacity(64), 0)))
        .collect();

    let outcome = match builder.run(|entry| {
        let wid = (entry.worker_id() % workers) as usize;
        if !entry.is_file() {
            if entry.is_dir() {
                let rel = entry.relative_path_bytes();
                if rel.is_empty() {
                    return WalkState::Continue;
                }
                if is_git_rel(rel) {
                    return WalkState::SkipDir;
                }
                let mut rel = entry.relative_path_lossy().into_owned();
                rel.push('/');
                shards[wid].lock().1.push(rel);
            }

            return WalkState::Continue;
        }

        if is_git_rel(entry.relative_path_bytes()) {
            return WalkState::Continue;
        }

        // `basename()` returns `&str` for files only.
        let basename = entry.basename().unwrap_or("");
        let is_binary = is_known_binary_extension_basename(basename);

        let size = entry.size().unwrap_or(0);
        // zlob reports mtime in ns since the Unix epoch; FileItem wants secs.
        let modified = entry
            .modified_ns()
            .map(|ns| (ns / 1_000_000_000).max(0) as u64)
            .unwrap_or(0);

        // Lossy pair: the offset must index the decoded string, not the raw
        // bytes, or it lands inside a U+FFFD on invalid-UTF-8 names (#799).
        let basename_offset = entry.basename_offset_in_relative_lossy() as u16;
        // zlob emits '/'-separated relative paths, which is fff's canonical
        // internal form on every platform — store them verbatim.
        let relative_path = entry.relative_path_lossy().into_owned();
        let item = FileItem::new_raw(basename_offset, size, modified, None, is_binary);

        let mut guard = shards[wid].lock();
        guard.0.push((item, relative_path));
        guard.2 += 1;
        if guard.2 >= PROGRESS_FLUSH {
            let n = guard.2;
            guard.2 = 0;
            drop(guard);
            synced_files_count.fetch_add(n, Ordering::Relaxed);
        }

        WalkState::Continue
    }) {
        Ok(outcome) => outcome,
        Err(e) => {
            // Preserve whatever we collected before the failure so the caller
            // can still surface a partial index instead of nothing.
            tracing::error!(?e, "zlob walk failed");
            return Err(crate::Error::WalkFailed(format!("{e:?}")));
        }
    };

    let mut pairs = Vec::new();
    let mut dirs = Vec::new();
    for shard in &shards {
        let mut guard = shard.lock();
        pairs.append(&mut guard.0);
        dirs.append(&mut guard.1);
    }
    // Always report the exact final total regardless of the last step.
    synced_files_count.store(pairs.len(), Ordering::Relaxed);

    // Retain the ignore rules only when the walk actually gathered some
    // (git roots with .gitignore/.ignore). Otherwise callers fall back.
    let ignore_rules = outcome
        .rules()
        .is_some()
        .then(|| WalkIgnoreRules { inner: outcome });

    Ok(WalkOutput {
        pairs,
        dirs,
        ignore_rules,
    })
}

#[cfg(test)]
mod tests {
    use super::is_git_rel;

    #[test]
    fn git_prefix_prune() {
        assert!(is_git_rel(b".git"));
        assert!(is_git_rel(b".git/objects/pack/a.pack"));
        assert!(is_git_rel(b"sub/.git"));
        assert!(is_git_rel(b"sub/.git/HEAD"));
        assert!(!is_git_rel(b".github/workflows/ci.yml"));
        assert!(!is_git_rel(b"src/main.rs"));
        assert!(!is_git_rel(b".gitignore"));
    }
}
