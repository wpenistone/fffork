use crate::file_picker::is_known_binary_extension_basename;
use crate::ignore::IGNORED_DIRS;
use crate::types::FileItem;
use crate::walk::{WalkIgnoreRules, WalkOutput, WalkSpan};
use std::cell::UnsafeCell;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;
use zlob::walk::{WalkBuilder, WalkFlags, WalkMetadata, WalkState};

const PROGRESS_FLUSH: usize = 512;

// Real `.git` directories never arrive: GITIGNORE skips them before the
// callback. Nested `.git/` is yielded only if that name was a followed symlink dir.
#[inline]
fn skip_git_rel(rel: &[u8], follow_symlinks: bool) -> bool {
    let base = rel.rsplit(|b| *b == b'/').next().unwrap_or(rel);
    if base == b".git" {
        return true;
    }
    follow_symlinks && is_git_rel(rel)
}

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

// Exclusive to one zlob `worker_id` until `WalkBuilder::run` returns.
struct Shard {
    bytes: UnsafeCell<Vec<u8>>,
    files: UnsafeCell<Vec<(FileItem, u32, u32)>>,
    dirs: UnsafeCell<Vec<(u32, u32)>>,
    pending: UnsafeCell<usize>,
}

// SAFETY: zlob invokes each worker_id on one thread until run() returns.
unsafe impl Sync for Shard {}

struct ShardParts<'a> {
    bytes: &'a mut Vec<u8>,
    files: &'a mut Vec<(FileItem, u32, u32)>,
    dirs: &'a mut Vec<(u32, u32)>,
    pending: &'a mut usize,
}

impl Shard {
    fn new() -> Self {
        Self {
            bytes: UnsafeCell::new(Vec::with_capacity(64 * 1024)),
            files: UnsafeCell::new(Vec::with_capacity(1024)),
            dirs: UnsafeCell::new(Vec::with_capacity(64)),
            pending: UnsafeCell::new(0),
        }
    }

    #[inline]
    fn parts(&self) -> ShardParts<'_> {
        // Same worker_id never runs this on two threads at once.
        unsafe {
            ShardParts {
                bytes: &mut *self.bytes.get(),
                files: &mut *self.files.get(),
                dirs: &mut *self.dirs.get(),
                pending: &mut *self.pending.get(),
            }
        }
    }
}

// Copies `rel` as UTF-8 (lossy when the OS bytes are not) and returns
// `(off, len, basename_off)` in that stored text.
#[inline]
fn store_rel(buf: &mut Vec<u8>, rel: &[u8], slash: bool) -> (u32, u32, usize) {
    let start = buf.len();
    if std::str::from_utf8(rel).is_ok() {
        buf.extend_from_slice(rel);
    } else {
        buf.extend_from_slice(String::from_utf8_lossy(rel).as_bytes());
    }
    let path_end = buf.len();
    if slash {
        buf.push(b'/');
    }
    let stored_base = buf[start..path_end]
        .iter()
        .rposition(|b| *b == b'/')
        .map(|i| i + 1)
        .unwrap_or(0);
    (start as u32, (buf.len() - start) as u32, stored_base)
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

    // One shard per worker_id. zlob never shares an id across threads, so
    // pushes do not take a lock.
    let workers = builder.max_workers().max(1);
    let shards: Vec<Shard> = (0..workers).map(|_| Shard::new()).collect();

    let outcome = match builder.run(|entry| {
        // `worker_id` is < `max_workers` and exclusive to this thread.
        let mut parts = shards[entry.worker_id()].parts();
        // Followed links keep the Symlink kind and lstat metadata, so stat
        // the target to index the ones resolving to files (#912).
        let target = if entry.is_file() {
            None
        } else if follow_symlinks && entry.is_symlink() {
            match std::fs::metadata(entry.path()) {
                Ok(meta) if meta.is_file() => Some(meta),
                _ => return WalkState::Continue,
            }
        } else {
            if entry.is_dir() {
                let rel = entry.relative_path_bytes();
                if rel.is_empty() {
                    return WalkState::Continue;
                }
                if skip_git_rel(rel, follow_symlinks) {
                    return WalkState::SkipDir;
                }
                let (off, len, _) = store_rel(parts.bytes, rel, true);
                parts.dirs.push((off, len));
            }

            return WalkState::Continue;
        };

        let rel = entry.relative_path_bytes();
        if skip_git_rel(rel, follow_symlinks) {
            return WalkState::Continue;
        }

        let (off, len, base_off) = store_rel(parts.bytes, rel, false);
        let (size, modified) = match &target {
            Some(meta) => (
                meta.len(),
                meta.modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs()),
            ),
            // zlob reports mtime in ns since the Unix epoch; FileItem wants secs.
            None => (
                entry.size().unwrap_or(0),
                entry
                    .modified_ns()
                    .map(|ns| (ns / 1_000_000_000).max(0) as u64)
                    .unwrap_or(0),
            ),
        };

        let mut item = if target.is_none() {
            let name = entry.basename().unwrap_or("");
            let mut item = FileItem::new_raw(
                base_off as u16,
                size,
                modified,
                None,
                is_known_binary_extension_basename(name),
            );
            item.mark_definition_language(name);
            item
        } else {
            let start = off as usize;
            // Stored bytes are UTF-8; `base_off` is the basename in that text.
            let name = unsafe {
                std::str::from_utf8_unchecked(&parts.bytes[start + base_off..start + len as usize])
            };
            let mut item = FileItem::new_raw(
                base_off as u16,
                size,
                modified,
                None,
                is_known_binary_extension_basename(name),
            );
            item.mark_definition_language(name);
            item
        };

        parts.files.push((item, off, len));
        *parts.pending += 1;
        if *parts.pending >= PROGRESS_FLUSH {
            let n = *parts.pending;
            *parts.pending = 0;
            drop(parts);
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

    let mut buffers = Vec::with_capacity(shards.len());
    let mut pairs = Vec::new();
    let mut dirs = Vec::new();
    for shard in &shards {
        let parts = shard.parts();
        if parts.files.is_empty() && parts.dirs.is_empty() && parts.bytes.is_empty() {
            continue;
        }
        let buf_i = buffers.len() as u16;
        if *parts.pending > 0 {
            synced_files_count.fetch_add(*parts.pending, Ordering::Relaxed);
            *parts.pending = 0;
        }
        for (item, off, len) in parts.files.drain(..) {
            pairs.push((item, WalkSpan { buf: buf_i, off, len }));
        }
        for (off, len) in parts.dirs.drain(..) {
            dirs.push(WalkSpan { buf: buf_i, off, len });
        }
        buffers.push(std::mem::take(parts.bytes));
    }
    // Always report the exact final total regardless of the last step.
    synced_files_count.store(pairs.len(), Ordering::Relaxed);

    // Retain the ignore rules only when the walk actually gathered some
    // (git roots with .gitignore/.ignore). Otherwise callers fall back.
    let ignore_rules = outcome
        .rules()
        .is_some()
        .then(|| WalkIgnoreRules { inner: outcome });

    Ok(WalkOutput::from_parts(buffers, pairs, dirs, ignore_rules))
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
