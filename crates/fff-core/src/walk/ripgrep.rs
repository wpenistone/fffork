use crate::ignore::non_git_repo_overrides;
use crate::types::FileItem;
use crate::walk::{WalkOutput, WalkSpan};
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[tracing::instrument(skip_all, name = "ripgrep walker", level = "info")]
pub(crate) fn walk_collect_files(
    base_path: &Path,
    is_git_repo: bool,
    follow_symlinks: bool,
    threads: usize,
    synced_files_count: &Arc<AtomicUsize>,
) -> crate::Result<WalkOutput> {
    let mut walk_builder = WalkBuilder::new(base_path);
    walk_builder
        // this is a very important guard for the user opening ~/ or other root non-git dir
        .hidden(!is_git_repo)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .ignore(true)
        .follow_links(follow_symlinks)
        .threads(threads);

    if !is_git_repo && let Some(overrides) = non_git_repo_overrides(base_path) {
        walk_builder.overrides(overrides);
    }

    let walker = walk_builder.build_parallel();

    // Threads merge once, on drop. The hot path only appends to thread-local bytes.
    struct Collected {
        buffers: Vec<Vec<u8>>,
        pairs: Vec<(FileItem, WalkSpan)>,
        dirs: Vec<WalkSpan>,
    }
    let collected = parking_lot::Mutex::new(Collected {
        buffers: Vec::new(),
        pairs: Vec::new(),
        dirs: Vec::new(),
    });

    struct FffVisitor<'a> {
        collected: &'a parking_lot::Mutex<Collected>,
        synced_files_count: &'a Arc<AtomicUsize>,
        base_path: PathBuf,
        local_bytes: Vec<u8>,
        local_pairs: Vec<(FileItem, u32, u32)>,
        local_dirs: Vec<(u32, u32)>,
        local_count: usize,
    }

    impl<'a> FffVisitor<'a> {
        fn flush_progress(&mut self) {
            if self.local_count == 0 {
                return;
            }
            self.synced_files_count
                .fetch_add(self.local_count, Ordering::Relaxed);
            self.local_count = 0;
        }
    }

    impl<'a> ignore::ParallelVisitor for FffVisitor<'a> {
        fn visit(&mut self, result: Result<ignore::DirEntry, ignore::Error>) -> ignore::WalkState {
            let Ok(entry) = result else {
                return ignore::WalkState::Continue;
            };

            if entry.file_type().is_some_and(|ft| ft.is_file()) {
                let path = entry.path();

                // Match July walker: filter .git files, do not touch dirs here.
                if is_git_file(path) {
                    return ignore::WalkState::Continue;
                }

                let rel = path.strip_prefix(&self.base_path).unwrap_or(path);
                let Some((off, len, base_off)) = push_canonical(&mut self.local_bytes, rel, false)
                else {
                    return ignore::WalkState::Continue;
                };
                let (size, modified) = match entry.metadata().ok() {
                    Some(m) => (
                        m.len(),
                        m.modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
                            .map_or(0, |d| d.as_secs()),
                    ),
                    None => (0, 0),
                };
                let mut item = FileItem::new_raw(
                    base_off,
                    size,
                    modified,
                    None,
                    crate::file_picker::is_known_binary_extension(path),
                );
                {
                    let rel_bytes =
                        &self.local_bytes[off as usize..off as usize + len as usize];
                    // Canonical text is UTF-8; basename starts at `base_off`.
                    let name = unsafe {
                        std::str::from_utf8_unchecked(&rel_bytes[base_off as usize..])
                    };
                    item.mark_definition_language(name);
                }

                self.local_pairs.push((item, off, len));
                self.local_count += 1;
                if self.local_count >= 512 {
                    self.flush_progress();
                }
            } else if entry.depth() > 0 && entry.file_type().is_some_and(|ft| ft.is_dir()) {
                let path = entry.path();
                // Never descend into `.git/` — object stores are huge (multi-GB)
                // and are not searchable paths. `hidden(!is_git_repo)` still
                // allows other dot-dirs (.github, .config).
                if is_git_file(path) {
                    return ignore::WalkState::Skip;
                }
                if let Ok(rel) = path.strip_prefix(&self.base_path) {
                    let _ = push_canonical(&mut self.local_bytes, rel, true).map(|(off, len, _)| {
                        self.local_dirs.push((off, len));
                    });
                }
            }
            ignore::WalkState::Continue
        }
    }

    impl<'a> Drop for FffVisitor<'a> {
        fn drop(&mut self) {
            self.flush_progress();
            if self.local_pairs.is_empty() && self.local_dirs.is_empty() {
                return;
            }
            let mut guard = self.collected.lock();
            let buf_i = guard.buffers.len() as u16;
            guard.buffers.push(std::mem::take(&mut self.local_bytes));
            for (item, off, len) in self.local_pairs.drain(..) {
                guard.pairs.push((
                    item,
                    WalkSpan {
                        buf: buf_i,
                        off,
                        len,
                    },
                ));
            }
            for (off, len) in self.local_dirs.drain(..) {
                guard.dirs.push(WalkSpan {
                    buf: buf_i,
                    off,
                    len,
                });
            }
        }
    }

    struct FffVisitorBuilder<'a> {
        collected: &'a parking_lot::Mutex<Collected>,
        synced_files_count: &'a Arc<AtomicUsize>,
        base_path: PathBuf,
    }

    impl<'a> ignore::ParallelVisitorBuilder<'a> for FffVisitorBuilder<'a> {
        fn build(&mut self) -> Box<dyn ignore::ParallelVisitor + 'a> {
            Box::new(FffVisitor {
                collected: self.collected,
                synced_files_count: self.synced_files_count,
                base_path: self.base_path.clone(),
                local_bytes: Vec::with_capacity(64 * 1024),
                local_pairs: Vec::with_capacity(512),
                local_dirs: Vec::with_capacity(64),
                local_count: 0,
            })
        }
    }

    let mut visitor_builder = FffVisitorBuilder {
        collected: &collected,
        synced_files_count,
        base_path: base_path.to_path_buf(),
    };
    walker.visit(&mut visitor_builder);

    let collected = collected.into_inner();
    synced_files_count.store(collected.pairs.len(), Ordering::Relaxed);
    Ok(WalkOutput::from_parts(
        collected.buffers,
        collected.pairs,
        collected.dirs,
        None,
    ))
}

fn is_git_file(path: &Path) -> bool {
    crate::watch::is_git_file(path)
}

// Append canonical `/` text. `slash` marks a directory. Empty paths are skipped.
fn push_canonical(buf: &mut Vec<u8>, rel: &Path, slash: bool) -> Option<(u32, u32, u16)> {
    let lossy = rel.to_string_lossy();
    let canonical = crate::path_utils::to_canonical_slashes(&lossy);
    if canonical.is_empty() {
        return None;
    }
    let start = buf.len();
    buf.extend_from_slice(canonical.as_bytes());
    let path_end = buf.len();
    if slash {
        buf.push(b'/');
    }
    let base = buf[start..path_end]
        .iter()
        .rposition(|b| *b == b'/')
        .map(|i| i + 1)
        .unwrap_or(0);
    Some((start as u32, (buf.len() - start) as u32, base as u16))
}
