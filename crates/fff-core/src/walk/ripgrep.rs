use crate::ignore::non_git_repo_overrides;
use crate::types::FileItem;
use crate::walk::WalkOutput;
use crate::watch::is_git_file;
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

    // Single lock for both collections: every entry is either a file or a
    // dir. Workers accumulate into thread-local batches and merge on drop so
    // the walk stays parallel with minimal contention.
    type Collected = (Vec<(FileItem, String)>, Vec<String>);
    let collected = parking_lot::Mutex::new(Collected::default());

    struct FffVisitor<'a> {
        collected: &'a parking_lot::Mutex<Collected>,
        synced_files_count: &'a Arc<AtomicUsize>,
        base_path: PathBuf,
        local_pairs: Vec<(FileItem, String)>,
        local_dirs: Vec<String>,
        local_count: usize,
    }

    impl<'a> FffVisitor<'a> {
        fn flush_local(&mut self) {
            if self.local_pairs.is_empty() && self.local_dirs.is_empty() {
                return;
            }
            let mut guard = self.collected.lock();
            guard.0.append(&mut self.local_pairs);
            guard.1.append(&mut self.local_dirs);
            drop(guard);
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

                // Ignore walkers sometimes surface files inside `.git/`
                // when the base is itself a git repo — skip them.
                if is_git_file(path) {
                    return ignore::WalkState::Continue;
                }

                let metadata = entry.metadata().ok();
                let (file_item, rel_path) =
                    FileItem::new_from_walk(path, &self.base_path, None, metadata.as_ref());

                self.local_pairs.push((file_item, rel_path));
                self.local_count += 1;
                // Flush before the local batch grows unbounded under bursty
                // closed-loop walkers, so the final append is not one huge
                // critical section and peak RSS stays flatter.
                if self.local_pairs.len() >= 512 {
                    self.flush_local();
                }
            } else if entry.depth() > 0 && entry.file_type().is_some_and(|ft| ft.is_dir()) {
                let path = entry.path();
                if !is_git_file(path)
                    && let Ok(rel) = path.strip_prefix(&self.base_path)
                {
                    let mut rel = crate::path_utils::to_canonical_slashes(&rel.to_string_lossy())
                        .into_owned();
                    rel.push('/');
                    self.local_dirs.push(rel);
                }
            }
            ignore::WalkState::Continue
        }
    }

    impl<'a> Drop for FffVisitor<'a> {
        fn drop(&mut self) {
            self.flush_local();
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
                local_pairs: Vec::with_capacity(2048),
                local_dirs: Vec::with_capacity(256),
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

    let (pairs, dirs) = collected.into_inner();
    Ok(WalkOutput {
        pairs,
        dirs,
        ignore_rules: None,
    })
}
