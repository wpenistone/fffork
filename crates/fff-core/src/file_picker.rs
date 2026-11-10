//! Core file picker: filesystem indexing, background watching, and fuzzy search.
//!
//! [`FilePicker`] is the central component of fff-search. It:
//!
//! 1. **Indexes** a directory tree in a background thread, collecting every
//!    non-ignored file into a path-sorted `Vec<FileItem>`.
//! 2. **Watches** the filesystem via the `notify` crate, applying
//!    create/modify/delete events to the index in real time.
//! 3. **Owns files**: Provides a values for search and provides a good entry point for
//!    fuzzy search and live grep
//!
//! # Lifecycle
//!
//! ```text
//!   new_with_shared_state()
//!     │
//!     ├─> background scan thread ──> populates SharedPicker
//!     └─> file-system watcher    ──> live updates SharedPicker
//!
//!   search()         <── borrows &self, delegates to fuzzy_search
//!   grep()           <── static, borrows &[FileItem] (live content search)
//!   trigger_rescan() <── synchronous re-index
//!   cancel()         <── shuts down background work
//! ```
//!
//! # Thread Safety
//!
//! `FilePicker` itself is **not** `Sync`!
//! all concurrent access goes through [`crate::SharedFilePicker`]

use crate::FFFStringStorage;
use crate::constants::{MAX_OVERFLOW_FILES, PATH_BUF_SIZE};
use crate::error::Error;
use crate::dbs::frecency::ModificationBasis;
use crate::frecency::{FrecencyReader, FrecencyTracker};
use crate::git::{is_modified_status, GitDiscovery, GitStatusCache};
use crate::git_recency::{self, GitRecencyConfig};
use crate::grep::{GrepResult, GrepSearchOptions, grep_search, multi_grep_search};
use crate::index::{BigramFilter, BigramOverlay};
use crate::query_tracker::QueryTracker;
use crate::scan::{ScanConfig, ScanJob, ScanSignals};
use crate::score::{
    fuzzy_match_and_score_files_presence, fuzzy_match_byte_offsets_for_page,
};
use crate::shared::{SharedFilePicker, SharedFrecency};
use crate::simd_path::ArenaPtr;
use crate::stable_vec::StableVec;
use crate::types::{
    ContentCacheBudget, DirItem, DirSearchResult, FileItem, MixedItemRef, MixedSearchResult,
    PaginationArgs, Score, ScoringContext, SearchResult,
};
use crate::walk::WalkSpan;
use crate::watch::BackgroundWatcher;
use ahash::{AHashMap, AHashSet};
use fff_query_parser::{Constraint, FFFQuery};
use git2::{Repository, Status};
use rayon::prelude::*;
use std::fmt::Debug;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread::JoinHandle;
use std::time::SystemTime;
use tracing::{Level, debug, error, info, warn};

use crate::parallelism::BACKGROUND_THREAD_POOL;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FFFMode {
    #[default]
    Neovim,
    Ai,
}

impl FFFMode {
    pub fn is_ai(self) -> bool {
        self == FFFMode::Ai
    }
}

/// Configuration for a single fuzzy search invocation.
///
/// Passed to [`FilePicker::search`] to control threading, pagination,
/// and scoring behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct FuzzySearchOptions<'a> {
    pub max_threads: usize,
    pub current_file: Option<&'a str>,
    pub project_path: Option<&'a Path>,
    pub combo_boost_score_multiplier: i32,
    pub min_combo_count: u32,
    pub pagination: PaginationArgs,
}

#[derive(Debug, Clone)]
pub(crate) struct FileSync {
    pub(crate) git: Option<GitDiscovery>,
    // Base path re-rooted into the main checkout when indexing a linked git worktree,
    // frecency and query history are keyed by it so all worktrees share one history
    shared_base: Option<PathBuf>,
    /// Base files laid out in two partitions, each internally sorted by
    /// (parent_dir, filename):
    ///   `files[..indexable_count]` - indexable
    ///   `files[indexable_count..base_count]` - original-unindexable
    ///   `files[base_count..]` - overflow
    files: StableVec<FileItem>,
    indexable_count: usize,
    base_count: usize,
    /// Number of active present files that exists in the file system
    pub(crate) live_count: usize,
    /// Sorted directory table. `StableVec` so post-scan snapshots can keep
    /// the allocation alive across a picker drop without copying, and so
    /// concurrent readers observe a consistent view via the same shared
    /// allocation. Dir frecency is updated through the per-entry atomic
    /// (`DirItem::max_access_frecency`) without `&mut` aliasing.
    /// Layout mirrors `files`: `dirs[..base_dirs_count]` is the sorted
    /// scan-built region, `dirs[base_dirs_count..]` holds watcher-appended dirs.
    dirs: StableVec<DirItem>,
    base_dirs_count: usize,
    /// Number of dirs with at least one live file (mirrors `live_count`).
    live_dirs_count: usize,
    /// Shared builder for overflow file paths. Each overflow file's ChunkedString
    /// uses `arena_override` pointing into this builder's arena.
    overflow_builder: Option<crate::simd_path::ChunkedPathStoreBuilder>,
    bigram_index: Option<Arc<BigramFilter>>,
    bigram_overlay: Option<Arc<parking_lot::RwLock<BigramOverlay>>>,
    /// Chunk-level deduped path store. Arc so post-scan snapshots can hold
    /// the arena alive while iterating file paths.
    chunked_paths: Option<Arc<crate::simd_path::ChunkedPathStore>>,
    /// Ignore rules the walker assembled (zlob backend only). Shared with the
    /// background watcher so filesystem events can be filtered without libgit2.
    pub(crate) ignore_rules: Option<Arc<crate::walk::WalkIgnoreRules>>,
    /// Cached max path length (see `PathLenCache`). Arc because `FileSync` is
    /// cloned for post-scan snapshots.
    pub(crate) max_path_len: Arc<PathLenCache>,
    /// Content byte-presence masks, indexed by file position.
    pub(crate) content_masks: Arc<ContentMaskStore>,
    /// Path-presence masks, indexed like `files`. Same values as `FileItem::presence`.
    presence: crate::types::PresenceTable,
}

/// Side table of per-file content byte-presence masks.
///
/// Kept out of `FileItem` (packed to 48 B for cache reasons) and indexed by
/// file position. `AtomicU64` because the indexer writes while grep reads.
#[derive(Debug, Default)]
pub(crate) struct ContentMaskStore {
    /// Sized during the scan, before any reader exists, so get stays
    /// lock-free on grep's per-file path. Uninitialised means 'no masks yet'
    /// and every lookup misses, which is safe.
    slots: std::sync::OnceLock<Vec<std::sync::atomic::AtomicU64>>,
    /// Set once any non-empty mask lands. Grep skips the table when this
    /// is false. Mask 0 stays unknown either way.
    has_data: AtomicBool,
}

impl ContentMaskStore {
    pub(crate) fn with_capacity(n: usize) -> Self {
        let s = Self {
            slots: std::sync::OnceLock::new(),
            has_data: AtomicBool::new(false),
        };
        s.ensure_capacity(n);
        s
    }

    /// Size the table. Called once during the scan, before grep can run.
    /// NOTE: OnceLock initialises on first use only, so this must be called
    /// before any set, not after.
    pub(crate) fn ensure_capacity(&self, n: usize) {
        let _ = self.slots.get_or_init(|| {
            (0..n)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect()
        });
    }

    #[inline]
    pub(crate) fn set(&self, index: usize, mask: crate::types::ContentMask) {
        if let Some(slots) = self.slots.get()
            && let Some(slot) = slots.get(index)
        {
            let raw = mask.get();
            slot.store(raw, Ordering::Relaxed);
            if raw != 0 {
                self.has_data.store(true, Ordering::Relaxed);
            }
        }
    }

    /// True once any mask was recorded. False means every lookup misses,
    /// so callers can skip mask work (and its per-file bookkeeping).
    #[inline]
    pub(crate) fn any_recorded(&self) -> bool {
        self.has_data.load(Ordering::Relaxed)
    }

    /// `None` when the index is out of range or no mask was recorded.
    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<crate::types::ContentMask> {
        let slots = self.slots.get()?;
        let raw = slots.get(index)?.load(Ordering::Relaxed);
        if raw == 0 {
            None
        } else {
            Some(crate::types::ContentMask::from_raw(raw))
        }
    }
}

impl FileSync {
    fn new() -> Self {
        Self {
            files: StableVec::from_vec_with_reserve(Vec::new(), MAX_OVERFLOW_FILES),
            indexable_count: 0,
            base_count: 0,
            live_count: 0,
            dirs: StableVec::from_vec_with_reserve(Vec::new(), MAX_OVERFLOW_FILES),
            base_dirs_count: 0,
            live_dirs_count: 0,
            overflow_builder: None,
            git: None,
            shared_base: None,
            bigram_index: None,
            bigram_overlay: None,
            chunked_paths: None,
            max_path_len: Arc::new(PathLenCache::new()),
            content_masks: Arc::new(ContentMaskStore::with_capacity(0)),
            presence: crate::types::PresenceTable::default(),
            ignore_rules: None,
        }
    }

    #[inline]
    fn arena_base_ptr(&self) -> ArenaPtr {
        self.chunked_paths
            .as_ref()
            .map(|s| s.as_arena_ptr())
            .unwrap_or(ArenaPtr::null())
    }

    #[inline]
    fn arena_overflow_ptr(&self) -> ArenaPtr {
        self.overflow_builder
            .as_ref()
            .map(|b| b.as_arena_ptr())
            .unwrap_or(ArenaPtr::null())
    }

    #[inline]
    fn arena_for_file(&self, file: &FileItem) -> ArenaPtr {
        if file.is_overflow() {
            self.arena_overflow_ptr()
        } else {
            self.arena_base_ptr()
        }
    }

    #[inline]
    fn files(&self) -> &[FileItem] {
        &self.files
    }

    #[inline]
    fn overflow_files(&self) -> &[FileItem] {
        &self.files[self.base_count..]
    }

    #[inline]
    fn get_file_mut(&mut self, index: usize) -> Option<(ArenaPtr, &mut FileItem)> {
        Some((
            if index < self.base_count {
                self.arena_base_ptr()
            } else {
                self.arena_overflow_ptr()
            },
            self.files.get_mut(index)?,
        ))
    }

    #[inline]
    fn find_file_index(&self, path: &Path, base_path: &Path) -> Option<usize> {
        // Strip base_path prefix to get the relative path. On Windows this
        // can fail for 8.3 short names or a different casing; fall back to
        // canonicalize-then-strip so watcher events still land on the right
        // `FileItem`.
        let rel_path_owned: String = match path.strip_prefix(base_path) {
            Ok(r) => r.to_string_lossy().into_owned(),
            Err(_) => {
                #[cfg(windows)]
                {
                    canonical_relative_path(path, base_path)?
                }
                #[cfg(not(windows))]
                {
                    return None;
                }
            }
        };
        // The dir table and stored file paths are '/'-canonical; fold the
        // native relative path so the byte-wise comparisons below match.
        self.find_by_relative_path(&crate::path_utils::to_canonical_slashes(&rel_path_owned))
    }

    // Lookup for a base-relative, '/'-canonical path — the form paths are
    // stored in, so no normalization is needed.
    fn find_by_relative_path(&self, rel_path: &str) -> Option<usize> {
        let arena = self.arena_base_ptr();

        // Split into directory (with trailing '/') and filename.
        let parent_end = rel_path
            .rfind(std::path::is_separator)
            .map(|i| i + 1)
            .unwrap_or(0);
        let dir_rel = &rel_path[..parent_end];
        let filename = &rel_path[parent_end..];

        // Binary search dirs to find the parent directory index.
        // Dir items store the relative path including trailing '/' (e.g. "src/components/").
        // Only the scan-built region is sorted; watcher-appended dirs are not.
        let mut dir_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
        let dir_idx = self.dirs[..self.base_dirs_count]
            .binary_search_by(|d| d.read_relative_path(arena, &mut dir_buf).cmp(dir_rel))
            .ok();

        if let Some(dir_idx) = dir_idx {
            let dir_idx = dir_idx as u32;
            let cmp_key = |f: &FileItem| {
                f.parent_dir_index.cmp(&dir_idx).then_with(|| {
                    let fname = f.file_name(arena);
                    fname.as_str().cmp(filename)
                })
            };

            if self.indexable_count > 0
                && let Ok(pos) = self.files[..self.indexable_count].binary_search_by(cmp_key)
            {
                return Some(pos);
            }

            if self.indexable_count < self.base_count
                && let Ok(rel_pos) =
                    self.files[self.indexable_count..self.base_count].binary_search_by(cmp_key)
            {
                return Some(self.indexable_count + rel_pos);
            }
        }

        // Overflow region: linear scan by full relative path.
        if self.base_count < self.files.len() {
            let overflow_arena = self.arena_overflow_ptr();
            if let Some(pos) = self.files[self.base_count..]
                .iter()
                .position(|f| f.relative_path_eq(overflow_arena, rel_path))
            {
                return Some(self.base_count + pos);
            }
        }

        None
    }

    /// Marks every dir matching `predicate` as deleted. Mirrors how dir-level
    /// FS events (remove/move-out) invalidate whole subtrees.
    fn tombstone_dirs_with_arena<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&DirItem, ArenaPtr) -> bool,
    {
        let base_arena = self.arena_base_ptr();
        let overflow_arena = self.arena_overflow_ptr();
        let base_dirs_count = self.base_dirs_count;

        let mut removed = 0usize;
        for (idx, dir) in self.dirs.iter_mut().enumerate() {
            if dir.is_deleted() {
                continue;
            }
            let arena = if idx < base_dirs_count {
                base_arena
            } else {
                overflow_arena
            };
            if predicate(dir, arena) && dir.set_deleted(true) {
                removed += 1;
            }
        }
        self.live_dirs_count -= removed;
    }

    /// Restores a dir to the live state (file appeared under it again).
    fn revive_dir(&mut self, dir_idx: u32) {
        if let Some(dir) = self.dirs.get_mut(dir_idx as usize)
            && dir.set_deleted(false)
        {
            self.live_dirs_count += 1;
        }
    }

    /// Finds the dir index for a '/'-canonical relative dir path
    /// (with trailing '/', empty string for the base dir itself).
    fn find_dir_index(&self, dir_rel: &str) -> Option<usize> {
        let arena = self.arena_base_ptr();
        let mut dir_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
        if let Ok(idx) = self.dirs[..self.base_dirs_count]
            .binary_search_by(|d| d.read_relative_path(arena, &mut dir_buf).cmp(dir_rel))
        {
            return Some(idx);
        }

        // Watcher-appended region: unsorted, small (bounded by overflow cap).
        let overflow_arena = self.arena_overflow_ptr();
        self.dirs[self.base_dirs_count..]
            .iter()
            .position(|d| d.read_relative_path(overflow_arena, &mut dir_buf) == dir_rel)
            .map(|pos| self.base_dirs_count + pos)
    }

    /// Finds or appends the DirItem for `dir_rel`, returning its index.
    /// `None` when the dir table's overflow capacity is exhausted.
    fn find_or_add_dir(&mut self, dir_rel: &str) -> Option<u32> {
        if let Some(idx) = self.find_dir_index(dir_rel) {
            return Some(idx as u32);
        }

        let builder = self.overflow_builder.get_or_insert_with(|| {
            crate::simd_path::ChunkedPathStoreBuilder::new(MAX_OVERFLOW_FILES)
        });
        let chunked = builder.add_dir_immediate(dir_rel);

        let last_seg = if dir_rel.is_empty() {
            0
        } else {
            let trimmed = dir_rel.trim_end_matches(std::path::is_separator);
            trimmed
                .rfind(std::path::is_separator)
                .map(|i| i + 1)
                .unwrap_or(0) as u16
        };

        let idx = self.dirs.len();
        if !self.dirs.push(DirItem::new_overflow(chunked, last_seg)) {
            return None;
        }
        self.live_dirs_count += 1;
        Some(idx as u32)
    }
}

/// Running max of `FileItem.path.byte_len` across live files. Monotonic on
/// insert (`fetch_max`), marked stale when a tombstone removes the current
/// max and recomputed lazily on the next read. Search needs this per query to
/// size its thread estimate, and walking the file list for it is O(n) on every
/// keystroke.
#[derive(Debug)]
pub(crate) struct PathLenCache {
    value: std::sync::atomic::AtomicU32,
    stale: std::sync::atomic::AtomicBool,
}

impl PathLenCache {
    fn new() -> Self {
        Self {
            value: std::sync::atomic::AtomicU32::new(0),
            stale: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub(crate) fn observe(&self, byte_len: u16) {
        self.value.fetch_max(byte_len as u32, Ordering::Relaxed);
    }

    pub(crate) fn forget(&self, byte_len: u16) {
        if self.value.load(Ordering::Relaxed) == byte_len as u32 {
            self.stale.store(true, Ordering::Relaxed);
        }
    }

    pub(crate) fn get(&self) -> usize {
        if self.stale.swap(false, Ordering::Relaxed) {
            self.stale.store(true, Ordering::Relaxed); // caller recomputes, then clears
        }
        self.value.load(Ordering::Relaxed) as usize
    }

    /// Recompute after a removal invalidated the cache. Only lowers the value,
    /// and only while no larger insert lands (CAS loop) so a concurrent
    /// `observe` is never lost.
    pub(crate) fn recompute(&self, files: &[FileItem]) {
        let mut m = 0u32;
        for f in files.iter() {
            if !f.is_deleted() {
                m = m.max(f.path.byte_len as u32);
            }
        }
        let mut cur = self.value.load(Ordering::Relaxed);
        while cur > m {
            match self
                .value
                .compare_exchange_weak(cur, m, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
        self.stale.store(false, Ordering::Relaxed);
    }
}

impl FileItem {
    pub fn new(path: PathBuf, base_path: &Path, git_status: Option<Status>) -> (Self, String) {
        let metadata = std::fs::metadata(&path).ok();
        Self::new_with_metadata(path, base_path, git_status, metadata.as_ref())
    }

    /// Create a FileItem using pre-fetched metadata to avoid a redundant stat syscall.
    /// Returns `(FileItem, relative_path)`. The FileItem's `path` field is
    /// empty; callers must populate it via `set_path` or `build_chunked_path_store_and_assign`.
    fn new_with_metadata(
        path: PathBuf,
        base_path: &Path,
        git_status: Option<Status>,
        metadata: Option<&std::fs::Metadata>,
    ) -> (Self, String) {
        let path_buf = pathdiff::diff_paths(&path, base_path).unwrap_or_else(|| path.clone());
        // The index is '/'-canonical on every platform; fold native separators.
        let relative_path =
            crate::path_utils::to_canonical_slashes(&path_buf.to_string_lossy()).into_owned();

        let (size, modified) = match metadata {
            Some(metadata) => {
                let size = metadata.len();
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());

                (size, modified)
            }
            None => (0, 0),
        };

        let is_binary = is_known_binary_extension(&path);

        let filename_start = relative_path
            .rfind(std::path::is_separator)
            .map(|i| i + 1)
            .unwrap_or(0) as u16;

        let mut item = Self::new_raw(filename_start, size, modified, git_status, is_binary);
        item.mark_definition_language(&relative_path);
        (item, relative_path)
    }

    /// Create a FileItem with an empty ChunkedString from a path on disk.
    ///
    /// Returns `(file_item, relative_path_string)`. The relative path must be
    /// kept alongside the FileItem until `build_chunked_path_store_and_assign`
    /// populates each item's `path` field from the shared arena.
    pub fn new_from_walk(
        path: &Path,
        base_path: &Path,
        git_status: Option<Status>,
        metadata: Option<&std::fs::Metadata>,
    ) -> (Self, String) {
        let (size, modified) = match metadata {
            Some(metadata) => {
                let size = metadata.len();
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());
                (size, modified)
            }
            None => (0, 0),
        };

        Self::new_from_walk_parts(path, base_path, git_status, size, modified)
    }

    /// Like [`Self::new_from_walk`] but takes already-extracted size and
    /// modification time (Unix seconds) instead of a `std::fs::Metadata`.
    /// Used by the zlob walker backend, which fetches metadata in bulk.
    pub fn new_from_walk_parts(
        path: &Path,
        base_path: &Path,
        git_status: Option<Status>,
        size: u64,
        modified: u64,
    ) -> (Self, String) {
        let is_binary = is_known_binary_extension(path);

        let rel = pathdiff::diff_paths(path, base_path).unwrap_or_else(|| path.to_path_buf());
        // The index is '/'-canonical on every platform; fold native separators.
        let rel_str = crate::path_utils::to_canonical_slashes(&rel.to_string_lossy()).into_owned();
        let fname_offset = rel_str
            .rfind(std::path::is_separator)
            .map(|i| i + 1)
            .unwrap_or(0) as u16;

        let mut item = Self::new_raw(fname_offset, size, modified, git_status, is_binary);
        item.mark_definition_language(&rel_str);
        (item, rel_str)
    }

    /// Zlob-walker fast path: skip the `pathdiff::diff_paths` PathBuf alloc by
    /// taking the already-relative slice and the basename-offset that zlob's
    /// scanner computed during traversal. ~80–120 ms saved on a chromium scan
    /// (500k entries × one fewer alloc + no component walk).
    ///
    /// `relative_path` is root-relative bytes; `basename_offset` is the byte
    /// offset where the basename begins (e.g. zlob's `entry.path_bytes().len()
    /// - entry.file_name().as_os_str().as_encoded_bytes().len()` minus the
    /// `relative_offset`).
    pub fn new_from_walk_bytes(
        path: &Path,
        relative_path: &[u8],
        basename_offset: u16,
        git_status: Option<Status>,
        size: u64,
        modified: u64,
    ) -> (Self, String) {
        let is_binary = is_known_binary_extension(path);
        // SAFETY-ish: paths on macOS/Linux are bytes; lossy conversion mirrors
        // the existing `to_string_lossy()` behavior on non-UTF8 names.
        let decoded = String::from_utf8_lossy(relative_path).into_owned();
        // The caller's offset indexes the raw bytes; re-measure it against the
        // decoded string so it never lands inside a U+FFFD (#799).
        let dir_bytes = relative_path
            .get(..basename_offset as usize)
            .unwrap_or(relative_path);
        let basename_offset = if std::str::from_utf8(dir_bytes).is_ok() {
            basename_offset
        } else {
            String::from_utf8_lossy(dir_bytes).len() as u16
        };
        let mut item = Self::new_raw(basename_offset, size, modified, git_status, is_binary);
        item.mark_definition_language(&decoded);
        (item, decoded)
    }

    pub(crate) fn update_access_frecency_score(
        &self,
        reader: &FrecencyReader,
        arena: ArenaPtr,
        base_path: &Path,
        shared_base: Option<&Path>,
        mode: FFFMode,
    ) -> i32 {
        let mut abs_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
        let abs = self.write_absolute_path(arena, base_path, &mut abs_buf);
        let access_score = match shared_base {
            Some(shared_base) => {
                let mut shared_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
                let shared = self.write_absolute_path(arena, shared_base, &mut shared_buf);
                reader.access_score(abs, Some(shared), mode)
            }
            None => reader.access_score(abs, None, mode),
        };

        let stored = access_score as i16;
        self.set_access_frecency_score(stored);
        stored as i32
    }

    // Caller holds one basis. A clean file that is already 0 does not store.
    pub(crate) fn apply_modification_frecency(&self, basis: &ModificationBasis) {
        let git = self.git_status();
        if !git.is_some_and(is_modified_status) {
            if self.modification_frecency_score() != 0 {
                self.set_modification_frecency_score(0);
            }
            return;
        }
        self.set_modification_frecency_score(basis.score_modified(self.modified()) as i16);
    }

    pub(crate) fn update_frecency_scores(
        &self,
        reader: &FrecencyReader,
        arena: ArenaPtr,
        base_path: &Path,
        shared_base: Option<&Path>,
        mode: FFFMode,
        basis: &ModificationBasis,
    ) -> i32 {
        let access = self.update_access_frecency_score(reader, arena, base_path, shared_base, mode);
        self.apply_modification_frecency(basis);
        access
    }
}

/// Options for creating a [`FilePicker`].
pub struct FilePickerOptions {
    pub base_path: String,
    /// Pre-populate mmap caches for top-frecency files after the initial scan
    pub enable_mmap_cache: bool,
    /// Build content index after the initial scan for faster content-aware filtering
    pub enable_content_indexing: bool,
    /// Mode of the picker impact the way file watcher events are handled and the scoring logic
    pub mode: FFFMode,
    /// Explicit cache budget. When `None`, the budget is auto-computed from
    /// the repo size after the initial scan completes.
    pub cache_budget: Option<ContentCacheBudget>,
    /// When `false` no background watcher will be created
    pub watch: bool,
    /// Follow symbolic links during file indexing
    pub follow_symlinks: bool,
    /// Allow indexing the filesystem root (`/`)
    pub enable_fs_root_scanning: bool,
    /// Allow indexing the user's home directory. Off by default for the same
    /// reason as `enable_fs_root_scanning`
    pub enable_home_dir_scanning: bool,
    /// Ranking boost for files that participated in recent commits of the
    /// current branch. Enabled with default limits unless overridden.
    pub git_recency: GitRecencyConfig,
    /// Do not discover .git, run git status, or apply git recency. Worktree
    /// search only — required for mega-corpus benches (sumo_fork has a 2.7GB
    /// .git that is not a search corpus). Indexes above 200k files skip
    /// automatically unless FFF_FORCE_GIT=1.
    pub skip_git: bool,
}

impl Default for FilePickerOptions {
    fn default() -> Self {
        Self {
            base_path: ".".into(),
            enable_mmap_cache: false,
            enable_content_indexing: false,
            mode: FFFMode::default(),
            cache_budget: None,
            watch: true,
            follow_symlinks: false,
            enable_fs_root_scanning: false,
            enable_home_dir_scanning: false,
            git_recency: GitRecencyConfig::default(),
            skip_git: false,
        }
    }
}

/// Live files above this count skip git status/recency automatically.
/// File count is a free post-walk proxy; full status on huge trees
/// dominates query time. `FFF_FORCE_GIT=1` opts back in; explicit
/// `skip_git` always wins.
pub(crate) const AUTO_SKIP_GIT_FILE_COUNT: usize = 200_000;

/// Single decision point for skipping git work.
pub(crate) fn should_skip_git(explicit_skip: bool, live_count: usize, forced: bool) -> bool {
    explicit_skip || (!forced && live_count > AUTO_SKIP_GIT_FILE_COUNT)
}

/// `FFF_FORCE_GIT` set disables auto-skip.
pub(crate) fn git_force() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_FORCE_GIT").is_some())
}

fn arena_stats_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_ARENA_STATS").is_some())
}

fn no_prefix_history() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_NO_TRIE").is_some())
}

pub struct FilePicker {
    pub mode: FFFMode,
    pub base_path: PathBuf,
    sync_data: FileSync,
    pub(crate) signals: ScanSignals,
    pub(crate) background_watcher: Option<BackgroundWatcher>,
    /// Single serialized writer for all git-status updates (scan, watcher,
    /// FFI). Owned by the picker so it exists before the first scan; its
    /// consumer thread is spawned lazily once a git workdir is discovered.
    pub(crate) git_status_worker: Arc<crate::git_status_worker::GitStatusWorker>,
    cache_budget: Arc<ContentCacheBudget>,
    has_explicit_cache_budget: bool,
    scanned_files_count: Arc<AtomicUsize>,
    enable_mmap_cache: bool,
    enable_content_indexing: bool,
    watch: bool,
    follow_symlinks: bool,
    enable_fs_root_scanning: bool,
    enable_home_dir_scanning: bool,
    git_recency_config: GitRecencyConfig,
    skip_git: bool,
    trace_span: tracing::Span,
    trace_id: String,
    /// Live candidate narrowing for typing. Sound only when the new query
    /// extends the previous one **and** `max_typos` is unchanged (then
    /// Match(longer,T) ⊆ Match(shorter,T)). Invalidated on index change.
    live: parking_lot::Mutex<LiveNarrow>,
    /// Bumped whenever the file set changes so live state is dropped.
    index_generation: std::sync::atomic::AtomicU64,
    // Status writes do not bump index_generation. A GitStatus hit set
    // is reusable only while this epoch is unchanged.
    git_status_epoch: std::sync::atomic::AtomicU64,
    /// Exact-repeat path-match memo for live-narrow subsets. Keyed by index
    /// generation, query, and the exact candidate list.
    match_cache: parking_lot::Mutex<crate::score::MatchCache>,
    // One completed page. A key hit rebuilds it and skips match, score, and offsets.
    scored_page: parking_lot::Mutex<Option<ScoredPage>>,
    /// Per-keystroke search epoch: each fuzzy search bumps it at entry,
    /// workers bail their slice when it moves. Per-picker so concurrent
    /// searches on different pickers never abort each other.
    search_epoch: std::sync::atomic::AtomicU64,
}

// Page path buffers. The last owner on this thread keeps the allocation.
thread_local! {
    static PATH_BYTES: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    static PATH_ROWS: std::cell::RefCell<Vec<crate::types::CachedFileRow>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static PAGE_IDX: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn take_path_bytes() -> Vec<u8> {
    let mut buf = PATH_BYTES.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    buf.clear();
    buf
}

fn take_path_rows() -> Vec<crate::types::CachedFileRow> {
    let mut buf = PATH_ROWS.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    buf.clear();
    buf
}

fn park_path_bytes(bytes: Arc<Vec<u8>>) {
    if let Ok(buf) = Arc::try_unwrap(bytes) {
        PATH_BYTES.with(|slot| park_larger(buf, slot));
    }
}

fn park_path_rows(rows: Arc<Vec<crate::types::CachedFileRow>>) {
    if let Ok(buf) = Arc::try_unwrap(rows) {
        PATH_ROWS.with(|slot| park_larger(buf, slot));
    }
}

fn take_page_idx() -> Vec<u32> {
    let mut buf = PAGE_IDX.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    buf.clear();
    buf
}

fn park_page_idx(indices: std::sync::Arc<Vec<u32>>) {
    if let Ok(buf) = std::sync::Arc::try_unwrap(indices) {
        park_page_idx_vec(buf);
    }
}

fn park_page_idx_vec(buf: Vec<u32>) {
    PAGE_IDX.with(|slot| park_larger(buf, slot));
}

fn reuse_string(mut buf: String, next: &str) -> String {
    buf.clear();
    buf.push_str(next);
    buf
}

fn reuse_opt_string(buf: Option<String>, next: Option<&str>) -> Option<String> {
    match (buf, next) {
        (Some(buf), Some(next)) => Some(reuse_string(buf, next)),
        (None, Some(next)) => Some(next.to_string()),
        (_, None) => None,
    }
}

fn reuse_opt_bytes(buf: Option<Vec<u8>>, next: Option<&[u8]>) -> Option<Vec<u8>> {
    match (buf, next) {
        (Some(mut buf), Some(next)) => {
            buf.clear();
            buf.extend_from_slice(next);
            Some(buf)
        }
        (None, Some(next)) => Some(next.to_vec()),
        (_, None) => None,
    }
}

fn park_larger<T>(mut buf: Vec<T>, slot: &std::cell::RefCell<Vec<T>>) {
    buf.clear();
    let mut slot = slot.borrow_mut();
    if buf.capacity() > slot.capacity() {
        *slot = buf;
    }
}

impl FilePicker {
    #[inline]
    pub(crate) fn index_generation(&self) -> u64 {
        self.index_generation.load(Ordering::Relaxed)
    }

    #[inline]
    pub(crate) fn bump_index_generation(&self) {
        {
            // Generation and `active` move together. A publish holding this
            // lock either installs the pre-bump set or sees the new generation.
            let mut live = self.live.lock();
            self.index_generation.fetch_add(1, Ordering::Relaxed);
            live.active = false;
        }
        self.match_cache.lock().clear();
    }

    #[inline]
    fn current_git_status_epoch(&self) -> u64 {
        self.git_status_epoch.load(Ordering::Relaxed)
    }

    #[inline]
    fn bump_git_status_epoch(&self) {
        self.git_status_epoch.fetch_add(1, Ordering::Relaxed);
        crate::types::bump_score_input_epoch();
    }

    fn replay_scored_page<'a>(
        &'a self,
        files: &'a [FileItem],
        key: &PageKey<'_>,
    ) -> Option<(
        Vec<&'a FileItem>,
        crate::types::SharedScores,
        usize,
        crate::types::SharedOffsets,
        crate::types::PathSnapshot,
    )> {
        let (indices, scores, offsets, total_matched, paths) = {
            let guard = self.scored_page.lock();
            let page = guard.as_ref()?;
            if !page.matches(key) {
                return None;
            }
            if page.indices.len() != page.scores.len() || page.indices.len() != page.offsets.len()
            {
                return None;
            }
            (
                std::sync::Arc::clone(&page.indices),
                std::sync::Arc::clone(&page.scores),
                std::sync::Arc::clone(&page.offsets),
                page.total_matched,
                crate::types::PathSnapshot {
                    bytes: std::sync::Arc::clone(&page.path_bytes),
                    rows: std::sync::Arc::clone(&page.path_rows),
                },
            )
        };
        let mut items = Vec::with_capacity(indices.len());
        for &idx in indices.iter() {
            items.push(files.get(idx as usize)?);
        }
        Some((items, scores, total_matched, offsets, paths))
    }

    fn snapshot_paths(&self, items: &[&FileItem]) -> crate::types::PathSnapshot {
        let mut bytes = take_path_bytes();
        // One reserve from the path cap when the reused byte buffer is shorter.
        let want = items.len().saturating_mul(self.max_path_len().clamp(1, 512));
        if bytes.capacity() < want {
            bytes.reserve(want);
        }
        let mut rows = take_path_rows();
        rows.reserve(items.len());
        for item in items {
            let len = item.relative_path_len();
            let start = bytes.len() as u32;
            let at = bytes.len();
            if len > 0 {
                let arena = self.arena_for(item);
                // The copy writes every new byte, so the buffer is not zeroed first.
                bytes.reserve(len);
                unsafe {
                    item.path.copy_trimmed_ptr(
                        arena,
                        0,
                        len as u32,
                        bytes.as_mut_ptr().add(at),
                    );
                    bytes.set_len(at + len);
                }
            }
            let hot = item.score_hot();
            rows.push(crate::types::CachedFileRow {
                start,
                len: len as u32,
                name_off: item.filename_offset_in_relative_path().min(len) as u32,
                size: item.size(),
                modified: item.modified(),
                access_frecency_score: hot.access,
                modification_frecency_score: hot.modification,
                total_frecency_score: hot.access as i32 + hot.modification as i32,
                git_status: crate::git::format_git_status(item.git_status()),
                git_recency_score: hot.git_recency,
                is_binary: hot.binary,
            });
        }
        crate::types::PathSnapshot {
            bytes: std::sync::Arc::new(bytes),
            rows: std::sync::Arc::new(rows),
        }
    }

    fn remember_scored_page(
        &self,
        abort: crate::parallelism::AbortToken<'_>,
        files: &[FileItem],
        items: &[&FileItem],
        scores: &crate::types::SharedScores,
        offsets: &crate::types::SharedOffsets,
        paths: &crate::types::PathSnapshot,
        total_matched: usize,
        key: &PageKey<'_>,
    ) {
        // Finished page only: abort, or a generation/epoch move, is a partial result.
        if abort.aborted()
            || self.index_generation() != key.generation
            || crate::types::score_input_epoch() != key.score_epoch
            || items.len() != scores.len()
            || items.len() != offsets.len()
        {
            return;
        }
        let Some(indices) = file_slots(files, items) else {
            return;
        };
        let recycled = {
            let mut guard = self.scored_page.lock();
            let old = guard.take();
            let (query, current_file, combo_path, recycled) = if let Some(mut old) = old {
                let query = reuse_string(std::mem::take(&mut old.query), key.query);
                let current_file = reuse_opt_string(old.current_file.take(), key.current_file);
                let combo_path = reuse_opt_bytes(old.combo_path.take(), key.combo_path);
                (
                    query,
                    current_file,
                    combo_path,
                    Some((
                        old.path_bytes,
                        old.path_rows,
                        old.indices,
                        old.offsets,
                        old.scores,
                    )),
                )
            } else {
                (
                    key.query.to_string(),
                    key.current_file.map(str::to_string),
                    key.combo_path.map(|p| p.to_vec()),
                    None,
                )
            };
            *guard = Some(ScoredPage {
                generation: key.generation,
                score_epoch: key.score_epoch,
                query,
                max_typos: key.max_typos,
                constraints_fp: key.constraints_fp,
                current_file,
                combo_path,
                combo_open_count: key.combo_open_count,
                min_combo_count: key.min_combo_count,
                combo_multiplier: key.combo_multiplier,
                offset: key.offset,
                limit: key.limit,
                indices: std::sync::Arc::new(indices),
                scores: std::sync::Arc::clone(scores),
                offsets: std::sync::Arc::clone(offsets),
                path_bytes: std::sync::Arc::clone(&paths.bytes),
                path_rows: std::sync::Arc::clone(&paths.rows),
                total_matched,
            });
            recycled
        };
        if let Some((bytes, rows, indices, offsets, scores)) = recycled {
            park_path_bytes(bytes);
            park_path_rows(rows);
            park_page_idx(indices);
            crate::score::park_offset_rows(offsets);
            crate::score::park_page_scores(scores);
        }
    }

    // Skip the atomic store when the status word is unchanged.
    fn store_git_status(file: &FileItem, next: Option<Status>) -> bool {
        if file.git_status() == next {
            return false;
        }
        file.set_git_status(next);
        true
    }

    /// Max path byte length across live files (0 when the index is empty).
    /// Maintained incrementally so search never rescans the file list.
    /// Shared handle to the content byte-presence masks, for the scan pipeline.
    pub(crate) fn content_mask_store(&self) -> std::sync::Arc<ContentMaskStore> {
        self.sync_data.content_masks.clone()
    }

    pub(crate) fn max_path_len(&self) -> usize {
        let cache = &self.sync_data.max_path_len;
        if cache.stale.load(Ordering::Relaxed) {
            cache.recompute(&self.sync_data.files);
        }
        cache.get()
    }
}

#[derive(Default)]
struct LiveNarrow {
    generation: u64,
    query: String,
    max_typos: u16,
    /// Parsed filter set for `query`; narrowing may only reuse a hit set
    /// when the new query has the exact same constraints.
    constraints_fp: u64,
    /// File indices (into sync_data.files) that matched the previous query.
    candidates: Arc<Vec<u32>>,
    git_status_epoch: u64,
    /// Files already scored at previous length — set only when we stored a hit set.
    active: bool,
    /// Recent hit sets by query: shortening the query (backspace) reuses
    /// the matching entry instead of rescanning. Bounded below.
    history: Vec<BackEntry>,
    /// Visited prefixes for forward retype: history only serves exact
    /// repeats, this serves any cached prefix of a new query. Small
    /// selective sets only (bounded count and size).
    prefix: Vec<BackEntry>,
}

struct BackEntry {
    query: Arc<str>,
    max_typos: u16,
    generation: u64,
    constraints_fp: u64,
    git_status_epoch: u64,
    candidates: Arc<Vec<u32>>,
}

struct PageKey<'a> {
    generation: u64,
    score_epoch: u64,
    query: &'a str,
    max_typos: u16,
    constraints_fp: u64,
    current_file: Option<&'a str>,
    combo_path: Option<&'a [u8]>,
    combo_open_count: u32,
    min_combo_count: u32,
    combo_multiplier: i32,
    offset: usize,
    limit: usize,
}

struct ScoredPage {
    generation: u64,
    score_epoch: u64,
    query: String,
    max_typos: u16,
    constraints_fp: u64,
    current_file: Option<String>,
    combo_path: Option<Vec<u8>>,
    combo_open_count: u32,
    min_combo_count: u32,
    combo_multiplier: i32,
    offset: usize,
    limit: usize,
    indices: std::sync::Arc<Vec<u32>>,
    scores: crate::types::SharedScores,
    offsets: crate::types::SharedOffsets,
    path_bytes: std::sync::Arc<Vec<u8>>,
    path_rows: std::sync::Arc<Vec<crate::types::CachedFileRow>>,
    total_matched: usize,
}

impl ScoredPage {
    fn matches(&self, key: &PageKey<'_>) -> bool {
        self.query == key.query
            && self.generation == key.generation
            && self.score_epoch == key.score_epoch
            && self.max_typos == key.max_typos
            && self.constraints_fp == key.constraints_fp
            && self.offset == key.offset
            && self.limit == key.limit
            && self.min_combo_count == key.min_combo_count
            && self.combo_multiplier == key.combo_multiplier
            && self.combo_open_count == key.combo_open_count
            && self.current_file.as_deref() == key.current_file
            && self.combo_path.as_deref() == key.combo_path
    }
}

fn file_slots(files: &[FileItem], items: &[&FileItem]) -> Option<Vec<u32>> {
    let stride = std::mem::size_of::<FileItem>();
    if stride == 0 {
        return None;
    }
    let base = files.as_ptr() as usize;
    let mut out = take_page_idx();
    out.reserve(items.len());
    for item in items {
        let addr = std::ptr::from_ref(*item) as usize;
        if addr < base || (addr - base) % stride != 0 {
            park_page_idx_vec(out);
            return None;
        }
        let idx = (addr - base) / stride;
        if idx >= files.len() {
            park_page_idx_vec(out);
            return None;
        }
        let Ok(idx) = u32::try_from(idx) else {
            park_page_idx_vec(out);
            return None;
        };
        out.push(idx);
    }
    Some(out)
}

fn hash_bytes(h: &mut u64, bytes: &[u8]) {
    const PRIME: u64 = 0x100000001b3;
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let v = u64::from_le(unsafe {
            core::ptr::read_unaligned(bytes.as_ptr().add(i) as *const u64)
        });
        *h ^= v as u8 as u64;
        *h = h.wrapping_mul(PRIME);
        *h ^= (v >> 8) as u8 as u64;
        *h = h.wrapping_mul(PRIME);
        *h ^= (v >> 16) as u8 as u64;
        *h = h.wrapping_mul(PRIME);
        *h ^= (v >> 24) as u8 as u64;
        *h = h.wrapping_mul(PRIME);
        *h ^= (v >> 32) as u8 as u64;
        *h = h.wrapping_mul(PRIME);
        *h ^= (v >> 40) as u8 as u64;
        *h = h.wrapping_mul(PRIME);
        *h ^= (v >> 48) as u8 as u64;
        *h = h.wrapping_mul(PRIME);
        *h ^= (v >> 56) as u8 as u64;
        *h = h.wrapping_mul(PRIME);
        i += 8;
    }
    for &b in &bytes[i..] {
        *h ^= b as u64;
        *h = h.wrapping_mul(PRIME);
    }
    *h ^= 0xff;
    *h = h.wrapping_mul(PRIME);
}

// Canonical enough for equality: every variant and payload contributes.
fn hash_constraint(h: &mut u64, c: &Constraint<'_>) {
    match c {
        Constraint::Extension(s) => {
            hash_bytes(h, b"ext");
            hash_bytes(h, s.as_bytes());
        }
        Constraint::Glob(s) => {
            hash_bytes(h, b"glob");
            hash_bytes(h, s.as_bytes());
        }
        Constraint::Parts(parts) => {
            hash_bytes(h, b"parts");
            hash_bytes(h, &(parts.len() as u64).to_le_bytes());
            for p in parts.iter() {
                hash_bytes(h, p.as_bytes());
            }
        }
        Constraint::Text(s) => {
            hash_bytes(h, b"text");
            hash_bytes(h, s.as_bytes());
        }
        Constraint::Exclude(parts) => {
            hash_bytes(h, b"exclude");
            hash_bytes(h, &(parts.len() as u64).to_le_bytes());
            for p in parts.iter() {
                hash_bytes(h, p.as_bytes());
            }
        }
        Constraint::PathSegment(s) => {
            hash_bytes(h, b"segment");
            hash_bytes(h, s.as_bytes());
        }
        Constraint::FilePath(s) => {
            hash_bytes(h, b"filepath");
            hash_bytes(h, s.as_bytes());
        }
        Constraint::FileType(s) => {
            hash_bytes(h, b"filetype");
            hash_bytes(h, s.as_bytes());
        }
        Constraint::GitStatus(status) => {
            hash_bytes(h, b"gitstatus");
            hash_bytes(h, &[*status as u8]);
        }
        Constraint::Not(inner) => {
            hash_bytes(h, b"not");
            hash_constraint(h, inner);
        }
    }
}

fn constraints_fingerprint(constraints: &[Constraint<'_>]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    hash_bytes(&mut h, &(constraints.len() as u64).to_le_bytes());
    for c in constraints {
        hash_constraint(&mut h, c);
    }
    h
}

// Subset scoring chases an index per file; the gate it skips is much
// larger. Narrow only when the set drops at least 1/16 of the corpus.
fn narrow_pays(subset_len: usize, files_len: usize) -> bool {
    (subset_len as u64) * 16 < (files_len as u64) * 15
}

fn constraints_use_git_status(constraints: &[Constraint<'_>]) -> bool {
    constraints.iter().any(constraint_uses_git_status)
}

fn constraint_uses_git_status(c: &Constraint<'_>) -> bool {
    match c {
        Constraint::GitStatus(_) => true,
        Constraint::Not(inner) => constraint_uses_git_status(inner),
        _ => false,
    }
}

#[inline]
fn git_epoch_ok(constrained: bool, stored: u64, current: u64) -> bool {
    !constrained || stored == current
}

// Parts the matcher actually scores. Fold is built once per search.
struct PartFolds<'a> {
    nps: [std::mem::MaybeUninit<crate::presence::NeedlePresence>; 8],
    needles: [&'a [u8]; 8],
    n: usize,
}

impl<'a> PartFolds<'a> {
    fn build(query: &'a FFFQuery<'a>, max_typos: u16) -> Self {
        let mut folds = Self {
            // Only `n` slots are written. The rest are never read.
            nps: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            needles: [b""; 8],
            n: 0,
        };
        match &query.fuzzy_query {
            fff_query_parser::FuzzyQuery::Text(t) => folds.push(t, max_typos),
            fff_query_parser::FuzzyQuery::Parts(parts) => {
                for part in parts.iter() {
                    folds.push(part, max_typos);
                }
            }
            _ => {}
        }
        folds
    }

    fn push(&mut self, part: &'a str, max_typos: u16) {
        if part.len() < 2 || self.n == self.nps.len() {
            return;
        }
        let typos = max_typos.min(part.len() as u16);
        let np = crate::presence::NeedlePresence::new(part.as_bytes(), typos);
        if !np.should_check() {
            return;
        }
        self.needles[self.n] = part.as_bytes();
        self.nps[self.n].write(np);
        self.n += 1;
    }

    #[inline(always)]
    fn np(&self, i: usize) -> &crate::presence::NeedlePresence {
        debug_assert!(i < self.n);
        unsafe { self.nps[i].assume_init_ref() }
    }

    #[inline(always)]
    fn allows(&self, mask: u64) -> bool {
        if mask == 0 || self.n == 0 {
            return true;
        }
        if self.n == 1 {
            return self.np(0).allows(mask, self.needles[0]);
        }
        for i in 0..self.n {
            if !self.np(i).allows(mask, self.needles[i]) {
                return false;
            }
        }
        true
    }
}

// Hit slots increase, but they are not adjacent. The mask eight hits ahead covers that gap.
#[inline(always)]
fn prefetch_hit_mask(presence: &[u64], candidates: &[u32], n: usize) {
    #[cfg(target_arch = "x86_64")]
    if let Some(&next) = candidates.get(n + 8) {
        let idx = next as usize;
        if idx < presence.len() {
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    presence.as_ptr().add(idx) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (presence, candidates, n);
}

fn filter_hits(
    candidates: &[u32],
    presence: &[u64],
    files: &[FileItem],
    folds: &PartFolds<'_>,
) -> Vec<u32> {
    let mut orig = crate::matcher::take_idx_buf();
    orig.reserve(candidates.len());
    // Hit indices are file slots. A full mask table is addressed directly.
    if presence.len() >= files.len() {
        if folds.n == 1 && folds.np(0).distinct() {
            let needle_mask = folds.np(0).distinct_mask();
            let need_hits = folds.np(0).min_hits();
            let cover = folds.np(0).cover_eq();
            for (n, &i) in candidates.iter().enumerate() {
                prefetch_hit_mask(presence, candidates, n);
                debug_assert!((i as usize) < presence.len());
                let mask = unsafe { *presence.get_unchecked(i as usize) };
                let hit = mask & needle_mask;
                let keep = mask == 0
                    || if cover {
                        hit == needle_mask
                    } else {
                        hit.count_ones() as usize >= need_hits
                    };
                if keep {
                    orig.push(i);
                }
            }
            return orig;
        }
        for (n, &i) in candidates.iter().enumerate() {
            prefetch_hit_mask(presence, candidates, n);
            debug_assert!((i as usize) < presence.len());
            let mask = unsafe { *presence.get_unchecked(i as usize) };
            if folds.allows(mask) {
                orig.push(i);
            }
        }
        return orig;
    }
    for (n, &i) in candidates.iter().enumerate() {
        prefetch_hit_mask(presence, candidates, n);
        let Some(mask) = presence.get(i as usize).copied().or_else(|| {
            files.get(i as usize).map(|f| f.presence_mask())
        }) else {
            continue;
        };
        if folds.allows(mask) {
            orig.push(i);
        }
    }
    orig
}

enum NarrowPlan {
    Full,
    Narrow { orig: crate::matcher::CandList },
    // Previous hits narrow to nothing: Match(new) is a subset
    // of old hits (same typos+generation), so it is empty too.
    Empty,
}

fn plan_owned(orig: Vec<u32>, files_len: usize) -> NarrowPlan {
    if orig.is_empty() {
        crate::matcher::reclaim_idx_buf(orig);
        NarrowPlan::Empty
    } else if narrow_pays(orig.len(), files_len) {
        // The cache holds this same list. A later keystroke does not copy it.
        NarrowPlan::Narrow {
            orig: crate::matcher::CandList::Shared(std::sync::Arc::new(orig)),
        }
    } else {
        crate::matcher::reclaim_idx_buf(orig);
        NarrowPlan::Full
    }
}

fn plan_indices(indices: Arc<Vec<u32>>, files_len: usize) -> NarrowPlan {
    if indices.is_empty() {
        NarrowPlan::Empty
    } else if narrow_pays(indices.len(), files_len) {
        NarrowPlan::Narrow {
            orig: crate::matcher::CandList::Shared(indices),
        }
    } else {
        NarrowPlan::Full
    }
}

// Longest cached prefix of `q` (same typos+generation): serves
// retype-after-backspace, where history's exact match misses.
fn longest_prefix(
    cache: &[BackEntry],
    q: &str,
    max_typos: u16,
    generation: u64,
    constraints_fp: u64,
    git_constrained: bool,
    git_epoch: u64,
) -> Option<Arc<Vec<u32>>> {
    if no_prefix_history() {
        return None;
    }
    cache
        .iter()
        .filter(|e| {
            e.generation == generation
                && e.max_typos == max_typos
                && e.constraints_fp == constraints_fp
                && git_epoch_ok(git_constrained, e.git_status_epoch, git_epoch)
                && !e.query.is_empty()
                && q.len() > e.query.len()
                && q.starts_with(e.query.as_ref())
        })
        .max_by_key(|e| e.query.len())
        .map(|e| Arc::clone(&e.candidates))
}

impl std::fmt::Debug for FilePicker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilePicker")
            .field("base_path", &self.base_path)
            .field("sync_data", &self.sync_data)
            .field(
                "is_scanning",
                &self.signals.scanning.load(Ordering::Relaxed),
            )
            .field(
                "scanned_files_count",
                &self.scanned_files_count.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl FFFStringStorage for &FilePicker {
    #[inline]
    fn arena_for(&self, file: &FileItem) -> crate::simd_path::ArenaPtr {
        self.sync_data.arena_for_file(file)
    }

    #[inline]
    fn base_arena(&self) -> crate::simd_path::ArenaPtr {
        self.sync_data.arena_base_ptr()
    }

    #[inline]
    fn overflow_arena(&self) -> crate::simd_path::ArenaPtr {
        self.sync_data.arena_overflow_ptr()
    }
}

thread_local! {
    static NARROW_HITS: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    static NARROW_HIT_CAP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// Working hit indices. Drop parks a larger buffer; publish records capacity for the next take.
struct ReusedHits(Vec<u32>);

impl ReusedHits {
    fn take() -> Self {
        let mut buf = NARROW_HITS.with(|c| std::mem::take(&mut *c.borrow_mut()));
        buf.clear();
        let want = NARROW_HIT_CAP.with(|saved| saved.get());
        if buf.capacity() < want {
            buf.reserve(want);
        }
        Self(buf)
    }

    fn into_arc(mut self) -> Arc<Vec<u32>> {
        let cap = self.0.capacity();
        NARROW_HIT_CAP.with(|saved| {
            if cap > saved.get() {
                saved.set(cap);
            }
        });
        Arc::new(std::mem::take(&mut self.0))
    }
}

impl std::ops::Deref for ReusedHits {
    type Target = Vec<u32>;
    fn deref(&self) -> &Vec<u32> {
        &self.0
    }
}

impl std::ops::DerefMut for ReusedHits {
    fn deref_mut(&mut self) -> &mut Vec<u32> {
        &mut self.0
    }
}

impl Drop for ReusedHits {
    fn drop(&mut self) {
        self.0.clear();
        NARROW_HITS.with(|c| {
            let mut slot = c.borrow_mut();
            if self.0.capacity() > slot.capacity() {
                *slot = std::mem::take(&mut self.0);
            }
        });
    }
}

impl FilePicker {
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub fn has_git_repo(&self) -> bool {
        self.sync_data.git.is_some()
    }

    /// Ignore rules the walker assembled during the last scan (zlob backend
    /// only). The background watcher uses these to filter events without
    /// libgit2. `None` when the backend doesn't surface rules or no ignore
    /// files were present.
    pub(crate) fn ignore_rules(&self) -> Option<Arc<crate::walk::WalkIgnoreRules>> {
        self.sync_data.ignore_rules.clone()
    }

    pub fn has_mmap_cache(&self) -> bool {
        self.enable_mmap_cache
    }

    pub fn has_content_indexing(&self) -> bool {
        self.enable_content_indexing
    }

    pub fn has_watcher(&self) -> bool {
        self.watch
    }

    pub fn is_watcher_ready(&self) -> bool {
        self.background_watcher.is_some() && self.signals.watcher_ready.load(Ordering::Acquire)
    }

    pub fn follows_symlinks(&self) -> bool {
        self.follow_symlinks
    }

    pub fn fs_root_scanning_enabled(&self) -> bool {
        self.enable_fs_root_scanning
    }

    pub fn home_dir_scanning_enabled(&self) -> bool {
        self.enable_home_dir_scanning
    }

    pub fn skips_git(&self) -> bool {
        self.skip_git
    }

    pub fn git_recency_config(&self) -> GitRecencyConfig {
        self.git_recency_config
    }

    pub fn trace_id(&self) -> &str {
        &self.trace_id
    }

    pub fn trace_span(&self) -> tracing::Span {
        self.trace_span.clone()
    }

    pub fn mode(&self) -> FFFMode {
        self.mode
    }

    pub fn cache_budget(&self) -> &ContentCacheBudget {
        &self.cache_budget
    }

    pub fn bigram_index(&self) -> Option<&BigramFilter> {
        self.sync_data.bigram_index.as_deref()
    }

    pub fn bigram_overlay(&self) -> Option<&parking_lot::RwLock<BigramOverlay>> {
        self.sync_data.bigram_overlay.as_deref()
    }

    pub fn get_file_mut(&mut self, index: usize) -> Option<(ArenaPtr, &mut FileItem)> {
        self.sync_data.get_file_mut(index)
    }

    /// Absolute path to the repository root if the indexed tree lives
    /// inside a git working directory. `None` for non-git bases.
    pub fn git_root(&self) -> Option<&Path> {
        self.sync_data.git.as_ref().map(|git| git.workdir.as_path())
    }

    /// Path keying the project history: frecency, query history and combo boosts.
    /// Inside a linked git worktree it is the main checkout twin of the base path.
    pub fn shared_base_path(&self) -> &Path {
        self.sync_data
            .shared_base
            .as_deref()
            .unwrap_or(&self.base_path)
    }

    /// Main checkout twin of `path` when the picker indexes a linked git worktree.
    pub fn shared_path_for(&self, path: impl AsRef<Path>) -> Option<PathBuf> {
        let shared_base = self.sync_data.shared_base.as_deref()?;
        crate::git::shared_twin(path.as_ref(), &self.base_path, shared_base)
    }

    pub(crate) fn worktree_shared_base(&self) -> Option<&Path> {
        self.sync_data.shared_base.as_deref()
    }

    pub fn has_explicit_cache_budget(&self) -> bool {
        self.has_explicit_cache_budget
    }

    pub fn set_cache_budget(&mut self, budget: ContentCacheBudget) {
        self.cache_budget = Arc::new(budget);
    }

    /// Get all indexed files sorted by path.
    /// Note: Files are stored sorted by PATH for efficient insert/remove.
    /// For frecency-sorted results, use search() which sorts matched results.
    pub fn get_files(&self) -> &[FileItem] {
        self.sync_data.files()
    }

    /// Sequentially touch per-file search metadata after the scan so the
    /// first keystroke does not pay the page-fault cost of FileItem +
    /// presence_mask (227k × ~64 B ≈ 14 MB of cold pages).
    /// Read-only; does not change results. Call once before `scanning=false`.
    pub fn prefault_search_metadata(&self) {
        let files = self.sync_data.files();
        let mut acc = 0u64;
        for f in files.iter() {
            acc = acc
                .wrapping_add(f.presence_mask())
                .wrapping_add(f.path.byte_len as u64);
        }
        for &mask in self.sync_data.presence.as_slice() {
            acc = acc.wrapping_add(mask);
        }
        std::hint::black_box(acc);
        // Overflow arena FileItems share the same layout.
        for f in self.sync_data.overflow_files().iter() {
            acc = acc.wrapping_add(f.presence_mask());
        }
        std::hint::black_box(acc);
    }

    /// Count of live (non-tombstoned) files. O(1).
    #[inline]
    pub fn live_file_count(&self) -> usize {
        self.sync_data.live_count
    }

    pub fn get_overflow_files(&self) -> &[FileItem] {
        self.sync_data.overflow_files()
    }

    /// Get the directory table (sorted by path).
    pub fn get_dirs(&self) -> &[DirItem] {
        &self.sync_data.dirs
    }

    /// Actual heap bytes used: (chunked_path_store, 0, 0).
    /// The second element is 0 because leaked overflow stores aren't tracked.
    pub fn arena_bytes(&self) -> (usize, usize, usize) {
        let chunked = self
            .sync_data
            .chunked_paths
            .as_ref()
            .map_or(0, |s| s.heap_bytes());

        (chunked, 0, 0)
    }

    #[tracing::instrument(level = "debug", skip_all)]
    pub(crate) fn for_each_dir(&self, mut f: impl FnMut(&Path) -> ControlFlow<()>) {
        let dir_table = &self.sync_data.dirs;
        let base = self.base_path.as_path();

        if !dir_table.is_empty() {
            let arena = self.arena_base_ptr();
            let overflow_arena = self.sync_data.arena_overflow_ptr();
            let mut path_buf = PathBuf::with_capacity(crate::simd_path::PATH_BUF_SIZE);
            let mut prev_relative_path = String::new();

            let mut scratch_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
            for dir_item in dir_table.iter() {
                if dir_item.is_deleted() {
                    continue;
                }
                let item_arena = if dir_item.is_overflow() {
                    overflow_arena
                } else {
                    arena
                };
                let full_relative_path = dir_item.read_relative_path(item_arena, &mut scratch_buf);
                let relative_path = full_relative_path.trim_end_matches(std::path::is_separator);

                if relative_path.is_empty() {
                    // Files directly under base_path
                    prev_relative_path.clear();
                    continue;
                }

                let mut i = common_dir_prefix_len(&prev_relative_path, relative_path);
                // If we stopped on a separator, skip it — we want to start
                // emitting at the first unseen segment, not re-emit the
                // already-emitted prefix path.
                if i < relative_path.len()
                    && std::path::is_separator(relative_path.as_bytes()[i] as char)
                {
                    i += 1;
                }

                // Walk the suffix of `relative_path` one segment at a time, emitting
                // each previously unseen ancestor up to and including `relative_path`.
                while i < relative_path.len() {
                    let next_sep = relative_path[i..]
                        .find(std::path::is_separator)
                        .map(|off| i + off)
                        .unwrap_or(relative_path.len());
                    let ancestor_rel = &relative_path[..next_sep];

                    path_buf.clear();
                    path_buf.push(base);
                    path_buf.push(ancestor_rel);

                    // we can't really emit iterator here unfortunately
                    if matches!(f(path_buf.as_path()), ControlFlow::Break(())) {
                        return;
                    }

                    i = next_sep + 1;
                }

                prev_relative_path.clear();
                prev_relative_path.push_str(relative_path);
            }
            return;
        }

        // fallback that should never be happening, but it is possible to get the file
        // path from the absolute path using components api as well:
        let files = self.sync_data.files();
        let arena = self.arena_base_ptr();
        let mut current = self.base_path.clone();
        let mut path_buf = [0u8; PATH_BUF_SIZE];

        for file in files {
            let abs = file.write_absolute_path(arena, base, &mut path_buf);
            let Some(parent) = abs.parent() else {
                continue;
            };
            if parent == current.as_path() {
                continue;
            }

            while current.as_path() != base && !parent.starts_with(&current) {
                current.pop();
            }

            let Ok(remainder) = parent.strip_prefix(&current) else {
                continue;
            };
            for component in remainder.components() {
                current.push(component);
                if matches!(f(current.as_path()), ControlFlow::Break(())) {
                    return;
                }
            }
        }
    }

    /// Create a new FilePicker from options.
    /// Always prefer new_with_shared_state for the consumer application, use this only if you know
    /// what you are doing. This won't spawn the backgraound watcher and won't walk the file tree.
    pub fn new(options: FilePickerOptions) -> Result<Self, Error> {
        crate::git::tune_libgit2_for_local_reads();
        // Probes cost ~30ms on first touch; pay at open, never on a keystroke.
        crate::machine::warm_calibration();

        let path = PathBuf::from(&options.base_path);
        if !path.exists() {
            error!("Base path does not exist: {}", options.base_path);
            return Err(Error::InvalidPath(path));
        }
        // Relative bases (".", "sub/dir") are resolved against the cwd so
        // they can be compared with the absolute paths reported by the OS
        // watcher. Purely lexical: no symlinks are resolved. The
        // `components()` pass drops interior `.` segments ("/cwd/.").
        let path = if path.is_relative() {
            std::env::current_dir()
                .map(|cwd| cwd.join(&path).components().collect())
                .unwrap_or(path)
        } else {
            path
        };
        if path.parent().is_none() && !options.enable_fs_root_scanning {
            error!("Refusing to index filesystem root: {}", path.display());
            return Err(Error::FilesystemRoot(path));
        }
        // Path equality compares components, so "~/" (as git root discovery returns it) matches too.
        if !options.enable_home_dir_scanning && dirs::home_dir().is_some_and(|home| path == home) {
            error!("Refusing to index home directory: {}", path.display());
            return Err(Error::FilesystemRoot(path));
        }

        // Windows-only: canonicalize with dunce so the base path does NOT
        // have the `\\?\` UNC prefix that `std::fs::canonicalize` adds.
        // libgit2's `repo.workdir()`
        #[cfg(windows)]
        let path = crate::path_utils::canonicalize(&path).unwrap_or(path);

        let has_explicit_budget = options.cache_budget.is_some();
        let initial_budget = options.cache_budget.unwrap_or_default();

        let trace_id = crate::log::generate_trace_id();
        let trace_span = crate::log::trace_span(&trace_id, "picker");

        Ok(FilePicker {
            background_watcher: None,
            git_status_worker: crate::git_status_worker::GitStatusWorker::new(),
            base_path: path,
            cache_budget: Arc::new(initial_budget),
            has_explicit_cache_budget: has_explicit_budget,
            signals: crate::scan::ScanSignals::default(),
            mode: options.mode,
            scanned_files_count: Arc::new(AtomicUsize::new(0)),
            sync_data: FileSync::new(),
            enable_mmap_cache: options.enable_mmap_cache,
            enable_content_indexing: options.enable_content_indexing,
            watch: options.watch,
            follow_symlinks: options.follow_symlinks,
            enable_fs_root_scanning: options.enable_fs_root_scanning,
            enable_home_dir_scanning: options.enable_home_dir_scanning,
            git_recency_config: options.git_recency,
            skip_git: options.skip_git,
            trace_span,
            trace_id,
            live: parking_lot::Mutex::new(LiveNarrow::default()),
            index_generation: std::sync::atomic::AtomicU64::new(0),
            git_status_epoch: std::sync::atomic::AtomicU64::new(0),
            match_cache: parking_lot::Mutex::new(crate::score::MatchCache::default()),
            scored_page: parking_lot::Mutex::new(None),
            search_epoch: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create a picker, place it into the shared handle, and spawn background
    /// indexing + file-system watcgenerate_trace_id the default entry point.
    pub fn new_with_shared_state(
        shared_picker: SharedFilePicker,
        shared_frecency: SharedFrecency,
        options: FilePickerOptions,
    ) -> Result<(), Error> {
        let mut picker = Self::new(options)?;
        picker.git_status_worker = crate::git_status_worker::GitStatusWorker::with_pause_flag(
            shared_picker.git_status_pause_flag(),
        );

        info!(
            "Spawning background threads: base_path={}, warmup={}, content_indexing={}, mode={:?}",
            picker.base_path.display(),
            picker.enable_mmap_cache,
            picker.enable_content_indexing,
            picker.mode,
        );

        let warmup = picker.enable_mmap_cache;
        let content_indexing = picker.enable_content_indexing;
        let watch = picker.watch;
        let mode = picker.mode;
        let follow_symlinks = picker.follow_symlinks;
        let enable_fs_root_scanning = picker.enable_fs_root_scanning;
        let enable_home_dir_scanning = picker.enable_home_dir_scanning;
        let skip_git = picker.skip_git;

        let signals = picker.scan_signals();
        let scanned_files_counter = picker.scanned_files_counter();
        let path = picker.base_path.clone();
        let trace_span = picker.trace_span.clone();

        // Pre-arm `scanning` BEFORE publishing the new picker. `ScanJob::spawn`
        // also sets it, but that runs after this function returns; consumers
        // (e.g. lua `wait_for_initial_scan` after `restart_index_in_path`)
        // that grab the signal Arc between publish and spawn would otherwise
        // observe scanning=false and skip the wait, racing the walker. The
        // race is wide on Windows CI where notify is slow.
        signals
            .scanning
            .store(true, std::sync::atomic::Ordering::Release);

        // Update the watch base before publishing the new picker.
        shared_picker.rebase_watches(&path);

        {
            let mut guard = shared_picker.write()?;
            *guard = Some(picker);
            // dropping old picker flips its `cancelled` flag → bg threads exit cleanly
        }

        ScanJob::new_initial(
            shared_picker,
            shared_frecency,
            path,
            mode,
            signals,
            scanned_files_counter,
            trace_span,
            ScanConfig {
                warmup,
                content_indexing,
                watch,
                auto_cache_budget: true,
                install_watcher: true,
                follow_symlinks,
                enable_fs_root_scanning,
                enable_home_dir_scanning,
                skip_git,
            },
        )
        .spawn();

        Ok(())
    }

    /// Synchronous filesystem scan — populates `self` with indexed files.
    ///
    /// Use this when you need direct access to the picker without shared state:
    /// ```ignore
    /// let mut picker = FilePicker::new(options)?;
    /// picker.collect_files()?;
    /// // picker.get_files() is now populated
    /// ```
    pub fn collect_files(&mut self) -> Result<(), Error> {
        self.signals.scanning.store(true, Ordering::Relaxed);
        self.scanned_files_count.store(0, Ordering::Relaxed);

        let git = if self.skip_git {
            None
        } else {
            FileSync::discover_git(&self.base_path)
        };

        let empty_frecency = SharedFrecency::default();
        let sync = FileSync::walk_filesystem(
            &self.base_path,
            git,
            &self.scanned_files_count,
            &empty_frecency,
            self.mode,
            self.follow_symlinks,
        )?;

        self.sync_data = sync;
        self.bump_index_generation();

        // Spawn the status read once the live count is known, so
        // mega-repos skip it instead of joining a uselessly long read.
        let skip_git = should_skip_git(self.skip_git, self.live_file_count(), git_force());
        if !self.skip_git && skip_git {
            info!(
                live = self.live_file_count(),
                "auto-skipping git status on huge index (FFF_FORCE_GIT=1 overrides)"
            );
        }
        let git_handle = if skip_git {
            None
        } else {
            self.sync_data
                .git
                .as_ref()
                .map(|git| FileSync::spawn_git_status(git.workdir.clone()))
        };

        if !self.has_explicit_cache_budget {
            let file_count = self.sync_data.files().len();
            self.cache_budget = Arc::new(ContentCacheBudget::new_for_repo(file_count));
        } else {
            self.cache_budget.reset();
        }

        // Files are searchable before git status/recency finish. Joining those
        // here blocked `scanning=false` for minutes on multi-GB .git repos.
        self.signals.scanning.store(false, Ordering::Relaxed);

        if let Some(handle) = git_handle
            && let Ok(Some(git_cache)) = handle.join()
        {
            // Dirty-only cache: a clean file is absent and already starts clean.
            // An empty map has nothing to store.
            if git_cache.statuses_len() > 0 {
                let mut path_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
                let arena = self.arena_base_ptr();
                let mut modified_set_changed = false;
                for file in self.sync_data.files.iter() {
                    let next = git_cache.lookup_status(file.write_absolute_path(
                        arena,
                        &self.base_path,
                        &mut path_buf,
                    ));
                    if Self::store_git_status(file, next) {
                        modified_set_changed = true;
                    }
                }
                if modified_set_changed {
                    self.bump_git_status_epoch();
                }
            }
        }

        if !skip_git
            && let Some(workdir) = self.git_root().map(Path::to_path_buf)
            && let Ok(repo) = Repository::open(&workdir)
                .inspect_err(|e| debug!(?e, ?workdir, "git recency: failed to open repo"))
        {
            let recency =
                git_recency::compute_git_recency(&repo, &self.git_recency_config, &self.base_path);
            self.apply_git_recency(recency.as_ref());
        }

        Ok(())
    }

    /// Perform fuzzy search on files with a pre-parsed query.
    ///
    /// The query should be parsed using [`crate::FFFQuery`] before calling
    /// this function. If a [`crate::QueryTracker`] is provided, the search will
    /// automatically look up the last selected file for this query and boost it
    #[tracing::instrument(skip_all, name = "Fuzzy file search", fields(query = query.raw_query))]
    pub fn fuzzy_search<'q>(
        &self,
        query: &'q FFFQuery<'q>,
        query_tracker: Option<&QueryTracker>,
        options: FuzzySearchOptions<'q>,
    ) -> SearchResult<'_> {
        let _query = crate::parallelism::QueryActiveGuard::new();
        let abort = crate::parallelism::AbortToken::live(&self.search_epoch);
        let files = self.get_files();

        let total_files = self.live_file_count();
        let location = query.location;

        // Get effective query for max_typos calculation (without location suffix)
        let effective_query = match &query.fuzzy_query {
            fff_query_parser::FuzzyQuery::Text(t) => *t,
            fff_query_parser::FuzzyQuery::Parts(parts) if !parts.is_empty() => parts[0],
            _ => query.raw_query.trim(),
        };
        // Mirrors the scoring junction: short/empty queries take the
        // frecency path and capture no hit set, so their empty buffer
        // must not enter history (repeats would read NarrowPlan::Empty).
        let fuzzy_ran = match &query.fuzzy_query {
            fff_query_parser::FuzzyQuery::Text(t) => t.len() >= 2,
            fff_query_parser::FuzzyQuery::Parts(parts) => !parts.is_empty(),
            _ => false,
        };

        // Tiny queries are too small to amortize Rayon dispatch when the
        // caller asked for a small pool. Never override an explicit request
        // for 3+ threads — that collapsed 4T parallel efficiency.
        let max_threads = if options.max_threads == 1
            || (effective_query.len() <= 1 && options.max_threads <= 2)
            || (effective_query.len() <= 2 && options.max_threads == 2 && total_files < 20_000)
        {
            1
        } else if options.max_threads == 0 {
            // Same width as the machine profile. The count is read once per process.
            crate::parallelism::effective_core_count()
        } else {
            options.max_threads
        };

        debug!(
            raw_query = ?query.raw_query,
            pagination = ?options.pagination,
            ?max_threads,
            current_file = ?options.current_file,
            "Fuzzy search",
        );

        // Path-store layout stats and the machine profile these numbers are
        // interpreted against. The gate is bound by arena load latency, so the
        // working-set shape matters more than anything else.
        // Path-store layout stats walk every file, so they are gated separately from
        // FFF_PHASE_TIMING: leaving them on the timing path cost ~560us on a
        // 227k-file query and skewed every phase measurement taken with it.
        let phase = crate::score::phase_timing();
        let t_total = phase.then(std::time::Instant::now);
        let arena_stats = arena_stats_enabled();
        if phase {
            let m = crate::machine::machine();
            eprintln!(
                "PHASE machine cores={} phys={} cache_line={} llc_mb={} latency_ns={:.1} read_bytes_per_ns={:.2} work_per_thread={} scan_cap={}",
                m.logical_cores,
                m.physical_cores,
                m.cache_line,
                m.llc_bytes / (1024 * 1024),
                m.mem_latency_ns,
                m.read_bytes_per_ns(),
                crate::machine::match_work_per_thread(),
                crate::machine::scan_thread_cap(),
            );
            if arena_stats && let Some(store) = self.sync_data.chunked_paths.as_ref() {
                let slots: usize = self
                    .sync_data
                    .files
                    .iter()
                    .map(|f| f.path.chunk_count())
                    .sum();
                let unique = store.unique_chunks_for_stats();
                let files = self.sync_data.files.len();
                // Directory prefix = bytes before the filename. Files are sorted
                // by (parent_dir, filename), so that prefix is shared by every
                // sibling and could be stored once per directory instead of
                // once per file - which is what the index table pays for today.
                let path_bytes: usize = self
                    .sync_data
                    .files
                    .iter()
                    .map(|f| f.path.byte_len as usize)
                    .sum();
                let prefix_bytes: usize = self
                    .sync_data
                    .files
                    .iter()
                    .map(|f| f.path.filename_offset as usize)
                    .sum();
                eprintln!(
                    "PHASE arena files={files} chunk_slots={slots} unique_chunks={unique} \
                     dedup_ratio={:.2} heap_bytes={} file_bytes={} avg_path_bytes={:.1} \
                     avg_prefix_bytes={:.1} prefix_share={:.2}",
                    if unique > 0 {
                        slots as f64 / unique as f64
                    } else {
                        0.0
                    },
                    store.heap_bytes(),
                    files * std::mem::size_of::<FileItem>(),
                    if files > 0 {
                        path_bytes as f64 / files as f64
                    } else {
                        0.0
                    },
                    if files > 0 {
                        prefix_bytes as f64 / files as f64
                    } else {
                        0.0
                    },
                    if path_bytes > 0 {
                        prefix_bytes as f64 / path_bytes as f64
                    } else {
                        0.0
                    },
                );
            }
        }

        // small queries with a large number of results can match absolutely everything
        let max_typos = (effective_query.len() as u16 / 4).clamp(2, 6);
        // Look up the last file selected for this query (combo-boost scoring)
        let t_tracker = phase.then(std::time::Instant::now);
        let last_same_query_entry =
            query_tracker
                .zip(options.project_path)
                .and_then(|(tracker, project_path)| {
                    tracker
                        .get_last_query_entry(
                            query.raw_query,
                            project_path,
                            options.min_combo_count,
                        )
                        .ok()
                        .flatten()
                });
        let tracker_us = t_tracker.map(|t| t.elapsed().as_micros()).unwrap_or(0);

        let context = ScoringContext {
            query,
            max_typos,
            max_threads,
            project_path: options.project_path,
            current_file: options.current_file,
            last_same_query_match: last_same_query_entry,
            combo_boost_score_multiplier: options.combo_boost_score_multiplier,
            min_combo_count: options.min_combo_count,
            pagination: options.pagination,
            check_deleted: self.live_file_count() != files.len(),
        };

        let time = tracing::enabled!(tracing::Level::INFO).then(std::time::Instant::now);

        let base_arena = self.sync_data.arena_base_ptr();
        let overflow_arena = self.sync_data.arena_overflow_ptr();

        // Typing live-narrow: when the query only extends the previous one and
        // max_typos is unchanged, Match(longer,T) ⊆ Match(shorter,T).
        //
        // Fast path holds the live lock only to filter the previous hit set
        // into (file_index, &FileItem) pairs — no clone of the 226k u32
        // candidate vector, and the kept indices stay aligned with the subset
        // so hit mapping is exact even when the presence mask drops items.
        // Narrowing keys on the full raw query: parts[1..] and constraints
        // change the hit set even when effective_query() does not. The
        // fingerprint additionally stops an appended constraint (e.g.
        // `foo` -> `foo *.rs`) from reusing an unfiltered hit set.
        let query_key = context.query.raw_query;
        let constraints_fp = constraints_fingerprint(&context.query.constraints);
        let git_constrained = constraints_use_git_status(&context.query.constraints);
        // Snapshots taken before the plan. A status refresh moves the git
        // epoch; those hits must not be published as the next narrow input.
        let generation_at_start = self.index_generation();
        let git_epoch_at_start = self.current_git_status_epoch();
        let (combo_path, combo_open_count) = match context.last_same_query_match.as_ref() {
            Some(entry) => (
                Some(entry.file_path.as_os_str().as_encoded_bytes()),
                entry.open_count,
            ),
            None => (None, 0),
        };
        let page_key = PageKey {
            generation: generation_at_start,
            score_epoch: crate::types::score_input_epoch(),
            query: query_key,
            max_typos,
            constraints_fp,
            current_file: context.current_file,
            combo_path,
            combo_open_count,
            min_combo_count: context.min_combo_count,
            combo_multiplier: context.combo_boost_score_multiplier,
            offset: context.pagination.offset,
            limit: context.pagination.limit,
        };
        // Exact key hit rebuilds the page and skips match, score, and byte offsets.
        if let Some((items, scores, total_matched, match_byte_offsets, paths)) =
            self.replay_scored_page(files, &page_key)
        {
            info!(
                ?query,
                completed_in = ?time.map(|t| t.elapsed()).unwrap_or_default(),
                total_matched,
                returned_count = items.len(),
                pagination = ?options.pagination,
                "Fuzzy search completed",
            );
            return SearchResult {
                items,
                scores,
                match_byte_offsets,
                paths: Some(paths),
                total_matched,
                total_files,
                location,
            };
        }
        let mut hit_buf = ReusedHits::take();
        // Starts before the narrow plan so the presence-mask prune is included.
        let t_msa = phase.then(std::time::Instant::now);
        // Every matcher part, built once. need < 3 does not read the mask table.
        let folds = PartFolds::build(context.query, max_typos);
        let (items, scores, total_matched) = {
            let masks = self.sync_data.presence.as_slice();
            // Clone the hit list, then drop the lock before the presence scan.
            enum Held {
                Full,
                Indices(Arc<Vec<u32>>),
                Filter(Arc<Vec<u32>>),
            }
            let held = {
                let live = self.live.lock();
                let can_narrow = live.active
                    && live.max_typos == max_typos
                    && live.constraints_fp == constraints_fp
                    && query_key.len() > live.query.len()
                    && query_key.starts_with(live.query.as_str())
                    && live.generation == generation_at_start
                    && git_epoch_ok(
                        git_constrained,
                        live.git_status_epoch,
                        git_epoch_at_start,
                    );
                if can_narrow {
                    let shared = Arc::clone(&live.candidates);
                    if folds.n == 0 {
                        Held::Indices(shared)
                    } else {
                        Held::Filter(shared)
                    }
                } else if let Some(entry) = live.history.iter().rfind(|e| {
                    e.generation == generation_at_start
                        && e.max_typos == max_typos
                        && e.constraints_fp == constraints_fp
                        && e.query.as_ref() == query_key
                        && git_epoch_ok(
                            git_constrained,
                            e.git_status_epoch,
                            git_epoch_at_start,
                        )
                }) {
                    Held::Indices(Arc::clone(&entry.candidates))
                } else if let Some(shared) = longest_prefix(
                    &live.prefix,
                    query_key,
                    max_typos,
                    generation_at_start,
                    constraints_fp,
                    git_constrained,
                    git_epoch_at_start,
                ) {
                    if folds.n == 0 {
                        Held::Indices(shared)
                    } else {
                        Held::Filter(shared)
                    }
                } else {
                    Held::Full
                }
            };
            let plan = match held {
                Held::Full => NarrowPlan::Full,
                Held::Indices(shared) => plan_indices(shared, files.len()),
                Held::Filter(shared) => {
                    let orig = filter_hits(&shared, masks, files, &folds);
                    plan_owned(orig, files.len())
                }
            };
            match plan {
                NarrowPlan::Empty => (vec![], vec![], 0),
                NarrowPlan::Narrow { orig } => {
                    hit_buf.clear();
                    crate::score::with_hit_capture(&mut hit_buf, || {
                        crate::score::fuzzy_match_and_score_subset(
                            files,
                            self.sync_data.presence.as_slice(),
                            orig,
                            &context,
                            self.sync_data.base_count,
                            base_arena,
                            overflow_arena,
                            self.max_path_len(),
                            abort,
                            Some(&self.match_cache),
                            generation_at_start,
                        )
                    })
                }
                NarrowPlan::Full => crate::score::with_hit_capture(&mut hit_buf, || {
                    fuzzy_match_and_score_files_presence(
                        files,
                        self.sync_data.presence.as_slice(),
                        &context,
                        self.sync_data.base_count,
                        base_arena,
                        overflow_arena,
                        self.max_path_len(),
                        abort,
                    )
                }),
            }
        };

        // A cancelled search returns whatever groups finished. Those indices
        // are not a superset, so they must not arm the next keystroke.
        if !abort.aborted() {
            let mut live = self.live.lock();
            let git_hits_current = !git_constrained
                || self.current_git_status_epoch() == git_epoch_at_start;
            if self.index_generation() == generation_at_start && git_hits_current {
                live.generation = generation_at_start;
                live.git_status_epoch = git_epoch_at_start;
                // Same keystroke text keeps this buffer and the history Arc.
                if live.query.as_str() != query_key {
                    live.query.clear();
                    live.query.push_str(query_key);
                }
                live.max_typos = max_typos;
                live.constraints_fp = constraints_fp;
                live.active = !hit_buf.is_empty();
                // Move the indices into the Arc. The next search does not write that Arc.
                live.candidates = hit_buf.into_arc();
                // Backspace reuses this hit set. Stale generations never match.
                let q = query_key;
                let store_history = fuzzy_ran && narrow_pays(live.candidates.len(), files.len());
                let store_prefix = fuzzy_ran
                    && !live.candidates.is_empty()
                    && live.candidates.len() <= 65_536;
                let query_owned = if store_history || store_prefix {
                    Some(
                        live.history
                            .iter()
                            .chain(live.prefix.iter())
                            .find(|e| e.query.as_ref() == q)
                            .map(|e| Arc::clone(&e.query))
                            .unwrap_or_else(|| Arc::<str>::from(q)),
                    )
                } else {
                    None
                };
                live.history
                    .retain(|e| e.query.as_ref() != q || e.max_typos != max_typos);
                if let Some(query_owned) = query_owned {
                    let snapshot = Arc::clone(&live.candidates);
                    let generation = live.generation;
                    let git_status_epoch = live.git_status_epoch;
                    if store_history {
                        live.history.push(BackEntry {
                            query: Arc::clone(&query_owned),
                            max_typos,
                            generation,
                            constraints_fp,
                            git_status_epoch,
                            candidates: Arc::clone(&snapshot),
                        });
                        while live.history.len() > 8 {
                            live.history.remove(0);
                        }
                    }
                    if store_prefix {
                        live.prefix
                            .retain(|e| e.query.as_ref() != q || e.max_typos != max_typos);
                        live.prefix.push(BackEntry {
                            query: query_owned,
                            max_typos,
                            generation,
                            constraints_fp,
                            git_status_epoch,
                            candidates: snapshot,
                        });
                        while live.prefix.len() > 8 {
                            live.prefix.remove(0);
                        }
                    }
                }
            }
        }
        // `msa_us` covers match+score, including the narrow plan above.
        let msa_us = t_msa.map(|t| t.elapsed().as_micros()).unwrap_or(0);
        let post_us = msa_us;
        let paths = self.snapshot_paths(&items);
        let t_offsets = phase.then(std::time::Instant::now);
        let match_byte_offsets = fuzzy_match_byte_offsets_for_page(
            query,
            &paths.bytes,
            &paths.rows,
            max_typos,
        );
        if phase {
            let now_us = t_total.map(|t| t.elapsed().as_micros()).unwrap_or(0);
            let offsets_us = t_offsets.map(|t| t.elapsed().as_micros()).unwrap_or(0);
            eprintln!(
                "PHASE total_us={now_us} msa_us={post_us} post_us={} tracker_us={tracker_us} offsets_us={offsets_us} page={}",
                now_us.saturating_sub(post_us + offsets_us + tracker_us),
                items.len()
            );
        }

        let scores = std::sync::Arc::new(scores);
        let match_byte_offsets = std::sync::Arc::new(match_byte_offsets);
        self.remember_scored_page(
            abort,
            files,
            &items,
            &scores,
            &match_byte_offsets,
            &paths,
            total_matched,
            &page_key,
        );

        info!(
            ?query,
            completed_in = ?time.map(|t| t.elapsed()).unwrap_or_default(),
            total_matched,
            returned_count = items.len(),
            pagination = ?options.pagination,
            "Fuzzy search completed",
        );

        SearchResult {
            items,
            scores,
            match_byte_offsets,
            paths: Some(paths),
            total_matched,
            total_files,
            location,
        }
    }

    /// Perform fuzzy search on indexed directories.
    ///
    /// Returns directories ranked by fuzzy match quality + frecency.
    pub fn fuzzy_search_directories<'q>(
        &self,
        query: &'q FFFQuery<'q>,
        options: FuzzySearchOptions<'q>,
    ) -> DirSearchResult<'_> {
        let _query = crate::parallelism::QueryActiveGuard::new();
        let dirs = self.get_dirs();
        let max_threads = if options.max_threads == 0 {
            crate::parallelism::effective_core_count()
        } else {
            options.max_threads
        };

        let total_dirs = self.sync_data.live_dirs_count;

        let effective_query = match &query.fuzzy_query {
            fff_query_parser::FuzzyQuery::Text(t) => *t,
            fff_query_parser::FuzzyQuery::Parts(parts) if !parts.is_empty() => parts[0],
            _ => query.raw_query.trim(),
        };

        let max_typos = (effective_query.len() as u16 / 4).clamp(2, 6);

        let context = ScoringContext {
            query,
            max_typos,
            max_threads,
            project_path: options.project_path,
            current_file: options.current_file,
            last_same_query_match: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            pagination: options.pagination,
            check_deleted: true,
        };

        let arena = self.sync_data.arena_base_ptr();
        let overflow_arena = self.sync_data.arena_overflow_ptr();
        let time = tracing::enabled!(tracing::Level::INFO).then(std::time::Instant::now);

        let (items, scores, total_matched) =
            crate::score::fuzzy_match_and_score_dirs(dirs, &context, arena, overflow_arena);

        info!(
            ?query,
            completed_in = ?time.map(|t| t.elapsed()).unwrap_or_default(),
            total_matched,
            returned_count = items.len(),
            "Directory search completed",
        );

        DirSearchResult {
            items,
            scores,
            total_matched,
            total_dirs,
        }
    }

    /// Perform a mixed fuzzy search across both files and directories.
    ///
    /// Returns a single flat list where files and directories are interleaved
    /// by total score in descending order.
    ///
    /// If the raw query ends with a path separator (`/`), only directories
    /// are searched — files are skipped entirely. The caller should parse the
    /// query with `DirSearchConfig` so that trailing `/` is kept as fuzzy
    /// text instead of becoming a `PathSegment` constraint.
    pub fn fuzzy_search_mixed<'q>(
        &self,
        query: &'q FFFQuery<'q>,
        query_tracker: Option<&QueryTracker>,
        options: FuzzySearchOptions<'q>,
    ) -> MixedSearchResult<'_> {
        let _query = crate::parallelism::QueryActiveGuard::new();
        let location = query.location;
        let page_offset = options.pagination.offset;
        let page_limit = if options.pagination.limit > 0 {
            options.pagination.limit
        } else {
            100
        };

        let dirs_only =
            query.raw_query.ends_with(std::path::MAIN_SEPARATOR) || query.raw_query.ends_with('/');

        // Run file search and dir search with no pagination (we merge then paginate).
        let internal_limit = page_offset.saturating_add(page_limit).saturating_mul(2);

        let dir_options = FuzzySearchOptions {
            pagination: PaginationArgs {
                offset: 0,
                limit: internal_limit,
            },
            ..options
        };
        let dir_results = self.fuzzy_search_directories(query, dir_options);

        if dirs_only {
            let total_matched = dir_results.total_matched;
            let total_dirs = dir_results.total_dirs;

            let mut merged: Vec<(MixedItemRef<'_>, Score)> =
                Vec::with_capacity(dir_results.items.len());
            for (dir, score) in dir_results.items.into_iter().zip(dir_results.scores) {
                merged.push((MixedItemRef::Dir(dir), score));
            }

            if page_offset >= merged.len() {
                return MixedSearchResult {
                    items: vec![],
                    scores: vec![],
                    total_matched,
                    total_files: self.live_file_count(),
                    total_dirs,
                    location,
                };
            }

            let end = (page_offset + page_limit).min(merged.len());
            let page = merged.drain(page_offset..end);
            let (items, scores): (Vec<_>, Vec<_>) = page.unzip();

            return MixedSearchResult {
                items,
                scores,
                total_matched,
                total_files: self.live_file_count(),
                total_dirs,
                location,
            };
        }

        let file_options = FuzzySearchOptions {
            pagination: PaginationArgs {
                offset: 0,
                limit: internal_limit,
            },
            ..options
        };
        let mut file_results = self.fuzzy_search(query, query_tracker, file_options);

        // Merge by score descending.
        let total_matched = file_results.total_matched + dir_results.total_matched;
        let total_files = file_results.total_files;
        let total_dirs = dir_results.total_dirs;

        let mut merged: Vec<(MixedItemRef<'_>, Score)> =
            Vec::with_capacity(file_results.items.len() + dir_results.items.len());

        for (file, score) in std::mem::take(&mut file_results.items)
            .into_iter()
            .zip(file_results.scores.iter().cloned())
        {
            merged.push((MixedItemRef::File(file), score));
        }
        for (dir, score) in dir_results.items.into_iter().zip(dir_results.scores) {
            merged.push((MixedItemRef::Dir(dir), score));
        }

        // Sort merged results by total score descending.
        merged.sort_unstable_by_key(|b| std::cmp::Reverse(b.1.total));

        // Paginate.
        if page_offset >= merged.len() {
            return MixedSearchResult {
                items: vec![],
                scores: vec![],
                total_matched,
                total_files,
                total_dirs,
                location,
            };
        }

        let end = (page_offset + page_limit).min(merged.len());
        let page = merged.drain(page_offset..end);
        let (items, scores): (Vec<_>, Vec<_>) = page.unzip();

        MixedSearchResult {
            items,
            scores,
            total_matched,
            total_files,
            total_dirs,
            location,
        }
    }

    /// Glob search: filter indexed files by a single glob pattern, rank by
    /// frecency, and paginate. Bypasses the regular query parser entirely —
    /// useful when callers already have a literal glob (`*.rs`, `**/*.test.ts`)
    /// and want neither fuzzy matching nor multi-token constraint parsing.
    ///
    /// Pipeline: `apply_constraints(Glob) → score_filtered_by_frecency → sort_and_paginate`.
    /// Same ranking semantics as `fuzzy_search` when the fuzzy query is empty.
    pub fn glob<'p>(
        &'p self,
        pattern: &'p str,
        options: FuzzySearchOptions<'p>,
    ) -> SearchResult<'p> {
        let query = FFFQuery {
            raw_query: pattern,
            constraints: std::iter::once(fff_query_parser::Constraint::Glob(pattern)).collect(),
            fuzzy_query: fff_query_parser::FuzzyQuery::Empty,
            location: None,
        };

        // `fuzzy_search` short-circuits to `score_filtered_by_frecency` when
        // `fuzzy_query` is `Empty`, then runs the same `sort_and_paginate`
        // path. Reusing it keeps the ranking guarantees identical without
        // exposing the private scoring helpers.
        self.fuzzy_search(&query, None, options)
    }

    /// Perform a live grep search across indexed files.
    ///
    /// If `options.abort_signal` is set it overrides the picker's internal
    /// cancellation flag, giving the caller full control over when to stop.
    pub fn grep(&self, query: &FFFQuery<'_>, options: &GrepSearchOptions) -> GrepResult<'_> {
        let _query = crate::parallelism::QueryActiveGuard::new();
        let overlay_guard = self.sync_data.bigram_overlay.as_ref().map(|o| o.read());
        let arena = self.arena_base_ptr();
        let overflow_arena = self.sync_data.arena_overflow_ptr();
        let cancel = options
            .abort_signal
            .as_deref()
            .unwrap_or(&self.signals.cancelled);

        let run_grep = || {
            grep_search(
                self.get_files(),
                self.live_file_count(),
                query,
                options,
                self.cache_budget(),
                self.sync_data.bigram_index.as_deref(),
                overlay_guard.as_deref(),
                cancel,
                &self.base_path,
                arena,
                overflow_arena,
                Some(&self.sync_data.content_masks),
            )
        };

        if rayon::current_thread_index().is_some() {
            run_grep()
        } else {
            crate::parallelism::run_on_search_pool(run_grep)
        }
    }

    /// Multi-pattern grep search across indexed files.
    pub fn multi_grep(
        &self,
        patterns: &[&str],
        constraints: &[fff_query_parser::Constraint<'_>],
        options: &GrepSearchOptions,
    ) -> GrepResult<'_> {
        let _query = crate::parallelism::QueryActiveGuard::new();
        let overlay_guard = self.sync_data.bigram_overlay.as_ref().map(|o| o.read());
        let arena = self.arena_base_ptr();
        let overflow_arena = self.sync_data.arena_overflow_ptr();
        let cancel = options
            .abort_signal
            .as_deref()
            .unwrap_or(&self.signals.cancelled);

        crate::parallelism::run_on_search_pool(|| {
            multi_grep_search(
                self.get_files(),
                patterns,
                constraints,
                options,
                self.cache_budget(),
                self.sync_data.bigram_index.as_deref(),
                overlay_guard.as_deref(),
                cancel,
                &self.base_path,
                arena,
                overflow_arena,
            )
        })
    }

    // Returns an ongoing or finisshed scan progress
    pub fn get_scan_progress(&self) -> ScanProgress {
        let scanned_count = self.scanned_files_count.load(Ordering::Relaxed);
        let is_scanning = self.signals.scanning.load(Ordering::Relaxed);

        ScanProgress {
            scanned_files_count: scanned_count,
            is_scanning,
            is_watcher_ready: self.signals.watcher_ready.load(Ordering::Relaxed),
            is_warmup_complete: !self.enable_content_indexing
                || self.sync_data.bigram_index.is_some(),
        }
    }

    pub(crate) fn set_bigram_index(&mut self, index: BigramFilter) {
        self.sync_data.bigram_index = Some(Arc::new(index));
        // once the index is reset automatically reset the overaly
        self.sync_data.bigram_overlay = Some(Arc::new(parking_lot::RwLock::new(
            BigramOverlay::new(self.sync_data.indexable_count),
        )));
    }

    pub(crate) fn scan_signals(&self) -> crate::scan::ScanSignals {
        self.signals.clone()
    }

    pub(crate) fn scanned_files_counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.scanned_files_count)
    }

    /// Capture raw pointers to the picker's internal arrays for off-lock use.
    ///
    /// Sets `post_scan_indexing_active` and returns a snapshot that clears it
    /// on drop. This is the ONLY approved way to escape the lock for
    /// long-running parallel work (git status, warmup, bigram).
    ///
    /// Returns `None` if `post_scan_indexing_active` is already set — this
    /// means another post-scan is in flight and we must not create a second
    /// set of dangling pointers.
    ///
    /// # Safety
    /// 1. `walk_filesystem` reserved `MAX_OVERFLOW_FILES` capacity on the
    ///    files Vec at creation — watcher pushes cannot reallocate it.
    /// 2. `post_scan_indexing_active` is set — prevents `commit_new_sync`
    ///    from replacing the Vec (ScanJob::new checks this flag).
    /// 3. Only `[..base_count]` is accessed — base files use the immutable
    ///    base arena. Overflow files use a different arena.
    pub(crate) unsafe fn post_scan_snapshot(&self) -> Option<PostScanUnsafeSnapshot> {
        if self
            .signals
            .post_scan_indexing_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            tracing::error!(
                "Can not acquire post scan unsafe snapshot, someone already acquired it"
            );
            return None;
        }

        Some(PostScanUnsafeSnapshot {
            files: self.sync_data.files.clone(),
            arena: self.sync_data.chunked_paths.as_ref().map(Arc::clone),
            base_count: self.sync_data.base_count,
            indexable_count: self.sync_data.indexable_count,
            base_path: self.base_path.clone(),
            post_scan_flag: Arc::clone(&self.signals.post_scan_indexing_active),
            _budget: Arc::clone(&self.cache_budget),
        })
    }

    pub(crate) fn commit_new_sync(&mut self, sync: FileSync) {
        self.sync_data = sync;
        self.cache_budget.reset();
        self.bump_index_generation();
    }

    #[inline]
    pub(crate) fn arena_base_ptr(&self) -> ArenaPtr {
        self.sync_data.arena_base_ptr()
    }

    /// Update git statuses for files, using the provided shared frecency tracker.
    pub(crate) fn update_git_statuses(
        &mut self,
        status_cache: GitStatusCache,
        shared_frecency: &SharedFrecency,
    ) -> Result<(), Error> {
        debug!(
            statuses_count = status_cache.statuses_len(),
            "Updating git status",
        );

        let mode = self.mode;
        let bp = self.base_path.clone();
        let shared_base = self.sync_data.shared_base.clone();
        let frecency = shared_frecency.read()?;
        let reader = frecency.as_ref().map(FrecencyTracker::reader);
        let basis = reader.as_ref().map(|r| r.modification_basis(mode));

        let mut modified_set_changed = false;
        status_cache
            .into_iter()
            .try_for_each(|(path, status)| -> Result<(), Error> {
                let index = self.sync_data.find_file_index(&path, &self.base_path);
                if let Some(index) = index
                    && let Some(file) = self.sync_data.files().get(index)
                {
                    let arena = self.sync_data.arena_for_file(file);
                    if Self::store_git_status(file, Some(status)) {
                        modified_set_changed = true;
                    }
                    // Score just stored is returned. No tracker: read the word on the file.
                    let score = if let (Some(reader), Some(basis)) =
                        (reader.as_ref(), basis.as_ref())
                    {
                        file.update_frecency_scores(
                            reader,
                            arena,
                            &bp,
                            shared_base.as_deref(),
                            mode,
                            basis,
                        )
                    } else {
                        file.access_frecency_score() as i32
                    };
                    let dir_idx = file.parent_dir_index as usize;
                    if let Some(dir) = self.sync_data.dirs.get(dir_idx) {
                        dir.update_frecency_if_larger(score);
                    }
                } else {
                    // Expected on sparse checkouts: git reports a status for
                    // a path that isn't materialized on disk and therefore
                    // isn't in the file index. Don't spam the log (#404).
                    debug!(?path, "Git status for path not in index, skipping");
                }
                Ok(())
            })?;

        if modified_set_changed {
            // Git filters key the status epoch. The file-set generation stays
            // so the path-match memo and non-git hit sets remain valid.
            self.bump_git_status_epoch();
        }

        Ok(())
    }

    // Replaces every recency score with a freshly computed set. `None` zeroes
    // them, so a vanished window (orphan HEAD, repo gone) leaves no stale boost.
    pub(crate) fn apply_git_recency(&mut self, scores: Option<&AHashMap<String, i16>>) {
        if !self.git_recency_config.enabled {
            return;
        }

        for file in self.sync_data.files.iter() {
            // Most files stay at zero; a relaxed load avoids dirtying the line.
            if file.git_recency_score() != 0 {
                file.set_git_recency_score(0);
            }
        }

        let Some(scores) = scores else { return };

        let mut applied = 0usize;
        for (relative_path, score) in scores {
            if let Some(index) = self.sync_data.find_by_relative_path(relative_path)
                && let Some(file) = self.sync_data.files().get(index)
            {
                file.set_git_recency_score(*score);
                applied += 1;
            }
        }

        debug!(files_scored = applied, "Git recency scores applied");
    }

    pub fn update_single_file_frecency(
        &mut self,
        file_path: impl AsRef<Path>,
        frecency_tracker: &FrecencyTracker,
    ) -> Result<(), Error> {
        let path = file_path.as_ref();

        let Some(index) = self.sync_data.find_file_index(path, &self.base_path) else {
            return Ok(());
        };

        let shared_base = self.sync_data.shared_base.clone();
        if let Some(file) = self.sync_data.files().get(index) {
            let arena = self.sync_data.arena_for_file(file);
            let reader = frecency_tracker.reader();
            let basis = reader.modification_basis(self.mode);
            let score = file.update_frecency_scores(
                &reader,
                arena,
                &self.base_path,
                shared_base.as_deref(),
                self.mode,
                &basis,
            );

            let dir_idx = file.parent_dir_index as usize;
            if let Some(dir) = self.sync_data.dirs.get(dir_idx) {
                dir.update_frecency_if_larger(score);
            }
        }

        Ok(())
    }

    pub fn get_file_by_path(&self, path: impl AsRef<Path>) -> Option<&FileItem> {
        self.sync_data
            .find_file_index(path.as_ref(), &self.base_path)
            .and_then(|index| self.sync_data.files().get(index))
    }

    pub fn get_mut_file_by_path(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Option<(ArenaPtr, &mut FileItem)> {
        let path = path.as_ref();
        let index = self.sync_data.find_file_index(path, &self.base_path);
        index.and_then(|i| self.sync_data.get_file_mut(i))
    }

    /// Handle the event of certain file being modified or adds a neww file if it is not added
    /// If this function returns `None` it means that picker is in the invalid state, or the capacity
    /// of index is exhausted and a new rescan needs to be triggered.
    #[tracing::instrument(skip(self),level = Level::DEBUG)]
    pub fn handle_create_or_modify(&mut self, path: impl AsRef<Path> + Debug) -> Option<&FileItem> {
        let path = path.as_ref();

        if let Some(idx) = self.sync_data.find_file_index(path, &self.base_path) {
            let slot = if idx < self.sync_data.base_count {
                FileSlot::Base(idx)
            } else {
                FileSlot::Overflow(idx)
            };

            return self.handle_file_modify(path, slot);
        }

        self.add_new_file(path)
    }

    #[tracing::instrument(skip_all, fields(path = ?path), level = Level::DEBUG)]
    fn handle_file_modify(&mut self, path: &Path, slot: FileSlot) -> Option<&FileItem> {
        let overlay = self.sync_data.bigram_overlay.as_ref().map(Arc::clone);
        let pos = slot.index();

        // this is the only way to actually know if the file is on disk, we CAN NOT
        // rely on the watcher to proive the latest state of the file, do the actual check
        let metadata = match std::fs::metadata(path) {
            Ok(m) => {
                self.untombstone_file(pos);

                m
            }
            Err(_) => {
                self.tombstone_file(pos);
                return None;
            }
        };

        let file = self.sync_data.files().get(pos)?;

        let size = metadata.len();
        let modified_time = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        file.update_metadata(&self.cache_budget, modified_time, Some(size));

        // Re-classify binary status from current content (chunked, fixed
        // buffer). Already-binary files are left alone.
        if !file.is_binary() {
            let mut chunk = [0u8; crate::types::BINARY_CLASSIFICATION_CHUNK_SIZE];
            let mask = file.detect_binary_per_byte(path, &mut chunk);
            // Refresh the content mask: the bytes just changed, so a mask
            // recorded at scan time is stale and would make grep reject this
            // file until the next rescan. Not doing this broke 6 bigram
            // overlay coherence tests once masks were populated for all files.
            self.sync_data.content_masks.set(pos, mask);
        } else {
            // Binary now: no mask, so grep reads (and rejects) it.
            self.sync_data
                .content_masks
                .set(pos, crate::types::ContentMask::default());
        }

        // Indexable base-region files feed fresh content to the bigram overlay.
        if matches!(slot, FileSlot::Base(_))
            && let Some(ref overlay) = overlay
        {
            let in_indexable = {
                let guard = overlay.read();
                pos < guard.base_file_count()
            };

            if in_indexable && let Ok(content) = std::fs::read(path) {
                overlay.write().modify_file(pos, &content);
            }
        }

        self.sync_data.files().get(pos)
    }

    /// Adds a new file to picker, if the file can not be added returns `None`
    /// which indicates that it's time to trigger a new sync
    #[tracing::instrument(skip(self))]
    pub fn add_new_file(&mut self, path: &Path) -> Option<&FileItem> {
        if self.get_overflow_files().len() >= MAX_OVERFLOW_FILES {
            return None;
        }

        // On Windows `pathdiff::diff_paths` is byte-wise, so a short-name
        // input never shares a prefix with the canonicalized base_path and
        // the resulting relative path becomes absolute. Canonicalize first.
        #[cfg(windows)]
        let canonical_buf: Option<PathBuf> = if path.starts_with(&self.base_path) {
            None
        } else if let Ok(c) = crate::path_utils::canonicalize(path) {
            Some(c)
        } else {
            tracing::error!(path = ?path.display(), "Failed to canonicalize file path to add");
            return None;
        };

        #[cfg(windows)]
        let path_for_index: &Path = canonical_buf.as_deref().unwrap_or(path);
        #[cfg(not(windows))]
        let path_for_index: &Path = path;

        let (mut file_item, rel_path) =
            FileItem::new(path_for_index.to_path_buf(), &self.base_path, None);

        // One read classifies and records the mask. Mask 0 stays unknown.
        let mask = file_item.detect_binary_per_byte(
            path_for_index,
            &mut [0u8; crate::types::BINARY_CLASSIFICATION_CHUNK_SIZE],
        );

        let builder = self.sync_data.overflow_builder.get_or_insert_with(|| {
            // we know that overflow would never create more files during the file
            crate::simd_path::ChunkedPathStoreBuilder::new(MAX_OVERFLOW_FILES)
        });

        file_item.set_path(builder.add_file_immediate(&rel_path, file_item.path.filename_offset));
        let presence = crate::presence::mask_of(rel_path.as_bytes());
        file_item.set_presence(presence);
        file_item.set_overflow(true);

        // Keep the dir table consistent: register (or revive) the parent dir
        // so directory search reflects watcher-added files immediately.
        let dir_rel = crate::path_utils::to_canonical_slashes(
            &rel_path[..file_item.path.filename_offset as usize],
        );

        if let Some(dir_idx) = self.sync_data.find_or_add_dir(&dir_rel) {
            file_item.parent_dir_index = dir_idx;
        }
        let parent_dir = file_item.parent_dir_index;

        let new_path_len = file_item.path.byte_len;

        if !self.sync_data.files.push(file_item) {
            return None;
        }
        let new_index = self.sync_data.files.len() - 1;
        self.sync_data.content_masks.set(new_index, mask);
        self.sync_data.presence.push(presence);
        self.sync_data.max_path_len.observe(new_path_len);

        self.sync_data.live_count += 1;
        // Dir may have been tombstoned by an earlier removal; a new file
        // under it proves it exists again.
        self.sync_data.revive_dir(parent_dir);
        self.bump_index_generation();
        self.sync_data.files.last()
    }

    fn tombstone_file(&mut self, index: usize) {
        let file = &self.sync_data.files[index];
        if file.is_deleted() {
            return;
        }

        file.set_deleted(true);
        file.invalidate_mmap(&self.cache_budget);
        file.set_git_status(None);
        // Drop the stale content mask so a resurrected file is never skipped.
        self.sync_data
            .content_masks
            .set(index, crate::types::ContentMask::default());
        // Removing the longest path can only lower the cached max; recompute
        // lazily on the next read instead of walking the file list here.
        self.sync_data.max_path_len.forget(file.path.byte_len);

        // Only base-region files participate in the bigram overlay
        if index < self.sync_data.base_count
            && let Some(ref overlay) = self.sync_data.bigram_overlay
        {
            overlay.write().delete_file(index);
        }

        self.sync_data.live_count -= 1;
        self.bump_index_generation();
    }

    fn untombstone_file(&mut self, index: usize) {
        let file = &self.sync_data.files[index];
        if !file.is_deleted() {
            return;
        }
        file.set_deleted(false);
        let parent_dir = file.parent_dir_index;

        self.sync_data.live_count += 1;
        // The path exists on disk again, so its parent dir does too.
        self.sync_data.revive_dir(parent_dir);
        self.bump_index_generation();
    }

    /// Marks file as deleted, make sure that if you call this yourself these changes can be reverted
    /// by the internal mechanics if the file actually exists on the disk, use only if you know that
    /// the file going to be disapperaed or if you do not have the watcher installed
    pub fn remove_file_by_path(&mut self, path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        if let Some(index) = self.sync_data.find_file_index(path, &self.base_path) {
            self.tombstone_file(index);
            true
        } else {
            false
        }
    }

    pub fn remove_all_files_in_dir(&mut self, dir: impl AsRef<Path>) -> usize {
        self.remove_all_files_in_dirs_inner(std::iter::once(dir.as_ref()), None)
    }

    /// Tombstones files under any of `dirs` in a single index scan.
    pub(crate) fn remove_all_files_in_dirs_with_callback<'a>(
        &mut self,
        dirs: impl IntoIterator<Item = &'a Path>,
        mut callback: impl FnMut(&Path),
    ) -> usize {
        self.remove_all_files_in_dirs_inner(dirs, Some(&mut callback))
    }

    pub(crate) fn remove_all_files_in_dirs<'a>(
        &mut self,
        dirs: impl IntoIterator<Item = &'a Path>,
    ) -> usize {
        self.remove_all_files_in_dirs_inner(dirs, None)
    }

    fn remove_all_files_in_dirs_inner<'a>(
        &mut self,
        dirs: impl IntoIterator<Item = &'a Path>,
        mut callback: Option<&mut dyn FnMut(&Path)>,
    ) -> usize {
        let mut dir_prefixes = Vec::new();
        for dir_path in dirs {
            let Some(relative_dir) = self
                .to_relative_path(dir_path)
                .map(|path| path.into_owned())
            else {
                continue;
            };

            if relative_dir.is_empty() {
                dir_prefixes.push(String::new());
            } else {
                // Stored relative paths are '/'-canonical on every platform.
                dir_prefixes.push(format!("{relative_dir}/"));
            }
        }

        if dir_prefixes.is_empty() {
            return 0;
        }

        let base_path = self.base_path.clone();
        let cache_budget = &self.cache_budget;
        let mut path_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
        // Translate prefixes to dir-table indices once: the file loop then
        // tests one integer per file instead of re-reading every arena path
        // per prefix. Equivalent to the prefix match (table paths are the
        // files' actual parents); out-of-table parents fall back to it.
        let mut subtree: AHashSet<u32> = AHashSet::new();
        {
            let base_arena = self.sync_data.arena_base_ptr();
            let overflow_arena = self.sync_data.arena_overflow_ptr();
            let base_dirs_count = self.sync_data.base_dirs_count;
            let mut dir_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
            for (di, dir) in self.sync_data.dirs.iter().enumerate() {
                let arena = if di < base_dirs_count {
                    base_arena
                } else {
                    overflow_arena
                };
                let rel = dir.read_relative_path(arena, &mut dir_buf);
                if dir_prefixes.iter().any(|prefix| rel.starts_with(prefix)) {
                    subtree.insert(di as u32);
                }
            }
        }
        let dirs_len = self.sync_data.dirs.len();
        let base_count = self.sync_data.base_count;
        let base_arena = self.sync_data.arena_base_ptr();
        let overflow_arena = self.sync_data.arena_overflow_ptr();
        let mut tombstoned = 0usize;
        for (idx, file) in self.sync_data.files.iter().enumerate() {
            if file.is_deleted() {
                continue;
            }
            let arena = if idx < base_count {
                base_arena
            } else {
                overflow_arena
            };
            let hit = if subtree.contains(&file.parent_dir_index) {
                true
            } else if (file.parent_dir_index as usize) >= dirs_len {
                dir_prefixes
                    .iter()
                    .any(|prefix| file.relative_path_starts_with(arena, prefix))
            } else {
                false
            };
            if !hit {
                continue;
            }
            file.invalidate_mmap(cache_budget);
            if let Some(callback) = callback.as_mut() {
                callback(file.write_absolute_path(arena, &base_path, &mut path_buf));
            }
            file.set_deleted(true);
            tombstoned += 1;
        }
        self.sync_data.live_count -= tombstoned;

        // The whole subtree is gone: tombstone the dirs too so directory
        // search stops surfacing them.
        let mut dir_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
        self.sync_data.tombstone_dirs_with_arena(|dir, arena| {
            let rel = dir.read_relative_path(arena, &mut dir_buf);
            dir_prefixes.iter().any(|prefix| rel.starts_with(prefix))
        });

        if tombstoned > 0 {
            self.bump_index_generation();
        }
        tombstoned
    }

    /// Use this to prevent any substantial background threads from acquiring the locks
    pub fn cancel(&self) {
        self.signals.cancelled.store(true, Ordering::Release);
    }

    /// Stop the background filesystem watcher. Non-blocking.
    pub fn stop_background_monitor(&mut self) {
        if let Some(mut watcher) = self.background_watcher.take() {
            watcher.stop();
        }
        self.signals.watcher_ready.store(false, Ordering::Release);
    }

    /// Quick way to check if scan is going without acquiring a lock for [Self::get_scan_progress]
    pub fn is_scan_active(&self) -> bool {
        self.signals.scanning.load(Ordering::Relaxed)
    }

    pub fn is_post_scan_active(&self) -> bool {
        self.signals
            .post_scan_indexing_active
            .load(Ordering::Acquire)
    }

    /// Return a clone of the watcher-ready flag so callers can poll it without
    /// holding a lock on the picker.
    pub fn watcher_signal(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.signals.watcher_ready)
    }

    /// Convert an absolute path to a relative path string (relative to base_path).
    /// Returns None if the path doesn't start with base_path.
    ///
    /// On Windows the picker canonicalizes its base via `dunce`, so caller
    /// paths that still carry 8.3 short names or a different casing would
    /// fail a naive prefix check. Fall back to canonicalizing (or, when the
    /// file was just deleted, canonicalizing its parent) before stripping.
    #[allow(clippy::needless_return)]
    fn to_relative_path<'a>(&self, path: &'a Path) -> Option<std::borrow::Cow<'a, str>> {
        if let Ok(stripped) = path.strip_prefix(&self.base_path)
            && let Some(s) = stripped.to_str()
        {
            // Callers compare against '/'-canonical stored paths.
            return Some(crate::path_utils::to_canonical_slashes(s));
        }

        #[cfg(windows)]
        {
            let rel = canonical_relative_path(path, &self.base_path)?;
            return Some(std::borrow::Cow::Owned(rel));
        }

        #[cfg(not(windows))]
        None
    }
}

/// Resolve a possibly-short-name Windows path to the picker's canonical base.
/// Used by the Windows-only fallbacks in `to_relative_path` and
/// `find_file_index` so events still match tombstoned entries.
#[cfg(windows)]
fn canonical_relative_path(path: &Path, base: &Path) -> Option<String> {
    if let Ok(canonical) = crate::path_utils::canonicalize(path)
        && let Ok(stripped) = canonical.strip_prefix(base)
        && let Some(s) = stripped.to_str()
    {
        return Some(crate::path_utils::to_canonical_slashes(s).into_owned());
    }

    // Deleted files can't be canonicalized — canonicalize the parent and
    // re-attach the filename.
    let parent = path.parent()?;
    let file_name = path.file_name()?;
    let canonical_parent = crate::path_utils::canonicalize(parent).ok()?;
    let stripped_parent = canonical_parent.strip_prefix(base).ok()?;
    let mut rel = stripped_parent.to_path_buf();
    rel.push(file_name);
    rel.to_str()
        .map(|s| crate::path_utils::to_canonical_slashes(s).into_owned())
}

impl Drop for FilePicker {
    fn drop(&mut self) {
        // Cancel any in-flight ScanJob bound to this picker's signals so
        // it cannot mutate the replacement picker after a swap.
        self.signals.cancelled.store(true, Ordering::Release);
        // Wake the git-status consumer so it exits; never joined (it takes
        // the picker write lock, a blocking join here could deadlock).
        self.git_status_worker.signal_shutdown();
    }
}

#[derive(Debug, Clone, Copy)]
enum FileSlot {
    Base(usize),
    Overflow(usize),
}

impl FileSlot {
    fn index(self) -> usize {
        match self {
            FileSlot::Base(i) | FileSlot::Overflow(i) => i,
        }
    }
}

/// Snapshot of FilePicker state for off-lock post-scan work.
///
/// Each data field is an Arc-shared clone of the picker's backing
/// allocation, so dropping the `FilePicker` (e.g. via
/// `SharedFilePicker::write().take()`) cannot free memory this
/// snapshot is still reading — UAF is impossible by construction.
///
/// Implements `Drop` to clear `post_scan_indexing_active`. Since only
/// one snapshot can exist at a time (enforced by the flag check in
/// `post_scan_snapshot`) and it is always created/dropped within
/// `ScanJob::run`, `scan_job_running == false` implies no live snapshot.
pub(crate) struct PostScanUnsafeSnapshot {
    pub files: StableVec<FileItem>,
    pub arena: Option<Arc<crate::simd_path::ChunkedPathStore>>,
    // TODO figure this out
    pub _budget: Arc<crate::types::ContentCacheBudget>,
    pub base_count: usize,
    pub indexable_count: usize,
    pub base_path: PathBuf,
    post_scan_flag: Arc<AtomicBool>,
}

impl Drop for PostScanUnsafeSnapshot {
    fn drop(&mut self) {
        self.post_scan_flag.store(false, Ordering::Release);
    }
}

// SAFETY: every data field is Arc-shared and outlives the snapshot
// via its own refcount. The mutable cast in `apply_git_status_and_frecency`
// is consumed on the scan thread under the single-writer discipline.
unsafe impl Send for PostScanUnsafeSnapshot {}
unsafe impl Sync for PostScanUnsafeSnapshot {}

/// A point-in-time snapshot of the file-scanning progress.
///
/// Returned by [`FilePicker::get_scan_progress`]. Useful for displaying
/// a progress indicator while the initial scan is running.
#[derive(Debug, Clone)]
pub struct ScanProgress {
    pub scanned_files_count: usize,
    pub is_scanning: bool,
    pub is_watcher_ready: bool,
    pub is_warmup_complete: bool,
}

impl FileSync {
    pub(crate) fn discover_git(base_path: &Path) -> Option<GitDiscovery> {
        // `discover` walks ancestors: a corpus staged under another repo would
        // inherit that parent git and change hidden/gitignore indexing. Prefer
        // the repo at base_path itself, else keep monorepo-subdir behavior.
        let git = if base_path.join(".git").exists() {
            Repository::open(base_path)
                .ok()
                .and_then(GitDiscovery::from_repo)
        } else {
            GitDiscovery::discover(base_path)
        };
        match &git {
            Some(git) => {
                debug!(workdir = ?git.workdir, git_dir = ?git.git_dir, "Git repository found")
            }
            None => warn!("No git repository found for path: {}", base_path.display()),
        }

        git
    }

    pub(crate) fn spawn_git_status(git_workdir: PathBuf) -> JoinHandle<Option<GitStatusCache>> {
        std::thread::spawn(move || {
            GitStatusCache::read_git_status(
                Some(git_workdir.as_path()),
                &mut crate::git::initial_scan_status_options(),
            )
        })
    }

    /// Returns files immediately (searchable) and a handle to the in-progress
    /// git status computation. This avoids blocking on `git status` which can
    /// take 10+ seconds on very large repos (e.g. chromium).
    #[tracing::instrument(skip_all, name = "walk_filesystem", level = Level::INFO)]
    pub(crate) fn walk_filesystem(
        base_path: &Path,
        git: Option<GitDiscovery>,
        synced_files_count: &Arc<AtomicUsize>,
        shared_frecency: &SharedFrecency,
        mode: FFFMode,
        follow_symlinks: bool,
    ) -> Result<FileSync, Error> {
        let scan_start = std::time::Instant::now();
        info!("SCAN: Starting filesystem walk and git status (async)");

        // Walk files (the fast part, typically 2-3s even on huge repos).
        let is_git_repo = git.is_some();
        let shared_base = git.as_ref().and_then(|git| git.shared_base(base_path));
        if let Some(shared_base) = &shared_base {
            info!(
                ?shared_base,
                "Linked git worktree, sharing the main checkout history"
            );
        }
        // Limit walk parallelism: the ignore walker spawns its own workers and
        // coexisting with the BG pool inflates scan peak RSS on flat corpora.
        let bg_threads = BACKGROUND_THREAD_POOL.current_num_threads().min(4);

        let (buffers, mut pairs, mut walked_dirs, ignore_rules) = crate::walk::walk_collect_files(
            base_path,
            is_git_repo,
            follow_symlinks,
            bg_threads,
            synced_files_count,
        )?
        .into_parts();
        let ignore_rules = ignore_rules.map(Arc::new);

        // group walked dirs and files with a dir part to the same order
        crate::parallelism::run_on_bg_pool(|| {
            rayon::join(
                || {
                    pairs.par_sort_unstable_by(|(a, span_a), (b, span_b)| {
                        // `filename_offset` is a char boundary in the stored UTF-8.
                        let path_a = crate::walk::span_bytes(&buffers, *span_a);
                        let path_b = crate::walk::span_bytes(&buffers, *span_b);
                        let (a_dir, a_file) = path_a.split_at(a.path.filename_offset as usize);
                        let (b_dir, b_file) = path_b.split_at(b.path.filename_offset as usize);
                        a_dir.cmp(b_dir).then_with(|| a_file.cmp(b_file))
                    });
                },
                || {
                    walked_dirs.par_sort_unstable_by(|a, b| {
                        crate::walk::span_bytes(&buffers, *a).cmp(crate::walk::span_bytes(&buffers, *b))
                    });
                },
            );
        });
        walked_dirs.dedup_by(|a, b| {
            crate::walk::span_bytes(&buffers, *a) == crate::walk::span_bytes(&buffers, *b)
        });

        let mut builder = crate::simd_path::ChunkedPathStoreBuilder::new(pairs.len());
        let (mut files, dirs) =
            populates_dirs_files_chunked_storage(&buffers, pairs, &walked_dirs, &mut builder);
        drop(walked_dirs);
        drop(buffers);
        let chunked_paths = builder.finish();
        let arena = chunked_paths.as_arena_ptr();

        // Apply frecency scores (access-based only — git status not yet available).
        // DirItem.max_access_frecency is AtomicI32, so parallel threads write directly.
        let frecency = shared_frecency
            .read()
            .map_err(|_| Error::AcquireFrecencyLock)?;

        if let Some(frecency) = frecency.as_ref()
            && !frecency.is_empty()
        {
            if let Some(shared_base) = shared_base.as_deref() {
                // Linked worktree: per-file scoring resolves the main-checkout
                // twin, which batch-by-hash lookup cannot express.
                let dirs_ref = &dirs;
                BACKGROUND_THREAD_POOL.install(|| {
                    files.par_iter_mut().for_each_init(
                        || frecency.reader(),
                        |reader, file| {
                            // Status is unset in this pass; modification stays 0.
                            let score = file.update_access_frecency_score(
                                reader,
                                arena,
                                base_path,
                                Some(shared_base),
                                mode,
                            );
                            if score > 0 {
                                let dir_idx = file.parent_dir_index as usize;
                                if let Some(dir) = dirs_ref.get(dir_idx) {
                                    dir.update_frecency_if_larger(score);
                                }
                            }
                        },
                    );
                });
            } else {
                // Batch hydration in chunks: one LMDB read txn per chunk, no
                // per-file canonicalize. Chunking bounds peak RSS and page-fault
                // churn on 200k+ trees (4T fault spikes in paired runs).
                const FREQ_CHUNK: usize = 8 * 1024;
                let mut offset = 0;
                while offset < files.len() {
                    let end = (offset + FREQ_CHUNK).min(files.len());
                    let hashes: Vec<[u8; 32]> = crate::parallelism::run_on_bg_pool(|| {
                        files[offset..end]
                            .par_iter()
                            .map(|file| {
                                let mut abs_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
                                let abs = file.write_absolute_path(arena, base_path, &mut abs_buf);
                                FrecencyTracker::path_hash_raw(abs).unwrap_or([0u8; 32])
                            })
                            .collect()
                    });
                    let scores = frecency.get_access_scores_for_hashes(&hashes, mode);
                    let dirs_ref = &dirs;
                    crate::parallelism::run_on_bg_pool(|| {
                        files[offset..end].par_iter_mut().zip(scores).for_each(
                            |(file, access_score)| {
                                let stored = access_score as i16;
                                file.set_access_frecency_score(stored);
                                // Status is unset here, so the modification word stays 0.
                                let score = stored as i32;
                                if score > 0 {
                                    let dir_idx = file.parent_dir_index as usize;
                                    if let Some(dir) = dirs_ref.get(dir_idx) {
                                        dir.update_frecency_if_larger(score);
                                    }
                                }
                            },
                        );
                    });
                    offset = end;
                }
            }
        }
        drop(frecency);

        // un-indexable files that are binary or not fitting the size cap has to beplaced in the end
        let is_indexable = |f: &FileItem| {
            !f.is_binary()
                && f.size() > 0
                && f.size() <= crate::constants::MAX_INDEXABLE_FILE_SIZE as u64
        };

        // Sort order is (indexable, parent_dir, filename). Downstream
        // binary_search assumes this order; filename cmp does not allocate.
        BACKGROUND_THREAD_POOL.install(|| {
            files.par_sort_unstable_by(|a, b| {
                (!is_indexable(a))
                    .cmp(&(!is_indexable(b)))
                    .then_with(|| a.parent_dir_index.cmp(&b.parent_dir_index))
                    .then_with(|| a.path.cmp_filename(arena, &b.path))
            });
        });
        let indexable_count = files.partition_point(is_indexable);

        // Ask the allocator to return freed pages to the OS.
        hint_allocator_collect();

        let file_item_size = std::mem::size_of::<FileItem>();
        let files_vec_bytes = files.len() * file_item_size;
        let dir_table_bytes = dirs.len() * std::mem::size_of::<DirItem>()
            + dirs.iter().map(|d| d.path.byte_len as usize).sum::<usize>();

        let total_time = scan_start.elapsed();
        info!(
            "SCAN: Walk completed in {:?} ({} files, {} dirs, \
         chunked_store={:.2}MB, files_vec={:.2}MB, dirs={:.2}MB, FileItem={}B)",
            total_time,
            files.len(),
            dirs.len(),
            chunked_paths.heap_bytes() as f64 / 1_048_576.0,
            files_vec_bytes as f64 / 1_048_576.0,
            dir_table_bytes as f64 / 1_048_576.0,
            file_item_size,
        );

        let base_count = files.len();
        let base_dirs_count = dirs.len();

        // Seed the cached max path length from the files just scanned.
        // `effective_match_threads` needs a real haystack length or it
        // under-provisions threads.
        let scanned_max_path_len = files.iter().map(|f| f.path.byte_len).max().unwrap_or(0);
        let max_path_len = Arc::new(PathLenCache::new());
        max_path_len.observe(scanned_max_path_len);
        // Spare slots so a file added later keeps the mask from its one read.
        let content_masks = Arc::new(ContentMaskStore::with_capacity(
            files.len().saturating_add(MAX_OVERFLOW_FILES),
        ));
        let mut presence = crate::types::PresenceTable::from_files(&files);
        presence.reserve_additional(MAX_OVERFLOW_FILES);

        Ok(FileSync {
            files: StableVec::from_vec_with_reserve(files, MAX_OVERFLOW_FILES),
            indexable_count,
            base_count,
            live_count: base_count,
            max_path_len,
            content_masks,
            presence,
            dirs: StableVec::from_vec_with_reserve(dirs, MAX_OVERFLOW_FILES),
            base_dirs_count,
            live_dirs_count: base_dirs_count,
            overflow_builder: None,
            git,
            shared_base,
            bigram_index: None,
            bigram_overlay: None,
            chunked_paths: Some(Arc::new(chunked_paths)),
            ignore_rules,
        })
    }
}

/// Pre-populate mmap caches for cold tail files so the first grep search
/// doesn't pay the mmap creation + page fault cost.
#[allow(dead_code)]
#[tracing::instrument(skip(files), name = "warmup_mmaps", level = Level::DEBUG)]
pub(crate) fn warmup_mmaps(
    files: &[FileItem],
    budget: &ContentCacheBudget,
    base_path: &Path,
    arena: ArenaPtr,
) {
    // for most of the use cases mmaps limit would be significantly smaller than arepo
    for file in files.iter() {
        if file.is_likely_hot()
            || file.is_binary()
            || file.size() == 0
            || file.size() > budget.max_file_size
        {
            continue;
        }

        let _ = file.get_cached_content(arena, base_path, budget);

        if budget.is_exhausted() {
            break;
        }
    }
}

/// This does both thing (yes sorry all the OOP morons)
/// in one go: populates files chunked storage and builds the dir table from
/// `walked_dirs` (every dir the walker visited: sorted, '/'-terminated,
/// deduped), merging file parents in a single lockstep sweep so dirs with no
/// files (empty subtrees, pure ancestors) are indexed and searchable too.
fn populates_dirs_files_chunked_storage(
    buffers: &[Vec<u8>],
    pairs: Vec<(FileItem, WalkSpan)>,
    walked_dirs: &[WalkSpan],
    chunk_storage: &mut crate::simd_path::ChunkedPathStoreBuilder,
) -> (Vec<FileItem>, Vec<DirItem>) {
    let mut dirs: Vec<DirItem> = Vec::with_capacity(walked_dirs.len() + 1);
    let mut dir_iter = walked_dirs.iter().peekable();

    // Root-level files sort first and their "" parent is never a walker dir.
    if pairs
        .first()
        .is_some_and(|(f, _)| f.path.filename_offset == 0)
    {
        push_dir_item(&mut dirs, chunk_storage, "");
    }

    // Detects contiguous same-dir runs (pairs are sorted by dir) so the
    // merge below runs once per directory, not once per file.  `prev_dir`
    // is owned because each iteration consumes its `rel` String.
    let mut prev_dir: String = String::new();
    let mut current_dir_idx: u32 = 0;

    let mut out_files: Vec<FileItem> = Vec::with_capacity(pairs.len());
    for (mut file, span) in pairs {
        let rel = crate::walk::span_path(buffers, span);
        let dir_part: &str = &rel[..file.path.filename_offset as usize];

        if prev_dir != dir_part {
            // Flush walked dirs up to and including this file's parent,
            // keeping the table sorted for the find_dir_index binary search.
            loop {
                let Some(span) = dir_iter.peek().map(|s| **s) else {
                    break;
                };
                let dir = crate::walk::span_path(buffers, span);
                if dir >= dir_part {
                    break;
                }
                dir_iter.next();
                push_dir_item(&mut dirs, chunk_storage, dir);
            }

            if dir_iter
                .peek()
                .map(|s| **s)
                .is_some_and(|span| crate::walk::span_path(buffers, span) == dir_part)
            {
                let span = dir_iter.next().unwrap();
                push_dir_item(
                    &mut dirs,
                    chunk_storage,
                    crate::walk::span_path(buffers, *span),
                );
            } else {
                // Parents the walker reported with a non-dir kind
                // (e.g. followed symlinks) aren't in the list.
                push_dir_item(&mut dirs, chunk_storage, dir_part);
            }

            current_dir_idx = (dirs.len() - 1) as u32;
            prev_dir.clear();
            prev_dir.push_str(dir_part);
        }

        file.path = chunk_storage.add_file_immediate(&rel, file.path.filename_offset);
        file.set_presence(crate::presence::mask_of(rel.as_bytes()));
        file.parent_dir_index = current_dir_idx;
        if crate::types::dir_is_vendored(dir_part) {
            file.set_vendored();
        }
        out_files.push(file);
    }

    for dir in dir_iter {
        push_dir_item(&mut dirs, chunk_storage, crate::walk::span_path(buffers, *dir));
    }

    (out_files, dirs)
}

fn push_dir_item(
    dirs: &mut Vec<DirItem>,
    chunk_storage: &mut crate::simd_path::ChunkedPathStoreBuilder,
    dir_part: &str,
) {
    let dir_string = chunk_storage.add_dir_immediate(dir_part);

    // Compute last-segment offset: for "src/components/" -> 4 (points to "components/")
    let last_seg = if dir_part.is_empty() {
        0
    } else {
        let trimmed = dir_part.trim_end_matches(std::path::is_separator);
        trimmed
            .rfind(std::path::is_separator)
            .map(|i| i + 1)
            .unwrap_or(0) as u16
    };

    dirs.push(DirItem::new(dir_string, last_seg));
}

/// Fast extension-based binary detection. Avoids opening files during scan.
/// Covers the vast majority of binary files in typical repositories.
#[inline]
#[doc(hidden)]
pub fn is_known_binary_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    is_binary_extension_str(ext)
}

/// Like [`is_known_binary_extension`] but takes a basename string directly,
/// avoiding `Path::extension()` overhead. Mirrors `Path::extension()`
/// semantics: dotfiles with no other dots → no extension. Used by the zlob
/// walker, which already has the basename slice from traversal.
#[cfg(feature = "zlob")]
#[inline]
pub(crate) fn is_known_binary_extension_basename(name: &str) -> bool {
    match name.rfind('.') {
        Some(pos) if pos > 0 && pos < name.len() - 1 => is_binary_extension_str(&name[pos + 1..]),
        _ => false,
    }
}

#[inline]
fn is_binary_extension_str(ext: &str) -> bool {
    matches!(
        ext,
        // Images
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" | "webp" | "tiff" | "tif" | "avif" |
        "heic" | "heif" | "jxl" | "jp2" | "j2k" | "psd" | "icns" | "cur" | "cr2" |
        "nef" | "dng" | "tga" |
        // GPU / VFX texture formats
        "rgbe" | "hdr" | "exr" | "dds" | "ktx" | "ktx2" | "pvr" | "astc" |
        // Adobe Illustrator (PDF wrapper) / Apple webarchive / MIME HTML archive
        "ai" | "webarchive" | "mhtml" |
        // Video/Audio
        "mp4" | "avi" | "mov" | "wmv" | "mkv" | "mp3" | "wav" | "flac" | "ogg" | "m4a" |
        "aac" | "webm" | "flv" | "mpg" | "mpeg" | "wma" | "opus" | "pcm" | "reapeaks" |
        // Compressed/Archives
        "zip" | "tar" | "gz" | "bz2" | "xz" | "7z" | "rar" | "zst" | "lz4" | "lzma" |
        "cab" | "cpio" | "jsonlz4" |
        // Packages/Installers
        "deb" | "rpm" | "apk" | "dmg" | "msi" | "iso" | "nupkg" | "whl" | "egg" |
        "appimage" | "flatpak" | "crx" | "pak" |
        // Executables/Libraries
        "exe" | "dll" | "so" | "dylib" | "o" | "a" | "lib" | "bin" | "elf" |
        // Documents (binary office formats)
        "pdf" | "doc" | "docx" | "xls" | "xlsx" | "ppt" | "pptx" |
        // Databases
        "db" | "sqlite" | "sqlite3" | "mdb" |
        // SQLite / LevelDB auxiliary files
        "sqlite-wal" | "sqlite-shm" | "sqlite3-wal" | "sqlite3-shm" |
        "db-wal" | "db-shm" | "ldb" |
        // Fonts
        "ttf" | "otf" | "woff" | "woff2" | "eot" |
        // Compiled/Runtime
        "class" | "pyc" | "pyo" | "wasm" | "dex" | "jar" | "war" |
        // OCaml / Swift / Objective-C build artefacts
        "cmi" | "cmt" | "cmti" | "cmx" | "nib" |
        "swiftdeps" | "swiftdeps~" | "swiftdoc" | "swiftmodule" | "swiftsourceinfo" |
        // ML/Data Science
        "npy" | "npz" | "h5" | "hdf5" | "pt" | "onnx" |
        "safetensors" | "tfrecord" | "tflite" | "gguf" | "ggml" | "joblib" |
        // 3D/Game assets
        "glb" | "blend" | "blp" |
        // Gzipped-XML / binary maps
        "dia" | "bcmap" |
        // Protobuf wire format
        "pb" |
        // Data/serialized
        "parquet" | "arrow" |
        // IDE/OS metadata
        "suo"
    )
}

/// Length of the longest shared directory prefix of two relative dir
/// paths (without a trailing separator), measured as the number of bytes
/// up to and including the last shared separator — plus the full shorter
/// path when it is itself a directory prefix of the longer one.
///
/// Examples:
///   `"src/components"` vs `"src/routes"`   → 4  (`"src/"` emitted once)
///   `"lib/deep/nested"` vs `"lib/deep"`   → 8  (`"lib/deep"` is a prefix)
///   `"lib/deep"` vs `"lib/deeper"`        → 4  (only `"lib/"` is shared)
///   `"lib"` vs `"src"`                    → 0
///
/// Used by [`FilePicker::for_each_watch_dir`] to avoid re-emitting
/// ancestors that were already yielded for the previous (sorted) sibling.
fn common_dir_prefix_len(a: &str, b: &str) -> usize {
    let max = a.len().min(b.len());
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut last_sep = 0;
    let mut i = 0;
    while i < max && a_bytes[i] == b_bytes[i] {
        if std::path::is_separator(a_bytes[i] as char) {
            last_sep = i + 1;
        }
        i += 1;
    }
    // If one string is a prefix of the other and the next byte in the
    // longer one is a separator, the full shorter path is a shared dir.
    if i == max && i > 0 {
        let longer = if a.len() > b.len() { a_bytes } else { b_bytes };
        if i < longer.len() && std::path::is_separator(longer[i] as char) {
            return i;
        }
    }
    last_sep
}

/// Keep mimalloc off 2 MiB huge pages: with THP the arena is resident at
/// 2 MiB granularity and idle index memory inflates RSS by ~2x. Env overrides win.
/// Must run before the first allocation (see `fff_nvim`'s init-array hook).
#[cfg(feature = "mimalloc-collect")]
pub extern "C" fn tune_mimalloc() {
    // SAFETY: getenv/mi_option_set touch static tables only; no allocation happens here.
    unsafe {
        let user_set = !libc::getenv(c"MIMALLOC_ALLOW_LARGE_OS_PAGES".as_ptr()).is_null()
            || !libc::getenv(c"MIMALLOC_LARGE_OS_PAGES".as_ptr()).is_null();
        if !user_set {
            libmimalloc_sys::mi_option_set(libmimalloc_sys::mi_option_large_os_pages, 0);
        }
    }
}

/// Ask the global allocator to return freed pages to the OS.
/// Enabled via the `mimalloc-collect` feature (set by fff-nvim).
/// No-op when the feature is off (tests, system allocator).
pub(crate) fn hint_allocator_collect() {
    #[cfg(feature = "mimalloc-collect")]
    {
        // Collect BACKGROUND_THREAD_POOL workers — that's where the bigram
        // builder allocated memory. `rayon::broadcast` would target the global
        // pool, which is the wrong set of threads.
        BACKGROUND_THREAD_POOL.broadcast(|_| unsafe { libmimalloc_sys::mi_collect(true) });

        // Main thread too.
        unsafe { libmimalloc_sys::mi_collect(true) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_home_dir_with_trailing_separator() {
        let Some(home) = dirs::home_dir() else { return };
        for base in [home.clone(), home.join("")] {
            let result = FilePicker::new(FilePickerOptions {
                base_path: base.to_string_lossy().into_owned(),
                watch: false,
                ..Default::default()
            });
            assert!(
                matches!(result, Err(Error::FilesystemRoot(_))),
                "home dir accepted as {}",
                base.display()
            );
        }
    }

    /// The watcher must watch every ancestor directory up to `base_path`,
    /// not just the immediate parents of indexed files. The dir table is
    /// built from the walker's visited dirs, so pure ancestors (dirs that
    /// contain only subdirectories) must be present and emitted exactly once.
    #[test]
    fn extract_watch_dirs_includes_pure_ancestor_dirs() {
        let dir = tempfile::tempdir().unwrap();
        // On Windows the picker canonicalizes base_path with dunce; match that
        // here so the stored dir paths line up with assertions built from
        // `base.join(..)` (which otherwise would carry an 8.3 short name).
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();

        // Tree:
        //   base/src/components/button.txt    (src/components has a file)
        //   base/src/routes/home.txt          (src/routes has a file)
        //   base/lib/deep/nested/util.txt     (lib and lib/deep have no files)
        for rel in [
            "src/components/button.txt",
            "src/routes/home.txt",
            "lib/deep/nested/util.txt",
        ] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }

        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();

        let mut watch_dirs: Vec<PathBuf> = Vec::new();
        picker.for_each_dir(|p| {
            watch_dirs.push(p.to_path_buf());
            std::ops::ControlFlow::Continue(())
        });
        let watch_set: std::collections::HashSet<PathBuf> = watch_dirs.iter().cloned().collect();

        // Immediate parents (in sync_data.dirs) must be present.
        for rel in ["src/components", "src/routes", "lib/deep/nested"] {
            assert!(
                watch_set.contains(&base.join(rel)),
                "expected immediate parent {rel} in watch dirs, got {watch_set:?}",
            );
        }

        // Pure-ancestor dirs (NOT in sync_data.dirs) must also be present.
        for rel in ["src", "lib", "lib/deep"] {
            assert!(
                watch_set.contains(&base.join(rel)),
                "expected pure-ancestor {rel} in watch dirs, got {watch_set:?}",
            );
        }

        // No duplicates — streaming dedup must not emit the same dir twice.
        assert_eq!(
            watch_dirs.len(),
            watch_set.len(),
            "duplicate watch dir emitted: {watch_dirs:?}",
        );

        // Base path itself is NOT walked into the result — the walker stops
        // at `current == base`. The outer `debouncer.watch(base_path, ...)`
        // call in create_debouncer covers it separately.
        assert!(
            !watch_set.contains(base),
            "base path must not be in watch dirs (covered by the top-level watch call)",
        );
    }

    /// Regression guard for #725: dirs that are EMPTY at scan time are merged
    /// into `sync_data.dirs` so they are searchable and get an inotify watch;
    /// files created in them later must be detected.
    #[test]
    fn for_each_dir_includes_empty_directories() {
        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();

        // Tree:
        //   base/init.lua                  (file directly under base)
        //   base/commands/                 (empty at scan — the #725 repro)
        //   base/src/main.rs               (src is indexed)
        //   base/src/plugins/extra/        (empty chain under an indexed dir)
        std::fs::create_dir_all(base.join("commands")).unwrap();
        std::fs::create_dir_all(base.join("src/plugins/extra")).unwrap();
        std::fs::write(base.join("init.lua"), b"x").unwrap();
        std::fs::write(base.join("src/main.rs"), b"x").unwrap();

        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();

        let mut watch_dirs: Vec<PathBuf> = Vec::new();
        picker.for_each_dir(|p| {
            watch_dirs.push(p.to_path_buf());
            std::ops::ControlFlow::Continue(())
        });
        let watch_set: std::collections::HashSet<PathBuf> = watch_dirs.iter().cloned().collect();

        for rel in ["commands", "src/plugins", "src/plugins/extra", "src"] {
            assert!(
                watch_set.contains(&base.join(rel)),
                "expected {rel} in watch dirs, got {watch_set:?}",
            );
        }

        // Dirs covered by indexed files must not be duplicated.
        assert_eq!(
            watch_dirs.len(),
            watch_set.len(),
            "duplicate watch dir emitted: {watch_dirs:?}",
        );
    }

    #[test]
    fn dir_table_merges_walked_dirs_with_file_parents() {
        let mut bytes = Vec::new();
        let mut pairs: Vec<(FileItem, WalkSpan)> = ["src/main.rs", "src/deep/lib.rs", "root.txt"]
            .iter()
            .map(|p| {
                let (item, rel) = FileItem::new(PathBuf::from(p), Path::new(""), None);
                let off = bytes.len() as u32;
                bytes.extend_from_slice(rel.as_bytes());
                (
                    item,
                    WalkSpan {
                        buf: 0,
                        off,
                        len: rel.len() as u32,
                    },
                )
            })
            .collect();
        {
            let file_bufs = std::slice::from_ref(&bytes);
            pairs.sort_by(|(a, sa), (b, sb)| {
                let pa = crate::walk::span_bytes(file_bufs, *sa);
                let pb = crate::walk::span_bytes(file_bufs, *sb);
                pa[..a.path.filename_offset as usize]
                    .cmp(&pb[..b.path.filename_offset as usize])
                    .then_with(|| pa.cmp(pb))
            });
        }

        // Sorted '/'-terminated walker output: file parents + an empty dir +
        // a sibling sharing a prefix with a file parent.
        let mut walked = Vec::new();
        for s in ["empty/", "src/", "src/deep/", "src/deeper/"] {
            let off = bytes.len() as u32;
            bytes.extend_from_slice(s.as_bytes());
            walked.push(WalkSpan {
                buf: 0,
                off,
                len: s.len() as u32,
            });
        }

        let mut builder = crate::simd_path::ChunkedPathStoreBuilder::new(pairs.len());
        let (files, dirs) =
            populates_dirs_files_chunked_storage(std::slice::from_ref(&bytes), pairs, &walked, &mut builder);
        let store = builder.finish();
        let arena = store.as_arena_ptr();

        let table: Vec<String> = dirs.iter().map(|d| d.relative_path(arena)).collect();
        // Sorted: "" (root files) first, all walked dirs present exactly once.
        assert_eq!(table, ["", "empty/", "src/", "src/deep/", "src/deeper/"]);

        // Every file's parent_dir_index points at its own dir entry.
        for file in &files {
            let dir = &dirs[file.parent_dir_index as usize];
            let rel = file.relative_path(arena);
            assert!(
                rel.starts_with(&dir.relative_path(arena)),
                "file {rel} must live under its parent dir",
            );
        }
    }

    #[test]
    fn common_dir_prefix_len_cases() {
        assert_eq!(common_dir_prefix_len("", ""), 0);
        assert_eq!(common_dir_prefix_len("", "src"), 0);
        assert_eq!(common_dir_prefix_len("lib", "src"), 0);
        assert_eq!(common_dir_prefix_len("src/components", "src/routes"), 4);
        assert_eq!(common_dir_prefix_len("lib/deep/nested", "lib/deep"), 8);
        assert_eq!(common_dir_prefix_len("lib/deep", "lib/deep/nested"), 8);
        assert_eq!(common_dir_prefix_len("lib/deep", "lib/deeper"), 4);
        assert_eq!(common_dir_prefix_len("src", "src"), 0);
        // "src" is emitted-as-dir; "src/x" extends it — full "src" is shared.
        assert_eq!(common_dir_prefix_len("src", "src/x"), 3);
    }

    #[test]
    fn directory_removal_collects_each_tombstoned_path() {
        let dir = tempfile::tempdir().unwrap();
        let base = crate::path_utils::canonicalize(dir.path()).unwrap();
        let removed_dir = base.join("removed");
        let kept = base.join("kept.txt");
        let first = removed_dir.join("a.txt");
        let second = removed_dir.join("nested/b.txt");
        std::fs::create_dir_all(second.parent().unwrap()).unwrap();
        std::fs::write(&first, b"a").unwrap();
        std::fs::write(&second, b"b").unwrap();
        std::fs::write(&kept, b"kept").unwrap();

        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_string_lossy().into_owned(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();

        let mut removed = Vec::new();
        assert_eq!(
            picker.remove_all_files_in_dirs_with_callback(
                std::iter::once(removed_dir.as_path()),
                |path| {
                    removed.push(path.to_path_buf());
                }
            ),
            2
        );
        removed.sort_unstable();
        assert_eq!(removed, vec![first, second]);
        assert!(picker.get_file_by_path(&kept).is_some());

        let outside = base.parent().unwrap().join("outside");
        assert_eq!(picker.remove_all_files_in_dir(&outside), 0);
        assert!(picker.get_file_by_path(&kept).is_some());
    }

    /// A live post-scan snapshot aliases the file buffer, so `get_file_mut`
    /// must refuse mutable access. Watcher/frecency paths must still update
    /// metadata through the per-entry atomics, and appended overflow files
    /// must remain visible; otherwise the snapshot window silently loses
    /// filesystem events (C-004).
    #[test]
    fn post_scan_snapshot_keeps_watcher_updates_live() {
        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        let existing = base.join("src/existing.rs");
        std::fs::create_dir_all(existing.parent().unwrap()).unwrap();
        std::fs::write(&existing, b"x").unwrap();

        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();
        let idx = picker
            .sync_data
            .find_file_index(&existing, base)
            .expect("scanned file must be indexed");
        assert_eq!(picker.get_file_by_path(&existing).unwrap().size(), 1);

        let snapshot = unsafe { picker.post_scan_snapshot() }.expect("snapshot");
        assert!(picker.get_file_mut(idx).is_none());

        std::fs::write(&existing, b"hello").unwrap();
        let updated = picker
            .handle_create_or_modify(&existing)
            .expect("watcher update");
        assert_eq!(updated.size(), 5);
        assert_eq!(picker.get_file_by_path(&existing).unwrap().size(), 5);
        assert!(
            picker.sync_data.content_masks.get(idx).is_some(),
            "content mask must refresh while the snapshot is held"
        );
        assert_eq!(snapshot.files[idx].size(), 5);

        let added = base.join("src/added.rs");
        std::fs::write(&added, b"new").unwrap();
        assert!(picker.add_new_file(&added).is_some());
        assert_eq!(picker.get_file_by_path(&added).unwrap().size(), 3);
        assert!(
            picker
                .get_overflow_files()
                .iter()
                .any(|f| f.relative_path(&picker).ends_with("src/added.rs")),
            "watcher-added overflow file must be searchable"
        );

        drop(snapshot);
        assert!(!picker.signals.post_scan_indexing_active.load(Ordering::Acquire));
        assert!(picker.get_file_mut(idx).is_some());
    }

    /// Live-narrow caches candidate indices per index generation.
    /// Mutations must bump the generation and disarm narrow, or extended
    /// queries search a stale set.
    #[test]
    fn live_narrow_invalidated_by_index_mutations() {
        use crate::types::PaginationArgs;

        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        for rel in ["src/alpha_1.rs", "src/alpha_2.rs"] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();
        assert_eq!(picker.live_file_count(), 2);

        let opts = || FuzzySearchOptions {
            max_threads: 1,
            current_file: None,
            project_path: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
        };
        let parser = fff_query_parser::QueryParser::default();
        let names_of = |picker: &FilePicker, text: &str| {
            let parsed = parser.parse(text);
            let out = picker.fuzzy_search(&parsed, None, opts());
            let names: Vec<String> = out.items.iter().map(|f| f.relative_path(picker)).collect();
            if let Some(paths) = &out.paths {
                for (item, row) in out.items.iter().zip(paths.rows.iter()) {
                    let start = row.start as usize;
                    let end = start + row.len as usize;
                    assert_eq!(&paths.bytes[start..end], item.relative_path(picker).as_bytes());
                }
            }
            (out.total_matched, names)
        };

        let (total, _) = names_of(&picker, "alph");
        assert_eq!(total, 2);
        assert!(picker.live.lock().active);
        let g0 = picker.index_generation();

        // Added file bumps generation, disarms narrow, and surfaces.
        let added = base.join("src/alpha_3.rs");
        std::fs::write(&added, b"x").unwrap();
        assert!(picker.add_new_file(&added).is_some());
        assert!(!picker.live.lock().active);
        assert!(picker.index_generation() > g0);
        let (total, names) = names_of(&picker, "alpha");
        assert_eq!(total, 3);
        assert!(names.iter().any(|n| n.ends_with("alpha_3.rs")), "{names:?}");
        let g1 = picker.index_generation();

        // Removed file invalidates the same way and stops surfacing.
        assert!(picker.remove_file_by_path(&added));
        assert!(!picker.live.lock().active);
        assert!(picker.index_generation() > g1);
        let (total, names) = names_of(&picker, "alpha");
        assert_eq!(total, 2);
        assert!(
            !names.iter().any(|n| n.ends_with("alpha_3.rs")),
            "{names:?}"
        );
    }

    /// Extended queries narrow the previous hit set instead of rescanning,
    /// and must agree with a fresh full-corpus search (same files, same
    /// order, same totals) — including the empty extension.
    #[test]
    fn live_narrow_extension_matches_full_search() {
        use crate::types::PaginationArgs;

        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        for rel in ["src/alpha_1.rs", "src/alpha_2.rs", "src/beta.rs"] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
        let mut fresh = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        fresh.collect_files().unwrap();
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();

        let opts = || FuzzySearchOptions {
            max_threads: 1,
            current_file: None,
            project_path: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
        };
        let parser = fff_query_parser::QueryParser::default();
        let run = |picker: &FilePicker, text: &str| {
            let parsed = parser.parse(text);
            let out = picker.fuzzy_search(&parsed, None, opts());
            let names: Vec<String> = out.items.iter().map(|f| f.relative_path(picker)).collect();
            let totals: Vec<i32> = out.scores.iter().map(|s| s.total).collect();
            (out.total_matched, names, totals)
        };

        // Prime the live set, then extend (narrow) and dead-end (empty).
        let _ = run(&picker, "alph");
        assert!(picker.live.lock().active);
        for text in ["alpha", "alpha_", "alphzzz"] {
            assert_eq!(
                run(&picker, text),
                run(&fresh, text),
                "narrow diverged for {text:?}"
            );
        }
    }

    /// A raw-query extension must gate narrowing, not just `parts[0]`.
    /// `a zzz` -> `ab yyy` grows the effective first part but is not a raw
    /// prefix; reusing the old (zzz-filtered) set would drop `ab_yyy.rs`.
    /// Multipart extensions that *are* raw prefixes must still match a full
    /// scan.
    #[test]
    fn live_narrow_multipart_identity_matches_full_search() {
        use crate::types::PaginationArgs;

        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        for rel in [
            "src/aa_zzz.rs",
            "src/ab_yyy.rs",
            "src/alpha_one.rs",
            "src/alpha_two.rs",
            "src/beta.rs",
        ] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
        let mut fresh = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        fresh.collect_files().unwrap();
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();

        let opts = || FuzzySearchOptions {
            max_threads: 1,
            current_file: None,
            project_path: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
        };
        let parser = fff_query_parser::QueryParser::default();
        let run = |picker: &FilePicker, text: &str| {
            let parsed = parser.parse(text);
            let out = picker.fuzzy_search(&parsed, None, opts());
            let names: Vec<String> = out.items.iter().map(|f| f.relative_path(picker)).collect();
            let totals: Vec<i32> = out.scores.iter().map(|s| s.total).collect();
            (out.total_matched, names, totals)
        };
        let force_full = |picker: &FilePicker| {
            let mut live = picker.live.lock();
            live.active = false;
            live.history.clear();
            live.prefix.clear();
        };

        // Multipart change disguised as a first-part extension.
        let _ = run(&picker, "a zzz");
        assert!(picker.live.lock().active);
        let narrowed = run(&picker, "ab yyy");
        force_full(&fresh);
        assert_eq!(narrowed, run(&fresh, "ab yyy"));
        assert!(
            narrowed.1.iter().any(|n| n.ends_with("ab_yyy.rs")),
            "new match was dropped: {:?}",
            narrowed.1
        );

        // A true multipart raw extension should narrow and still match full.
        let _ = run(&picker, "src alpha");
        assert!(picker.live.lock().active);
        let narrowed = run(&picker, "src alpha one");
        force_full(&fresh);
        assert_eq!(narrowed, run(&fresh, "src alpha one"));
        assert!(
            narrowed.1.iter().any(|n| n.ends_with("alpha_one.rs")),
            "{:?}",
            narrowed.1
        );
    }

    /// Overflow files live in a second arena, and an appended constraint
    /// changes the filter even when the raw query is a prefix extension.
    #[test]
    fn live_narrow_overflow_and_constraint_extension_match_full_search() {
        use crate::types::PaginationArgs;

        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        for rel in ["src/alpha_1.rs", "src/alpha_2.rs", "src/beta.rs"] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();
        let added = base.join("src/alpha_3.rs");
        std::fs::write(&added, b"x").unwrap();
        assert!(picker.add_new_file(&added).is_some());
        assert!(!picker.get_overflow_files().is_empty());

        let opts = || FuzzySearchOptions {
            max_threads: 1,
            current_file: None,
            project_path: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
        };
        let parser = fff_query_parser::QueryParser::default();
        let run = |picker: &FilePicker, text: &str| {
            let parsed = parser.parse(text);
            let out = picker.fuzzy_search(&parsed, None, opts());
            let names: Vec<String> = out.items.iter().map(|f| f.relative_path(picker)).collect();
            let totals: Vec<i32> = out.scores.iter().map(|s| s.total).collect();
            (out.total_matched, names, totals)
        };
        let force_full = |picker: &FilePicker| {
            let mut live = picker.live.lock();
            live.active = false;
            live.history.clear();
            live.prefix.clear();
        };

        // Prime with an unfiltered prefix, then narrow across the overflow
        // boundary. alpha_3.rs must survive.
        let _ = run(&picker, "alph");
        let narrow_alpha = run(&picker, "alpha");
        assert!(narrow_alpha.1.iter().any(|n| n.ends_with("alpha_3.rs")));
        force_full(&picker);
        assert_eq!(narrow_alpha, run(&picker, "alpha"));

        // Same prefix plus a new constraint must not reuse the old hit set.
        let _ = run(&picker, "alph");
        let narrow_filtered = run(&picker, "alph *.rs");
        force_full(&picker);
        assert_eq!(narrow_filtered, run(&picker, "alph *.rs"));
    }

    /// Shortening the query reuses the matching history entry instead of
    /// rescanning, and must agree with a fresh full-corpus search.
    #[test]
    fn live_narrow_backspace_matches_full_search() {
        use crate::types::PaginationArgs;

        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        for rel in ["src/alpha_1.rs", "src/alpha_2.rs", "src/beta.rs"] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
        let mut fresh = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        fresh.collect_files().unwrap();
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();

        let opts = || FuzzySearchOptions {
            max_threads: 1,
            current_file: None,
            project_path: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
        };
        let parser = fff_query_parser::QueryParser::default();
        let run = |picker: &FilePicker, text: &str| {
            let parsed = parser.parse(text);
            let out = picker.fuzzy_search(&parsed, None, opts());
            let names: Vec<String> = out.items.iter().map(|f| f.relative_path(picker)).collect();
            let totals: Vec<i32> = out.scores.iter().map(|s| s.total).collect();
            (out.total_matched, names, totals)
        };

        // Forward chain populates history; backspaces must reuse it.
        for text in ["alph", "alpha", "alpha_"] {
            assert_eq!(
                run(&picker, text),
                run(&fresh, text),
                "forward diverged for {text:?}"
            );
        }
        assert!(!picker.live.lock().history.is_empty());
        for text in ["alpha", "alph", "alpha_", "alphzzz", "alph"] {
            assert_eq!(
                run(&picker, text),
                run(&fresh, text),
                "backspace diverged for {text:?}"
            );
        }
    }

    #[test]
    fn narrow_margin_follows_measurement() {
        assert!(narrow_pays(0, 227_405));
        assert!(narrow_pays(20_778, 227_405));
        assert!(!narrow_pays(226_747, 227_405));
        assert!(!narrow_pays(227_405, 227_405));
        assert!(narrow_pays(2, 3));
    }

    #[test]
    fn longest_prefix_picks_compatible_entry() {
        let cache = vec![
            BackEntry {
                query: "al".into(),
                max_typos: 2,
                generation: 7,
                constraints_fp: 0,
                git_status_epoch: 0,
                candidates: vec![1].into(),
            },
            BackEntry {
                query: "alph".into(),
                max_typos: 2,
                generation: 7,
                constraints_fp: 0,
                git_status_epoch: 0,
                candidates: vec![1, 2].into(),
            },
            BackEntry {
                query: "alph".into(),
                max_typos: 3,
                generation: 7,
                constraints_fp: 0,
                git_status_epoch: 0,
                candidates: vec![9].into(),
            },
            BackEntry {
                query: "alph".into(),
                max_typos: 2,
                generation: 8,
                constraints_fp: 0,
                git_status_epoch: 0,
                candidates: vec![8].into(),
            },
        ];
        assert_eq!(
            longest_prefix(&cache, "alphax", 2, 7, 0, false, 0)
                .as_deref()
                .map(Vec::as_slice),
            Some(vec![1, 2].as_slice())
        );
        assert_eq!(
            longest_prefix(&cache, "alph", 2, 7, 0, false, 0)
                .as_deref()
                .map(Vec::as_slice),
            Some(vec![1].as_slice())
        );
        assert_eq!(
            longest_prefix(&cache, "beta", 2, 7, 0, false, 0)
                .as_deref()
                .map(Vec::as_slice),
            None
        );
        assert_eq!(
            longest_prefix(&cache, "alphax", 3, 7, 0, false, 0)
                .as_deref()
                .map(Vec::as_slice),
            Some(vec![9].as_slice())
        );
        assert_eq!(
            longest_prefix(&cache, "alphax", 2, 9, 0, false, 0)
                .as_deref()
                .map(Vec::as_slice),
            None
        );
        // A changed constraint set must never reuse an unfiltered hit set.
        assert_eq!(
            longest_prefix(&cache, "alphax", 2, 7, 1, false, 0)
                .as_deref()
                .map(Vec::as_slice),
            None
        );
    }

    /// Forward retype after a sideways move reuses the cached prefix
    /// instead of rescanning, and must agree with a fresh search.
    #[test]
    fn prefix_cache_serves_forward_retype() {
        use crate::types::PaginationArgs;

        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        for rel in ["src/alpha_1.rs", "src/alpha_2.rs", "src/beta.rs"] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
        let mut fresh = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        fresh.collect_files().unwrap();
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();

        let opts = || FuzzySearchOptions {
            max_threads: 1,
            current_file: None,
            project_path: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
        };
        let parser = fff_query_parser::QueryParser::default();
        let run = |picker: &FilePicker, text: &str| {
            let parsed = parser.parse(text);
            let out = picker.fuzzy_search(&parsed, None, opts());
            let names: Vec<String> = out.items.iter().map(|f| f.relative_path(picker)).collect();
            let totals: Vec<i32> = out.scores.iter().map(|s| s.total).collect();
            (out.total_matched, names, totals)
        };

        // "alp" primes the prefix cache; "alph" moves live on; "alpxyz"
        // shares only the cached prefix (live/history both miss it).
        assert_eq!(run(&picker, "alp"), run(&fresh, "alp"));
        assert_eq!(run(&picker, "alph"), run(&fresh, "alph"));
        assert!(!picker.live.lock().prefix.is_empty());
        assert_eq!(
            run(&picker, "alpxyz"),
            run(&fresh, "alpxyz"),
            "prefix retype diverged"
        );
    }

    #[test]
    fn short_query_repeats_match_first_search() {
        use crate::types::PaginationArgs;

        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        for rel in ["src/alpha_1.rs", "src/alpha_2.rs", "src/beta.rs"] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();

        let opts = || FuzzySearchOptions {
            max_threads: 1,
            current_file: None,
            project_path: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
        };
        let parser = fff_query_parser::QueryParser::default();
        let run = |picker: &FilePicker, text: &str| {
            let parsed = parser.parse(text);
            let out = picker.fuzzy_search(&parsed, None, opts());
            (out.total_matched, out.items.len())
        };

        // Single-char queries take the frecency path (no hit capture);
        // repeats and backspaces onto them must recompute, not read Empty.
        let first = run(&picker, "a");
        assert_eq!(first.0, 3);
        assert_eq!(run(&picker, "a"), first);
        assert_eq!(run(&picker, "alph"), (2, 2));
        assert_eq!(run(&picker, "a"), first);
    }

    /// Binary sniff reads a bounded prefix: early NULs flag binary, late
    /// NULs past the cap stay text, and only fully-read files earn masks.
    #[test]
    fn binary_sniff_caps_at_prefix() {
        use crate::types::BINARY_CLASSIFICATION_CAP;

        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        std::fs::create_dir_all(base.join("src")).unwrap();

        let tiny = base.join("src/tiny.txt");
        std::fs::write(&tiny, b"hello world").unwrap();
        // `.dat` is not a known-binary extension, so content decides.
        let early = base.join("src/early.dat");
        let mut early_bytes = vec![b'x'; 64];
        early_bytes[10] = 0;
        std::fs::write(&early, &early_bytes).unwrap();

        // NUL at 1 MiB: past the old 256 KiB window but inside the
        // 2 MiB cap, so it must still be detected.
        let big = BINARY_CLASSIFICATION_CAP + 32 * 1024;
        let mut late_bytes = vec![b'y'; big];
        late_bytes[1024 * 1024] = 0;
        let late = base.join("src/late.dat");
        std::fs::write(&late, &late_bytes).unwrap();
        let text = base.join("src/big.txt");
        std::fs::write(&text, vec![b'z'; big]).unwrap();
        // NUL past the cap stays unread: text, mask unknown so grep reads it.
        let huge = BINARY_CLASSIFICATION_CAP + 64 * 1024;
        let mut huge_bytes = vec![b'w'; huge];
        huge_bytes[huge - 8 * 1024] = 0;
        let huge = base.join("src/huge.dat");
        std::fs::write(&huge, &huge_bytes).unwrap();

        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        for p in [&tiny, &early, &late, &text, &huge] {
            assert!(picker.add_new_file(p).is_some());
        }
        // `add_new_file` classifies but does not store masks; assert on the
        // detector return (empty mask reads back as unknown downstream).
        let sniff = |picker: &FilePicker, p: &std::path::Path| {
            let f = picker.get_file_by_path(p).unwrap();
            let mut chunk = [0u8; crate::types::BINARY_CLASSIFICATION_CHUNK_SIZE];
            let mask = f.detect_binary_per_byte(p, &mut chunk);
            (f.is_binary(), mask)
        };

        let (binary, mask) = sniff(&picker, &tiny);
        assert!(!binary);
        assert_eq!(mask.allows(b"hello world", 0), Some(true));
        let (binary, mask) = sniff(&picker, &early);
        assert!(binary);
        assert_eq!(mask.allows(b"x", 0), None);
        // Late NUL inside the cap: binary, mask unknown so grep skips it.
        let (binary, mask) = sniff(&picker, &late);
        assert!(binary);
        assert_eq!(mask.allows(b"y", 0), None);
        let (binary, mask) = sniff(&picker, &text);
        assert!(!binary);
        assert_eq!(mask.allows(b"z", 0), None);
        // NUL past the cap: stays text, mask unknown so grep reads it.
        let (binary, mask) = sniff(&picker, &huge);
        assert!(!binary);
        assert_eq!(mask.allows(b"w", 0), None);
    }

    #[test]
    fn git_skip_decision_matrix() {
        assert!(!should_skip_git(false, 0, false));
        assert!(!should_skip_git(false, 199_999, false));
        assert!(!should_skip_git(false, AUTO_SKIP_GIT_FILE_COUNT, false));
        assert!(should_skip_git(false, AUTO_SKIP_GIT_FILE_COUNT + 1, false));
        assert!(should_skip_git(false, 1_000_000, false));
        assert!(!should_skip_git(false, 1_000_000, true));
        assert!(should_skip_git(true, 0, false));
        assert!(should_skip_git(true, 0, true));
    }

    fn git_repo_fixture() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let base = crate::path_utils::canonicalize(dir.path()).unwrap();
        Repository::init(base.as_path()).unwrap();
        let path = base.join("src/main.rs");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"fn main() {}").unwrap();
        (dir, base)
    }

    #[test]
    fn explicit_skip_git_stays_skipped() {
        let (_dir, base) = git_repo_fixture();
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            skip_git: true,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();
        assert!(!picker.has_git_repo());
    }

    #[test]
    fn small_git_repo_keeps_git() {
        // Far below the auto threshold: discovery + status path run.
        let (_dir, base) = git_repo_fixture();
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();
        assert!(picker.has_git_repo());
    }

    /// Dir removal tombstones exactly the subtree: survivors, callback set,
    /// live count, search totals, idempotency, root, and overflow files.
    #[test]
    fn remove_all_files_in_dir_covers_subtree() {
        let dir = tempfile::tempdir().unwrap();
        let base_buf = crate::path_utils::canonicalize(dir.path()).unwrap();
        let base = base_buf.as_path();
        for rel in [
            "src/a/f1.rs",
            "src/a/f2.rs",
            "src/a/nested/f3.rs",
            "src/b/f4.rs",
            "root.rs",
        ] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();
        assert_eq!(picker.live_file_count(), 5);

        let search_total = |picker: &FilePicker| {
            let parser = fff_query_parser::QueryParser::default();
            let parsed = parser.parse("");
            picker
                .fuzzy_search(
                    &parsed,
                    None,
                    FuzzySearchOptions {
                        max_threads: 1,
                        current_file: None,
                        project_path: None,
                        combo_boost_score_multiplier: 0,
                        min_combo_count: 0,
                        pagination: crate::types::PaginationArgs {
                            offset: 0,
                            limit: 50,
                        },
                    },
                )
                .total_matched
        };

        let mut removed = Vec::new();
        assert_eq!(
            picker.remove_all_files_in_dirs_with_callback(
                std::iter::once(base.join("src/a").as_path()),
                |path| removed.push(path.to_path_buf()),
            ),
            3
        );
        removed.sort_unstable();
        assert_eq!(
            removed,
            vec![
                base.join("src/a/f1.rs"),
                base.join("src/a/f2.rs"),
                base.join("src/a/nested/f3.rs"),
            ]
        );
        assert_eq!(picker.live_file_count(), 2);
        assert_eq!(search_total(&picker), 2);

        // Idempotent: second removal finds nothing.
        assert_eq!(picker.remove_all_files_in_dir(base.join("src/a")), 0);
        assert_eq!(picker.live_file_count(), 2);

        // Overflow files (watcher-added) fall under the same subtree rule.
        let extra = base.join("src/c/f5.rs");
        std::fs::create_dir_all(extra.parent().unwrap()).unwrap();
        std::fs::write(&extra, b"x").unwrap();
        assert!(picker.add_new_file(&extra).is_some());
        assert_eq!(picker.live_file_count(), 3);
        assert_eq!(picker.remove_all_files_in_dir(base.join("src")), 2);
        assert_eq!(picker.live_file_count(), 1);
        assert_eq!(search_total(&picker), 1);

        // Root removal takes the rest.
        assert_eq!(picker.remove_all_files_in_dir(&base), 1);
        assert_eq!(picker.live_file_count(), 0);
        assert_eq!(search_total(&picker), 0);
    }
}
