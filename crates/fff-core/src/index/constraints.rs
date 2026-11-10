//! Constraint-based prefiltering for search queries.

use fff_query_parser::{Constraint, GitStatusFilter};
use smallvec::SmallVec;

use crate::git::is_modified_status;
use crate::simd_path::ArenaPtr;
use crate::simd_string_utils::memmem::{find_planned, plan_for, PairPlan};

const PAR_THRESHOLD: usize = 10_000;
const PAR_CHUNKS_PER_THREAD: usize = 4;

pub(crate) trait Constrainable {
    fn write_file_name(&self, arena: ArenaPtr, out: &mut String);
    fn git_status(&self) -> Option<git2::Status>;
    fn write_relative_path(&self, arena: ArenaPtr, out: &mut String);
    // Same bytes as `write_relative_path`. Overrides skip the scratch `String`.
    #[cfg(feature = "zlob")]
    fn append_relative_path(&self, arena: ArenaPtr, out: &mut Vec<u8>, scratch: &mut String) {
        self.write_relative_path(arena, scratch);
        out.extend_from_slice(scratch.as_bytes());
    }
    // Case-sensitive substring of the relative path. Default copies; overrides do not.
    fn path_contains(&self, arena: ArenaPtr, finder: &memchr::memmem::Finder<'_>) -> bool {
        let mut owned = String::new();
        self.write_relative_path(arena, &mut owned);
        finder.find(owned.as_bytes()).is_some()
    }
    // ASCII case-insensitive substring. Default copies; overrides read chunks.
    fn path_contains_folded(
        &self,
        arena: ArenaPtr,
        needle_lower: &[u8],
        plan: PairPlan,
    ) -> bool {
        let mut owned = String::new();
        self.write_relative_path(arena, &mut owned);
        find_planned(owned.as_bytes(), needle_lower, plan).is_some()
    }
    // Directory segment. `segment_lower` is ASCII-folded. Default copies.
    fn path_has_segment(&self, arena: ArenaPtr, segment_lower: &[u8]) -> bool {
        let mut owned = String::new();
        self.write_relative_path(arena, &mut owned);
        match std::str::from_utf8(segment_lower) {
            Ok(seg) => path_contains_segment(&owned, seg),
            Err(_) => false,
        }
    }
    fn is_overflow(&self) -> bool;

    // Suffix at a separator boundary. Default copies; overrides read the arena.
    fn ends_with_path_suffix(&self, arena: ArenaPtr, suffix: &str) -> bool {
        let mut owned = String::new();
        self.write_relative_path(arena, &mut owned);
        path_ends_with_suffix(&owned, suffix)
    }

    fn has_extension(&self, arena: ArenaPtr, extension: &str, scratch: &mut String) -> bool {
        self.write_file_name(arena, scratch);
        file_has_extension(scratch, extension)
    }
}

/// Stored/canonical paths use `/`; also accept `\` so a Windows user typing
/// a native separator in a query still matches.
#[inline]
fn is_path_sep(b: u8) -> bool {
    b == b'/' || b == b'\\'
}

fn bytes_match(a: u8, b: u8) -> bool {
    if is_path_sep(a) && is_path_sep(b) {
        true
    } else {
        a.eq_ignore_ascii_case(&b)
    }
}

// First index at or after `i` where either side is `/` or `\`. Lengths match.
#[inline(always)]
fn next_sep_either(hay: &[u8], needle: &[u8], i: usize) -> usize {
    let h = memchr::memchr2(b'/', b'\\', &hay[i..]).map(|p| i + p);
    let n = memchr::memchr2(b'/', b'\\', &needle[i..]).map(|p| i + p);
    match (h, n) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => hay.len(),
    }
}

// A non-separator run of 4 or more uses the wide compare. Separators stay one byte.
fn span_eq(hay: &[u8], needle: &[u8]) -> bool {
    let mut i = 0usize;
    while i < hay.len() {
        let run = next_sep_either(hay, needle, i);
        let len = run - i;
        if len >= 4 {
            if !crate::simd_path::eq_ignore_ascii_bytes(&hay[i..run], &needle[i..run]) {
                return false;
            }
        } else if (i..run).any(|j| !hay[j].eq_ignore_ascii_case(&needle[j])) {
            return false;
        }
        if run == hay.len() {
            return true;
        }
        if !bytes_match(hay[run], needle[run]) {
            return false;
        }
        i = run + 1;
    }
    true
}

#[inline]
fn path_slice_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && span_eq(a, b)
}

/// Path ends with suffix at a path-separator boundary (case-insensitive).
#[inline]
pub fn path_ends_with_suffix(path: &str, suffix: &str) -> bool {
    let path_bytes = path.as_bytes();
    let suffix_bytes = suffix.as_bytes();
    if path_bytes.len() < suffix_bytes.len() {
        return false;
    }

    let start = path.len() - suffix.len();

    // Multi-byte UTF-8 may put `start` inside a char.
    if !path.is_char_boundary(start) {
        return false;
    }

    if !path_slice_eq(&path_bytes[start..], suffix_bytes) {
        return false;
    }

    // Exact or preceded by a separator. Only the byte directly before
    // `start` matters: separators are single-byte ASCII, so a non-ASCII
    // byte there (lead or continuation) means no boundary. Scanning
    // further back skips whole non-ASCII words and false-matches.
    if start == 0 {
        return true;
    }
    is_path_sep(path_bytes[start - 1])
}

// Same rule as `path_ends_with_suffix`. One index base; each later slot is the next entry.
pub(crate) fn chunked_ends_with_path_suffix(
    path: &crate::simd_path::ChunkedString,
    arena: ArenaPtr,
    suffix: &str,
) -> bool {
    let suffix = suffix.as_bytes();
    let total = path.byte_len as usize;
    if total < suffix.len() {
        return false;
    }
    let start = total - suffix.len();
    let simd = crate::simd_path::SIMD_CHUNK_BYTES;
    if suffix.is_empty() {
        if start == 0 {
            return true;
        }
        let prev = start - 1;
        let chunk = crate::simd_path::ChunkedString::chunk_at(
            arena,
            unsafe { *path.slot_index(arena, prev / simd) },
        );
        return is_path_sep(chunk[prev % simd]);
    }

    let slot = start / simd;
    let within = start % simd;
    let index_start = path.slot_index(arena, slot);
    let head = crate::simd_path::ChunkedString::chunk_at(arena, unsafe { *index_start });
    if head[within] & 0xC0 == 0x80 {
        return false;
    }
    let mut index = index_start;
    let mut chunk = head;
    let mut at = within;
    let mut npos = 0usize;
    while npos < suffix.len() {
        let n = (simd - at).min(suffix.len() - npos);
        if !span_eq(&chunk[at..at + n], &suffix[npos..npos + n]) {
            return false;
        }
        npos += n;
        if npos < suffix.len() {
            at = 0;
            index = unsafe { index.add(1) };
            chunk = crate::simd_path::ChunkedString::chunk_at(arena, unsafe { *index });
        }
    }
    if start == 0 {
        return true;
    }
    let prev = if within == 0 {
        crate::simd_path::ChunkedString::chunk_at(arena, unsafe { *index_start.sub(1) })[simd - 1]
    } else {
        head[within - 1]
    };
    is_path_sep(prev)
}

#[inline]
pub fn file_has_extension(file_name: &str, ext: &str) -> bool {
    let name_bytes = file_name.as_bytes();
    let ext_bytes = ext.as_bytes();
    if name_bytes.len() <= ext_bytes.len() + 1 {
        return false;
    }
    let start = name_bytes.len() - ext_bytes.len() - 1;
    if start > 0 && !file_name.is_char_boundary(start) {
        return false;
    }
    name_bytes.get(start) == Some(&b'.')
        && crate::simd_path::eq_ignore_ascii_bytes(&name_bytes[start + 1..], ext_bytes)
}

/// Matches multi-segment queries like `libswscale/aarch64`.
#[inline]
pub fn path_contains_segment(path: &str, segment: &str) -> bool {
    let path_bytes = path.as_bytes();
    let segment_bytes = segment.as_bytes();
    let segment_len = segment_bytes.len();

    if path_bytes.len() > segment_len
        && is_path_sep(path_bytes[segment_len])
        && path.is_char_boundary(segment_len)
        && path_slice_eq(&path_bytes[..segment_len], segment_bytes)
    {
        return true;
    }

    if path_bytes.len() < segment_len + 2 {
        return false;
    }

    for i in 0..path_bytes.len().saturating_sub(segment_len + 1) {
        if is_path_sep(path_bytes[i]) {
            let start = i + 1;
            let end = start + segment_len;
            if end < path_bytes.len()
                && is_path_sep(path_bytes[end])
                && path.is_char_boundary(start)
                && path.is_char_boundary(end)
                && path_slice_eq(&path_bytes[start..end], segment_bytes)
            {
                return true;
            }
        }
    }
    false
}

/// Returns `None` if no constraints are present, `Some(filtered)` otherwise.
///
/// Constraint semantics:
/// - All `Extension` constraints OR together (file matches if ANY extension hits).
///   They're split out up front so the per-item loop reads the OR predicate as a
///   single short-circuit check, not as N AND-merged sub-constraints.
/// - Every other constraint kind ANDs (file matches only if ALL hold). They're
///   evaluated in order with short-circuit on first failure.
pub(crate) fn apply_constraints<'a, T: Constrainable + Sync>(
    items: &'a [T],
    constraints: &[Constraint<'_>],
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
) -> Option<Vec<&'a T>> {
    if constraints.is_empty() {
        return None;
    }
    let plan = ConstraintPlan::build(constraints, items, base_arena, overflow_arena);
    Some(plan.run(items, base_arena, overflow_arena))
}

#[cfg(feature = "zlob")]
pub(crate) type GlobPattern = zlob::ZlobPattern;
#[cfg(all(not(feature = "zlob"), feature = "ripgrep"))]
pub(crate) type GlobPattern = globset::GlobMatcher;

/// `PERIOD` on top of `RECOMMENDED`: we filter an index that already contains
/// dotfiles, so `**/x` must cross `.config`/`.pi` components. Without it zlob
/// applies fnmatch's leading-dot rule and diverges from the globset backend.
#[cfg(feature = "zlob")]
const GLOB_FLAGS: zlob::ZlobFlags = zlob::ZlobFlags::RECOMMENDED.union(zlob::ZlobFlags::PERIOD);

struct InlineGlob<'q> {
    pattern: Option<GlobPattern>,
    needle: Option<&'q str>,
}

/// How `Constraint::Glob` is evaluated for each item.
enum GlobStrategy<'q> {
    None,
    // Zlob pure-glob: one batch over a single copy of each path.
    Prepass(Vec<Vec<bool>>),
    // Compiled patterns. `needle` is a required literal run, if any.
    Inline(Vec<InlineGlob<'q>>),
    // `*literal*`: `*` matches any byte, so this is a substring test.
    Contains(SmallVec<[&'q str; 4]>),
}

/// Bundles preprocessed constraints for the per-item evaluator.
pub(crate) struct ConstraintPlan<'q, 'c> {
    /// OR semantics — file passes if ANY extension matches. Empty = no ext filter.
    extensions: SmallVec<[&'q str; 8]>,
    /// AND semantics — file passes only if ALL match.
    rest: SmallVec<[&'c Constraint<'q>; 8]>,
    glob: GlobStrategy<'q>,
    /// FilePath and inline globs copy one path. Text and segments do not.
    needs_path: bool,
}

pub(crate) struct ConstraintsBuffers {
    fname: String,
    path: String,
    path_ready: bool,
    // Query text and segment are stable for the scan. Fold each once per worker.
    text_ptr: usize,
    text_len: usize,
    text_lower: SmallVec<[u8; 64]>,
    text_plan: PairPlan,
    seg_ptr: usize,
    seg_len: usize,
    seg_lower: SmallVec<[u8; 64]>,
    // One finder per `*literal*` glob, rebuilt when the literal changes.
    glob_finders: Vec<memchr::memmem::Finder<'static>>,
    // Parallel to inline globs. None = no required literal.
    inline_finders: Vec<Option<memchr::memmem::Finder<'static>>>,
}

impl ConstraintsBuffers {
    pub(crate) fn new() -> Self {
        Self {
            fname: String::with_capacity(64),
            path: String::with_capacity(64),
            path_ready: false,
            text_ptr: 0,
            text_len: usize::MAX,
            text_lower: SmallVec::new(),
            text_plan: plan_for(b""),
            seg_ptr: 0,
            seg_len: usize::MAX,
            seg_lower: SmallVec::new(),
            glob_finders: Vec::new(),
            inline_finders: Vec::new(),
        }
    }

    fn cache_inline(&mut self, globs: &[InlineGlob<'_>]) {
        let same = self.inline_finders.len() == globs.len()
            && self.inline_finders.iter().zip(globs).all(|(slot, glob)| {
                match (slot, glob.needle) {
                    (None, None) => true,
                    (Some(finder), Some(needle)) => finder.needle() == needle.as_bytes(),
                    _ => false,
                }
            });
        if same {
            return;
        }
        self.inline_finders.clear();
        self.inline_finders.extend(globs.iter().map(|glob| {
            glob.needle
                .map(|needle| memchr::memmem::Finder::new(needle).into_owned())
        }));
    }

    fn cache_globs(&mut self, needles: &[&str]) {
        let same = self.glob_finders.len() == needles.len()
            && self
                .glob_finders
                .iter()
                .zip(needles)
                .all(|(finder, needle)| finder.needle() == needle.as_bytes());
        if same {
            return;
        }
        self.glob_finders.clear();
        self.glob_finders
            .extend(needles.iter().map(|n| memchr::memmem::Finder::new(*n).into_owned()));
    }

    fn cache_text(&mut self, text: &str) {
        let ptr = text.as_ptr() as usize;
        if ptr == self.text_ptr && text.len() == self.text_len {
            return;
        }
        self.text_ptr = ptr;
        self.text_len = text.len();
        self.text_lower.clear();
        self.text_lower.extend_from_slice(text.as_bytes());
        self.text_lower.make_ascii_lowercase();
        self.text_plan = plan_for(&self.text_lower);
    }

    fn cache_segment(&mut self, segment: &str) {
        let ptr = segment.as_ptr() as usize;
        if ptr == self.seg_ptr && segment.len() == self.seg_len {
            return;
        }
        self.seg_ptr = ptr;
        self.seg_len = segment.len();
        self.seg_lower.clear();
        self.seg_lower.extend_from_slice(segment.as_bytes());
        self.seg_lower.make_ascii_lowercase();
    }
}

impl<'q, 'c> ConstraintPlan<'q, 'c> {
    pub(crate) fn build<T: Constrainable>(
        constraints: &'c [Constraint<'q>],
        items: &[T],
        base_arena: ArenaPtr,
        overflow_arena: ArenaPtr,
    ) -> Self {
        let mut extensions = SmallVec::new();
        let mut rest: SmallVec<[&'c Constraint<'q>; 8]> = SmallVec::new();
        for c in constraints {
            match c {
                Constraint::Extension(ext) => extensions.push(*ext),
                _ => rest.push(c),
            }
        }
        let has_pre_filter = !extensions.is_empty() || rest.iter().any(|&c| !is_glob_node(c));
        let glob = build_glob_strategy(&rest, has_pre_filter, items, base_arena, overflow_arena);
        // Inline globs and file-path suffixes copy one path. Text, segments,
        // and `*literal*` globs read arena chunks.
        let inline_globs = matches!(glob, GlobStrategy::Inline(_));
        let needs_path = rest.iter().any(|c| touches_item_path(c, inline_globs));

        Self {
            extensions,
            rest,
            glob,
            needs_path,
        }
    }

    fn run<'a, T: Constrainable + Sync>(
        &self,
        items: &'a [T],
        base_arean: ArenaPtr,
        overflow_arena: ArenaPtr,
    ) -> Vec<&'a T> {
        if items.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            // Fixed spans keep the collect indexed; the unindexed fallback
            // pays per-task buffers plus a serial merge on the caller.
            let threads = rayon::current_num_threads().max(1);
            let span = items.len().div_ceil(threads * PAR_CHUNKS_PER_THREAD).max(1);
            let n_chunks = items.len().div_ceil(span);
            let parts: Vec<Vec<&'a T>> = (0..n_chunks)
                .into_par_iter()
                .map_init(ConstraintsBuffers::new, |scratch, c| {
                    let lo = c * span;
                    let hi = ((c + 1) * span).min(items.len());
                    (lo..hi)
                        .filter_map(|i| {
                            self.matches(&items[i], i, base_arean, overflow_arena, scratch)
                                .then_some(&items[i])
                        })
                        .collect()
                })
                .collect();
            let mut out = Vec::with_capacity(parts.iter().map(Vec::len).sum());
            for mut part in parts {
                out.append(&mut part);
            }
            out
        } else {
            let mut scratch = ConstraintsBuffers::new();
            items
                .iter()
                .enumerate()
                .filter_map(|(i, item)| {
                    self.matches(item, i, base_arean, overflow_arena, &mut scratch)
                        .then_some(item)
                })
                .collect()
        }
    }

    #[inline]
    pub(crate) fn matches<T: Constrainable>(
        &self,
        item: &T,
        index: usize,
        base_arena: ArenaPtr,
        overflow_arena: ArenaPtr,
        scratch: &mut ConstraintsBuffers,
    ) -> bool {
        let arena = if item.is_overflow() {
            overflow_arena
        } else {
            base_arena
        };

        if self.needs_path {
            scratch.path_ready = false;
        }
        if let GlobStrategy::Contains(needles) = &self.glob {
            scratch.cache_globs(needles);
        } else if let GlobStrategy::Inline(globs) = &self.glob {
            scratch.cache_inline(globs);
        }
        if !self.passes_extensions(item, arena, scratch) {
            return false;
        }

        let mut glob_idx = 0;
        self.rest.iter().all(|c| {
            evaluate(
                item,
                index,
                c,
                &self.glob,
                &mut glob_idx,
                false,
                arena,
                scratch,
            )
        })
    }

    #[inline]
    fn passes_extensions<T: Constrainable>(
        &self,
        item: &T,
        arena: ArenaPtr,
        scratch: &mut ConstraintsBuffers,
    ) -> bool {
        if self.extensions.is_empty() {
            return true;
        }
        self.extensions
            .iter()
            .any(|ext| item.has_extension(arena, ext, &mut scratch.fname))
    }
}

#[inline]
fn item_path<'a, T: Constrainable>(
    item: &T,
    arena: ArenaPtr,
    scratch: &'a mut ConstraintsBuffers,
) -> &'a str {
    if !scratch.path_ready {
        item.write_relative_path(arena, &mut scratch.path);
        scratch.path_ready = true;
    }
    &scratch.path
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn evaluate<T: Constrainable>(
    item: &T,
    index: usize,
    constraint: &Constraint<'_>,
    glob: &GlobStrategy<'_>,
    glob_idx: &mut usize,
    negate: bool,
    arena: ArenaPtr,
    scratch: &mut ConstraintsBuffers,
) -> bool {
    let raw = match constraint {
        Constraint::Glob(_) => {
            let m = match glob {
                GlobStrategy::None => true,
                GlobStrategy::Prepass(masks) => masks
                    .get(*glob_idx)
                    .and_then(|mask| mask.get(index).copied())
                    .unwrap_or(false),
                GlobStrategy::Inline(globs) => {
                    // Miss cannot match. A hit is not a match.
                    let miss = scratch
                        .inline_finders
                        .get(*glob_idx)
                        .and_then(|slot| slot.as_ref())
                        .is_some_and(|finder| !item.path_contains(arena, finder));
                    if miss {
                        false
                    } else {
                        let path = item_path(item, arena, scratch);
                        globs
                            .get(*glob_idx)
                            .and_then(|glob| glob.pattern.as_ref())
                            .map(|pattern| compiled_matches(pattern, path))
                            .unwrap_or(false)
                    }
                }
                GlobStrategy::Contains(_) => scratch
                    .glob_finders
                    .get(*glob_idx)
                    .is_some_and(|finder| item.path_contains(arena, finder)),
            };
            *glob_idx += 1;
            m
        }
        // Reachable only via `Not(Extension(_))` — bare extensions are split out
        // up front and handled in `passes_extensions`.
        Constraint::Extension(ext) => item.has_extension(arena, ext, &mut scratch.fname),
        Constraint::PathSegment(segment) => {
            scratch.cache_segment(segment);
            item.path_has_segment(arena, &scratch.seg_lower)
        }
        Constraint::FilePath(suffix) => item.ends_with_path_suffix(arena, suffix),
        Constraint::Text(text) => {
            // `!text` still visits every path. Chunks stay in the arena.
            scratch.cache_text(text);
            let plan = scratch.text_plan;
            item.path_contains_folded(arena, &scratch.text_lower, plan)
        }
        Constraint::GitStatus(filter) => matches_git_status(item.git_status(), filter),
        Constraint::Not(inner) => {
            return evaluate(item, index, inner, glob, glob_idx, !negate, arena, scratch);
        }
        // Pass-throughs — handled at higher levels.
        Constraint::Parts(_) | Constraint::Exclude(_) | Constraint::FileType(_) => true,
    };
    if negate { !raw } else { raw }
}

#[inline]
fn matches_git_status(status: Option<git2::Status>, filter: &GitStatusFilter) -> bool {
    match (status, filter) {
        (Some(s), GitStatusFilter::Modified) => is_modified_status(s),
        (Some(s), GitStatusFilter::Untracked) => s.contains(git2::Status::WT_NEW),
        (Some(s), GitStatusFilter::Staged) => s.intersects(
            git2::Status::INDEX_NEW
                | git2::Status::INDEX_MODIFIED
                | git2::Status::INDEX_DELETED
                | git2::Status::INDEX_RENAMED
                | git2::Status::INDEX_TYPECHANGE,
        ),
        (Some(s), GitStatusFilter::Unmodified) => s.is_empty(),
        (None, GitStatusFilter::Unmodified) => true,
        (None, _) => false,
    }
}

#[inline]
#[cfg(feature = "zlob")]
pub(crate) fn compiled_matches(p: &GlobPattern, path: &str) -> bool {
    p.matches_default(path)
}

#[inline]
#[cfg(all(not(feature = "zlob"), feature = "ripgrep"))]
pub(crate) fn compiled_matches(p: &GlobPattern, path: &str) -> bool {
    p.is_match(path)
}

/// Append indices (into `rels`) of paths matching `p`, in input order.
/// zlob backend: ONE FFI call for the whole batch.
#[cfg(feature = "zlob")]
pub(crate) fn glob_matches_into(p: &GlobPattern, rels: &[&str], out: &mut Vec<usize>) {
    match p.match_indices(rels, p.flags()) {
        Ok(ix) => out.extend_from_slice(ix.as_slice()),
        Err(e) => {
            tracing::warn!(?e, "zlob batch match failed, falling back to per-path");
            out.extend((0..rels.len()).filter(|&i| p.matches_default(rels[i])));
        }
    }
}

#[cfg(all(not(feature = "zlob"), feature = "ripgrep"))]
pub(crate) fn glob_matches_into(p: &GlobPattern, rels: &[&str], out: &mut Vec<usize>) {
    out.extend((0..rels.len()).filter(|&i| p.is_match(rels[i])));
}

// Zlob batches a pure glob in one call. Other backends match per path, so
// materializing every path first only adds a copy.
fn build_glob_strategy<'q, T: Constrainable>(
    rest: &[&Constraint<'q>],
    has_pre_filter: bool,
    items: &[T],
    arena: ArenaPtr,
    overflow_arena: ArenaPtr,
) -> GlobStrategy<'q> {
    if !contains_glob(rest) {
        return GlobStrategy::None;
    }
    // `*literal*` is a substring on both glob backends, so it never copies paths.
    if let Some(needles) = contains_literals(rest) {
        return GlobStrategy::Contains(needles);
    }
    #[cfg(feature = "zlob")]
    if !has_pre_filter {
        let buf = PathBuffer::collect(items, arena, overflow_arena);
        let path_refs = buf.as_strs();
        return GlobStrategy::Prepass(precompute_masks(rest, &path_refs));
    }
    #[cfg(not(feature = "zlob"))]
    let _ = (has_pre_filter, items, arena, overflow_arena);
    GlobStrategy::Inline(compile_globs(rest))
}

fn touches_item_path(c: &Constraint<'_>, inline_globs: bool) -> bool {
    match c {
        Constraint::FilePath(_) => false,
        Constraint::Glob(_) => inline_globs,
        Constraint::Not(inner) => touches_item_path(inner, inline_globs),
        _ => false,
    }
}

/// `Glob` or `Not(Glob)`. Other constraints can reject a file before a glob runs.
fn is_glob_node(c: &Constraint<'_>) -> bool {
    match c {
        Constraint::Glob(_) => true,
        Constraint::Not(inner) => is_glob_node(inner),
        _ => false,
    }
}

fn contains_glob(rest: &[&Constraint<'_>]) -> bool {
    rest.iter().any(|c| is_glob_node(c))
}

// Leading and trailing `*` only, with a metacharacter-free literal between them.
fn star_literal(pattern: &str) -> Option<&str> {
    let b = pattern.as_bytes();
    if b.len() < 3 || b[0] != b'*' || b[b.len() - 1] != b'*' {
        return None;
    }
    let inner = &pattern[1..b.len() - 1];
    if inner.bytes().any(|c| matches!(c, b'*' | b'?' | b'[' | b'\\' | b'{' | b'}')) {
        return None;
    }
    Some(inner)
}

fn contains_literals<'q>(rest: &[&Constraint<'q>]) -> Option<SmallVec<[&'q str; 4]>> {
    let mut needles = SmallVec::new();
    for c in rest {
        let mut ok = true;
        walk_globs(c, &mut |pattern| match star_literal(pattern) {
            Some(lit) => needles.push(lit),
            None => ok = false,
        });
        if !ok {
            return None;
        }
    }
    if needles.is_empty() { None } else { Some(needles) }
}

// One contiguous copy of each relative path for the zlob batch.
// Stored paths stay `/`-canonical; the matcher sees those bytes.
#[cfg(feature = "zlob")]
struct PathBuffer {
    bytes: Vec<u8>,
    offsets: Vec<(usize, usize)>,
}

#[cfg(feature = "zlob")]
impl PathBuffer {
    fn collect<T: Constrainable>(items: &[T], arena: ArenaPtr, overflow_arena: ArenaPtr) -> Self {
        let mut bytes = Vec::<u8>::new();
        let mut offsets = Vec::with_capacity(items.len());
        let mut tmp = String::with_capacity(64);
        for item in items {
            let item_arena = if item.is_overflow() {
                overflow_arena
            } else {
                arena
            };
            let start = bytes.len();
            item.append_relative_path(item_arena, &mut bytes, &mut tmp);
            offsets.push((start, bytes.len() - start));
        }
        Self { bytes, offsets }
    }

    fn as_strs(&self) -> Vec<&str> {
        self.offsets
            .iter()
            .map(|&(off, len)| unsafe {
                std::str::from_utf8_unchecked(&self.bytes[off..off + len])
            })
            .collect()
    }
}

#[cfg(feature = "zlob")]
fn precompute_masks(rest: &[&Constraint<'_>], paths: &[&str]) -> Vec<Vec<bool>> {
    let mut out = Vec::new();
    for c in rest {
        walk_globs(c, &mut |pattern| {
            out.push(match_glob_pattern(pattern, paths))
        });
    }
    out
}

fn compile_globs<'q>(rest: &[&Constraint<'q>]) -> Vec<InlineGlob<'q>> {
    let mut out = Vec::new();
    for c in rest {
        walk_globs(c, &mut |pattern| {
            out.push(InlineGlob {
                pattern: compile_one(pattern),
                needle: required_literal(pattern),
            });
        });
    }
    out
}

// Longest case-sensitive literal every match contains (`[]` / `{}` excluded).
// Shorter than 3 keeps the path copy.
fn required_literal(pattern: &str) -> Option<&str> {
    fn consider(
        pattern: &str,
        escapes: bool,
        best_at: &mut usize,
        best_len: &mut usize,
        mut start: usize,
        end: usize,
    ) {
        let bytes = pattern.as_bytes();
        // `**/` can match an empty prefix, so that slash is not required.
        if start < end
            && bytes[start] == b'/'
            && start >= 2
            && bytes[start - 1] == b'*'
            && bytes[start - 2] == b'*'
            && !escaped_meta(bytes, escapes, start - 2)
        {
            start += 1;
        }
        if end <= start {
            return;
        }
        let len = end - start;
        if len >= 3 && len > *best_len {
            *best_at = start;
            *best_len = len;
        }
    }

    let bytes = pattern.as_bytes();
    let escapes = cfg!(feature = "zlob") || !cfg!(windows);
    let mut best_at = 0usize;
    let mut best_len = 0usize;
    let mut run: Option<usize> = None;
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' {
            if depth == 0
                && let Some(start) = run.take()
            {
                consider(pattern, escapes, &mut best_at, &mut best_len, start, i);
            }
            i += 1;
            if escapes && i < bytes.len() && is_glob_meta(bytes[i]) {
                i += 1;
            }
            continue;
        }
        if c == b'[' {
            if depth == 0
                && let Some(start) = run.take()
            {
                consider(pattern, escapes, &mut best_at, &mut best_len, start, i);
            }
            i = skip_class(bytes, i);
            continue;
        }
        if c == b'{' {
            if depth == 0
                && let Some(start) = run.take()
            {
                consider(pattern, escapes, &mut best_at, &mut best_len, start, i);
            }
            depth += 1;
            i += 1;
            continue;
        }
        if c == b'}' {
            if depth > 0 {
                depth -= 1;
                run = None;
            } else if let Some(start) = run.take() {
                consider(pattern, escapes, &mut best_at, &mut best_len, start, i);
            }
            i += 1;
            continue;
        }
        if cfg!(feature = "zlob") && c == b'~' && depth == 0 {
            if let Some(start) = run.take() {
                consider(pattern, escapes, &mut best_at, &mut best_len, start, i);
            }
            let expand = i == 0 || bytes[i - 1] == b'/';
            i += 1;
            if expand {
                while i < bytes.len() && bytes[i] != b'/' && !is_glob_meta(bytes[i]) {
                    i += 1;
                }
            }
            continue;
        }
        if depth > 0 {
            run = None;
            i += 1;
            continue;
        }
        if matches!(c, b'*' | b'?' | b']') {
            if let Some(start) = run.take() {
                consider(pattern, escapes, &mut best_at, &mut best_len, start, i);
            }
            i += 1;
            continue;
        }
        if run.is_none() {
            run = Some(i);
        }
        i += 1;
    }
    if depth == 0
        && let Some(start) = run
    {
        consider(pattern, escapes, &mut best_at, &mut best_len, start, bytes.len());
    }
    if best_len >= 3 {
        Some(&pattern[best_at..best_at + best_len])
    } else {
        None
    }
}

fn is_glob_meta(b: u8) -> bool {
    matches!(b, b'*' | b'?' | b'[' | b']' | b'{' | b'}' | b'\\')
}

fn escaped_meta(bytes: &[u8], escapes: bool, at: usize) -> bool {
    if !escapes {
        return false;
    }
    let mut n = 0usize;
    let mut j = at;
    while j > 0 && bytes[j - 1] == b'\\' {
        n += 1;
        j -= 1;
    }
    n % 2 == 1
}

fn skip_class(bytes: &[u8], open: usize) -> usize {
    let mut i = open + 1;
    if i < bytes.len() && matches!(bytes[i], b'!' | b'^') {
        i += 1;
    }
    let mut first = true;
    while i < bytes.len() {
        if cfg!(feature = "zlob") && bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
            first = false;
            continue;
        }
        if bytes[i] == b']' && !first {
            return i + 1;
        }
        i += 1;
        first = false;
    }
    bytes.len()
}

/// Visit every Glob (including ones nested under Not) in constraint walk order.
/// Order matters: `glob_idx` in the per-item evaluator increments by one per Glob node.
fn walk_globs<'a, F: FnMut(&'a str)>(c: &Constraint<'a>, f: &mut F) {
    match c {
        Constraint::Glob(p) => f(p),
        Constraint::Not(inner) => walk_globs(inner, f),
        _ => {}
    }
}

#[cfg(feature = "zlob")]
pub(crate) fn compile_one(pattern: &str) -> Option<GlobPattern> {
    zlob::ZlobPattern::compile(pattern, GLOB_FLAGS).ok()
}

#[cfg(all(not(feature = "zlob"), feature = "ripgrep"))]
pub(crate) fn compile_one(pattern: &str) -> Option<GlobPattern> {
    globset::Glob::new(pattern)
        .ok()
        .map(|g| g.compile_matcher())
}

// Positive-match bits. A missing column stays false, which `Not` flips.
#[cfg(feature = "zlob")]
fn match_glob_pattern(pattern: &str, paths: &[&str]) -> Vec<bool> {
    let mut mask = vec![false; paths.len()];
    let Ok(hits) = zlob::zlob_match_paths_indices(pattern, paths, GLOB_FLAGS) else {
        return mask;
    };
    for i in hits.to_iter() {
        if i < mask.len() {
            mask[i] = true;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestItem {
        relative_path: &'static str,
        file_name: &'static str,
    }

    impl Constrainable for TestItem {
        fn write_file_name(&self, _arena: ArenaPtr, out: &mut String) {
            out.clear();
            out.push_str(self.file_name);
        }

        fn write_relative_path(&self, _arena: ArenaPtr, out: &mut String) {
            out.clear();
            out.push_str(self.relative_path);
        }

        fn git_status(&self) -> Option<git2::Status> {
            None
        }

        fn is_overflow(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_file_has_extension() {
        assert!(file_has_extension("file.rs", "rs"));
        assert!(file_has_extension("file.RS", "rs")); // case-insensitive
        assert!(file_has_extension("file.test.rs", "rs"));
        assert!(file_has_extension("a.rs", "rs"));

        assert!(!file_has_extension("file.tsx", "rs"));
        assert!(!file_has_extension("rs", "rs")); // too short
        assert!(!file_has_extension(".rs", "rs")); // just extension
        assert!(!file_has_extension("file.rsx", "rs")); // different extension
        assert!(!file_has_extension("filers", "rs")); // no dot
    }

    #[test]
    fn chunked_path_suffix_matches_the_string_rule() {
        let paths = [
            "src/lib.rs",
            "src/lib.rsx",
            "SRC/LIB.RS",
            "src\\lib.rs",
            "lib.rs",
            "very/deeply/nested/directory/structure/with/many/levels/lib.rs",
        ];
        let files: Vec<crate::types::FileItem> = paths
            .iter()
            .map(|_| crate::types::FileItem::new_raw(0, 0, 0, None, false))
            .collect();
        let owned: Vec<String> = paths.iter().map(|p| (*p).to_string()).collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&owned, &files);
        let arena = store.as_arena_ptr();
        for (path, chunked) in paths.iter().zip(&strings) {
            for suffix in ["lib.rs", "src/lib.rs", "LIB.RS", "src\\lib.rs", "levels/lib.rs"] {
                assert_eq!(
                    chunked_ends_with_path_suffix(chunked, arena, suffix),
                    path_ends_with_suffix(path, suffix),
                    "{path} / {suffix}"
                );
            }
        }
    }

    #[test]
    fn test_path_contains_segment() {
        // Segment at start
        assert!(path_contains_segment("src/lib.rs", "src"));
        assert!(path_contains_segment("SRC/lib.rs", "src")); // case-insensitive

        // Segment in middle
        assert!(path_contains_segment("app/src/lib.rs", "src"));
        assert!(path_contains_segment("app/SRC/lib.rs", "src"));

        // Multiple levels
        assert!(path_contains_segment("core/workflow/src/main.rs", "src"));
        assert!(path_contains_segment(
            "core/workflow/src/main.rs",
            "workflow"
        ));
        assert!(path_contains_segment("core/workflow/src/main.rs", "core"));

        // Should not match partial segments
        assert!(!path_contains_segment("source/lib.rs", "src"));
        assert!(!path_contains_segment("mysrc/lib.rs", "src"));

        // Should not match filename
        assert!(!path_contains_segment("lib/src", "src"));

        // Multi-segment constraints
        assert!(path_contains_segment(
            "libswscale/aarch64/input.S",
            "libswscale/aarch64"
        ));
        assert!(path_contains_segment(
            "foo/libswscale/aarch64/input.S",
            "libswscale/aarch64"
        ));
        assert!(path_contains_segment(
            "foo/LibSwscale/AArch64/input.S",
            "libswscale/aarch64"
        )); // case-insensitive
        assert!(!path_contains_segment(
            "xlibswscale/aarch64/input.S",
            "libswscale/aarch64"
        )); // partial match at start
        assert!(!path_contains_segment(
            "foo/libswscale/aarch64x/input.S",
            "libswscale/aarch64"
        )); // partial match at end
        assert!(path_contains_segment(
            "crates/fff-core/src/grep.rs",
            "fff-core/src"
        ));

        // Edge cases
        assert!(!path_contains_segment("", "src"));
        assert!(!path_contains_segment("src", "src")); // no trailing slash
    }

    #[cfg(windows)]
    #[test]
    fn test_path_contains_segment_accepts_backslash() {
        assert!(path_contains_segment("src\\lib.rs", "src"));
        assert!(path_contains_segment(
            "app\\modules\\src\\services\\x.lua",
            "src"
        ));
        assert!(path_contains_segment("app\\SRC\\x.lua", "src"));

        assert!(path_contains_segment(
            "foo\\libswscale\\aarch64\\input.S",
            "libswscale/aarch64"
        ));
        assert!(path_contains_segment(
            "crates\\fff-core\\src\\grep.rs",
            "fff-core/src"
        ));

        assert!(!path_contains_segment("mysrc\\lib.rs", "src"));
        assert!(!path_contains_segment(
            "xlibswscale\\aarch64\\in.S",
            "libswscale/aarch64"
        ));
    }

    #[test]
    fn test_path_ends_with_suffix() {
        // Exact match
        assert!(path_ends_with_suffix(
            "libswscale/input.c",
            "libswscale/input.c"
        ));

        // Suffix match at / boundary
        assert!(path_ends_with_suffix(
            "foo/libswscale/input.c",
            "libswscale/input.c"
        ));

        // Deep nesting
        assert!(path_ends_with_suffix(
            "a/b/c/libswscale/input.c",
            "libswscale/input.c"
        ));

        // No boundary — partial directory name
        assert!(!path_ends_with_suffix(
            "xlibswscale/input.c",
            "libswscale/input.c"
        ));

        // Case insensitive
        assert!(path_ends_with_suffix(
            "foo/LibSwscale/Input.C",
            "libswscale/input.c"
        ));

        // Single file name
        assert!(path_ends_with_suffix("input.c", "input.c"));
        assert!(!path_ends_with_suffix("xinput.c", "input.c"));

        // Suffix longer than path
        assert!(!path_ends_with_suffix("input.c", "foo/input.c"));

        // Simple path
        assert!(path_ends_with_suffix("src/main.rs", "src/main.rs"));
        assert!(path_ends_with_suffix("crates/src/main.rs", "src/main.rs"));

        // Non-ASCII word directly before the suffix is not a boundary,
        // even though scanning further back would find a separator.
        assert!(!path_ends_with_suffix(
            "abc/приветfoo/bar.txt",
            "foo/bar.txt"
        ));
        assert!(path_ends_with_suffix(
            "abc/привет/foo/bar.txt",
            "foo/bar.txt"
        ));
        assert!(path_ends_with_suffix("привет/bar.txt", "bar.txt"));
    }

    #[cfg(windows)]
    #[test]
    fn test_path_ends_with_suffix_accepts_backslash() {
        assert!(path_ends_with_suffix(
            "app\\modules\\src\\services\\handler.lua",
            "services/handler.lua"
        ));
        assert!(path_ends_with_suffix(
            "foo\\libswscale\\input.c",
            "libswscale/input.c"
        ));
        assert!(!path_ends_with_suffix(
            "xlibswscale\\input.c",
            "libswscale/input.c"
        ));
    }

    #[test]
    fn test_path_ends_with_suffix_does_not_panic_on_unicode_suffix() {
        assert!(!path_ends_with_suffix("유니코드_파일_테스트.csv", "트.c"));
        assert!(path_ends_with_suffix(
            "data/유니코드_파일_테스트.csv",
            "유니코드_파일_테스트.csv"
        ));
    }

    #[test]
    fn test_path_ends_with_suffix_unicode_apostrophe_mismatch() {
        assert!(!path_ends_with_suffix(
            "dir/\u{2019}bar/file.txt",
            "'bar/file.txt"
        ));
    }

    #[test]
    fn test_path_ends_with_suffix_unicode_space_mismatch() {
        assert!(!path_ends_with_suffix(
            "dir/\u{202f}am/file.txt",
            " am/file.txt"
        ));
    }

    #[test]
    fn test_path_contains_segment_does_not_panic_on_unicode_segment() {
        assert!(!path_contains_segment("문서/notes.txt", "문x"));
        assert!(path_contains_segment("프로젝트/문서/notes.txt", "문서"));
    }

    #[test]
    fn test_path_contains_segment_unicode_no_panic() {
        assert!(!path_contains_segment(
            "Library/Cloud/Project\u{2019}s Folder/books.ttl",
            "Project's Folder"
        ));
    }

    #[test]
    fn test_file_has_extension_unicode_no_panic() {
        assert!(!file_has_extension("cat\u{00e9}.rs", "s"));
    }

    #[test]
    fn test_file_has_extension_unicode_filename() {
        assert!(file_has_extension("운영-가이드.md", "md"));
        assert!(file_has_extension("테스트.csv", "csv"));
        assert!(!file_has_extension("테스트.csv", "md"));
    }

    #[test]
    fn test_apply_constraints_file_path_with_unicode_suffix() {
        let arena_ptr = ArenaPtr::null();

        let item = TestItem {
            relative_path: "data/유니코드_파일_테스트.csv",
            file_name: "유니코드_파일_테스트.csv",
        };

        let exact = [Constraint::FilePath("유니코드_파일_테스트.csv")];
        let mismatch = [Constraint::FilePath("트.c")];

        let exact_items = [item.clone()];
        let exact_matches = apply_constraints(&exact_items, &exact, arena_ptr, arena_ptr)
            .expect("constraints applied");
        assert_eq!(exact_matches.len(), 1);

        let mismatch_items = [item];
        let mismatch_matches = apply_constraints(&mismatch_items, &mismatch, arena_ptr, arena_ptr)
            .expect("constraints applied");
        assert!(mismatch_matches.is_empty());
    }

    #[test]
    fn test_unicode_path_no_panic_real_korean_cases() {
        // Real Korean paths that caused panics
        let path1 = "Downloads/(커리큘럼) hermes agent_정승현님 - 1차 커리큘럼 (강사님 작성).csv";
        let path2 = "hermes-agent-lecture-materials/세부_커리큘럼_최종.csv";
        let path3 = "projects/fastcampus-hermes-agent-curriculum/chapters/part-02-Hermes-설치-및-기본-사용/section-02-doctor로-설치-상태-검증/research/03-fix가-자동-수정하는-것과-못하는-것.md";

        // These must not panic regardless of segment/suffix used
        assert!(!path_contains_segment(path1, "작성"));
        assert!(!path_ends_with_suffix(path1, "작성.csv"));
        assert!(!path_contains_segment(path2, "최종"));
        assert!(!path_ends_with_suffix(path2, "최종.csv"));
        assert!(!path_contains_segment(path3, "수정"));
        assert!(!path_ends_with_suffix(path3, "것.md"));

        // Positive cases should still work
        assert!(path_contains_segment(
            path2,
            "hermes-agent-lecture-materials"
        ));
        assert!(path_ends_with_suffix(
            path1,
            "(커리큘럼) hermes agent_정승현님 - 1차 커리큘럼 (강사님 작성).csv"
        ));
        assert!(path_ends_with_suffix(path2, "세부_커리큘럼_최종.csv"));
    }

    #[test]
    fn test_negated_glob_excludes_matching_files() {
        let arena_ptr = ArenaPtr::null();

        let items = vec![
            TestItem {
                relative_path: "src/main.rs",
                file_name: "main.rs",
            },
            TestItem {
                relative_path: "src/lib.ts",
                file_name: "lib.ts",
            },
            TestItem {
                relative_path: "include/fff.h",
                file_name: "fff.h",
            },
        ];

        // Not(Glob("**/*.rs")) should exclude .rs files
        let constraints = vec![Constraint::Not(Box::new(Constraint::Glob("**/*.rs")))];
        let result = apply_constraints(&items, &constraints, arena_ptr, arena_ptr).unwrap();
        let paths: Vec<&str> = result.iter().map(|i| i.relative_path).collect();
        assert!(
            !paths.contains(&"src/main.rs"),
            "rs file should be excluded"
        );
        assert!(paths.contains(&"src/lib.ts"), "ts file should be included");
        assert!(
            paths.contains(&"include/fff.h"),
            "h file should be included"
        );
    }

    #[test]
    fn test_inline_glob_path_matches_prepass() {
        // Mixed (extensions + glob) takes the inline-compiled path.
        // Pure glob takes the prepass bitmap path. Both must give identical results.
        let arena_ptr = ArenaPtr::null();
        let items = vec![
            TestItem {
                relative_path: "src/main.rs",
                file_name: "main.rs",
            },
            TestItem {
                relative_path: "src/lib.ts",
                file_name: "lib.ts",
            },
            TestItem {
                relative_path: "tests/foo.rs",
                file_name: "foo.rs",
            },
            TestItem {
                relative_path: "docs/readme.md",
                file_name: "readme.md",
            },
        ];

        let mixed = vec![Constraint::Extension("rs"), Constraint::Glob("src/**")];
        let mixed_paths: Vec<&str> = apply_constraints(&items, &mixed, arena_ptr, arena_ptr)
            .unwrap()
            .iter()
            .map(|i| i.relative_path)
            .collect();
        assert_eq!(mixed_paths, vec!["src/main.rs"]);

        let pure_glob = vec![Constraint::Glob("src/**")];
        let glob_paths: Vec<&str> = apply_constraints(&items, &pure_glob, arena_ptr, arena_ptr)
            .unwrap()
            .iter()
            .map(|i| i.relative_path)
            .collect();
        assert!(glob_paths.contains(&"src/main.rs"));
        assert!(glob_paths.contains(&"src/lib.ts"));
        assert_eq!(glob_paths.len(), 2);
    }

    #[test]
    fn test_inline_negated_glob_with_extension() {
        // Mixed Not(Glob) on inline path — exercise the negate=true branch in
        // glob_matches_inline through the Not->Glob recursion.
        let arena_ptr = ArenaPtr::null();
        let items = vec![
            TestItem {
                relative_path: "src/main.rs",
                file_name: "main.rs",
            },
            TestItem {
                relative_path: "vendor/foo.rs",
                file_name: "foo.rs",
            },
            TestItem {
                relative_path: "vendor/foo.ts",
                file_name: "foo.ts",
            },
        ];

        let constraints = vec![
            Constraint::Extension("rs"),
            Constraint::Not(Box::new(Constraint::Glob("vendor/**"))),
        ];
        let paths: Vec<&str> = apply_constraints(&items, &constraints, arena_ptr, arena_ptr)
            .unwrap()
            .iter()
            .map(|i| i.relative_path)
            .collect();
        assert_eq!(paths, vec!["src/main.rs"]);
    }
}
