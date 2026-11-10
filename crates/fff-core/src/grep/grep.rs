use super::prefilter::{chunk_plan, prefilter_with_filepath_retry_indexed};
use super::regex::{RegexMatcher, RegexSink, build_regex};
use super::sink::{
    SinkState, debug_assert_newline_terminator, lossy_owned, strip_line_terminators,
    truncate_display_bytes,
};
use super::types::{GrepMatch, GrepMode, GrepResult, GrepSearchOptions, PageFillGate};
use crate::index::{
    BigramFilter, BigramOverlay, bigram_boundary, fuzzy_candidates, literal_candidates,
    regex_candidates,
};
use crate::parallelism::search_pool_threads;
use crate::simd_string_utils::memmem;
use crate::types::{ContentCacheBudget, FileItem, MmapSlot};
use fff_grep::{
    Searcher, SearcherBuilder, Sink, SinkMatch,
    matcher::{Match, Matcher, NoError},
};
use fff_query_parser::{FFFQuery, GrepConfig, QueryParser};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tracing::Level;

#[allow(clippy::large_enum_variant)]
pub(super) enum NeedleFinder<'a> {
    CaseSensitive(memchr::memmem::Finder<'a>),
    /// Pre-lowered needle plus the rare-pair plan built once per search.
    CaseInsensitive {
        needle: &'a [u8],
        plan: crate::simd_string_utils::memmem::PairPlan,
    },
}

impl<'a> NeedleFinder<'a> {
    fn new(needle: &'a [u8], case_insensitive: bool) -> Self {
        if case_insensitive {
            Self::CaseInsensitive {
                needle,
                plan: memmem::plan_for(needle),
            }
        } else {
            Self::CaseSensitive(memchr::memmem::Finder::new(needle))
        }
    }

    #[inline]
    fn find(&self, haystack: &[u8]) -> Option<usize> {
        match self {
            Self::CaseSensitive(finder) => finder.find(haystack),
            Self::CaseInsensitive { needle, plan } => memmem::find_planned(haystack, needle, *plan),
        }
    }

    #[inline]
    fn needle(&self) -> &[u8] {
        match self {
            Self::CaseSensitive(finder) => finder.needle(),
            Self::CaseInsensitive { needle, .. } => needle,
        }
    }

    /// Compare `haystack` against a slice of the needle with the same case
    // semantics as `find`. Short prefix loop, not a memmem dispatch.
    #[inline]
    fn eq_fold(&self, haystack: &[u8], needle_seg: &[u8]) -> bool {
        match self {
            Self::CaseSensitive(_) => haystack == needle_seg,
            Self::CaseInsensitive { .. } => {
                haystack.len() == needle_seg.len()
                    && crate::simd_path::eq_ignore_ascii_bytes(haystack, needle_seg)
            }
        }
    }

/// Non-overlapping highlight spans. The next search starts after the previous hit.
    #[inline]
    fn for_each_occurrence(&self, haystack: &[u8], mut on_match: impl FnMut(usize)) {
        let step = self.needle().len().max(1);
        match self {
            Self::CaseSensitive(finder) => {
                for pos in finder.find_iter(haystack) {
                    on_match(pos);
                }
            }
            Self::CaseInsensitive { needle, plan } => {
                let mut start_pos = 0usize;
                while let Some(pos) = memmem::find_planned(&haystack[start_pos..], needle, *plan) {
                    on_match(start_pos + pos);
                    start_pos += pos + step;
                    if start_pos >= haystack.len() {
                        break;
                    }
                }
            }
        }
    }
}

struct PlainTextMatcher<'a> {
    finder: &'a NeedleFinder<'a>,
}

impl Matcher for PlainTextMatcher<'_> {
    type Error = NoError;

    #[inline]
    fn find_at(&self, haystack: &[u8], at: usize) -> Result<Option<Match>, NoError> {
        let hay = &haystack[at..];
        let needle_len = self.finder.needle().len();

        Ok(self
            .finder
            .find(hay)
            .map(|pos| Match::new(at + pos, at + pos + needle_len)))
    }

    #[inline]
    fn line_terminator(&self) -> Option<fff_grep::LineTerminator> {
        Some(fff_grep::LineTerminator::byte(b'\n'))
    }
}

struct PlainTextSink<'r> {
    state: SinkState<'r>,
    finder: &'r NeedleFinder<'r>,
    pattern_len: u32,
    multiline_segment_len: Option<usize>,
}

impl Sink for PlainTextSink<'_> {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        searcher: &Searcher,
        sink_match: &SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        debug_assert_newline_terminator(searcher);
        let line_bytes = sink_match.bytes();
        let (display_bytes, display_len, line_number, byte_offset) =
            SinkState::prepare_line(line_bytes, sink_match);

        let mut match_byte_offsets: SmallVec<[(u32, u32); 4]> = SmallVec::new();
        let mut col = 0usize;
        let mut first = true;

        if let Some(seg_len) = self.multiline_segment_len {
            // Multiline needle: the match starts on this line, so the needle's
            // first segment must be a suffix of the line. Highlight that suffix.
            let seg = &self.finder.needle()[..seg_len];
            if !seg.is_empty()
                && display_bytes.len() >= seg.len()
                && self
                    .finder
                    .eq_fold(&display_bytes[display_bytes.len() - seg.len()..], seg)
            {
                col = display_bytes.len() - seg.len();
                match_byte_offsets.push((col as u32, display_len));
            }
        } else {
            let pattern_len = self.pattern_len;
            self.finder.for_each_occurrence(display_bytes, |pos| {
                let abs_start = pos as u32;
                let abs_end = (abs_start + pattern_len).min(display_len);
                if first {
                    col = pos;
                    first = false;
                }
                match_byte_offsets.push((abs_start, abs_end));
            });
        }

        let Some(is_definition) = self.state.definition_mode.classify_match(
            self.state.file,
            strip_line_terminators(line_bytes),
            &match_byte_offsets,
            false,
        ) else {
            return Ok(true);
        };
        // Rejected lines never become matches, so the display string is built after the filter.
        let line_content = lossy_owned(display_bytes);
        let (context_before, context_after) = self.state.extract_context(sink_match);
        Ok(self.state.push_match(
            line_number,
            col,
            byte_offset,
            line_content,
            match_byte_offsets,
            is_definition,
            context_before,
            context_after,
        ))
    }

    fn finish(&mut self, _: &Searcher, _: &fff_grep::SinkFinish) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Perform a grep search across all indexed files.
//
/// When `query` is empty, returns git-modified/untracked files sorted by
/// frecency for the "welcome state" UI.
#[tracing::instrument(skip_all, fields(file_count = files.len()))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn grep_search<'a>(
    files: &'a [FileItem],
    total_files: usize,
    query: &FFFQuery<'_>,
    options: &GrepSearchOptions,
    budget: &ContentCacheBudget,
    bigram_index: Option<&BigramFilter>,
    bigram_overlay: Option<&BigramOverlay>,
    abort_signal: &AtomicBool,
    base_path: &Path,
    arena: crate::simd_path::ArenaPtr,
    overflow_arena: crate::simd_path::ArenaPtr,
    content_masks: Option<&'a crate::file_picker::ContentMaskStore>,
) -> GrepResult<'a> {
    let result = grep_search_parsed(
        files,
        query,
        options,
        total_files,
        budget,
        bigram_index,
        bigram_overlay,
        abort_signal,
        base_path,
        arena,
        overflow_arena,
        content_masks,
    );

    // Constraint parsing can swallow tokens the user meant literally (e.g. `!=`
    // becoming an exclusion). If the constrained search scanned everything and
    // found nothing, retry the whole raw query as literal text. This also holds
    // for later pages: an empty full scan at offset 0 stays empty at any offset,
    // so paging offsets consistently index the literal search's file list.
    let full_scan_empty = result.matches.is_empty() && result.next_file_offset == 0;
    if !full_scan_empty
        || query.constraints.is_empty()
        || options.classify_definitions == super::DefinitionMode::DefinitionOnly
        || abort_signal.load(Ordering::Relaxed)
    {
        return result;
    }

    // An empty section must not trigger a literal retry when the query matched the other section.
    if options.classify_definitions.is_filtered() {
        let probe = GrepSearchOptions {
            file_offset: 0,
            page_limit: 1,
            classify_definitions: Default::default(),
            ..options.clone()
        };
        let unfiltered = grep_search_parsed(
            files,
            query,
            &probe,
            total_files,
            budget,
            bigram_index,
            bigram_overlay,
            abort_signal,
            base_path,
            arena,
            overflow_arena,
            content_masks,
        );
        if !unfiltered.matches.is_empty() || unfiltered.next_file_offset != 0 {
            return result;
        }
    }

    let raw = query.raw_query.trim();
    if raw.is_empty() {
        return result;
    }

    // Keep any explicit FilePath scope (AI mode `path/to/file.ext` prefix) so the
    // fallback can't leak matches outside the file the user pinned. Only the
    // swallowed operator/glob tokens are dropped. See issue #756.
    let scoped_constraints: fff_query_parser::ConstraintVec<'_> = query
        .constraints
        .iter()
        .filter(|c| matches!(c, fff_query_parser::Constraint::FilePath(_)))
        .cloned()
        .collect();

    let literal_query = FFFQuery {
        raw_query: query.raw_query,
        constraints: scoped_constraints,
        fuzzy_query: fff_query_parser::FuzzyQuery::Text(raw),
        location: None,
    };

    let mut fallback = grep_search_parsed(
        files,
        &literal_query,
        options,
        total_files,
        budget,
        bigram_index,
        bigram_overlay,
        abort_signal,
        base_path,
        arena,
        overflow_arena,
        content_masks,
    );

    if fallback.matches.is_empty() {
        result
    } else {
        fallback.literal_fallback = true;
        fallback
    }
}

#[allow(clippy::too_many_arguments)]
fn grep_search_parsed<'a>(
    files: &'a [FileItem],
    query: &FFFQuery<'_>,
    options: &GrepSearchOptions,
    total_files: usize,
    budget: &ContentCacheBudget,
    bigram_index: Option<&BigramFilter>,
    bigram_overlay: Option<&BigramOverlay>,
    abort_signal: &AtomicBool,
    base_path: &Path,
    arena: crate::simd_path::ArenaPtr,
    overflow_arena: crate::simd_path::ArenaPtr,
    content_masks: Option<&'a crate::file_picker::ContentMaskStore>,
) -> GrepResult<'a> {
    // Masks are sized at scan but only filled when content indexing ran.
    // An empty store rejects nothing yet costs per-file work: drop it.
    let content_masks = content_masks.filter(|s| s.any_recorded());
    let constraints_from_query = &query.constraints[..];

    let grep_text = extract_grep_text(query);
    if grep_text.is_empty() {
        return GrepResult::empty(total_files, total_files);
    }

    let casing = options.effective_casing();
    let case_insensitive = casing.is_insensitive_for(&grep_text);

    let base_count = bigram_boundary(bigram_overlay, files.len());

    let mut regex_fallback_error: Option<String> = None;
    let regex = match options.mode {
        GrepMode::PlainText => None,
        GrepMode::Fuzzy => {
            let bigram_candidates = fuzzy_candidates(bigram_index, bigram_overlay, &grep_text);

            let (files_to_search, filtered_file_count) = prefilter_with_filepath_retry_indexed(
                files,
                constraints_from_query,
                bigram_candidates.as_deref(),
                base_count,
                options,
                arena,
                overflow_arena,
            );

            if files_to_search.is_empty() {
                return GrepResult::empty(total_files, filtered_file_count);
            }

            return super::fuzzy_grep::fuzzy_grep_search(
                &grep_text,
                &files_to_search,
                options,
                total_files,
                filtered_file_count,
                case_insensitive,
                budget,
                abort_signal,
                base_path,
                arena,
                overflow_arena,
                content_masks,
            );
        }
        GrepMode::Regex => build_regex(&grep_text, casing)
            .inspect_err(|err| {
                tracing::warn!("Regex compilation failed for {}. Error {}", grep_text, err);

                regex_fallback_error = Some(err.to_string());
            })
            .ok(),
    };

    let (multiline_segment_len, effective_pattern) = match replace_newline_escapes(&grep_text) {
        Some((replaced, first_newline_pos)) => (Some(first_newline_pos), replaced),
        None => (None, grep_text),
    };

    let is_multiline = multiline_segment_len.is_some();

    // when there is multiple line requested automatically expand the context to include all the lines
    let after_context = if is_multiline && regex.is_none() && options.after_context == 0 {
        effective_pattern.bytes().filter(|&b| b == b'\n').count()
    } else {
        options.after_context
    };

    // ASCII fold is a no-op when no byte changes; borrowing keeps the same needle.
    let lowered = if case_insensitive
        && crate::simd_path::needle_has_ascii_upper(effective_pattern.as_bytes())
    {
        let bytes = effective_pattern.as_bytes();
        let mut lower = vec![0u8; bytes.len()];
        crate::simd_path::fold_ascii_needle(bytes, &mut lower);
        Some(lower)
    } else {
        None
    };
    let needle_bytes = lowered.as_deref().unwrap_or(effective_pattern.as_bytes());
    let finder = NeedleFinder::new(needle_bytes, case_insensitive);
    let pattern_len = needle_bytes.len() as u32;

    // PlainText (or regex-fallback-to-plain): literal bigram query.
    // Regex: decompose the pattern HIR into an AND/OR bigram query tree.
    let bigram_candidates = if regex.is_none() {
        literal_candidates(bigram_index, bigram_overlay, &[&effective_pattern])
    } else {
        regex_candidates(bigram_index, bigram_overlay, &effective_pattern)
    };

    let (files_to_search, filtered_file_count) = prefilter_with_filepath_retry_indexed(
        files,
        constraints_from_query,
        bigram_candidates.as_deref(),
        base_count,
        options,
        arena,
        overflow_arena,
    );

    if files_to_search.is_empty() {
        return GrepResult {
            regex_fallback_error,
            ..GrepResult::empty(total_files, filtered_file_count)
        };
    }

    // `PlainTextMatcher` is used by the grep-searcher engine for line detection.
    // `PlainTextSink` / `RegexSink` handle highlight extraction independently via ripgrep create
    let plain_matcher = PlainTextMatcher { finder: &finder };

    let searcher = {
        let mut b = SearcherBuilder::new();
        b.line_number(true).multi_line(is_multiline);
        b
    }
    .build();

    // The content mask needs a literal needle. Plain text gives one; regex and
    // fuzzy do not, and testing a decomposed pattern would be unsound.
    let mask_needle: Option<crate::types::MaskNeedle> = match options.mode {
        GrepMode::PlainText => Some(crate::types::MaskNeedle::new(
            effective_pattern.as_bytes(),
            0,
        )),
        _ => None,
    };
    let should_prefilter = regex.is_none();
    // Literal, one line, no context: one pass, no searcher line walk.
    let direct_plain =
        should_prefilter && !is_multiline && options.before_context == 0 && after_context == 0;
    let mut result = perform_grep(
        &files_to_search,
        options,
        &GrepContext {
            total_files,
            filtered_file_count,
            budget,
            base_path,
            arena,
            overflow_arena,
            prefilter: should_prefilter.then_some(&finder),
            abort_signal,
            content_masks,
            mask_needle,
            direct_plain,
        },
        // The single sink-selection point: every mode's matcher/sink pairing
        // is decided here based on the compiled pattern.
        |file: &FileItem, file_bytes: &[u8], max_matches: usize| {
            let state = SinkState {
                file,
                definition_mode: options.classify_definitions,
                file_index: 0,
                matches: Vec::with_capacity(4),
                max_matches,
                before_context: options.before_context,
                after_context,
            };

            match regex {
                Some(ref re) => {
                    let regex_matcher = RegexMatcher {
                        regex: re,
                        is_multiline,
                    };
                    let mut sink = RegexSink { state, re };
                    if let Err(e) = searcher.search_slice(&regex_matcher, file_bytes, &mut sink) {
                        tracing::error!(error = %e, "Grep (regex) search failed");
                    }
                    sink.state.matches
                }
                None => {
                    let mut sink = PlainTextSink {
                        state,
                        finder: &finder,
                        pattern_len,
                        multiline_segment_len,
                    };
                    if let Err(e) = searcher.search_slice(&plain_matcher, file_bytes, &mut sink) {
                        tracing::error!(error = %e, "Grep (plain text) search failed");
                    }
                    sink.state.matches
                }
            }
        },
    );
    result.regex_fallback_error = regex_fallback_error;
    result
}

/// Replace unescaped `\n` escapes with real newlines in a single pass.
//
/// Returns `Some((replaced, first_newline_pos))` when the pattern contained at
/// least one real `\n` escape (the user wants multiline search), where
/// `first_newline_pos` is the byte offset of the first inserted newline in the
/// replaced string. Returns `None` when nothing had to be replaced: `\\n` is
/// preserved as-is (escaped backslash + literal `n`, e.g. `\\nvim-data`).
pub(super) fn replace_newline_escapes(text: &str) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    // Most queries have no backslash escape — skip the Vec entirely.
    if !bytes.contains(&b'\\') {
        return None;
    }
    let mut result = Vec::with_capacity(bytes.len());
    let mut first_newline_pos: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'n' {
                // Odd number of consecutive backslashes before 'n' -> real \n escape
                let mut backslash_count = 1;
                while backslash_count <= i && bytes[i - backslash_count] == b'\\' {
                    backslash_count += 1;
                }
                if backslash_count % 2 == 1 {
                    first_newline_pos.get_or_insert(result.len());
                    result.push(b'\n');
                    i += 2;
                    continue;
                }
            }
            result.push(bytes[i]);
            i += 1;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }

    let first_newline_pos = first_newline_pos?;
    let replaced = String::from_utf8(result).unwrap_or_else(|_| text.to_string());
    Some((replaced, first_newline_pos))
}

pub fn parse_grep_query(query: &str) -> FFFQuery<'_> {
    let parser = QueryParser::new(GrepConfig);
    parser.parse(query)
}

/// Extract the grep pattern text from the parsed query: all non-constraint
/// tokens joined with spaces, e.g. `"name = *.rs someth"` -> `"name = someth"`
/// with constraint `Extension("rs")`.
fn extract_grep_text(query: &FFFQuery<'_>) -> String {
    if !matches!(query.fuzzy_query, fff_query_parser::FuzzyQuery::Empty) {
        return query.grep_text();
    }

    // if constraint-only or empty query we use raw_query for backslash-escape handling
    let t = query.raw_query.trim();
    if t.starts_with('\\') && t.len() > 1 {
        let suffix = &t[1..];
        let parser = QueryParser::new(GrepConfig);
        if !parser.parse(suffix).constraints.is_empty() {
            return suffix.to_string();
        }
    }
    t.to_string()
}

#[derive(Clone, Copy)]
pub(super) struct GrepContext<'a, 'b> {
    pub(super) total_files: usize,
    pub(super) filtered_file_count: usize,
    pub(super) budget: &'a ContentCacheBudget,
    pub(super) base_path: &'a Path,
    pub(super) arena: crate::simd_path::ArenaPtr,
    pub(super) overflow_arena: crate::simd_path::ArenaPtr,
    pub(super) prefilter: Option<&'a NeedleFinder<'b>>,
    pub(super) abort_signal: &'a AtomicBool,
    // Content byte-presence masks, indexed by file position. Lets grep skip
    // `get_content_for_search` for files that provably cannot contain the
    // needle, which is ~95% of grep's wall time.
    pub(super) content_masks: Option<&'a crate::file_picker::ContentMaskStore>,
    // Literal needle bytes for the content mask pre-check. `None` disables it
    // (regex queries have no single literal to test for).
    // Needle with its byte classes precomputed, so the per-file check is one
    // AND + popcount instead of a scan over the needle bytes.
    pub(super) mask_needle: Option<crate::types::MaskNeedle>,
    // Literal, single line, no context: harvest hits without the line searcher.
    pub(super) direct_plain: bool,
}

// Files skipped by the content-mask prefilter since process start.
pub(crate) static CONTENT_MASK_REJECTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

// Files skipped by the whole-file memmem prefilter since process start.
// Counted separately: mask skips mean "unreadable without opening",
// memmem skips mean "opened but needle absent".
pub(crate) static MEMMEM_PREFILTER_REJECTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

// Warm the next chunk's file while this worker reads its own. Hint only.
#[cfg(unix)]
#[inline]
fn prefetch_ahead(
    files: &[(&FileItem, usize)],
    idx: usize,
    stride: usize,
    ctx: &GrepContext<'_, '_>,
) {
    let Some(&(next, orig_idx)) = files.get(idx.saturating_add(stride)) else {
        return;
    };
    if next.is_binary() || next.size() == 0 || next.size() > ctx.budget.max_file_size {
        return;
    }
    if let (Some(masks), Some(needle)) = (ctx.content_masks, ctx.mask_needle)
        && let Some(mask) = masks.get(orig_idx)
        && needle.allows(mask) == Some(false)
    {
        return;
    }
    let mut path_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
    let abs = next.write_absolute_path(ctx.arena_for_file(next), ctx.base_path, &mut path_buf);
    if let Ok(file) = std::fs::File::open(abs) {
        use std::os::fd::AsRawFd;
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED);
        }
    }
}

#[cfg(not(unix))]
#[inline]
fn prefetch_ahead(
    _files: &[(&FileItem, usize)],
    _idx: usize,
    _stride: usize,
    _ctx: &GrepContext<'_, '_>,
) {
}

impl GrepContext<'_, '_> {
    #[inline]
    fn arena_for_file(&self, file: &FileItem) -> crate::simd_path::ArenaPtr {
        if file.is_overflow() {
            self.overflow_arena
        } else {
            self.arena
        }
    }
}

#[tracing::instrument(
    skip_all,
    level = Level::DEBUG,
    fields(prefiltered_count = files_to_search.len())
)]
pub(super) fn perform_grep<'a, F>(
    files_to_search: &[(&'a FileItem, usize)],
    options: &GrepSearchOptions,
    ctx: &GrepContext<'_, '_>,
    search_file: F,
) -> GrepResult<'a>
where
    F: Fn(&FileItem, &[u8], usize) -> Vec<GrepMatch> + Sync,
{
    let time_budget = if options.time_budget_ms > 0 {
        Some(std::time::Duration::from_millis(options.time_budget_ms))
    } else {
        None
    };

    let search_start = std::time::Instant::now();
    // Directory FD for openat reads on Unix; None elsewhere and on
    // open failure (callers fall back to absolute-path opens).
    #[cfg(unix)]
    let base_dir = std::fs::File::open(ctx.base_path).ok();
    #[cfg(not(unix))]
    let base_dir: Option<std::fs::File> = None;
    let phase_timing = std::env::var_os("FFF_PHASE_TIMING").is_some();
    // Per-phase nanos, summed across chunk iterations. Cheap: two adds per file.
    let mut phase_read_ns = 0u64;
    let mut phase_scan_ns = 0u64;
    let page_limit = options.page_limit;
    // Lowest index an abort skipped; everything below it was searched, so it is
    // the resume point for the next page. usize::MAX means no abort happened.
    let first_skipped = AtomicUsize::new(usize::MAX);
    // `usize::MAX` means "collect everything" — no page can fill, so skip the
    // per-chunk gate overhead entirely.
    let gate_enabled = page_limit < usize::MAX;

    let mut result_files: Vec<&'a FileItem> = Vec::new();
    let mut all_matches: Vec<GrepMatch> = Vec::new();
    let mut files_consumed: usize = 0;
    let mut page_filled = false;
    let mut aborted = false;

    // Small chunks exit early on page fill, large chunks scan sparse
    // corpora in few barriers. Sized off the search pool, not the ambient
    // one (which collapses under nested installs).
    let threads = search_pool_threads();
    let (base_chunk, max_chunk) =
        chunk_plan(files_to_search.len(), ctx.total_files, page_limit, threads);
    let mut chunk_size = base_chunk;
    let mut chunk_start = 0;

    while chunk_start < files_to_search.len() {
        let chunk_end = (chunk_start + chunk_size).min(files_to_search.len());
        let chunk = &files_to_search[chunk_start..chunk_end];
        chunk_start = chunk_end;
        let chunk_offset = files_consumed;

        // Unless enforced, the budget stays dormant until something matched.
        let budget = time_budget.filter(|_| options.enforce_time_budget || all_matches.len() > 1);

        // Skip the gate for full scans, zero pages, and tiny chunks where
        // the Vec<AtomicU32> + CAS traffic costs more than it saves.
        let use_gate = gate_enabled && page_limit != 0 && chunk.len() >= threads.saturating_mul(2);
        let gate = use_gate.then(|| PageFillGate::new(chunk.len(), all_matches.len(), page_limit));

        let chunk_results: Vec<(usize, &'a FileItem, Vec<GrepMatch>, u64, u64)> = chunk
            .par_iter()
            .enumerate()
            .map_init(
                // 8K covers most source files without realloc; the shrink
                // below caps huge-file residue at 32K instead of 10MB.
                || {
                    (
                        Vec::with_capacity(64 * 1024),
                        MmapSlot::default(),
                        false,
                        0u64,
                        0u64,
                    )
                },
                |(buf, mmap_slot, aborted, read_ns, scan_ns), (local_idx, &(file, orig_idx))| {
                    let idx = chunk_offset + local_idx;
                    let found = (|| -> Option<Vec<GrepMatch>> {
                        // Page already full from lower-indexed files: skip. This
                        // is not an abort, so `first_skipped` is untouched and
                        // the resume cursor stays on the tipping file.
                        if gate.as_ref().is_some_and(|g| g.is_full()) {
                            return None;
                        }
                        // check the clock on every 8th file only; `aborted` latches for the
                        // rest of this worker's range. File 0 is never skipped so a page
                        // always consumes at least one file (cursor 0 means "done").
                        if !*aborted && local_idx % 8 == 0 && idx > 0 {
                            *aborted = ctx.abort_signal.load(Ordering::Relaxed)
                                || budget.is_some_and(|b| search_start.elapsed() > b);
                        }
                        if *aborted {
                            first_skipped.fetch_min(idx, Ordering::Relaxed);
                            return None;
                        }

                        prefetch_ahead(files_to_search, idx, chunk.len(), ctx);

                        let t_read = phase_timing.then(std::time::Instant::now);
                        // Content masks are keyed by picker position from the prefilter.
                        if let (Some(masks), Some(needle)) = (ctx.content_masks, ctx.mask_needle)
                            && let Some(mask) = masks.get(orig_idx)
                            && needle.allows(mask) == Some(false)
                        {
                            // Diagnostic only, and gated: an unconditional
                            // fetch_add here is a contended atomic on a single
                            // cache line, once per file. On a full-corpus grep
                            // that costs more than the check being measured.
                            // Tests always count (deterministic counter tests).
                            if phase_timing || cfg!(test) {
                                CONTENT_MASK_REJECTS
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            return None;
                        }
                        // A literal longer than the file cannot match. Skip the open.
                        if let Some(pf) = ctx.prefilter {
                            let n = pf.needle().len() as u64;
                            if n > 0 && file.size() < n {
                                return None;
                            }
                        }
                        let content = file.get_content_for_search(
                            buf,
                            mmap_slot,
                            ctx.arena_for_file(file),
                            ctx.base_path,
                            ctx.budget,
                            base_dir.as_ref(),
                        )?;
                        if let Some(t) = t_read {
                            *read_ns += t.elapsed().as_nanos() as u64;
                        }

                        if ctx.direct_plain {
                            let t_scan = phase_timing.then(std::time::Instant::now);
                            let harvested = ctx.prefilter.and_then(|pf| {
                                collect_plain_lines(
                                    file,
                                    content,
                                    pf,
                                    options.classify_definitions,
                                    options.max_matches_per_file,
                                )
                            });
                            if let Some(t) = t_scan {
                                *scan_ns += t.elapsed().as_nanos() as u64;
                            }
                            return match harvested {
                                None => {
                                    if phase_timing || cfg!(test) {
                                        MEMMEM_PREFILTER_REJECTS
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    None
                                }
                                Some(file_matches) if file_matches.is_empty() => None,
                                Some(file_matches) => Some(file_matches),
                            };
                        }

                        // Fast whole-file memmem check before entering the
                        // grep-searcher machinery. Skips Vec alloc, Searcher
                        // setup, and line-splitting for files that can't match.
                        if let Some(pf) = ctx.prefilter
                            && pf.find(content).is_none()
                        {
                            // Diagnostic only, and gated (see above).
                            if phase_timing || cfg!(test) {
                                MEMMEM_PREFILTER_REJECTS
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            return None;
                        }

                        let t_scan = phase_timing.then(std::time::Instant::now);
                        let file_matches = search_file(file, content, options.max_matches_per_file);
                        if let Some(t) = t_scan {
                            *scan_ns += t.elapsed().as_nanos() as u64;
                        }

                        if file_matches.is_empty() {
                            return None;
                        }

                        Some(file_matches)
                    })();

                    // Cap huge-file residue so one 10MB file doesn't pin RSS
                    // on this worker for the rest of the scan.
                    if buf.capacity() > 256 * 1024 {
                        buf.shrink_to(32 * 1024);
                    }

                    if let Some(g) = &gate {
                        g.record_done(local_idx, found.as_ref().map_or(0, Vec::len));
                    }

                    found.map(|m| (idx, file, m, *read_ns, *scan_ns))
                },
            )
            .flatten()
            .collect();

        // Every file in the chunk was visited unless an abort cut it short.
        let resume_at = first_skipped.load(Ordering::Relaxed);
        aborted = resume_at != usize::MAX;
        files_consumed = resume_at.min(chunk_offset + chunk.len());

        let before_matches = all_matches.len();
        for (batch_idx, file, file_matches, read_ns, scan_ns) in chunk_results {
            phase_read_ns += read_ns;
            phase_scan_ns += scan_ns;
            // Matches past the abort point are dropped; the next page re-searches them.
            if batch_idx >= resume_at {
                continue;
            }

            let file_result_idx = result_files.len();
            result_files.push(file);

            for mut m in file_matches {
                m.file_index = file_result_idx;
                if options.trim_whitespace {
                    m.trim_leading_whitespace();
                }
                all_matches.push(m);
            }

            if all_matches.len() >= page_limit {
                // Tighten files_consumed to the file that tipped us over so
                // the next page resumes right after it.
                files_consumed = batch_idx + 1;
                page_filled = true;
                break;
            }
        }

        if page_filled || aborted {
            break;
        }

        // AIMD: empty chunks double (sparse scan, few barriers); near-page
        // chunks reset to base so fill doesn't over-read; partial hits grow
        // additively. Budget pressure clamps back to base scale.
        let new_in_chunk = all_matches.len() - before_matches;
        if new_in_chunk == 0 {
            chunk_size = (chunk_size * 2).min(max_chunk);
        } else if page_limit != usize::MAX && all_matches.len() * 2 >= page_limit {
            chunk_size = base_chunk;
        } else {
            chunk_size = (chunk_size + base_chunk).min(max_chunk);
        }
        if time_budget.is_some_and(|b| search_start.elapsed() * 2 > b) {
            chunk_size = chunk_size.min(base_chunk * 2);
        }
    }

    // No match and no abort means the whole slice was searched.
    if result_files.is_empty() && !aborted {
        files_consumed = files_to_search.len();
    }

    let has_more = (aborted || page_filled) && files_consumed < files_to_search.len();

    let next_file_offset = if has_more {
        options.file_offset + files_consumed
    } else {
        0
    };

    if phase_timing {
        eprintln!(
            "PHASE grep files={} read_ms={:.1} scan_ms={:.1} total_ms={:.1} mask_skips={} memmem_skips={}",
            files_consumed,
            phase_read_ns as f64 / 1e6,
            phase_scan_ns as f64 / 1e6,
            search_start.elapsed().as_secs_f64() * 1e3,
            CONTENT_MASK_REJECTS.load(std::sync::atomic::Ordering::Relaxed),
            MEMMEM_PREFILTER_REJECTS.load(std::sync::atomic::Ordering::Relaxed),
        );
    }

    GrepResult {
        matches: all_matches,
        files_with_matches: result_files.len(),
        files: result_files,
        total_files_searched: files_consumed,
        total_files: ctx.total_files,
        filtered_file_count: ctx.filtered_file_count,
        next_file_offset,
        regex_fallback_error: None,
        literal_fallback: false,
    }
}

// One forward pass. Line bounds are taken only around hits, with the same
// columns, per-file cap, and definition filter as the searcher sink.
fn collect_plain_lines(
    file: &FileItem,
    content: &[u8],
    finder: &NeedleFinder<'_>,
    definition_mode: super::DefinitionMode,
    max_matches: usize,
) -> Option<Vec<GrepMatch>> {
    let mut matches = Vec::new();
    let mut pos = 0usize;
    let mut line_number = 1u64;
    let mut counted = 0usize;
    let mut saw = false;
    let pattern_len = finder.needle().len() as u32;
    while pos < content.len() {
        let Some(rel) = finder.find(&content[pos..]) else {
            break;
        };
        saw = true;
        let abs = pos + rel;
        let line_start = memchr::memrchr(b'\n', &content[..abs]).map_or(0, |i| i + 1);
        if line_start > counted {
            line_number +=
                memchr::memchr_iter(b'\n', &content[counted..line_start]).count() as u64;
            counted = line_start;
        }
        let line_end = memchr::memchr(b'\n', &content[line_start..])
            .map_or(content.len(), |i| line_start + i + 1);
        let stripped = strip_line_terminators(&content[line_start..line_end]);
        let display = truncate_display_bytes(stripped);
        let display_len = display.len() as u32;
        let mut offsets: SmallVec<[(u32, u32); 4]> = SmallVec::new();
        let mut col = 0usize;
        let mut first = true;
        finder.for_each_occurrence(display, |hit| {
            let start = hit as u32;
            let end = (start + pattern_len).min(display_len);
            if first {
                col = hit;
                first = false;
            }
            offsets.push((start, end));
        });
        let advanced = line_end > pos;
        pos = line_end;
        if let Some(is_definition) = definition_mode.classify_match(file, stripped, &offsets, false)
        {
            if matches.is_empty() {
                matches.reserve(4);
            }
            matches.push(GrepMatch {
                file_index: 0,
                line_number,
                col,
                byte_offset: line_start as u64,
                line_content: lossy_owned(display),
                match_byte_offsets: offsets,
                fuzzy_score: None,
                is_definition,
                context_before: Vec::new(),
                context_after: Vec::new(),
            });
            if max_matches != 0 && matches.len() >= max_matches {
                break;
            }
        }
        if !advanced {
            break;
        }
    }
    saw.then_some(matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grep::DefinitionMode;

    #[test]
    fn sinks_stop_when_the_retained_match_limit_is_reached() {
        struct Counted<'a> {
            inner: PlainTextSink<'a>,
            candidates: usize,
        }

        impl Sink for Counted<'_> {
            type Error = std::io::Error;

            fn matched(
                &mut self,
                searcher: &Searcher,
                mat: &SinkMatch<'_>,
            ) -> Result<bool, Self::Error> {
                self.candidates += 1;
                self.inner.matched(searcher, mat)
            }
        }

        let file =
            FileItem::new_from_walk_parts(Path::new("file.rs"), Path::new("."), None, 100, 0).0;
        let finder = NeedleFinder::new(b"Needle", false);
        let matcher = PlainTextMatcher { finder: &finder };
        for source in [
            "Needle();\nfn Needle() {}\nNeedle();\n",
            "fn Needle() {}\nNeedle();\nfn NeedleMore() {}\n",
        ] {
            for mode in [
                DefinitionMode::Off,
                DefinitionMode::Classify,
                DefinitionMode::DefinitionOnly,
                DefinitionMode::UsageOnly,
            ] {
                let first_is_definition = source.starts_with("fn ");
                let expected_candidates = if mode.retains(first_is_definition) {
                    1
                } else {
                    2
                };
                let mut sink = Counted {
                    inner: PlainTextSink {
                        state: SinkState {
                            file: &file,
                            definition_mode: mode,
                            file_index: 0,
                            matches: Vec::new(),
                            max_matches: 1,
                            before_context: 0,
                            after_context: 0,
                        },
                        finder: &finder,
                        pattern_len: 6,
                        multiline_segment_len: None,
                    },
                    candidates: 0,
                };
                SearcherBuilder::new()
                    .line_number(true)
                    .build()
                    .search_slice(&matcher, source.as_bytes(), &mut sink)
                    .unwrap();
                assert_eq!(sink.candidates, expected_candidates, "{mode:?}: {source}");
                assert_eq!(sink.inner.state.matches.len(), 1);
            }
        }
    }
}
