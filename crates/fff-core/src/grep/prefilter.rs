use super::types::GrepSearchOptions;
use crate::index::BigramFilter;
use crate::index::constraints::{ConstraintPlan, ConstraintsBuffers};
use crate::sort_buffer::sort_with_buffer;
use crate::types::FileItem;
use fff_query_parser::Constraint;

/// Prefilter with a FilePath-constraint fallback: when constraints yield 0
/// files and the query had FilePath constraints, retry without them (the path
/// token was likely part of the search text).
#[allow(clippy::too_many_arguments)]
pub(super) fn prefilter_with_filepath_retry<'a>(
    files: &'a [FileItem],
    constraints: &[Constraint<'_>],
    bigram_candidates: Option<&[u64]>,
    base_count: usize,
    options: &GrepSearchOptions,
    arena: crate::simd_path::ArenaPtr,
    overflow_arena: crate::simd_path::ArenaPtr,
) -> (Vec<&'a FileItem>, usize) {
    let (indexed, total) = prefilter_with_filepath_retry_indexed(
        files,
        constraints,
        bigram_candidates,
        base_count,
        options,
        arena,
        overflow_arena,
    );
    (indexed.into_iter().map(|(f, _)| f).collect(), total)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prefilter_with_filepath_retry_indexed<'a>(
    files: &'a [FileItem],
    constraints: &[Constraint<'_>],
    bigram_candidates: Option<&[u64]>,
    base_count: usize,
    options: &GrepSearchOptions,
    arena: crate::simd_path::ArenaPtr,
    overflow_arena: crate::simd_path::ArenaPtr,
) -> (Vec<(&'a FileItem, usize)>, usize) {
    let (files_to_search, filtered_file_count) = prefilter_files_indexed(
        files,
        constraints,
        bigram_candidates,
        base_count,
        options,
        arena,
        overflow_arena,
    );

    if !files_to_search.is_empty() {
        return (files_to_search, filtered_file_count);
    }

    let Some(stripped) = strip_file_path_constraint_if_present(constraints) else {
        return (files_to_search, filtered_file_count);
    };

    prefilter_files_indexed(
        files,
        &stripped,
        bigram_candidates,
        base_count,
        options,
        arena,
        overflow_arena,
    )
}

/// Shared chunk sizing for plain + fuzzy grep. Base stays small for cheap
/// page-fill exit; max stays large so zero-hit scans still ramp to few
/// barriers via AIMD. Full scans skip the gate entirely.
pub(super) fn chunk_plan(
    num_candidates: usize,
    total_files: usize,
    page_limit: usize,
    threads: usize,
) -> (usize, usize) {
    let base = (threads * 4).max(16).min(num_candidates.max(1));
    if page_limit == usize::MAX {
        // No early exit possible: one chunk, one barrier.
        let max = num_candidates.max(base);
        return (max, max);
    }
    let strong = total_files > 0 && num_candidates * 2 < total_files;
    if strong {
        return (base, base);
    }
    if page_limit > 0 && page_limit <= 256 {
        // Small pages must not over-read: cap growth at ~16 pages and at
        // the candidate count so tiny repos stay single-chunk.
        let max = (base * 4)
            .max(256)
            .min(page_limit.saturating_mul(16).max(base))
            .min(num_candidates.max(base));
        return (base, max);
    }
    // Cap at the candidate count so small repos don't reserve an 8K ceiling.
    let max = (base * 256).max(8 * 1024).min(num_candidates.max(base));
    (base, max)
}

/// Single pass prefilter that doesn't involve file reading
/// allocates only amount of memory required for storing references of the FileItems have to be
/// opened for grepping unaviodably, in the worst case allocates N * <word> memory if no prefilter needed
pub(crate) fn prefilter_files<'a>(
    files: &'a [FileItem],
    constraints: &[Constraint<'_>],
    bigram_candidates: Option<&[u64]>,
    base_count: usize,
    options: &GrepSearchOptions,
    arena: crate::simd_path::ArenaPtr,
    overflow_arena: crate::simd_path::ArenaPtr,
) -> (Vec<&'a FileItem>, usize) {
    let (indexed, total) = prefilter_files_indexed(
        files,
        constraints,
        bigram_candidates,
        base_count,
        options,
        arena,
        overflow_arena,
    );
    (indexed.into_iter().map(|(f, _)| f).collect(), total)
}

// Indexed core: carries the original file position so grep can feed the
// content-mask store without pointer div/mod in the hot parallel loop.
pub(super) fn prefilter_files_indexed<'a>(
    files: &'a [FileItem],
    constraints: &[Constraint<'_>],
    bigram_candidates: Option<&[u64]>,
    base_count: usize,
    options: &GrepSearchOptions,
    arena: crate::simd_path::ArenaPtr,
    overflow_arena: crate::simd_path::ArenaPtr,
) -> (Vec<(&'a FileItem, usize)>, usize) {
    let max_file_size = options.max_file_size;
    let plan = if constraints.is_empty() {
        None
    } else {
        Some(ConstraintPlan::build(
            constraints,
            files,
            arena,
            overflow_arena,
        ))
    };

    let mut scratch = ConstraintsBuffers::new();

    #[inline(always)]
    fn basic_prefilter(file: &FileItem, max: u64) -> bool {
        !file.is_deleted() && !file.is_binary() && file.size > 0 && file.size <= max
    }

    // squeeze as much prefilters into a single loop as possible
    let mut prefiltered: Vec<(&FileItem, usize)> = match bigram_candidates {
        Some(candidates) => {
            let boundary = base_count.min(files.len());
            let (indexed, tail) = files.split_at(boundary);

            let cap = BigramFilter::count_candidates(candidates) + tail.len();
            let mut out: Vec<(&FileItem, usize)> = Vec::with_capacity(cap);

            let full_words = boundary / 64;
            let last_word_bits = boundary % 64;

            // we need this because we already had a regression of the wrong bit
            // has been set for the very last word based on the overlay, it's pretty cheap
            macro_rules! evaluate_bigram_match_word {
                ($word:expr, $base:expr) => {{
                    let mut bits: u64 = $word;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        let file_idx = $base + bit;
                        bits &= bits - 1;

                        let f = unsafe { indexed.get_unchecked(file_idx) };
                        if !basic_prefilter(f, max_file_size) {
                            continue;
                        }
                        if let Some(plan) = plan.as_ref()
                            && !plan.matches(f, file_idx, arena, overflow_arena, &mut scratch)
                        {
                            continue;
                        }
                        out.push((f, file_idx));
                    }
                }};
            }

            // Full words: every set bit guaranteed `< boundary`.
            for (word_idx, &word) in candidates.iter().take(full_words).enumerate() {
                if word != 0 {
                    evaluate_bigram_match_word!(word, word_idx * 64);
                }
            }

            // Last partial word: mask bits past `boundary` once at word load.
            if last_word_bits != 0 {
                // this will get only (mod 64) bits from the last word guarantee that it's 0 padded
                let last_mask: u64 = (1u64 << last_word_bits) - 1;
                let word = candidates[full_words] & last_mask;
                if word != 0 {
                    evaluate_bigram_match_word!(word, full_words * 64);
                }
            }

            // Sequential processing for non-bigrammable files: they are always in the end
            for (offset, f) in tail.iter().enumerate() {
                if !basic_prefilter(f, max_file_size) {
                    continue;
                }
                if let Some(ref p) = plan
                    && !p.matches(f, boundary + offset, arena, overflow_arena, &mut scratch)
                {
                    continue;
                }
                out.push((f, boundary + offset));
            }

            out
        }
        // this will be executed if there is no bigram, in the worst case it will allocate
        // whole array of files but probability in the real repo of NO preflter working is so
        // low that we just ignore that, usually there would be at least a few files excluded
        None => {
            let mut out: Vec<(&FileItem, usize)> = Vec::new();
            for (idx, f) in files.iter().enumerate() {
                if !basic_prefilter(f, max_file_size) {
                    continue;
                }
                if let Some(ref p) = plan
                    && !p.matches(f, idx, arena, overflow_arena, &mut scratch)
                {
                    continue;
                }
                out.push((f, idx));
            }
            out
        }
    };

    let total_count = prefiltered.len();

    sort_with_buffer(&mut prefiltered, |a, b| {
        b.0.total_frecency_score()
            .cmp(&a.0.total_frecency_score())
            .then(b.0.modified.cmp(&a.0.modified))
    });

    if options.file_offset > 0 && options.file_offset < total_count {
        let paginated = prefiltered.split_off(options.file_offset);
        (paginated, total_count)
    } else if options.file_offset >= total_count {
        (Vec::new(), total_count)
    } else {
        (prefiltered, total_count)
    }
}

fn strip_file_path_constraint_if_present<'a>(
    constraints: &[Constraint<'a>],
) -> Option<fff_query_parser::ConstraintVec<'a>> {
    if !constraints
        .iter()
        .any(|c| matches!(c, Constraint::FilePath(_)))
    {
        return None;
    }

    let filtered: fff_query_parser::ConstraintVec<'a> = constraints
        .iter()
        .filter(|c| !matches!(c, Constraint::FilePath(_)))
        .cloned()
        .collect();

    Some(filtered)
}
