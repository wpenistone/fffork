use super::types::GrepSearchOptions;
use crate::index::BigramFilter;
use crate::index::constraints::{ConstraintPlan, ConstraintsBuffers};
use crate::sort_buffer::sort_with_buffer;
use crate::types::FileItem;
use fff_query_parser::Constraint;

/// On an empty FilePath-constrained result, retry without those path tokens.
/// Each item is `(file, picker index)` for the content mask.
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

    if !files_to_search.is_empty()
        || options.classify_definitions == super::DefinitionMode::DefinitionOnly
    {
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

// Shared chunk sizing for plain + fuzzy grep. Base stays small for cheap
// page-fill exit; max stays large so zero-hit scans still ramp to few
// barriers via AIMD. Full scans skip the gate entirely.
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
    let definition_only = options.classify_definitions == super::DefinitionMode::DefinitionOnly;
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
    fn basic_prefilter(file: &FileItem, max: u64, definition_only: bool) -> bool {
        !file.is_deleted()
            && !file.is_binary()
            && file.size() > 0
            && file.size() <= max
            && (!definition_only || file.supports_definitions())
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
                        if !basic_prefilter(f, max_file_size, definition_only) {
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
                if !basic_prefilter(f, max_file_size, definition_only) {
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
                if !basic_prefilter(f, max_file_size, definition_only) {
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
    sort_candidates_once(&mut prefiltered);

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

// One sequential load of frecency/mtime. The stable sort compares those keys
// and does not reload FileItem atomics on each comparison.
fn sort_candidates_once(items: &mut Vec<(&FileItem, usize)>) {
    let n = items.len();
    if n < 2 {
        return;
    }
    let mut keys: Vec<(i32, u64)> = Vec::with_capacity(n);
    for (file, _) in items.iter() {
        keys.push((file.total_frecency_score(), file.modified()));
    }
    let mut order: Vec<usize> = (0..n).collect();
    sort_with_buffer(&mut order, |&a, &b| {
        keys[b].0.cmp(&keys[a].0).then(keys[b].1.cmp(&keys[a].1))
    });
    // `order[i]` is the source index for position `i`.
    for i in 0..n {
        if order[i] == i {
            continue;
        }
        let mut current = i;
        loop {
            let next = order[current];
            if next == i {
                order[current] = current;
                break;
            }
            items.swap(current, next);
            order[current] = current;
            current = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grep::DefinitionMode;
    use crate::simd_path::ArenaPtr;

    #[test]
    fn definition_only_prefilters_before_sorting_and_offsets() {
        let files: Vec<_> = (0..70)
            .map(|idx| {
                let size = match idx {
                    9 => 0,
                    12 => 101,
                    _ => 10,
                };
                let mut file = FileItem::new_raw(0, size, 1000 - idx, None, idx == 6);
                file.mark_definition_language(["file.rs", "file.txt", "file.py"][idx as usize % 3]);
                file.set_deleted(idx == 8);
                file.set_overflow(idx >= 65);
                file
            })
            .collect();
        let all = [u64::MAX, 1];
        let even = [0x5555_5555_5555_5555, 1];
        for candidates in [None, Some(all.as_slice()), Some(even.as_slice())] {
            for mode in [
                DefinitionMode::Off,
                DefinitionMode::Classify,
                DefinitionMode::DefinitionOnly,
                DefinitionMode::UsageOnly,
            ] {
                let expected: Vec<_> = (0..70)
                    .filter(|idx| ![6, 8, 9, 12].contains(idx))
                    .filter(|idx| mode != DefinitionMode::DefinitionOnly || idx % 3 != 1)
                    .filter(|idx| {
                        *idx >= 65
                            || candidates.is_none_or(|bits| bits[idx / 64] & (1 << (idx % 64)) != 0)
                    })
                    .map(|idx| 1000 - idx as u64)
                    .collect();
                for offset in [0, 2, expected.len(), expected.len() + 1] {
                    let options = GrepSearchOptions {
                        max_file_size: 100,
                        classify_definitions: mode,
                        file_offset: offset,
                        ..Default::default()
                    };
                    let (result, count) = prefilter_files(
                        &files,
                        &[],
                        candidates,
                        65,
                        &options,
                        ArenaPtr::null(),
                        ArenaPtr::null(),
                    );
                    assert_eq!(count, expected.len());
                    assert_eq!(
                        result.iter().map(|file| file.modified()).collect::<Vec<_>>(),
                        expected[offset.min(expected.len())..],
                    );
                }
            }
        }
    }
}
