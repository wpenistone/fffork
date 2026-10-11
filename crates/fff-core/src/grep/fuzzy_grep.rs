use crate::parallelism::search_pool_threads;
use crate::simd_path::ArenaPtr;
use crate::types::{ContentCacheBudget, FileItem, MaskNeedle, MmapSlot};
use fff_grep::lines::LineStep;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::prefilter::chunk_plan;
use super::sink::{
    char_indices_to_byte_offsets, classify_definition, strip_line_terminators,
    truncate_display_bytes,
};
use super::types::{GrepMatch, GrepResult, GrepSearchOptions, PageFillGate};

#[allow(clippy::too_many_arguments)]
pub(super) fn fuzzy_grep_search<'a>(
    grep_text: &str,
    files_to_search: &[(&'a FileItem, usize)],
    options: &GrepSearchOptions,
    total_files: usize,
    filtered_file_count: usize,
    case_insensitive: bool,
    budget: &ContentCacheBudget,
    abort_signal: &AtomicBool,
    base_path: &Path,
    arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    content_masks: Option<&crate::file_picker::ContentMaskStore>,
) -> GrepResult<'a> {
    // Same empty-store fast path as plain grep (see grep_search_parsed).
    let content_masks = content_masks.filter(|s| s.any_recorded());
    // max_typos controls how many *needle* characters can be unmatched.
    // A transposition (e.g. "shcema" -> "schema") costs ~1 typo with
    // default gap penalties. We scale max_typos by needle length:
    //   1-2 chars -> 0 typos (exact subsequence only)
    //   3-5 chars -> 1 typo
    //   6+  chars -> 2 typos
    // Cap at 2: higher values (3+) let the SIMD prefilter pass lines
    // missing key characters entirely (e.g. query "flvencodeX" matching
    // lines without 'l' or 'v'). Quality comes from the post-match filters.
    let max_typos = (grep_text.len() / 3).min(2);
    let scoring = neo_frizbee::Scoring {
        // Use default gap penalties. Higher values (e.g. 20) cause
        // smith-waterman to prefer *dropping needle chars* over paying
        // gap costs, which inflates the typo count and breaks
        // transposition matching ("shcema" -> "schema" becomes 3 typos instead of 1)
        exact_match_bonus: 100,
        // gap_open_penalty: 4,
        // gap_extend_penalty: 2,
        prefix_bonus: 0,
        capitalization_bonus: if case_insensitive { 0 } else { 4 },
        ..neo_frizbee::Scoring::default()
    };

    let matcher = neo_frizbee::Matcher::new(
        grep_text,
        &neo_frizbee::Config {
            // Use the real max_typos so frizbee's SIMD prefilter actually rejects non-matching lines (~2 SIMD instructions per line vs full SW scoring).
            max_typos: Some(max_typos as u16),
            sort: neo_frizbee::SortStrategy::Unsorted,
            scoring,
            ..Default::default()
        },
    );

    // Minimum score threshold: 50% of a perfect contiguous match.
    // With default scoring (match_score=12, matching_case_bonus=4 = 16/char),
    // a transposition costs ~5 from a gap, keeping the score well above 50%
    let perfect_score = (grep_text.len() as u16) * 16;
    let min_score = (perfect_score * 50) / 100;

    // Target identifiers are often longer than the query due to delimiters
    // (e.g. query "flvencodepicture" -> "ff_flv_encode_picture_header" from ffmpeg)
    // Allow 3x needle length to accommodate underscore/dot-separated names
    let max_match_span = grep_text.len() * 3;
    let needle_len = grep_text.len();

    // Each delimiter (_, .) in the target creates a gap. A typical C/Rust
    // identifier like "ff_flv_encode_picture_header" has 4-5 underscores.
    // Scale generously so delimiter gaps don't reject valid matches.
    let max_gaps = (needle_len / 3).max(2);

    // Distinct lowercase needle bytes: one memchr2 per char covers both
    // cases, halving scans vs lower+upper entries. Count stays sound for
    // case-sensitive too (a wrong-case file passes, never a match rejected).
    let needle_bytes = grep_text.as_bytes();
    let mut seen = [false; 256];
    let mut unique_needle_chars: Vec<u8> = Vec::new();
    for &b in needle_bytes {
        let l = b.to_ascii_lowercase();
        if !seen[l as usize] {
            seen[l as usize] = true;
            unique_needle_chars.push(l);
        }
    }

    // How many distinct needle chars must appear in the file.
    // With max_typos allowed, we need at least (unique_count - max_typos)
    let min_chars_required = unique_needle_chars.len().saturating_sub(max_typos);
    // Indexed byte-class check before any open+read. Sound for fuzzy: a
    // match keeps at least len-typos needle bytes, so a failing AND+popcount
    // provably cannot match.
    let mask_needle = MaskNeedle::new(needle_bytes, max_typos as u16);

    let time_budget = if options.time_budget_ms > 0 {
        Some(std::time::Duration::from_millis(options.time_budget_ms))
    } else {
        None
    };
    let search_start = std::time::Instant::now();
    #[cfg(unix)]
    let base_dir = std::fs::File::open(base_path).ok();
    #[cfg(not(unix))]
    let base_dir: Option<std::fs::File> = None;
    // Lowest index an abort skipped, i.e. the resume point; usize::MAX means no abort.
    let first_skipped = AtomicUsize::new(usize::MAX);
    let max_matches_per_file = options.max_matches_per_file;

    let page_limit = options.page_limit;
    // Same pool sizing as plain grep: never the ambient count.
    let threads = search_pool_threads();
    let (base_chunk, max_chunk) =
        chunk_plan(files_to_search.len(), total_files, page_limit, threads);
    let gate_enabled = page_limit < usize::MAX;

    let mut chunk_size = base_chunk;
    let mut chunk_start = 0;
    let mut running_matches = 0usize;
    let mut per_file_results: Vec<(usize, &'a FileItem, Vec<GrepMatch>)> = Vec::new();

    while chunk_start < files_to_search.len() {
        let chunk_end = (chunk_start + chunk_size).min(files_to_search.len());
        let chunk = &files_to_search[chunk_start..chunk_end];
        let chunk_offset = chunk_start;
        chunk_start = chunk_end;

        // Same gate-skip policy as plain grep: tiny chunks and full scans
        // pay more in atomics than they save in skipped I/O.
        let use_gate = gate_enabled && page_limit != 0 && chunk.len() >= threads.saturating_mul(2);
        let gate = use_gate.then(|| PageFillGate::new(chunk.len(), running_matches, page_limit));

        let chunk_results: Vec<(usize, &'a FileItem, Vec<GrepMatch>)> = chunk
            .par_iter()
            .enumerate()
            .map_init(
                || {
                    (
                        matcher.clone(),
                        Vec::with_capacity(64 * 1024),
                        MmapSlot::default(),
                    )
                },
                |(matcher, buf, mmap_slot), (local_idx, (file, orig_idx))| {
                    let idx = chunk_offset + local_idx;
                    let found = (|| -> Option<Vec<GrepMatch>> {
                        if gate.as_ref().is_some_and(|g| g.is_full()) {
                            return None;
                        }
                        // File 0 is never skipped so a page always consumes at least one
                        // file (cursor 0 means "done").
                        if idx > 0
                            && (abort_signal.load(Ordering::Relaxed)
                                || time_budget.is_some_and(|b| search_start.elapsed() > b))
                        {
                            first_skipped.fetch_min(idx, Ordering::Relaxed);
                            return None;
                        }

                        // Indexed mask first: skips open+read, one AND+popcount.
                        if let Some(masks) = content_masks
                            && let Some(mask) = masks.get(*orig_idx)
                            && mask_needle.allows(mask) == Some(false)
                        {
                            return None;
                        }

                        let file_arena = if file.is_overflow() {
                            overflow_arena
                        } else {
                            arena
                        };

                        let file_bytes = file.get_content_for_search(
                            buf,
                            mmap_slot,
                            file_arena,
                            base_path,
                            budget,
                            base_dir.as_ref(),
                        )?;

                        if min_chars_required > 0 {
                            let mut chars_found = 0usize;
                            for &ch in &unique_needle_chars {
                                let hit = if ch.is_ascii_lowercase() {
                                    memchr::memchr2(ch, ch.to_ascii_uppercase(), file_bytes)
                                        .is_some()
                                } else {
                                    memchr::memchr(ch, file_bytes).is_some()
                                };
                                if hit {
                                    chars_found += 1;
                                    if chars_found >= min_chars_required {
                                        break;
                                    }
                                }
                            }
                            if chars_found < min_chars_required {
                                return None;
                            }
                        }

                        // Stream lines in small batches into the scorer instead
                        // of materializing the whole file's lines + metadata.
                        let mut stepper = LineStep::new(b'\n', 0, file_bytes.len());
                        let mut batch_lines: Vec<&str> = Vec::with_capacity(128);
                        let mut batch_meta: Vec<(u64, u64)> = Vec::with_capacity(128);
                        let mut file_matches: Vec<GrepMatch> = Vec::new();
                        let mut line_number: u64 = 1;
                        while let Some(line_match) = stepper.next_match(file_bytes) {
                            let byte_offset = line_match.start() as u64;
                            let trimmed = strip_line_terminators(&file_bytes[line_match]);
                            let ln = line_number;
                            line_number += 1;
                            if trimmed.is_empty() {
                                continue;
                            }
                            let Ok(line_str) = std::str::from_utf8(trimmed) else {
                                continue;
                            };
                            batch_lines.push(line_str);
                            batch_meta.push((ln, byte_offset));
                            if batch_lines.len() >= 128 {
                                push_fuzzy_batch(
                                    &mut *matcher,
                                    &batch_lines,
                                    &batch_meta,
                                    min_score,
                                    needle_len,
                                    max_typos,
                                    max_match_span,
                                    max_gaps,
                                    max_matches_per_file,
                                    options.classify_definitions,
                                    &mut file_matches,
                                );
                                batch_lines.clear();
                                batch_meta.clear();
                                if max_matches_per_file != 0
                                    && file_matches.len() >= max_matches_per_file
                                {
                                    break;
                                }
                            }
                        }
                        if !batch_lines.is_empty()
                            && (max_matches_per_file == 0
                                || file_matches.len() < max_matches_per_file)
                        {
                            push_fuzzy_batch(
                                &mut *matcher,
                                &batch_lines,
                                &batch_meta,
                                min_score,
                                needle_len,
                                max_typos,
                                max_match_span,
                                max_gaps,
                                max_matches_per_file,
                                options.classify_definitions,
                                &mut file_matches,
                            );
                        }

                        if file_matches.is_empty() {
                            return None;
                        }

                        Some(file_matches)
                    })();

                    if buf.capacity() > 256 * 1024 {
                        buf.shrink_to(32 * 1024);
                    }

                    if let Some(g) = &gate {
                        g.record_done(local_idx, found.as_ref().map_or(0, Vec::len));
                    }

                    found.map(|m| (idx, *file, m))
                },
            )
            .flatten()
            .collect();

        let before_matches = running_matches;
        for result in chunk_results {
            running_matches += result.2.len();
            per_file_results.push(result);
        }

        if running_matches >= page_limit || first_skipped.load(Ordering::Relaxed) != usize::MAX {
            break;
        }

        // AIMD shared with plain grep: sparse chunks double, near-page
        // chunks reset to base, partial hits grow additively.
        let new_in_chunk = running_matches - before_matches;
        if new_in_chunk == 0 {
            chunk_size = (chunk_size * 2).min(max_chunk);
        } else if page_limit != usize::MAX && running_matches * 2 >= page_limit {
            chunk_size = base_chunk;
        } else {
            chunk_size = (chunk_size + base_chunk).min(max_chunk);
        }
        if time_budget.is_some_and(|b| search_start.elapsed() * 2 > b) {
            chunk_size = chunk_size.min(base_chunk * 2);
        }
    }

    let abort_resume = Some(first_skipped.load(Ordering::Relaxed)).filter(|&i| i != usize::MAX);
    GrepResult::collect(
        per_file_results,
        files_to_search.len(),
        options,
        total_files,
        filtered_file_count,
        abort_resume,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_fuzzy_batch(
    matcher: &mut neo_frizbee::Matcher,
    batch_lines: &[&str],
    batch_meta: &[(u64, u64)],
    min_score: u16,
    needle_len: usize,
    max_typos: usize,
    max_match_span: usize,
    max_gaps: usize,
    max_matches_per_file: usize,
    classify: bool,
    file_matches: &mut Vec<GrepMatch>,
) {
    let scored = matcher.match_list_indices(batch_lines);
    for mut m in scored {
        if m.score < min_score {
            continue;
        }
        let line_idx = m.index as usize;
        let Some(&(ln, bo)) = batch_meta.get(line_idx) else {
            continue;
        };
        let Some(&raw_line) = batch_lines.get(line_idx) else {
            continue;
        };
        let truncated = truncate_display_bytes(raw_line.as_bytes());
        let display_line = if truncated.len() < raw_line.len() {
            &raw_line[..truncated.len()]
        } else {
            raw_line
        };
        if display_line.len() < raw_line.len() {
            let Some(re_m) = matcher
                .match_list_indices(std::slice::from_ref(&display_line))
                .into_iter()
                .next()
            else {
                continue;
            };
            m = re_m;
        }
        m.indices.sort_unstable();
        let min_matched = needle_len.saturating_sub(max_typos).max(1);
        if m.indices.len() < min_matched {
            continue;
        }
        if let (Some(&first), Some(&last)) = (m.indices.first(), m.indices.last()) {
            let span = (last - first + 1) as usize;
            if span > max_match_span {
                continue;
            }
            let density = (m.indices.len() * 100) / span;
            let min_density = if m.indices.len() >= needle_len {
                45
            } else {
                65
            };
            if density < min_density {
                continue;
            }
            let gaps = m.indices.windows(2).filter(|w| w[1] != w[0] + 1).count();
            if gaps > max_gaps {
                continue;
            }
        }
        let offsets = char_indices_to_byte_offsets(display_line, &m.indices);
        let col = offsets.first().map(|r| r.0 as usize).unwrap_or(0);
        file_matches.push(GrepMatch {
            file_index: 0,
            line_number: ln,
            col,
            byte_offset: bo,
            is_definition: classify_definition(classify, display_line),
            line_content: display_line.to_string(),
            match_byte_offsets: offsets,
            fuzzy_score: Some(m.score),
            context_before: Vec::new(),
            context_after: Vec::new(),
        });
        if max_matches_per_file != 0 && file_matches.len() >= max_matches_per_file {
            break;
        }
    }
}
