use crate::parallelism::search_pool_threads;
use crate::simd_path::ArenaPtr;
use crate::types::{ContentCacheBudget, FileItem, MaskNeedle, MmapSlot};
use fff_grep::lines::LineStep;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::prefilter::chunk_plan;
use super::sink::{char_indices_to_byte_offsets, strip_line_terminators, truncate_display_bytes};
use super::types::{GrepMatch, GrepResult, GrepSearchOptions, PageFillGate};

// Cleared, then filled on each search.
// Parked empty so the next search keeps the allocation.
thread_local! {
    static UNIQUE_NEEDLE_CHARS: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static PER_FILE_RESULTS: std::cell::RefCell<Vec<(usize, Vec<GrepMatch>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static MATCH_SPARE: std::cell::RefCell<Option<Vec<GrepMatch>>> =
        const { std::cell::RefCell::new(None) };
}

fn take_unique_needle_chars() -> Vec<u8> {
    let mut buf = UNIQUE_NEEDLE_CHARS.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    buf.clear();
    buf
}

fn park_unique_needle_chars(mut buf: Vec<u8>) {
    buf.clear();
    UNIQUE_NEEDLE_CHARS.with(|slot| {
        let slot = &mut *slot.borrow_mut();
        if buf.capacity() > slot.capacity() {
            *slot = buf;
        }
    });
}

fn take_per_file_results() -> Vec<(usize, Vec<GrepMatch>)> {
    let mut buf = PER_FILE_RESULTS.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    buf.clear();
    buf
}

fn park_per_file_results(mut buf: Vec<(usize, Vec<GrepMatch>)>) {
    buf.clear();
    PER_FILE_RESULTS.with(|slot| {
        let slot = &mut *slot.borrow_mut();
        if buf.capacity() > slot.capacity() {
            *slot = buf;
        }
    });
}

// Spare list on this thread, reserved for `n` rows; otherwise a new buffer of `n`.
fn alloc_match_buf(n: usize) -> Vec<GrepMatch> {
    MATCH_SPARE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(mut buf) = slot.take() {
            buf.clear();
            buf.reserve(n);
            buf
        } else {
            Vec::with_capacity(n)
        }
    })
}

fn recycle_match_buf(mut buf: Vec<GrepMatch>) {
    buf.clear();
    MATCH_SPARE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let keep = match slot.as_ref() {
            Some(cur) => buf.capacity() > cur.capacity(),
            None => buf.capacity() > 0,
        };
        if keep {
            *slot = Some(buf);
        }
    });
}

fn match_buf_len(slot: &Option<Vec<GrepMatch>>) -> usize {
    slot.as_ref().map_or(0, Vec::len)
}

fn ensure_match_buf(slot: &mut Option<Vec<GrepMatch>>, n: usize) -> &mut Vec<GrepMatch> {
    if slot.is_none() {
        *slot = Some(alloc_match_buf(n));
    } else if let Some(buf) = slot.as_mut() {
        buf.reserve(n);
    }
    slot.as_mut().unwrap()
}

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
    let mut unique_needle_chars = take_unique_needle_chars();
    unique_needle_chars.clear();
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
    // A class owned by one needle byte proves that byte. A shared class does not.
    let mut seen_class = 0u64;
    let mut private_classes = 0u64;
    for &ch in &unique_needle_chars {
        let bit = 1u64 << crate::presence::fold_bit(ch);
        if seen_class & bit != 0 {
            private_classes &= !bit;
        } else {
            seen_class |= bit;
            private_classes |= bit;
        }
    }
    let classes_distinct = private_classes == seen_class && min_chars_required > 0;
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
    let mut per_file_results = take_per_file_results();
    per_file_results.clear();

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
                            // Page filled mid-chunk: files from here on are
                            // unsearched, so mark the resume point (same as
                            // the abort path) instead of losing it.
                            first_skipped.fetch_min(idx, Ordering::Relaxed);
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

                        // Miss skips the read. A private class skips memchr.
                        // Mask 0 is unknown and falls through to a scan.
                        let mut skip_char_scan = false;
                        let mut mask_bits = 0u64;
                        if let Some(masks) = content_masks
                            && let Some(mask) = masks.get(*orig_idx)
                        {
                            match mask_needle.allows(mask) {
                                Some(false) => return None,
                                Some(true) => {
                                    let bits = mask.get();
                                    let private_present =
                                        (bits & private_classes).count_ones() as usize;
                                    let shared = unique_needle_chars.len().saturating_sub(
                                        private_classes.count_ones() as usize,
                                    );
                                    if private_present + shared < min_chars_required {
                                        return None;
                                    }
                                    if classes_distinct {
                                        skip_char_scan = true;
                                    } else {
                                        mask_bits = bits;
                                    }
                                }
                                None => {}
                            }
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

                        if min_chars_required > 0 && !skip_char_scan {
                            let mut chars_found = 0usize;
                            for &ch in &unique_needle_chars {
                                let bit = 1u64 << crate::presence::fold_bit(ch);
                                let hit = if mask_bits != 0 && private_classes & bit != 0 {
                                    mask_bits & bit != 0
                                } else if ch.is_ascii_lowercase() {
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
                        let mut file_matches: Option<Vec<GrepMatch>> = None;
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
                                    file,
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
                                    && match_buf_len(&file_matches) >= max_matches_per_file
                                {
                                    break;
                                }
                            }
                        }
                        if !batch_lines.is_empty()
                            && (max_matches_per_file == 0
                                || match_buf_len(&file_matches) < max_matches_per_file)
                        {
                            push_fuzzy_batch(
                                &mut *matcher,
                                file,
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

                        let file_matches = match file_matches {
                            Some(buf) if !buf.is_empty() => buf,
                            Some(buf) => {
                                recycle_match_buf(buf);
                                return None;
                            }
                            None => return None,
                        };

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
        for (idx, _file, matches) in chunk_results {
            running_matches += matches.len();
            per_file_results.push((idx, matches));
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

    park_unique_needle_chars(unique_needle_chars);
    let abort_resume = Some(first_skipped.load(Ordering::Relaxed)).filter(|&i| i != usize::MAX);
    let result = assemble_grep_result(
        &mut per_file_results,
        files_to_search,
        options,
        total_files,
        filtered_file_count,
        abort_resume,
    );
    park_per_file_results(per_file_results);
    result
}

// Same page cursor as `GrepResult::collect`.
fn assemble_grep_result<'a>(
    per_file_results: &mut Vec<(usize, Vec<GrepMatch>)>,
    files_to_search: &[(&'a FileItem, usize)],
    options: &GrepSearchOptions,
    total_files: usize,
    filtered_file_count: usize,
    abort_resume: Option<usize>,
) -> GrepResult<'a> {
    let files_to_search_len = files_to_search.len();
    let page_limit = options.page_limit;
    let mut result_files: Vec<&'a FileItem> = Vec::new();
    let mut all_matches: Vec<GrepMatch> = Vec::new();
    let mut files_consumed: usize = 0;
    let mut page_filled = false;

    let mut iter = per_file_results.drain(..);
    while let Some((batch_idx, mut file_matches)) = iter.next() {
        if abort_resume.is_some_and(|at| batch_idx >= at) {
            recycle_match_buf(file_matches);
            continue;
        }

        files_consumed = batch_idx + 1;
        let file_result_idx = result_files.len();
        result_files.push(files_to_search[batch_idx].0);

        for mut m in file_matches.drain(..) {
            m.file_index = file_result_idx;
            if options.trim_whitespace {
                m.trim_leading_whitespace();
            }
            all_matches.push(m);
        }
        recycle_match_buf(file_matches);

        if all_matches.len() >= page_limit {
            page_filled = true;
            break;
        }
    }
    for (_, file_matches) in iter {
        recycle_match_buf(file_matches);
    }

    match abort_resume {
        Some(resume_at) if !page_filled => files_consumed = resume_at.min(files_to_search_len),
        None if result_files.is_empty() => files_consumed = files_to_search_len,
        _ => {}
    }

    let has_more = files_consumed < files_to_search_len
        && (abort_resume.is_some() || all_matches.len() >= page_limit);
    let next_file_offset = if has_more {
        options.file_offset + files_consumed
    } else {
        0
    };

    GrepResult {
        matches: all_matches,
        files_with_matches: result_files.len(),
        files: result_files,
        total_files_searched: files_consumed,
        total_files,
        filtered_file_count,
        next_file_offset,
        regex_fallback_error: None,
        literal_fallback: false,
    }
}

fn fuzzy_display_line(raw_line: &str) -> &str {
    let truncated = truncate_display_bytes(raw_line.as_bytes());
    if truncated.len() < raw_line.len() {
        &raw_line[..truncated.len()]
    } else {
        raw_line
    }
}

#[allow(clippy::too_many_arguments)]
fn push_fuzzy_batch(
    matcher: &mut neo_frizbee::Matcher,
    file: &FileItem,
    batch_lines: &[&str],
    batch_meta: &[(u64, u64)],
    min_score: u16,
    needle_len: usize,
    max_typos: usize,
    max_match_span: usize,
    max_gaps: usize,
    max_matches_per_file: usize,
    definition_mode: super::DefinitionMode,
    file_matches: &mut Option<Vec<GrepMatch>>,
) {
    // Rows this batch will push. Sized before the owned list is reserved.
    struct Kept {
        scored_idx: u32,
        line_idx: u32,
        line_number: u64,
        byte_offset: u64,
        is_definition: bool,
    }

    let already = match_buf_len(file_matches);
    let mut kept: smallvec::SmallVec<[Kept; 8]> = smallvec::SmallVec::new();
    let mut scored = matcher.match_list_indices(batch_lines);
    for i in 0..scored.len() {
        if max_matches_per_file != 0 && already + kept.len() >= max_matches_per_file {
            break;
        }
        if scored[i].score < min_score {
            continue;
        }
        let line_idx = scored[i].index as usize;
        let Some(&(ln, bo)) = batch_meta.get(line_idx) else {
            continue;
        };
        let Some(&raw_line) = batch_lines.get(line_idx) else {
            continue;
        };
        let display_line = fuzzy_display_line(raw_line);
        if display_line.len() < raw_line.len() {
            let Some(re_m) = matcher
                .match_list_indices(std::slice::from_ref(&display_line))
                .into_iter()
                .next()
            else {
                continue;
            };
            scored[i] = re_m;
        }
        scored[i].indices.sort_unstable();
        let min_matched = needle_len.saturating_sub(max_typos).max(1);
        if scored[i].indices.len() < min_matched {
            continue;
        }
        if let (Some(&first), Some(&last)) = (scored[i].indices.first(), scored[i].indices.last()) {
            let span = (last - first + 1) as usize;
            if span > max_match_span {
                continue;
            }
            let density = (scored[i].indices.len() * 100) / span;
            let min_density = if scored[i].indices.len() >= needle_len {
                45
            } else {
                65
            };
            if density < min_density {
                continue;
            }
            let gaps = scored[i]
                .indices
                .windows(2)
                .filter(|w| w[1] != w[0] + 1)
                .count();
            if gaps > max_gaps {
                continue;
            }
        }
        let offsets = char_indices_to_byte_offsets(display_line, &scored[i].indices);
        let Some(is_definition) =
            definition_mode.classify_match(file, raw_line.as_bytes(), &offsets, true)
        else {
            continue;
        };
        kept.push(Kept {
            scored_idx: i as u32,
            line_idx: line_idx as u32,
            line_number: ln,
            byte_offset: bo,
            is_definition,
        });
    }

    if kept.is_empty() {
        return;
    }
    let out = ensure_match_buf(file_matches, kept.len());
    for row in kept {
        let m = &scored[row.scored_idx as usize];
        let raw_line = batch_lines[row.line_idx as usize];
        let display_line = fuzzy_display_line(raw_line);
        let offsets = char_indices_to_byte_offsets(display_line, &m.indices);
        let col = offsets.first().map(|r| r.0 as usize).unwrap_or(0);
        out.push(GrepMatch {
            file_index: 0,
            line_number: row.line_number,
            col,
            byte_offset: row.byte_offset,
            is_definition: row.is_definition,
            line_content: display_line.to_string(),
            match_byte_offsets: offsets,
            fuzzy_score: Some(m.score),
            context_before: Vec::new(),
            context_after: Vec::new(),
        });
    }
}
