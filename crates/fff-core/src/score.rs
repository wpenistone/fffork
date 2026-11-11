use crate::{
    index::constraints::apply_constraints,
    path_utils::{CurrentDir, DirPenaltyMemo},
    simd_path::{ArenaPtr, MAX_FRIZBEE_CHUNKS},
    sort_buffer::{apply_source_order, sort_prefix_with_buffer, sort_with_buffer},
    types::{DirItem, FileItem, Score, ScoringContext, VENDORED_PATH_PENALTY},
};
use fff_query_parser::{FFFQuery, FuzzyQuery};
use neo_frizbee::Scoring;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::MAIN_SEPARATOR;

use crate::matcher::*;
pub(crate) use crate::matcher::{capture_hit_indices, with_hit_capture};

static AVX2_PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static PHASE_TIMING_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static FORCE_SCALAR_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub(crate) fn phase_timing() -> bool {
    *PHASE_TIMING_FLAG.get_or_init(|| std::env::var_os("FFF_PHASE_TIMING").is_some())
}

fn force_scalar() -> bool {
    *FORCE_SCALAR_FLAG.get_or_init(|| std::env::var_os("FFF_FORCE_SCALAR").is_some())
}

// One env read. `FFF_NO_PREFIX_CACHE` turns the sibling walk memo off.
fn prefix_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FFF_NO_PREFIX_CACHE").is_none())
}

fn simd_available() -> bool {
    *AVX2_PROBE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("sse4.1")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

// Break-even for per-file presence rechecks: reject savings vs recheck cost.
fn presence_rechecks_pay(rate: f64, presence_ns: f64, gate_ns: f64) -> bool {
    (1.0 - rate) * gate_ns > presence_ns
}

// `presence[i]` matches `FileItems` slot `i`. A short slice falls back per file.
#[inline(always)]
fn scanned_presence(files: &FileItems<'_>, presence: &[u64], i: usize) -> u64 {
    let slot = match files {
        FileItems::All(_) => i,
        FileItems::Filtered { idx, .. } => idx[i] as usize,
    };
    if let Some(&mask) = presence.get(slot) {
        mask
    } else {
        files.index(i).presence_mask()
    }
}

// Full scan without a candidate list. `file_at` is one list shape for the group.
#[inline(always)]
fn walk_span<'a>(
    lo: usize,
    hi: usize,
    check_presence: bool,
    arena: ArenaPtr,
    mut file_at: impl FnMut(usize) -> &'a FileItem,
    mut skip: impl FnMut(usize) -> bool,
    mut visit: impl FnMut(&'a FileItem, u32),
) {
    for k in lo..hi {
        if skip(k) {
            continue;
        }
        let file = file_at(k);
        if !check_presence && k + 1 < hi {
            let nf = file_at(k + 1);
            nf.path.prefetch_for_gate(
                arena,
                nf.parent_dir_index == file.parent_dir_index,
            );
        }
        visit(file, k as u32);
    }
}

// A scoring column is 32 bytes. Pad so that column starts on a 64-byte line.
fn take_line_aligned(slot: &mut Vec<u8>, len: usize) -> (Vec<u8>, usize) {
    let need = len + 63;
    if slot.len() < need {
        slot.resize(need, 0);
    }
    let buf = std::mem::take(slot);
    let mis = buf.as_ptr() as usize & 63;
    let off = if mis == 0 { 0 } else { 64 - mis };
    (buf, off)
}

// Candidates already passed presence. `file_at` is one list shape for the whole group.
#[inline(always)]
fn walk_candidates<'a>(
    lo: usize,
    hi: usize,
    cand: &[u32],
    arena: ArenaPtr,
    mut file_at: impl FnMut(usize) -> &'a FileItem,
    mut visit: impl FnMut(&'a FileItem, u32),
) {
    for k in lo..hi {
        let idx = unsafe { *cand.get_unchecked(k) };
        let pos = idx as usize;
        let file = file_at(pos);
        if k + 1 < hi {
            let npos = unsafe { *cand.get_unchecked(k + 1) } as usize;
            let nf = file_at(npos);
            nf.path.prefetch_for_gate(arena, nf.parent_dir_index == file.parent_dir_index);
        }
        visit(file, idx);
    }
}

// One presence walk. `bits16` is that needle's accept set; `miss` is the same test for the tail.
#[inline(always)]
fn walk_present<const TIMING: bool>(
    files: &[FileItem],
    presence: &[u64],
    lo: usize,
    hi: usize,
    arena: ArenaPtr,
    mut bits16: impl FnMut(usize) -> u16,
    mut miss: impl FnMut(u64) -> bool,
    mut visit: impl FnMut(&FileItem, u32),
    rejects: &mut u64,
) {
    debug_assert!(hi <= files.len() && hi <= presence.len());
    let mut k = lo;
    while k + 16 <= hi {
        prefetch_presence_ahead(presence, k, 32);
        let mut bits = bits16(k);
        if TIMING {
            *rejects += (16 - bits.count_ones()) as u64;
        }
        while bits != 0 {
            let dk = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let file = unsafe { files.get_unchecked(k + dk) };
            if bits != 0 {
                let next = unsafe { files.get_unchecked(k + bits.trailing_zeros() as usize) };
                prefetch_next_accepted(file, next, arena);
            }
            visit(file, (k + dk) as u32);
        }
        k += 16;
    }
    for k in k..hi {
        let mask = unsafe { *presence.get_unchecked(k) };
        if miss(mask) {
            if TIMING {
                *rejects += 1;
            }
            continue;
        }
        visit(unsafe { files.get_unchecked(k) }, k as u32);
    }
}

// The presence line `dist` masks ahead. Sixteen masks are two lines.
#[inline(always)]
fn prefetch_presence_ahead(presence: &[u64], k: usize, dist: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        let ahead = k + dist;
        if ahead < presence.len() {
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    presence.as_ptr().add(ahead) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (presence, k, dist);
}

// Distinct letters, unrolled so the sixteen tests stay independent.
// Each lane matches `NeedlePresence::misses_distinct`, and mask 0 accepts.
#[inline(always)]
fn presence_bits16_distinct(
    presence: &[u64],
    k: usize,
    needle_mask: u64,
    need: usize,
    cover: bool,
) -> u16 {
    let lo = load_presence8(presence, k);
    let hi = load_presence8(presence, k + 8);
    let mut bits = 0u16;
    macro_rules! lane {
        ($mask:expr, $shift:expr, $ok:expr) => {{
            bits |= ($ok as u16) << $shift;
        }};
    }
    if cover {
        macro_rules! ok {
            ($mask:expr) => {{
                let mask = $mask;
                mask == 0 || (mask & needle_mask) == needle_mask
            }};
        }
        lane!(lo[0], 0, ok!(lo[0]));
        lane!(lo[1], 1, ok!(lo[1]));
        lane!(lo[2], 2, ok!(lo[2]));
        lane!(lo[3], 3, ok!(lo[3]));
        lane!(lo[4], 4, ok!(lo[4]));
        lane!(lo[5], 5, ok!(lo[5]));
        lane!(lo[6], 6, ok!(lo[6]));
        lane!(lo[7], 7, ok!(lo[7]));
        lane!(hi[0], 8, ok!(hi[0]));
        lane!(hi[1], 9, ok!(hi[1]));
        lane!(hi[2], 10, ok!(hi[2]));
        lane!(hi[3], 11, ok!(hi[3]));
        lane!(hi[4], 12, ok!(hi[4]));
        lane!(hi[5], 13, ok!(hi[5]));
        lane!(hi[6], 14, ok!(hi[6]));
        lane!(hi[7], 15, ok!(hi[7]));
        return bits;
    }
    macro_rules! lane_pop {
        ($mask:expr, $shift:expr) => {{
            let mask = $mask;
            let ok = mask == 0 || (mask & needle_mask).count_ones() as usize >= need;
            bits |= (ok as u16) << $shift;
        }};
    }
    lane_pop!(lo[0], 0);
    lane_pop!(lo[1], 1);
    lane_pop!(lo[2], 2);
    lane_pop!(lo[3], 3);
    lane_pop!(lo[4], 4);
    lane_pop!(lo[5], 5);
    lane_pop!(lo[6], 6);
    lane_pop!(lo[7], 7);
    lane_pop!(hi[0], 8);
    lane_pop!(hi[1], 9);
    lane_pop!(hi[2], 10);
    lane_pop!(hi[3], 11);
    lane_pop!(hi[4], 12);
    lane_pop!(hi[5], 13);
    lane_pop!(hi[6], 14);
    lane_pop!(hi[7], 15);
    bits
}

// Repeated letters whose multiplicity is at most two. No per-mask branch.
#[inline(always)]
fn presence_bits16_twice(
    presence: &[u64],
    k: usize,
    np: &crate::presence::NeedlePresence,
) -> u16 {
    let lo = load_presence8(presence, k);
    let hi = load_presence8(presence, k + 8);
    let mut bits = 0u16;
    macro_rules! lane {
        ($mask:expr, $shift:expr) => {{
            let mask = $mask;
            let ok = mask == 0 || np.allows_twice(mask);
            bits |= (ok as u16) << $shift;
        }};
    }
    lane!(lo[0], 0);
    lane!(lo[1], 1);
    lane!(lo[2], 2);
    lane!(lo[3], 3);
    lane!(lo[4], 4);
    lane!(lo[5], 5);
    lane!(lo[6], 6);
    lane!(lo[7], 7);
    lane!(hi[0], 8);
    lane!(hi[1], 9);
    lane!(hi[2], 10);
    lane!(hi[3], 11);
    lane!(hi[4], 12);
    lane!(hi[5], 13);
    lane!(hi[6], 14);
    lane!(hi[7], 15);
    bits
}

// Repeated letters, unrolled so each mask's test stays independent.
#[inline(always)]
fn presence_bits16_repeated(
    presence: &[u64],
    k: usize,
    np: &crate::presence::NeedlePresence,
    needle: &[u8],
) -> u16 {
    if np.twice_bounded() {
        return presence_bits16_twice(presence, k, np);
    }
    let lo = load_presence8(presence, k);
    let hi = load_presence8(presence, k + 8);
    let mut bits = 0u16;
    macro_rules! lane {
        ($mask:expr, $shift:expr) => {{
            let mask = $mask;
            let ok = mask == 0 || np.allows_repeated(mask, needle);
            bits |= (ok as u16) << $shift;
        }};
    }
    lane!(lo[0], 0);
    lane!(lo[1], 1);
    lane!(lo[2], 2);
    lane!(lo[3], 3);
    lane!(lo[4], 4);
    lane!(lo[5], 5);
    lane!(lo[6], 6);
    lane!(lo[7], 7);
    lane!(hi[0], 8);
    lane!(hi[1], 9);
    lane!(hi[2], 10);
    lane!(hi[3], 11);
    lane!(hi[4], 12);
    lane!(hi[5], 13);
    lane!(hi[6], 14);
    lane!(hi[7], 15);
    bits
}

// Unknown masks included. Unrolled so each mask's test stays independent.
#[inline(always)]
fn presence_bits16_unknown(
    presence: &[u64],
    k: usize,
    np: &crate::presence::NeedlePresence,
    needle: &[u8],
) -> u16 {
    let lo = load_presence8(presence, k);
    let hi = load_presence8(presence, k + 8);
    let mut bits = 0u16;
    macro_rules! lane {
        ($mask:expr, $shift:expr) => {{
            let ok = crate::presence::allows_or_unknown(np, $mask, needle);
            bits |= (ok as u16) << $shift;
        }};
    }
    lane!(lo[0], 0);
    lane!(lo[1], 1);
    lane!(lo[2], 2);
    lane!(lo[3], 3);
    lane!(lo[4], 4);
    lane!(lo[5], 5);
    lane!(lo[6], 6);
    lane!(lo[7], 7);
    lane!(hi[0], 8);
    lane!(hi[1], 9);
    lane!(hi[2], 10);
    lane!(hi[3], 11);
    lane!(hi[4], 12);
    lane!(hi[5], 13);
    lane!(hi[6], 14);
    lane!(hi[7], 15);
    bits
}

// The next survivor's path slot. A sibling only needs the filename; a new directory starts at byte 0.
#[inline(always)]
fn prefetch_next_accepted(current: &FileItem, next: &FileItem, arena: ArenaPtr) {
    let sibling = next.parent_dir_index == current.parent_dir_index;
    next.path.prefetch_for_gate(arena, sibling);
}

// The next ranked file's record, fetched while this match is scored.
#[inline(always)]
// Called eight matches ahead so the frecency line can return during the bound.
fn prefetch_ranked_file(file: &FileItem) {
    file.prefetch_score_hot();
}

// Eight presence words, issued before any popcnt so the line can return together.
#[inline(always)]
fn load_presence8(presence: &[u64], k: usize) -> [u64; 8] {
    let p = unsafe { presence.as_ptr().add(k) };
    unsafe {
        [
            *p,
            *p.add(1),
            *p.add(2),
            *p.add(3),
            *p.add(4),
            *p.add(5),
            *p.add(6),
            *p.add(7),
        ]
    }
}

fn split_presence(presence: &[u64], base_count: usize) -> (&[u64], &[u64]) {
    if presence.len() >= base_count {
        presence.split_at(base_count)
    } else {
        (&[], &[])
    }
}

// Positions are indexes into `files`. The returned view's own indexes are 0..n,
// so a later part's hit maps straight back onto the previous match list.
fn subset_files<'a>(files: &FileItems<'a>, positions: Vec<u32>) -> FileItems<'a> {
    let base = match files {
        FileItems::All(base) => *base,
        FileItems::Filtered { base, .. } => *base,
    };
    let idx = match files {
        FileItems::All(_) => positions,
        FileItems::Filtered { idx, .. } => positions.iter().map(|&p| idx[p as usize]).collect(),
    };
    FileItems::Filtered {
        base,
        idx: CandList::Owned(idx),
    }
}

// Each part is `match_fuzzy_scanned` on one needle. The stored score is the
// sum of those scores divided by the part count, once, at the end.
fn combine_scored_parts(
    valid_parts: &[&str],
    working_files: &FileItems<'_>,
    presence: &[u64],
    options: &neo_frizbee::Config,
    max_threads: usize,
    arena: ArenaPtr,
    max_path_len: usize,
    abort: crate::parallelism::AbortToken<'_>,
    before_filename: Option<&std::sync::atomic::AtomicBool>,
    check_deleted: bool,
) -> Vec<neo_frizbee::Match> {
    // Two parts: score the longer one first. It rejects more. The end column
    // stays the query-order first part, which is what the filename bonus reads.
    let reverse_pair = valid_parts.len() == 2 && valid_parts[1].len() > valid_parts[0].len();
    let mut order: SmallVec<[&str; 4]> = SmallVec::new();
    if reverse_pair {
        order.push(valid_parts[1]);
        order.push(valid_parts[0]);
    } else {
        order.extend(valid_parts.iter().copied());
    }

    let score_part = |part: &str,
                      files: &FileItems<'_>,
                      before: Option<&std::sync::atomic::AtomicBool>,
                      threads: usize|
     -> Vec<neo_frizbee::Match> {
        let mut part_options = *options;
        part_options.max_typos = options.max_typos.map(|t| t.min(part.len() as u16));
        match_fuzzy_scanned(
            std::slice::from_ref(&part),
            files,
            presence,
            &part_options,
            threads,
            arena,
            max_path_len,
            abort,
            before,
            check_deleted,
        )
    };

    let first = score_part(order[0], working_files, before_filename, max_threads);
    if first.is_empty() {
        return first;
    }
    let mut matches = first;
    let mut sums: Vec<u32> = matches.iter().map(|m| m.score as u32).collect();
    for (ord, part) in order[1..].iter().enumerate() {
        let parts_done = (ord + 2) as u32;
        let positions: Vec<u32> = matches.iter().map(|m| m.index).collect();
        let view = subset_files(working_files, positions);
        let sub_threads = effective_match_threads(
            view.len(),
            part.len(),
            max_path_len,
            options.max_typos.map(|t| t.min(part.len() as u16)),
            max_threads.max(1),
        );
        let sub_matches = score_part(part, &view, None, sub_threads);
        drop(view);
        if sub_matches.is_empty() {
            reclaim_match_buf(matches);
            return vec![];
        }
        let prev_matches = std::mem::take(&mut matches);
        let prev_sums = std::mem::take(&mut sums);
        let mut next_sums = Vec::with_capacity(sub_matches.len());
        matches = sub_matches
            .into_iter()
            .map(|sm| {
                let i = sm.index as usize;
                let prev = &prev_matches[i];
                let sum = prev_sums[i].saturating_add(sm.score as u32);
                next_sums.push(sum);
                neo_frizbee::Match {
                    index: prev.index,
                    score: (sum / parts_done).min(u16::MAX as u32) as u16,
                    end_col: if reverse_pair {
                        sm.end_col
                    } else {
                        prev.end_col
                    },
                    exact: prev.exact && sm.exact,
                }
            })
            .collect();
        sums = next_sums;
        reclaim_match_buf(prev_matches);
    }
    matches
}

fn match_fuzzy_parts(
    fuzzy_parts: &[&str],
    working_files: &FileItems<'_>,
    options: &neo_frizbee::Config,
    max_threads: usize,
    arena: ArenaPtr,
    max_path_len: usize,
    abort: crate::parallelism::AbortToken<'_>,
) -> Vec<neo_frizbee::Match> {
    match_fuzzy_scanned(
        fuzzy_parts,
        working_files,
        &[],
        options,
        max_threads,
        arena,
        max_path_len,
        abort,
        None,
        true,
    )
}

// `presence` is parallel to the `FileItems` base slice. An empty slice
// falls back to `FileItem::presence` (same bits).
fn match_fuzzy_scanned(
    fuzzy_parts: &[&str],
    working_files: &FileItems<'_>,
    presence: &[u64],
    options: &neo_frizbee::Config,
    max_threads: usize,
    arena: ArenaPtr,
    max_path_len: usize,
    abort: crate::parallelism::AbortToken<'_>,
    // Set when an accept ends before the filename. `None` skips the check.
    before_filename: Option<&std::sync::atomic::AtomicBool>,
    check_deleted: bool,
) -> Vec<neo_frizbee::Match> {
    // 1-2 part queries fit on the stack.
    let valid_parts: SmallVec<[&str; 4]> = fuzzy_parts
        .iter()
        .copied()
        .filter(|p| p.len() >= 2)
        .collect();

    if valid_parts.is_empty() {
        tracing::debug!("match_fuzzy_parts: no valid parts after filtering, returning empty");
        return vec![];
    }

    // One part is the gate+SW body below. Several parts each take that body;
    // their scores are one mean. A second engine would not be on that scale.
    if valid_parts.len() > 1 {
        return combine_scored_parts(
            &valid_parts,
            working_files,
            presence,
            options,
            max_threads,
            arena,
            max_path_len,
            abort,
            before_filename,
            check_deleted,
        );
    }

    // One needle. Part order lives in `combine_scored_parts`.
    let first_part: &str = valid_parts[0];
    let mut first_options = *options;
    first_options.max_typos = options.max_typos.map(|t| t.min(first_part.len() as u16));
    // Thread count from the work estimate alone; cached max path length keeps it O(1).
    let threads = effective_match_threads(
        working_files.len(),
        first_part.len(),
        max_path_len,
        first_options.max_typos,
        max_threads.max(1),
    );

    // Gate match-set + windows, then re-score with SW replicating frizbee's gap model
    // on the same trimmed slices with the same include_prefix semantics.
    let first_part_matches = {
        let part_timing = phase_timing();
        let t_prefilter = part_timing.then(std::time::Instant::now);
        // Gate match-set + windows; skip_sw encodes (start, end) in (score, end_col).
        let np = crate::presence::NeedlePresence::new(
            first_part.as_bytes(),
            first_options.max_typos.unwrap_or(u16::MAX),
        );
        // Probe a dispatch-sized sample; scan fully only when reject savings win.
        let gate_typos = first_options.max_typos.unwrap_or(u16::MAX);
        let mut gate_state = crate::gate::GateNeedle::new(first_part.as_bytes(), gate_typos);
        gate_state.set_check_deleted(check_deleted);
        let mut cand: Vec<u32> = Vec::new();
        let t_cand = part_timing.then(std::time::Instant::now);
        let n_files = working_files.len();
        let avg_bytes = max_path_len.max(1);
        let big_enough = crate::dispatch::plan_files(n_files, avg_bytes, usize::MAX).threads > 1;
        // Presence economics, set only when the gate probe was timed.
        let mut presence_probe: Option<(f64, f64, f64)> = None;
        // need < 3 matches almost every path: sampling would still load a mask per probe.
        if big_enough && n_files > 0 && np.should_check() {
            let n_probe = crate::dispatch::probe_len(n_files, avg_bytes);
            let stride = (n_files / n_probe).max(1);
            let needle_b = first_part.as_bytes();
            let t_probe = std::time::Instant::now();
            let mut pass = 0usize;
            let mut probed = 0usize;
            // Passing probe indices reuse for gate timing; bounded to keep the probe cheap.
            let mut pass_idx: SmallVec<[u32; 32]> = SmallVec::new();
            for i in (0..n_files).step_by(stride) {
                // Mask only. `hit_file` still rejects short paths, so this
                // does not load `FileItem`.
                let mask = scanned_presence(working_files, presence, i);
                if mask == 0 || np.allows(mask, needle_b) {
                    pass += 1;
                    if pass_idx.len() < 32 {
                        pass_idx.push(i as u32);
                    }
                }
                probed += 1;
                if probed >= n_probe {
                    break;
                }
            }
            let presence_ns = t_probe.elapsed().as_nanos() as f64 / probed.max(1) as f64;
            let rate = pass as f64 / probed.max(1) as f64;
            // Every sample passed: rechecks cannot pay, so skip the gate timing walks.
            let mut gate_ns = 0.0f64;
            if rate < 1.0 && !pass_idx.is_empty() {
                let t_gate = std::time::Instant::now();
                for &pi in pass_idx.iter() {
                    let f = working_files.index(pi as usize);
                    std::hint::black_box(gate_state.hit_file(f, arena));
                }
                gate_ns = t_gate.elapsed().as_nanos() as f64 / pass_idx.len() as f64;
            }
            if !pass_idx.is_empty() {
                presence_probe = Some((rate, presence_ns, gate_ns));
            }
            // Fire only when reject savings beat scan + recheck + dispatch costs.
            let fire = pass > 0
                && rate < 1.0
                && (n_files as f64 * (1.0 - rate) * gate_ns
                    > n_files as f64 * presence_ns
                        + crate::dispatch::fork_join_ns() as f64
                        + n_files as f64 * rate * presence_ns);
            if fire {
                let keep = |i: usize| {
                    let mask = scanned_presence(working_files, presence, i);
                    if mask != 0 && !np.allows(mask, needle_b) {
                        return None;
                    }
                    Some(i as u32)
                };
                cand = if max_threads < crate::machine::search_threads() {
                    // Binding width: one planned chunk per task, at most max_threads tasks.
                    let span = crate::dispatch::plan_files(n_files, avg_bytes, max_threads.max(1))
                        .chunk
                        .max(1);
                    let n_chunks = n_files.div_ceil(span);
                    crate::parallelism::run_on_search_pool(|| {
                        let parts: Vec<Vec<u32>> = (0..n_chunks)
                            .into_par_iter()
                            .map(|c| {
                                let lo = c * span;
                                let hi = ((c + 1) * span).min(n_files);
                                (lo..hi).filter_map(&keep).collect()
                            })
                            .collect();
                        parts.into_iter().flatten().collect()
                    })
                } else {
                    crate::parallelism::run_on_search_pool(|| {
                        (0..n_files).into_par_iter().filter_map(&keep).collect()
                    })
                };
            }
        }
        let _cand_clock = t_cand;
        // Re-index only when something was rejected; a full pass adds indirection.
        let use_cand = !cand.is_empty() && cand.len() < n_files;
        // `gate_n` is the gate's index space (candidates when the prefilter fired).
        let gate_n = if use_cand { cand.len() } else { n_files };
        // Rechecks are a sound pretest: scan members already passed it, so skipping never changes results.
        // Honors `FFF_NO_GATE_PREFILTER`.
        #[allow(clippy::unnecessary_map_or)]
        let check_presence = np.should_check()
            && std::env::var_os("FFF_NO_GATE_PREFILTER").is_none()
            && !use_cand
            && presence_probe.map_or(true, |(rate, pns, gns)| {
                presence_rechecks_pay(rate, pns, gns)
            });
        // Full scan: mask k is file k, so the slice load needs no bounds check.
        let direct_mask = check_presence
            && presence.len() >= n_files
            && matches!(working_files, FileItems::All(_));
        // Full-scan k is a file index. The layout match stays outside the file loop.
        let all_files: &[FileItem] = match working_files {
            FileItems::All(files) if direct_mask => files,
            _ => &[],
        };
        let t_gate_us = std::sync::atomic::AtomicU64::new(0);
        let t_gate_files = std::sync::atomic::AtomicU64::new(0);
        let t_gate_accepts = std::sync::atomic::AtomicU64::new(0);
        let t_gate_path_bytes = std::sync::atomic::AtomicU64::new(0);
        let t_gate_walked = std::sync::atomic::AtomicU64::new(0);
        let t_presence_rejects = std::sync::atomic::AtomicU64::new(0);
        let t_swslice_us = std::sync::atomic::AtomicU64::new(0);
        let t_prefix_hits = std::sync::atomic::AtomicU64::new(0);
        let t_prefix_fills = std::sync::atomic::AtomicU64::new(0);
        let t_prefix_skip = std::sync::atomic::AtomicU64::new(0);
        let t_tight_trim = std::sync::atomic::AtomicU64::new(0);
        let t_tight_files = std::sync::atomic::AtomicU64::new(0);
        let _prefilter_clock = t_prefilter;

        let has_upper = first_part.chars().any(|c| c.is_uppercase());
        let p = crate::interseq_sw::SwParams::for_query(has_upper);
        let needle = first_part.as_bytes();
        // Folded once per query. Exact compares must not refold the needle.
        let mut needle_lower_buf = [0u8; 128];
        let needle_folded = fill_ascii_lower(needle, &mut needle_lower_buf);
        let needle_lower = &needle_lower_buf[..needle.len().min(needle_lower_buf.len())];

        // Pin exact filename hits for `name.ext` queries (not a full-corpus walk).
        let looks_like_filename = {
            let q = first_part;
            q.len() >= 4 && q.contains('.') && !q.contains('/')
        };
        // Exact-name hits for one group. The next group on this thread reuses the list.
        thread_local! {
            static EXACT_HITS: std::cell::RefCell<Vec<u32>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        let take_exact_hits = || {
            let mut hits = EXACT_HITS.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
            hits.clear();
            hits
        };
        let reclaim_exact_hits = |mut hits: Vec<u32>| {
            hits.clear();
            EXACT_HITS.with(|slot| {
                let slot = &mut *slot.borrow_mut();
                if hits.capacity() > slot.capacity() {
                    *slot = hits;
                }
            });
        };

        // No cheap-rank collapse: it disagreed with SW and reordering regresses ranking.
        let t_sw = part_timing.then(std::time::Instant::now);
        {
            // Chunked scoring: L2-sized metadata keeps TLS scratch resident.
            thread_local! {
                static TRIMS: std::cell::RefCell<Vec<crate::matcher::Trim>> =
                    const { std::cell::RefCell::new(Vec::new()) };
                static SCORES: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
                static ECOLS: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
                static ORDER: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
                static PERM_SCORES: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
                static PERM_ECOLS: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            fn reclaim<T>(b: &mut Vec<T>, v: Vec<T>) {
                *b = v;
                // Cap retained TLS at chunk + one batch; huge match sets must not pin it.
                let retain = crate::dispatch::sw_metadata_chunk() + 64;
                if b.capacity() > retain {
                    b.shrink_to(retain);
                }
            }

            // `FFF_FORCE_SCALAR=1` skips AVX2; must stay result-identical.
            let force_scalar = force_scalar();
            let avx2_ok = if force_scalar {
                false
            } else {
                simd_available()
            };

            // u8 fast path: 32 lanes, valid when `u8_safe` and the needle fits 32 rows.
            // ARM NEON is a first-class path too: 16 lanes, same `u8_safe`
            // gate (NEON is baseline on aarch64, no detection needed).
            #[cfg(target_arch = "x86_64")]
            let use_u8 = avx2_ok && needle.len() <= 32 && p.u8_safe(needle.len());
            #[cfg(target_arch = "x86_64")]
            let batch_sz: usize = if use_u8 { 32 } else { 16 };
            #[cfg(target_arch = "aarch64")]
            let use_u8 = false;
            #[cfg(target_arch = "aarch64")]
            let batch_sz: usize = 16;
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let use_u8 = false;
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let batch_sz: usize = 16;
            // u16 rows are dynamic to 32; longer needles stay scalar.
            #[cfg(target_arch = "x86_64")]
            let use_simd = avx2_ok && needle.len() <= 32;
            #[cfg(not(target_arch = "x86_64"))]
            let _ = avx2_ok;
            // NEON tracks end columns byte-wide like u8: over-long chunks
            // go scalar below (x86 splits them to exact u16 halves).
            #[cfg(target_arch = "aarch64")]
            let use_simd = !force_scalar && needle.len() <= 32 && p.u8_safe(needle.len());
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let use_simd = false;
            // Sibling prefix walks memoized (`FFF_NO_PREFIX_CACHE` escapes); same results.
            let prefix_cache_on = prefix_cache_enabled();

            let t_collect_us = std::sync::atomic::AtomicU64::new(0);
            let t_batch_us = std::sync::atomic::AtomicU64::new(0);
            let t_assemble_us = std::sync::atomic::AtomicU64::new(0);
            // Diagnostic batch split; each add is timing-gated.
            let t_stage_us = std::sync::atomic::AtomicU64::new(0);
            let t_xpose_us = std::sync::atomic::AtomicU64::new(0);
            let t_kernel_us = std::sync::atomic::AtomicU64::new(0);
            let t_scalar_us = std::sync::atomic::AtomicU64::new(0);
            let t_prep_us = std::sync::atomic::AtomicU64::new(0);
            let t_scatter_us = std::sync::atomic::AtomicU64::new(0);
            let t_sort_us = std::sync::atomic::AtomicU64::new(0);
            let t_fan_serial = std::sync::atomic::AtomicU64::new(0);
            let t_fan_parallel = std::sync::atomic::AtomicU64::new(0);
            let timing = part_timing;

            // Chunks are disjoint with thread-local scratch, so one owner may fan out.
            // Saturated groups, and width below the pool, keep batches on this thread.
            let sw_body = |matches_chunk: &mut [neo_frizbee::Match],
                           exact_hits: &[u32],
                           allow_inner: bool| {
                let t_chunk = timing.then(std::time::Instant::now);
                let n_matches = matches_chunk.len();
                let mut trims = TRIMS.with(|c| take_tls(c, n_matches));
                trims.clear();
                // Uniform and small groups never sort by length, so the histogram stays unbuilt.
                let need_hist = n_matches > batch_sz;
                let mut cnt = std::mem::MaybeUninit::<[u32; 129]>::uninit();
                let mut max_trim = 0usize;
                let mut one_len = true;
                let mut seen_len = 0usize;
                {
                    let whole_path = gate_state.full_path_windows();
                    for (mi, m) in matches_chunk.iter().enumerate() {
                        // A start past byte 0 is already the window. Byte 0 still needs the length.
                        let trim = if whole_path {
                            // The window starts at 0 and ends at the path, so both bonuses apply.
                            crate::matcher::Trim {
                                file_idx: m.index,
                                start: m.score,
                                end: m.end_col,
                                flags: 1 | 2,
                            }
                        } else if m.score != 0 && m.score < m.end_col {
                            crate::matcher::Trim {
                                file_idx: m.index,
                                start: m.score,
                                end: m.end_col,
                                flags: 0,
                            }
                        } else {
                            // The scan stored end-of-path in `exact` until scoring overwrites it.
                            let hay_len = if m.score == 0 {
                                if m.exact {
                                    m.end_col as usize
                                } else {
                                    m.end_col as usize + 1
                                }
                            } else {
                                m.end_col as usize
                            };
                            crate::matcher::Trim::from_window(decode_match_window(m, hay_len))
                        };
                        let wlen = (trim.end - trim.start) as usize;
                        if one_len {
                            if mi == 0 {
                                seen_len = wlen;
                            } else if wlen != seen_len {
                                one_len = false;
                                if need_hist {
                                    // Earlier windows shared seen_len, so one store covers them.
                                    let c = cnt.write([0u32; 129]);
                                    c[seen_len.min(128)] = mi as u32;
                                    c[wlen.min(128)] += 1;
                                }
                            }
                        } else if need_hist {
                            // SAFETY: the first length mismatch initialized `cnt`.
                            let c = unsafe { cnt.assume_init_mut() };
                            c[wlen.min(128)] += 1;
                        }
                        max_trim = max_trim.max(wlen);
                        trims.push(trim);
                    }
                }
                if let Some(t_chunk) = t_chunk {
                    t_collect_us.fetch_add(
                        t_chunk.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }

                let t_batch = timing.then(std::time::Instant::now);
                // 64B-align batch outputs: 32 u16 fills a cache line, so par_chunks never false-share.
                let mut scores = SCORES.with(|c| take_tls_init(c, n_matches + 32));
                let mut end_cols = ECOLS.with(|c| take_tls_init(c, n_matches + 32));

                // Scalar path: one serial pass over trims with a reused window buffer.
                // (`max_trim` fused into the collect loop above.)
                // NEON tracks end columns byte-wide like u8 with no wide
                // fallback on ARM, so over-long chunks stay scalar here
                // (x86 splits them to exact u16 halves in dispatch).
                #[cfg(target_arch = "aarch64")]
                let scalar_direct = !use_simd || max_trim > 256;
                #[cfg(not(target_arch = "aarch64"))]
                let scalar_direct = !use_simd;
                if scalar_direct {
                    thread_local! {
                        static SWIN: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
                    }
                    let t_scalar = timing.then(std::time::Instant::now);
                    let mut win = SWIN.with(|c| std::mem::take(&mut *c.borrow_mut()));
                    win.clear();
                    if win.capacity() < max_trim {
                        win.reserve(max_trim);
                    }
                    let mut flip_buf = [0u8; 256];
                    let flip = scalar_flip(needle, &mut flip_buf);
                    let direct_short =
                        !needle.is_empty() && needle.len() <= 32 && flip.len() == needle.len();
                    macro_rules! score_scalar {
                        ($at:ident, $($src:expr),+) => {
                            for (i, t) in trims.iter().enumerate() {
                                let wlen = (t.end - t.start) as usize;
                                if wlen == 0 {
                                    scores[i] = 0;
                                    end_cols[i] = 0;
                                    continue;
                                }
                                let file = crate::matcher::FileItems::$at(
                                    $($src),+,
                                    t.file_idx as usize,
                                );
                        let (sc, ec) = if let Some(slice) = file.path.trimmed_in_chunk(
                            arena,
                            t.start as u32,
                            t.end as u32,
                        ) {
                            score_scalar_window_direct(
                                direct_short,
                                needle,
                                flip,
                                slice,
                                &p,
                                t.include_prefix(),
                            )
                        } else {
                            // The span crosses a slot. The one-slot borrow already missed.
                            unsafe {
                                file.path.copy_crossing(
                                    arena,
                                    t.start as usize,
                                    t.end as usize,
                                    win.as_mut_ptr(),
                                );
                                let dst = std::slice::from_raw_parts(win.as_ptr(), wlen);
                                score_scalar_window_direct(
                                    direct_short,
                                    needle,
                                    flip,
                                    dst,
                                    &p,
                                    t.include_prefix(),
                                )
                            }
                        };
                                scores[i] = sc;
                                end_cols[i] = ec;
                            }
                        };
                    }
                    match &working_files {
                        crate::matcher::FileItems::All(files) => score_scalar!(at_all, files),
                        crate::matcher::FileItems::Filtered { base, idx } => {
                            score_scalar!(at_filtered, base, idx)
                        }
                    }
                    SWIN.with(|c| reclaim(&mut c.borrow_mut(), win));
                    if let Some(t_scalar) = t_scalar {
                        t_scalar_us.fetch_add(
                            t_scalar.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        if let Some(t_batch) = t_batch {
                            t_batch_us.fetch_add(
                                t_batch.elapsed().as_micros() as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                    }
                } else {
                    // Same-length and small groups score in input order.
                    let direct = n_matches <= batch_sz || one_len;
                    // Pure indexed map: serial and parallel runs are bit-identical.
                    // u8 runs full 32-wide; over-long pieces run exact u16 halves.
                    let score_batch_w = |ord: &[u32],
                                         seq_base: Option<u32>,
                                         sc_out: &mut [u16],
                                         ec_out: &mut [u16],
                                         width: usize,
                                         max_len: usize| {
                        // `seq_base` scores trims[base + k] and does not read `ord`.
                        let n = match seq_base {
                            Some(_) => sc_out.len(),
                            None => ord.len(),
                        };
                        // `k` stays inside this batch, and every stored index came from `trims`.
                        let oi_at = |k: usize| -> u32 {
                            match seq_base {
                                Some(base) => base + k as u32,
                                None => unsafe { *ord.get_unchecked(k) },
                            }
                        };
                        if max_len == 0 {
                            sc_out.fill(0);
                            ec_out.fill(0);
                            return;
                        }
                        // Eight or fewer windows never fill a SIMD kernel. Score
                        // them from the arena; copy only a window that crosses a chunk.
                        if n <= 8 {
                            let t_scalar = timing.then(std::time::Instant::now);
                            thread_local! {
                                static CROSS: std::cell::RefCell<Vec<u8>> =
                                    const { std::cell::RefCell::new(Vec::new()) };
                            }
                            let mut cross = CROSS.with(|c| std::mem::take(&mut *c.borrow_mut()));
                            let mut flip_buf = [0u8; 256];
                            let flip = scalar_flip(needle, &mut flip_buf);
                            let direct_short =
                                !needle.is_empty() && needle.len() <= 32 && flip.len() == needle.len();
                            macro_rules! score_small {
                                ($at:ident, $($src:expr),+) => {
                                    for k in 0..n {
                                        let oi = oi_at(k);
                                        let tw = unsafe { trims.get_unchecked(oi as usize) };
                                        let file = crate::matcher::FileItems::$at(
                                            $($src),+,
                                            tw.file_idx as usize,
                                        );
                                        #[cfg(target_arch = "x86_64")]
                                        if k + 1 < n {
                                            let nt = unsafe { trims.get_unchecked(oi_at(k + 1) as usize) };
                                            let nf = crate::matcher::FileItems::$at(
                                                $($src),+,
                                                nt.file_idx as usize,
                                            );
                                            nf.path.prefetch_at(arena, nt.start as usize);
                                        }
                                let wlen = (tw.end - tw.start) as usize;
                                let (sc, ec) = if let Some(slice) = file.path.trimmed_in_chunk(
                                    arena,
                                    tw.start as u32,
                                    tw.end as u32,
                                ) {
                                    score_scalar_window_direct(
                                        direct_short,
                                        needle,
                                        flip,
                                        slice,
                                        &p,
                                        tw.include_prefix(),
                                    )
                                } else {
                                    if cross.capacity() < wlen {
                                        cross.reserve(wlen);
                                    }
                                    // The span crosses a slot. A zero-length window is scored as empty.
                                    unsafe {
                                        if wlen > 0 {
                                            file.path.copy_crossing(
                                                arena,
                                                tw.start as usize,
                                                tw.end as usize,
                                                cross.as_mut_ptr(),
                                            );
                                        }
                                        let dst = std::slice::from_raw_parts(cross.as_ptr(), wlen);
                                        score_scalar_window_direct(
                                            direct_short,
                                            needle,
                                            flip,
                                            dst,
                                            &p,
                                            tw.include_prefix(),
                                        )
                                    }
                                };
                                        sc_out[k] = sc;
                                        ec_out[k] = ec;
                                    }
                                };
                            }
                            match &working_files {
                                crate::matcher::FileItems::All(files) => score_small!(at_all, files),
                                crate::matcher::FileItems::Filtered { base, idx } => {
                                    score_small!(at_filtered, base, idx)
                                }
                            }
                            CROSS.with(|c| *c.borrow_mut() = cross);
                            if let Some(t_scalar) = t_scalar {
                                t_scalar_us.fetch_add(
                                    t_scalar.elapsed().as_micros() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            return;
                        }
                        // u8 path only at full width; halves are u16.
                        let u8path = use_u8 && width == 32;
                        let rows = max_len.div_ceil(8) * 8;
                        thread_local! {
                            static SOA: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
                            static STAGE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
                        }
                        let buf_len = rows * width;
                        let t_stage = timing.then(std::time::Instant::now);
                        let mut stage = STAGE.with(|c| {
                            let mut b = c.borrow_mut();
                            if b.len() < buf_len {
                                b.resize(buf_len, 0);
                            }
                            std::mem::take(&mut *b)
                        });
                        stage.truncate(buf_len);
                        // Byte-wide x86 lanes are independent and unused lanes are not stored.
                        // The unused lanes are one contiguous suffix of the stage.
                        if !(use_u8 && width == 32) && n < width {
                            stage[n * rows..width * rows].fill(0);
                        }

                        let mut prefix_mask: u32 = 0;
                        // Every window of a short query starts at byte 0.
                        let whole_path = gate_state.full_path_windows();
                        macro_rules! stage_hits {
                            ($simd:tt, $pad:tt, $pre:tt, $at:ident, $($src:expr),+) => {{
                                let mut slot = stage.as_mut_ptr();
                                for k in 0..n {
                                    let oi = oi_at(k);
                                    let t = unsafe { trims.get_unchecked(oi as usize) };
                                    let (fi, s, e) = (t.file_idx, t.start, t.end);
                                    let file = crate::matcher::FileItems::$at($($src),+, fi as usize);
                                    #[cfg(target_arch = "x86_64")]
                                    {
                                        // The next window covers the copy. Four ahead covers the line fill.
                                        if k + 1 < n {
                                            let nt = unsafe { trims.get_unchecked(oi_at(k + 1) as usize) };
                                            let nf = crate::matcher::FileItems::$at(
                                                $($src),+,
                                                nt.file_idx as usize,
                                            );
                                            nf.path.prefetch_at(arena, nt.start as usize);
                                        }
                                        if k + 4 < n {
                                            let nt = unsafe { trims.get_unchecked(oi_at(k + 4) as usize) };
                                            let nf = crate::matcher::FileItems::$at(
                                                $($src),+,
                                                nt.file_idx as usize,
                                            );
                                            nf.path.prefetch_at(arena, nt.start as usize);
                                        }
                                    }
                            let wlen = (e - s) as usize;
                            // `wlen <= max_len <= rows`, so the window and its padding stay in `stage`.
                            debug_assert!(slot as usize + max_len <= stage.as_ptr() as usize + stage.len());
                            // The copy writes the window. The tail past it is the batch padding.
                            unsafe {
                                let dst = slot;
                                #[cfg(target_arch = "x86_64")]
                                {
                                    if $simd {
                                        file.path.copy_window_avx2(
                                            arena,
                                            s as usize,
                                            e as usize,
                                            dst,
                                        );
                                    } else if !file.path.copy_in_slot(
                                        arena,
                                        s as u32,
                                        wlen,
                                        dst,
                                    ) {
                                        file.path.copy_crossing(arena, s as usize, e as usize, dst);
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                if !file.path.copy_in_slot(arena, s as u32, wlen, dst) {
                                    file.path.copy_crossing(arena, s as usize, e as usize, dst);
                                }
                                // A same-length batch already fills every column, so that loop skips this.
                                if $pad {
                                    if wlen < max_len {
                                        dst.add(wlen).write_bytes(0, max_len - wlen);
                                    }
                                }
                            }
                            // A short query sets every prefix bit after the copy.
                            if $pre {
                                if t.include_prefix() {
                                    prefix_mask |= 1u32 << k;
                                }
                            }
                            slot = unsafe { slot.add(rows) };
                                }
                            }};
                        }
                        macro_rules! stage_match {
                            ($simd:tt, $pad:tt, $pre:tt) => {
                                match &working_files {
                                    crate::matcher::FileItems::All(files) => {
                                        stage_hits!($simd, $pad, $pre, at_all, files)
                                    }
                                    crate::matcher::FileItems::Filtered { base, idx } => {
                                        stage_hits!($simd, $pad, $pre, at_filtered, base, idx)
                                    }
                                }
                            };
                        }
                        macro_rules! stage_kind {
                            ($simd:tt) => {
                                if one_len {
                                    if whole_path {
                                        stage_match!($simd, false, false);
                                    } else {
                                        stage_match!($simd, false, true);
                                    }
                                } else if whole_path {
                                    stage_match!($simd, true, false);
                                } else {
                                    stage_match!($simd, true, true);
                                }
                            };
                        }
                        #[cfg(target_arch = "x86_64")]
                        {
                            if use_simd {
                                stage_kind!(true);
                            } else {
                                stage_kind!(false);
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        stage_kind!(false);
                        if whole_path {
                            // Unused high lanes are not stored, so every bit can stay set.
                            prefix_mask = u32::MAX;
                        }
                        if let Some(t_stage) = t_stage {
                            t_stage_us.fetch_add(
                                t_stage.elapsed().as_micros() as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }

                        #[cfg(target_arch = "x86_64")]
                        if use_simd {
                            let t_xpose = timing.then(std::time::Instant::now);
                            let (mut soa, soa_off) =
                                SOA.with(|c| take_line_aligned(&mut c.borrow_mut(), buf_len));
                            let soa_bytes = &mut soa[soa_off..soa_off + buf_len];
                            debug_assert_eq!(soa_bytes.as_ptr() as usize & 63, 0);
                            // SAFETY: both buffers hold exactly rows*width bytes; rows/width are multiples of 8.
                            unsafe {
                                let lanes = if u8path { n.div_ceil(8) * 8 } else { width };
                                crate::interseq_sw::avx2::soa_from_stage(
                                    &stage, rows, width, lanes, soa_bytes,
                                );
                            }
                            if let Some(t_xpose) = t_xpose {
                                t_xpose_us.fetch_add(
                                    t_xpose.elapsed().as_micros() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            let t_kernel = timing.then(std::time::Instant::now);
                            // SAFETY: AVX2 gated by `use_simd`; outputs are sized to the lane count.
                            unsafe {
                                if u8path {
                                    crate::interseq_sw::avx2::with_cached_u8_fold(
                                        needle,
                                        &p,
                                        |fold| {
                                            crate::interseq_sw::avx2::score_batch_avx2_u8_dispatch::<false>(
                                                needle,
                                                soa_bytes,
                                                max_len,
                                                &p,
                                                prefix_mask,
                                                &mut sc_out[..n],
                                                &mut ec_out[..n],
                                                Some(fold),
                                            );
                                        },
                                    );
                                } else {
                                    crate::interseq_sw::avx2::score_batch_avx2(
                                        needle,
                                        soa_bytes,
                                        max_len,
                                        &p,
                                        prefix_mask as u16,
                                        &mut sc_out[..n],
                                        &mut ec_out[..n],
                                    );
                                }
                            }
                            if let Some(t_kernel) = t_kernel {
                                t_kernel_us.fetch_add(
                                    t_kernel.elapsed().as_micros() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            SOA.with(|c| *c.borrow_mut() = soa);
                            STAGE.with(|c| *c.borrow_mut() = stage);
                            return;
                        }
                        // First-class ARM path: scalar transpose (no SSE2
                        // tiles), NEON u8 kernel, same saturation rescore.
                        // Chunk-level max_trim gate above keeps max_len in
                        // byte range, so no wide fallback is needed here.
                        #[cfg(target_arch = "aarch64")]
                        if use_simd {
                            let t_xpose = timing.then(std::time::Instant::now);
                            let mut soa = SOA.with(|c| {
                                let mut b = c.borrow_mut();
                                if b.len() < buf_len {
                                    b.resize(buf_len, 0);
                                }
                                std::mem::take(&mut *b)
                            });
                            soa.truncate(buf_len);
                            crate::interseq_sw::soa_from_stage_scalar(
                                &stage, rows, width, &mut soa,
                            );
                            if let Some(t_xpose) = t_xpose {
                                t_xpose_us.fetch_add(
                                    t_xpose.elapsed().as_micros() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            let t_kernel = timing.then(std::time::Instant::now);
                            // SAFETY: NEON is baseline on aarch64; outputs
                            // are sized to the lane count.
                            unsafe {
                                crate::interseq_sw::neon::score_batch_neon_u8(
                                    needle,
                                    &soa,
                                    max_len,
                                    &p,
                                    prefix_mask as u16,
                                    &mut sc_out[..n],
                                    &mut ec_out[..n],
                                );
                            }
                            if let Some(t_kernel) = t_kernel {
                                t_kernel_us.fetch_add(
                                    t_kernel.elapsed().as_micros() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            SOA.with(|c| *c.borrow_mut() = soa);
                            STAGE.with(|c| *c.borrow_mut() = stage);
                            return;
                        }
                        let t_scalar = timing.then(std::time::Instant::now);
                        let mut flip_buf = [0u8; 256];
                        let flip = scalar_flip(needle, &mut flip_buf);
                        let direct_short =
                            !needle.is_empty() && needle.len() <= 32 && flip.len() == needle.len();
                        let mut slot = stage.as_ptr();
                        for k in 0..n {
                            let oi = oi_at(k);
                            let tw = unsafe { trims.get_unchecked(oi as usize) };
                            let (s, e, inc_prefix) = (tw.start, tw.end, tw.include_prefix());
                            let wlen = (e - s) as usize;
                            // The staged window is `rows` bytes. Only the live prefix is scored.
                            let src = unsafe { std::slice::from_raw_parts(slot, wlen) };
                            slot = unsafe { slot.add(rows) };
                            let (sc, ec) = score_scalar_window_direct(
                                direct_short, needle, flip, src, &p, inc_prefix,
                            );
                            sc_out[k] = sc;
                            ec_out[k] = ec;
                        }
                        if let Some(t_scalar) = t_scalar {
                            t_scalar_us.fetch_add(
                                t_scalar.elapsed().as_micros() as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                        STAGE.with(|c| *c.borrow_mut() = stage);
                    };
                    // Over-long batches run the exact u16 path in 16-wide halves.
                    let score_batch = |ord: &[u32], sc_out: &mut [u16], ec_out: &mut [u16]| {
                        // Uniform groups already recorded the only length.
                        let max_len = if one_len {
                            seen_len
                        } else {
                            let mut max_len = 0usize;
                            for &oi in ord.iter() {
                                let t = unsafe { trims.get_unchecked(oi as usize) };
                                let (s, e) = (t.start, t.end);
                                max_len = max_len.max((e - s) as usize);
                            }
                            max_len
                        };
                        if max_len == 0 {
                            sc_out.fill(0);
                            ec_out.fill(0);
                            return;
                        }
                        if use_u8 && max_len > 256 {
                            for ((o, s), e) in ord
                                .chunks(16)
                                .zip(sc_out.chunks_mut(16))
                                .zip(ec_out.chunks_mut(16))
                            {
                                score_batch_w(o, None, s, e, 16, max_len);
                            }
                        } else {
                            score_batch_w(ord, None, sc_out, ec_out, batch_sz, max_len);
                        }
                    };
                    // Same-length windows stay in input order and write scores in place.
                    if direct {
                        if max_trim == 0 {
                            scores[..n_matches].fill(0);
                            end_cols[..n_matches].fill(0);
                        } else {
                            let mut at = 0usize;
                            let lane = if use_u8 && max_trim > 256 { 16 } else { batch_sz };
                            while at < n_matches {
                                let n = lane.min(n_matches - at);
                                score_batch_w(
                                    &[],
                                    Some(at as u32),
                                    &mut scores[at..at + n],
                                    &mut end_cols[at..at + n],
                                    lane,
                                    max_trim,
                                );
                                at += n;
                            }
                        }
                    } else {
                    let t_prep = timing.then(std::time::Instant::now);
                    let mut order = ORDER.with(|c| take_tls_init(c, n_matches));
                    let t_sort = timing.then(std::time::Instant::now);
                    // SAFETY: a large group with mixed lengths initialized `cnt`.
                    let cnt = unsafe { cnt.assume_init_ref() };
                    let mut start = std::mem::MaybeUninit::<[u32; 129]>::uninit();
                    // SAFETY: the loop writes every slot before a slot is read.
                    let start = unsafe { &mut *start.as_mut_ptr() };
                    let mut acc = 0u32;
                    for l in 0..129 {
                        start[l] = acc;
                        acc += cnt[l];
                    }
                    let long_lo = start[128] as usize;
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..n_matches {
                        let t = unsafe { trims.get_unchecked(i) };
                        let (s, e) = (t.start, t.end);
                        let l = (e - s).min(128) as usize;
                        order[start[l] as usize] = i as u32;
                        start[l] += 1;
                    }
                    let long_hi = start[128] as usize;
                    // Lengths ≥128 share one bucket; order that run by exact length.
                    if long_hi - long_lo > batch_sz {
                        let mut lo_l = u16::MAX;
                        let mut hi_l = 0u16;
                        for &i in &order[long_lo..long_hi] {
                            let t = trims[i as usize];
                            let l = t.end.wrapping_sub(t.start);
                            lo_l = lo_l.min(l);
                            hi_l = hi_l.max(l);
                        }
                        if hi_l != lo_l {
                            order[long_lo..long_hi].sort_unstable_by_key(|&i| {
                                let t = trims[i as usize];
                                t.end.wrapping_sub(t.start)
                            });
                        }
                    }
                    if let Some(t_sort) = t_sort {
                        t_sort_us.fetch_add(
                            t_sort.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    let mut perm_scores = PERM_SCORES.with(|c| take_tls_init(c, n_matches + 32));
                    let mut perm_ecols = PERM_ECOLS.with(|c| take_tls_init(c, n_matches + 32));
                    fn align64_u16(buf: &mut [u16], n: usize) -> &mut [u16] {
                        let addr = buf.as_ptr() as usize;
                        let pad = ((64 - (addr % 64)) % 64) / 2;
                        &mut buf[pad..pad + n]
                    }
                    let ps_aligned = align64_u16(&mut perm_scores, n_matches);
                    let pe_aligned = align64_u16(&mut perm_ecols, n_matches);
                    if let Some(t_prep) = t_prep {
                        t_prep_us.fetch_add(
                            t_prep.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    // Break-even only when the outer join left workers idle. No clock otherwise.
                    let n_batches = order.len().div_ceil(batch_sz);
                    let fan_parallel = allow_inner
                        && crate::dispatch::parallel_batches(
                            n_batches,
                            crate::dispatch::batch_ns_estimate(use_u8),
                            crate::machine::search_threads(),
                        );
                    if fan_parallel {
                        if timing {
                            t_fan_parallel
                                .fetch_add(n_batches as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                        let t_fan = std::time::Instant::now();
                        crate::parallelism::run_on_search_pool(|| {
                            use rayon::prelude::*;
                            order
                                .par_chunks(batch_sz)
                                .zip(ps_aligned.par_chunks_mut(batch_sz))
                                .zip(pe_aligned.par_chunks_mut(batch_sz))
                                .for_each(|((ord, sc_out), ec_out)| {
                                    score_batch(ord, sc_out, ec_out);
                                });
                        });
                        crate::dispatch::observe_batch_ns(
                            use_u8,
                            t_fan.elapsed().as_nanos() as f64,
                            n_batches,
                        );
                    } else {
                        if timing {
                            t_fan_serial
                                .fetch_add(n_batches as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                        for ((ord, sc_out), ec_out) in order
                            .chunks(batch_sz)
                            .zip(ps_aligned.chunks_mut(batch_sz))
                            .zip(pe_aligned.chunks_mut(batch_sz))
                        {
                            score_batch(ord, sc_out, ec_out);
                        }
                    }

                    // Scatter through the inverse order map into chunk-local scores.
                    let t_scatter = timing.then(std::time::Instant::now);
                    for (i, &oi) in order.iter().enumerate() {
                        scores[oi as usize] = ps_aligned[i];
                        end_cols[oi as usize] = pe_aligned[i];
                    }
                    if let Some(t_scatter) = t_scatter {
                        t_scatter_us.fetch_add(
                            t_scatter.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    ORDER.with(|c| reclaim(&mut c.borrow_mut(), order));
                    PERM_SCORES.with(|c| reclaim(&mut c.borrow_mut(), perm_scores));
                    PERM_ECOLS.with(|c| reclaim(&mut c.borrow_mut(), perm_ecols));
                    }
                    if let Some(t_batch) = t_batch {
                        t_batch_us.fetch_add(
                            t_batch.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                }

                let t_asm = timing.then(std::time::Instant::now);
                macro_rules! assemble_hits {
                    ($at:ident, $($src:expr),+) => {{
                        let mut exact_at = 0usize;
                        for (i, m) in matches_chunk.iter_mut().enumerate() {
                            // One trim was stored per match, in this order.
                            let t = unsafe { trims.get_unchecked(i) };
                            let (fi, start_adj, end_pos, include_exact) =
                                (t.file_idx, t.start as u32, t.end as u32, t.include_exact());
                            // The stored end is the path length, so a different length cannot be exact.
                            let exact = include_exact && (end_pos as usize) == needle.len() && {
                                let file = crate::matcher::FileItems::$at($($src),+, fi as usize);
                                if needle_folded {
                                    file.path.window_eq_lower_ready(arena, 0, needle_lower)
                                } else {
                                    file.path.window_eq_ignore_ascii_ready(arena, 0, needle)
                                }
                            };
                    let bonus = if exact { 8u16 } else { 0 };
                    m.score = scores[i].saturating_add(bonus);
                    m.end_col = end_cols[i].saturating_add(start_adj.min(u16::MAX as u32) as u16);
                    m.exact = exact;
                            if exact_at < exact_hits.len() && exact_hits[exact_at] == m.index {
                                m.score = m.score.max(200);
                                m.exact = true;
                                exact_at += 1;
                            }
                        }
                    }};
                }
                match &working_files {
                    crate::matcher::FileItems::All(files) => assemble_hits!(at_all, files),
                    crate::matcher::FileItems::Filtered { base, idx } => {
                        assemble_hits!(at_filtered, base, idx)
                    }
                }
                if let Some(t_asm) = t_asm {
                    t_assemble_us.fetch_add(
                        t_asm.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }

                TRIMS.with(|c| reclaim(&mut c.borrow_mut(), trims));
                SCORES.with(|c| reclaim(&mut c.borrow_mut(), scores));
                ECOLS.with(|c| reclaim(&mut c.borrow_mut(), end_cols));
            };
            // Fused groups: gate, exact, and SW per file chunk while bytes are hot.
            // The split is planned at pool width; the requested width bounds execution,
            // never the partition (gate windows must not move with it). FFF_SW_CHUNK overrides.
            let per_file = crate::dispatch::match_bytes_per_file(
                first_part.len(),
                max_path_len,
                first_options.max_typos,
            );
            let pool = crate::parallelism::effective_core_count().max(1);
            let plan = crate::dispatch::plan_files(gate_n, per_file, pool);
            // Serial cutoff comes from the plan: fork/join costs more than the body.
            let serial = gate_n <= plan.chunk;
            let file_chunk: usize = crate::dispatch::sw_chunk_override().unwrap_or_else(|| {
                // Over-partition for work-stealing slack; floor bounds per-group costs.
                (plan.chunk / 4).max(1024)
            });
            let n_groups = gate_n.div_ceil(file_chunk);
            // Same split at every width. Inner batches stay serial once groups fill workers.
            let allow_inner = crate::dispatch::allow_inner_sw_fanout(n_groups, pool, max_threads);
            // Per worker. The next group reserves the most accepts this worker has seen.
            thread_local! {
                static GROUP_ACCEPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
            }
            let run_group = |g: usize, m: &mut Vec<neo_frizbee::Match>| {
                // Newer keystrokes own the box: drop aborted slices unpaid.
                if abort.aborted() {
                    return;
                }
                let mut exact_buf = take_exact_hits();
                let lo = g * file_chunk;
                let hi = ((g + 1) * file_chunk).min(gate_n);
                let t_gate = part_timing.then(std::time::Instant::now);
                let span = hi - lo;
                let guess = GROUP_ACCEPTS.with(|c| c.get()).min(span);
                let start = m.len();
                m.reserve(guess);
                // Per-group prefix-walk cache (groups run on different threads).
                let mut pcache = crate::gate::PrefixWalkCache::new();
                pcache.set_enabled(prefix_cache_on);
                let mut g_files = 0u64;
                let mut g_accepts = 0u64;
                let mut g_path = 0u64;
                let mut g_prej = 0u64;
                let mut saw_before_name = false;
                macro_rules! record_hit {
                    ($timing:tt, $file:expr, $hit:expr, $idx:expr) => {{
                        let file = $file;
                        if $timing {
                            g_files += 1;
                            g_path += file.path.byte_len as u64;
                        }
                        if let Some(hit) = $hit {
                            if $timing {
                                g_accepts += 1;
                            }
                            let name_span = if looks_like_filename {
                                Some(file.path.span_lens())
                            } else {
                                None
                            };
                            if !saw_before_name {
                                if let Some(flag) = before_filename {
                                    // Widened ends sit at EOF. A start before the name is a directory hit.
                                    let split = match name_span {
                                        Some((_, name_off)) => name_off as u32,
                                        None => file.path.filename_offset as u32,
                                    };
                                    if hit.start < split.saturating_sub(1) {
                                        saw_before_name = true;
                                        flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    }
                                }
                            }
                            // `exact` means the window already ends at the path. Scoring replaces it.
                            let path_len = match name_span {
                                Some((byte_len_u, _)) => byte_len_u as usize,
                                None => file.path.byte_len as usize,
                            };
                            m.push(neo_frizbee::Match {
                                score: hit.start as u16,
                                index: $idx,
                                exact: hit.end as usize == path_len,
                                end_col: hit.end as u16,
                            });
                            if let Some((byte_len_u, name_off)) = name_span {
                                let flen = byte_len_u as usize - name_off as usize;
                                if flen == needle.len() {
                                    let name_hit = if needle_folded {
                                        file.path.window_eq_lower_ready(
                                            arena,
                                            name_off as u32,
                                            needle_lower,
                                        )
                                    } else {
                                        file.path.window_eq_ignore_ascii_ready(
                                            arena,
                                            name_off as u32,
                                            needle,
                                        )
                                    };
                                    if name_hit {
                                        exact_buf.push($idx);
                                    }
                                }
                            }
                        }
                    }};
                }
                macro_rules! visit_file {
                    ($timing:tt, $file:expr, $idx:expr, cached) => {{
                        let file = $file;
                        record_hit!(
                            $timing,
                            file,
                            gate_state.hit_file_cached(file, arena, &mut pcache),
                            $idx
                        );
                    }};
                    ($timing:tt, $file:expr, $idx:expr, live) => {{
                        let file = $file;
                        record_hit!(
                            $timing,
                            file,
                            gate_state.hit_file_live::<false>(file, arena, &mut pcache),
                            $idx
                        );
                    }};
                    ($timing:tt, $file:expr, $idx:expr, live_del) => {{
                        let file = $file;
                        record_hit!(
                            $timing,
                            file,
                            gate_state.hit_file_live::<true>(file, arena, &mut pcache),
                            $idx
                        );
                    }};
                }
                // Timing is fixed for the search. The file loop does not branch on it.
                macro_rules! scan_files {
                    ($timing:tt) => {
                        if direct_mask {
                    // Needle shape and tombstones are fixed for the whole scan.
                    if gate_state.live_scan() {
                        let deletes = gate_state.deletes_checked();
                        if np.distinct() {
                            let needle_mask = np.distinct_mask();
                            let need_hits = np.min_hits();
                            let cover = np.cover_eq();
                            let bits = |k: usize| {
                                presence_bits16_distinct(
                                    presence, k, needle_mask, need_hits, cover,
                                )
                            };
                            let miss = |mask: u64| mask != 0 && np.misses_distinct(mask);
                            if deletes {
                                walk_present::<$timing>(
                                    all_files,
                                    presence,
                                    lo,
                                    hi,
                                    arena,
                                    bits,
                                    miss,
                                    |file, idx| visit_file!($timing, file, idx, live_del),
                                    &mut g_prej,
                                );
                            } else {
                                walk_present::<$timing>(
                                    all_files,
                                    presence,
                                    lo,
                                    hi,
                                    arena,
                                    bits,
                                    miss,
                                    |file, idx| visit_file!($timing, file, idx, live),
                                    &mut g_prej,
                                );
                            }
                        } else {
                            let bits =
                                |k: usize| presence_bits16_repeated(presence, k, &np, needle);
                            let miss =
                                |mask: u64| mask != 0 && !np.allows_repeated(mask, needle);
                            if deletes {
                                walk_present::<$timing>(
                                    all_files,
                                    presence,
                                    lo,
                                    hi,
                                    arena,
                                    bits,
                                    miss,
                                    |file, idx| visit_file!($timing, file, idx, live_del),
                                    &mut g_prej,
                                );
                            } else {
                                walk_present::<$timing>(
                                    all_files,
                                    presence,
                                    lo,
                                    hi,
                                    arena,
                                    bits,
                                    miss,
                                    |file, idx| visit_file!($timing, file, idx, live),
                                    &mut g_prej,
                                );
                            }
                        }
                    } else {
                        walk_present::<$timing>(
                            all_files,
                            presence,
                            lo,
                            hi,
                            arena,
                            |k| presence_bits16_unknown(presence, k, &np, needle),
                            |mask| !crate::presence::allows_or_unknown(&np, mask, needle),
                            |file, idx| visit_file!($timing, file, idx, cached),
                            &mut g_prej,
                        );
                    }
                } else if use_cand {
                    // Candidates already passed presence. The list shape is fixed for the group.
                    match &working_files {
                        crate::matcher::FileItems::All(files) => walk_candidates(
                            lo,
                            hi,
                            &cand,
                            arena,
                            |i| crate::matcher::FileItems::at_all(files, i),
                            |file, idx| visit_file!($timing, file, idx, cached),
                        ),
                        crate::matcher::FileItems::Filtered { base, idx } => walk_candidates(
                            lo,
                            hi,
                            &cand,
                            arena,
                            |i| crate::matcher::FileItems::at_filtered(base, idx, i),
                            |file, idx| visit_file!($timing, file, idx, cached),
                        ),
                    }
                } else {
                    macro_rules! skip_file {
                        () => {
                            |k: usize| {
                                if check_presence
                                    && !crate::presence::allows_or_unknown(
                                        &np,
                                        scanned_presence(working_files, presence, k),
                                        needle,
                                    )
                                {
                                    if $timing {
                                        g_prej += 1;
                                    }
                                    true
                                } else {
                                    false
                                }
                            }
                        };
                    }
                    match &working_files {
                        crate::matcher::FileItems::All(files) => walk_span(
                            lo,
                            hi,
                            check_presence,
                            arena,
                            |k| crate::matcher::FileItems::at_all(files, k),
                            skip_file!(),
                            |file, k| visit_file!($timing, file, k, cached),
                        ),
                        crate::matcher::FileItems::Filtered { base, idx } => walk_span(
                            lo,
                            hi,
                            check_presence,
                            arena,
                            |k| crate::matcher::FileItems::at_filtered(base, idx, k),
                            skip_file!(),
                            |file, k| visit_file!($timing, file, k, cached),
                        ),
                    }
                }
                    };
                }
                // The timing scan is a second copy of the file loop. Keep it out of this function.
                if part_timing {
                    run_timed_scan(|| scan_files!(true));
                } else {
                    scan_files!(false);
                }
                if let Some(t_gate) = t_gate {
                    t_prefix_hits.fetch_add(pcache.hits, std::sync::atomic::Ordering::Relaxed);
                    t_prefix_fills.fetch_add(pcache.fills, std::sync::atomic::Ordering::Relaxed);
                    t_prefix_skip.fetch_add(
                        pcache.bytes_skipped,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    t_tight_trim.fetch_add(pcache.tight_trim, std::sync::atomic::Ordering::Relaxed);
                    t_tight_files.fetch_add(pcache.tight_files, std::sync::atomic::Ordering::Relaxed);
                    t_gate_files.fetch_add(g_files, std::sync::atomic::Ordering::Relaxed);
                    t_gate_accepts.fetch_add(g_accepts, std::sync::atomic::Ordering::Relaxed);
                    t_gate_path_bytes.fetch_add(g_path, std::sync::atomic::Ordering::Relaxed);
                    t_gate_walked.fetch_add(pcache.walked, std::sync::atomic::Ordering::Relaxed);
                    t_presence_rejects.fetch_add(g_prej, std::sync::atomic::Ordering::Relaxed);
                    t_gate_us.fetch_add(
                        t_gate.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                let t_swslice = part_timing.then(std::time::Instant::now);
                sw_body(&mut m[start..], &exact_buf, allow_inner);
                reclaim_exact_hits(exact_buf);
                if let Some(t_swslice) = t_swslice {
                    t_swslice_us.fetch_add(
                        t_swslice.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                let n_acc = m.len() - start;
                GROUP_ACCEPTS.with(|c| {
                    if n_acc > c.get() {
                        c.set(n_acc);
                    }
                });
            };
            let one_group = |g: usize| {
                let mut m = take_group_match_buf();
                run_group(g, &mut m);
                m
            };
            let mut matches: Vec<neo_frizbee::Match> = take_match_buf();
            if serial {
                for g in 0..n_groups {
                    run_group(g, &mut matches);
                }
            } else if max_threads >= crate::machine::search_threads() {
                // Stay on the search pool. The global pool would add a second set of workers.
                crate::parallelism::run_on_search_pool(|| {
                    use rayon::prelude::*;
                    let mut scored: Vec<Vec<neo_frizbee::Match>> =
                        (0..n_groups).into_par_iter().map(one_group).collect();
                    let n_out: usize = scored.iter().map(|v| v.len()).sum();
                    matches.reserve(n_out);
                    for v in &mut scored {
                        matches.append(v);
                    }
                    reclaim_group_match_bufs(scored);
                });
            } else {
                // Same groups, one join. Width caps workers, not the split.
                let width = max_threads.max(1);
                let mut parts = crate::parallelism::map_search_width(n_groups, width, one_group);
                let n_out: usize = parts.iter().map(|v| v.len()).sum();
                matches.reserve(n_out);
                for v in &mut parts {
                    matches.append(v);
                }
                reclaim_group_match_bufs(parts);
            }

            if part_timing {
                let cand_ms = _cand_clock
                    .map(|t| t.elapsed().as_secs_f64() * 1e3)
                    .unwrap_or(0.0);
                eprintln!("PHASE cand_ms={cand_ms:.1} n={}", matches.len());
                let sw_us = t_sw.map(|t| t.elapsed().as_micros()).unwrap_or(0);
                let gate_us = t_gate_us.load(std::sync::atomic::Ordering::Relaxed);
                let swslice_us = t_swslice_us.load(std::sync::atomic::Ordering::Relaxed);
                let prefix_hit = t_prefix_hits.load(std::sync::atomic::Ordering::Relaxed);
                let prefix_fill = t_prefix_fills.load(std::sync::atomic::Ordering::Relaxed);
                let prefix_skip_kb = t_prefix_skip.load(std::sync::atomic::Ordering::Relaxed) / 1024;
                let gate_files = t_gate_files.load(std::sync::atomic::Ordering::Relaxed);
                let gate_acc = t_gate_accepts.load(std::sync::atomic::Ordering::Relaxed);
                let pres_rej = t_presence_rejects.load(std::sync::atomic::Ordering::Relaxed);
                let gate_walk_kb = t_gate_walked.load(std::sync::atomic::Ordering::Relaxed) / 1024;
                let gate_path_kb = t_gate_path_bytes.load(std::sync::atomic::Ordering::Relaxed) / 1024;
                eprintln!(
                    "PHASE parts gate_us={gate_us} swslice_us={swslice_us} fuse_us={sw_us} threads={threads} n={} files_n={n_files} use_cand={use_cand} cand_n={gate_n} gate_files={gate_files} gate_acc={gate_acc} pres_rej={pres_rej} gate_walk_kb={gate_walk_kb} gate_path_kb={gate_path_kb} prefix_hit={prefix_hit} prefix_fill={prefix_fill} prefix_skip_kb={prefix_skip_kb}",
                    matches.len()
                );
                // Census: mean trimmed bytes per acceptor sizes the SW prize.
                let tight_files = t_tight_files.load(std::sync::atomic::Ordering::Relaxed);
                if tight_files > 0 {
                    let trim = t_tight_trim.load(std::sync::atomic::Ordering::Relaxed);
                    eprintln!("PHASE tight files={tight_files} trim_kb={}", trim / 1024);
                }
                let prefilter_ms = _prefilter_clock
                    .map(|t| t.elapsed().as_secs_f64() * 1e3)
                    .unwrap_or(0.0);
                let collect_ms = t_collect_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let batch_ms = t_batch_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let assemble_ms =
                    t_assemble_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let stage_ms = t_stage_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let xpose_ms = t_xpose_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let kernel_ms = t_kernel_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let scalar_ms = t_scalar_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let prep_ms = t_prep_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let scatter_ms = t_scatter_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let sort_ms = t_sort_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
                let fan_serial = t_fan_serial.load(std::sync::atomic::Ordering::Relaxed);
                let fan_parallel = t_fan_parallel.load(std::sync::atomic::Ordering::Relaxed);
                eprintln!(
                    "PHASE interseq prefilter_ms={prefilter_ms:.1} collect_ms={collect_ms:.1} batch_ms={batch_ms:.1} assemble_ms={assemble_ms:.1} n={} stage_ms={stage_ms:.1} xpose_ms={xpose_ms:.1} kernel_ms={kernel_ms:.1} scalar_ms={scalar_ms:.1} prep_ms={prep_ms:.1} scatter_ms={scatter_ms:.1} sort_ms={sort_ms:.1} fan_serial={fan_serial} fan_parallel={fan_parallel}",
                    matches.len()
                );
            }
            matches
        }
    };

    first_part_matches
}

// The last query's match list. The next keystroke reuses the allocation.
thread_local! {
    static MATCH_BUF: std::cell::RefCell<Vec<neo_frizbee::Match>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn take_match_buf() -> Vec<neo_frizbee::Match> {
    MATCH_BUF.with(|slot| {
        let mut buf = std::mem::take(&mut *slot.borrow_mut());
        buf.clear();
        buf
    })
}

fn reclaim_match_buf(mut buf: Vec<neo_frizbee::Match>) {
    buf.clear();
    MATCH_BUF.with(|slot| {
        let slot = &mut *slot.borrow_mut();
        if buf.capacity() > slot.capacity() {
            *slot = buf;
        }
    });
}

// Group lists from the parallel scan. Workers borrow one and return it after the join.
static GROUP_MATCH_BUFS: parking_lot::Mutex<Vec<Vec<neo_frizbee::Match>>> =
    parking_lot::Mutex::new(Vec::new());

fn take_group_match_buf() -> Vec<neo_frizbee::Match> {
    if let Some(mut buf) = GROUP_MATCH_BUFS.lock().pop() {
        buf.clear();
        buf
    } else {
        Vec::new()
    }
}

fn reclaim_group_match_bufs(bufs: Vec<Vec<neo_frizbee::Match>>) {
    let mut pool = GROUP_MATCH_BUFS.lock();
    for buf in bufs {
        if pool.len() >= 64 {
            break;
        }
        pool.push(buf);
    }
}

// Exact repeat shares the match list. Scores are not stored.
// Generation, query, typos, and the candidate list are compared exactly.
const MATCH_CACHE_MAX_IDX: usize = 1_048_576;
const MATCH_CACHE_MAX_MATCHES: usize = 1_048_576;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatchSide {
    Base,
    Overflow,
}

struct CachedMatchSet {
    idx: std::sync::Arc<Vec<u32>>,
    matches: std::sync::Arc<Vec<neo_frizbee::Match>>,
}

// Owned when the list is not stored; shared on a hit or a store.
enum HeldMatches {
    Owned(Vec<neo_frizbee::Match>),
    Shared(std::sync::Arc<Vec<neo_frizbee::Match>>),
}

impl HeldMatches {
    fn as_slice(&self) -> &[neo_frizbee::Match] {
        match self {
            HeldMatches::Owned(v) => v.as_slice(),
            HeldMatches::Shared(v) => v.as_slice(),
        }
    }
}

impl Drop for HeldMatches {
    fn drop(&mut self) {
        if let HeldMatches::Owned(buf) = self {
            reclaim_match_buf(std::mem::take(buf));
        }
    }
}

#[derive(Default)]
pub(crate) struct MatchCache {
    generation: u64,
    query: Vec<String>,
    max_typos: u16,
    base: Option<CachedMatchSet>,
    overflow: Option<CachedMatchSet>,
    #[cfg(test)]
    hits: std::cell::Cell<u64>,
}

impl MatchCache {
    pub(crate) fn clear(&mut self) {
        self.base = None;
        self.overflow = None;
    }

    #[inline]
    fn slot(&self, side: MatchSide) -> &Option<CachedMatchSet> {
        match side {
            MatchSide::Base => &self.base,
            MatchSide::Overflow => &self.overflow,
        }
    }

    #[inline]
    fn slot_mut(&mut self, side: MatchSide) -> &mut Option<CachedMatchSet> {
        match side {
            MatchSide::Base => &mut self.base,
            MatchSide::Overflow => &mut self.overflow,
        }
    }

    #[inline]
    fn key_matches(&self, generation: u64, parts: &[&str], max_typos: u16) -> bool {
        self.generation == generation
            && self.max_typos == max_typos
            && self.query.len() == parts.len()
            && self
                .query
                .iter()
                .zip(parts.iter())
                .all(|(cached, part)| cached == part)
    }

    fn get(
        &self,
        generation: u64,
        parts: &[&str],
        max_typos: u16,
        side: MatchSide,
        idx: &[u32],
    ) -> Option<std::sync::Arc<Vec<neo_frizbee::Match>>> {
        if !self.key_matches(generation, parts, max_typos) {
            return None;
        }
        let entry = self.slot(side).as_ref()?;
        if entry.idx.as_slice() == idx {
            #[cfg(test)]
            self.hits.set(self.hits.get() + 1);
            Some(std::sync::Arc::clone(&entry.matches))
        } else {
            None
        }
    }

    fn get_list(
        &self,
        generation: u64,
        parts: &[&str],
        max_typos: u16,
        side: MatchSide,
        idx: &crate::matcher::CandList,
    ) -> Option<std::sync::Arc<Vec<neo_frizbee::Match>>> {
        if !self.key_matches(generation, parts, max_typos) {
            return None;
        }
        let entry = self.slot(side).as_ref()?;
        let same = match idx {
            crate::matcher::CandList::Shared(arc) => {
                std::sync::Arc::ptr_eq(arc, &entry.idx) || entry.idx.as_slice() == arc.as_slice()
            }
            crate::matcher::CandList::Owned(v) => entry.idx.as_slice() == v.as_slice(),
        };
        if same {
            #[cfg(test)]
            self.hits.set(self.hits.get() + 1);
            Some(std::sync::Arc::clone(&entry.matches))
        } else {
            None
        }
    }

    // Keep the stored query strings. A new keystroke overwrites them.
    fn store_query(&mut self, parts: &[&str]) {
        if self.query.len() > parts.len() {
            self.query.truncate(parts.len());
        }
        for (i, part) in parts.iter().enumerate() {
            if i < self.query.len() {
                self.query[i].clear();
                self.query[i].push_str(part);
            } else {
                self.query.push((*part).to_string());
            }
        }
    }

    fn put(
        &mut self,
        generation: u64,
        parts: &[&str],
        max_typos: u16,
        side: MatchSide,
        idx: Option<std::sync::Arc<Vec<u32>>>,
        matches: Option<std::sync::Arc<Vec<neo_frizbee::Match>>>,
    ) {
        if !self.key_matches(generation, parts, max_typos) {
            self.generation = generation;
            self.store_query(parts);
            self.max_typos = max_typos;
            recycle_cached(&mut self.base);
            recycle_cached(&mut self.overflow);
        }
        // Too large to memoize; drop the slot rather than hold a big copy.
        let slot = self.slot_mut(side);
        recycle_cached(slot);
        if let (Some(idx), Some(matches)) = (idx, matches)
            && idx.len() <= MATCH_CACHE_MAX_IDX
            && matches.len() <= MATCH_CACHE_MAX_MATCHES
        {
            *slot = Some(CachedMatchSet { idx, matches });
        }
    }
}

fn recycle_cached(slot: &mut Option<CachedMatchSet>) {
    let Some(old) = slot.take() else {
        return;
    };
    if let Ok(buf) = std::sync::Arc::try_unwrap(old.matches) {
        reclaim_match_buf(buf);
    }
    if let Ok(buf) = std::sync::Arc::try_unwrap(old.idx) {
        crate::matcher::reclaim_idx_buf(buf);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MatchCacheRef<'a> {
    cache: &'a parking_lot::Mutex<MatchCache>,
    generation: u64,
    side: MatchSide,
}

// Returns the vec to the filtered-index slot unless `into_cand` took it.
struct PooledIdx(Option<Vec<u32>>);

impl PooledIdx {
    fn take() -> Self {
        let mut v = crate::matcher::take_idx_buf();
        v.clear();
        Self(Some(v))
    }

    fn into_cand(mut self) -> crate::matcher::CandList {
        let v = self.0.take().unwrap();
        crate::matcher::CandList::Owned(v)
    }
}

impl Drop for PooledIdx {
    fn drop(&mut self) {
        if let Some(v) = self.0.take() {
            crate::matcher::reclaim_idx_buf(v);
        }
    }
}

impl std::ops::Deref for PooledIdx {
    type Target = Vec<u32>;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for PooledIdx {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().unwrap()
    }
}

// Scores only `idx` for typing live-narrow; `total_matched` is the subset
// hit count. Indices are full-corpus indices, so base and overflow files
// are split and each is read through its own arena.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fuzzy_match_and_score_subset<'a>(
    files: &'a [FileItem],
    presence: &[u64],
    idx: crate::matcher::CandList,
    context: &ScoringContext,
    base_count: usize,
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    max_path_len: usize,
    abort: crate::parallelism::AbortToken<'_>,
    cache: Option<&parking_lot::Mutex<MatchCache>>,
    generation: u64,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    debug_assert!(
        idx.iter().all(|&i| (i as usize) < files.len()),
        "live-narrow candidates must be corpus-relative"
    );
    // No overflow file: indices are already base-relative.
    if base_count >= files.len() {
        let (results, total, ranked) = match_and_score_scanned(
            FileItems::Filtered { base: files, idx },
            presence,
            context,
            base_arena,
            max_path_len,
            abort,
            0,
            false,
            cache.map(|cache| MatchCacheRef {
                cache,
                generation,
                side: MatchSide::Base,
            }),
        );
        let mut out = finish_page(results, context, total, ranked);
        out.2 = total;
        return out;
    }
    let (presence_base, presence_overflow) = split_presence(presence, base_count);
    // Overflow storage is allocated only after the first overflow index.
    // Both lists are seeded from the filtered-index slot and returned there.
    let n = idx.len();
    let mut base_idx = PooledIdx::take();
    base_idx.reserve(n);
    let mut overflow_idx = PooledIdx::take();
    for (seen, &i) in idx.iter().enumerate() {
        let i = i as usize;
        if i < base_count {
            base_idx.push(i as u32);
        } else {
            if overflow_idx.capacity() == 0 {
                overflow_idx.reserve(n - seen);
            }
            overflow_idx.push((i - base_count) as u32);
        }
    }

    let mut results = Vec::new();
    let mut total = 0usize;
    let mut ranked = true;
    let mut parts = 0u8;
    if !base_idx.is_empty() {
        let (mut base_results, base_total, base_ranked) = match_and_score_scanned(
            FileItems::Filtered {
                base: &files[..base_count],
                idx: base_idx.into_cand(),
            },
            presence_base,
            context,
            base_arena,
            max_path_len,
            abort,
            0,
            false,
            cache.map(|cache| MatchCacheRef {
                cache,
                generation,
                side: MatchSide::Base,
            }),
        );
        results.append(&mut base_results);
        total += base_total;
        parts += 1;
        ranked = base_ranked;
    }
    if !overflow_idx.is_empty() {
        let (mut overflow_results, overflow_total, overflow_ranked) = match_and_score_scanned(
            FileItems::Filtered {
                base: &files[base_count..],
                idx: overflow_idx.into_cand(),
            },
            presence_overflow,
            context,
            overflow_arena,
            max_path_len,
            abort,
            base_count as u32,
            !results.is_empty(),
            cache.map(|cache| MatchCacheRef {
                cache,
                generation,
                side: MatchSide::Overflow,
            }),
        );
        results.append(&mut overflow_results);
        total += overflow_total;
        parts += 1;
        ranked = if parts == 1 { overflow_ranked } else { false };
    }
    let mut out = finish_page(results, context, total, ranked);
    out.2 = total;
    out
}

// Match + score base and overflow files, each with its own arena.
#[tracing::instrument(skip_all, level = tracing::Level::DEBUG)]
pub(crate) fn fuzzy_match_and_score_files<'a>(
    files: &'a [FileItem],
    context: &ScoringContext,
    base_count: usize,
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    max_path_len: usize,
    abort: crate::parallelism::AbortToken<'_>,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    fuzzy_match_and_score_files_impl(
        files,
        &[],
        context,
        base_count,
        base_arena,
        overflow_arena,
        max_path_len,
        abort,
    )
}

pub(crate) fn fuzzy_match_and_score_files_presence<'a>(
    files: &'a [FileItem],
    presence: &[u64],
    context: &ScoringContext,
    base_count: usize,
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    max_path_len: usize,
    abort: crate::parallelism::AbortToken<'_>,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    fuzzy_match_and_score_files_impl(
        files,
        presence,
        context,
        base_count,
        base_arena,
        overflow_arena,
        max_path_len,
        abort,
    )
}

fn fuzzy_match_and_score_files_impl<'a>(
    files: &'a [FileItem],
    presence: &[u64],
    context: &ScoringContext,
    base_count: usize,
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    max_path_len: usize,
    abort: crate::parallelism::AbortToken<'_>,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    let (presence_base, presence_overflow) = split_presence(presence, base_count);
    // Overflow files first; exact ties break by index order.
    let (results, total_matched, ranked) = if files.len() > base_count {
        let (mut results, mut total, _) = match_and_score_scanned(
            FileItems::All(&files[base_count..]),
            presence_overflow,
            context,
            overflow_arena,
            max_path_len,
            abort,
            base_count as u32,
            false,
            None,
        );

        let (base_results, base_total, _) = match_and_score_scanned(
            FileItems::All(&files[..base_count]),
            presence_base,
            context,
            base_arena,
            max_path_len,
            abort,
            0,
            true,
            None,
        );
        results.extend(base_results);
        total += base_total;
        (results, total, false)
    } else {
        match_and_score_scanned(
            FileItems::All(files),
            presence,
            context,
            base_arena,
            max_path_len,
            abort,
            0,
            false,
            None,
        )
    };

    let t_sort = phase_timing().then(std::time::Instant::now);
    let out = finish_page(results, context, total_matched, ranked);
    if let Some(t_sort) = t_sort {
        eprintln!(
            "PHASE paginate_us={} n={}",
            t_sort.elapsed().as_micros(),
            out.2
        );
    }
    out
}

// Highlight rows for one page. The next page reuses the allocation.
thread_local! {
    static OFFSET_ROWS: std::cell::RefCell<Vec<SmallVec<[(u32, u32); 4]>>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static PATH_STRS: std::cell::RefCell<Vec<&'static str>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) fn park_offset_rows(rows: std::sync::Arc<Vec<SmallVec<[(u32, u32); 4]>>>) {
    if let Ok(mut buf) = std::sync::Arc::try_unwrap(rows) {
        buf.clear();
        OFFSET_ROWS.with(|slot| {
            let slot = &mut *slot.borrow_mut();
            if buf.capacity() > slot.capacity() {
                *slot = buf;
            }
        });
    }
}

fn take_offset_rows(n: usize) -> Vec<SmallVec<[(u32, u32); 4]>> {
    let mut rows = OFFSET_ROWS.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    rows.clear();
    rows.resize(n, SmallVec::new());
    rows
}

pub(crate) fn fuzzy_match_byte_offsets_for_page<'q>(
    query: &'q FFFQuery<'q>,
    bytes: &[u8],
    rows: &[crate::types::CachedFileRow],
    max_typos: u16,
) -> Vec<SmallVec<[(u32, u32); 4]>> {
    let mut one_part = [""; 1];
    let mut many_parts: Vec<&str> = Vec::new();
    let parts: &[&str] = match &query.fuzzy_query {
        FuzzyQuery::Text(text) if text.len() >= 2 => {
            one_part[0] = text;
            &one_part
        }
        FuzzyQuery::Parts(parts) => {
            many_parts.extend(parts.iter().copied().filter(|p| p.len() >= 2));
            &many_parts
        }
        _ => &[],
    };

    let mut ranges_by_item = take_offset_rows(rows.len());
    if parts.is_empty() || rows.is_empty() {
        return ranges_by_item;
    }

    // Page snapshot already holds each path. Highlight reads that buffer.
    let mut path_strs: Vec<&str> = PATH_STRS.with(|slot| {
        let buf = std::mem::take(&mut *slot.borrow_mut());
        // Cleared before it is stored again, so no stale pointer is kept.
        unsafe { std::mem::transmute(buf) }
    });
    path_strs.clear();
    path_strs.extend(rows.iter().map(|row| {
        let start = row.start as usize;
        let end = start + row.len as usize;
        let slice = bytes.get(start..end).unwrap_or(&[]);
        unsafe { std::str::from_utf8_unchecked(slice) }
    }));

    let has_uppercase = parts
        .iter()
        .any(|part| part.chars().any(|ch| ch.is_uppercase()));
    let config = neo_frizbee::Config {
        max_typos: Some(max_typos),
        sort: neo_frizbee::SortStrategy::Unsorted,
        scoring: Scoring {
            capitalization_bonus: if has_uppercase { 8 } else { 0 },
            matching_case_bonus: if has_uppercase { 4 } else { 0 },
            ..Default::default()
        },
        ..Default::default()
    };

    for (idx, part) in parts.iter().copied().enumerate() {
        let mut part_config = config;
        if idx > 0 {
            part_config.max_typos = config.max_typos.map(|t| t.min(part.len() as u16));
        }

        let matched_list =
            with_cached_matcher(part, &part_config, |m| m.match_list_indices(&path_strs));
        for mut matched in matched_list {
            let item_idx = matched.index as usize;
            let Some(path) = path_strs.get(item_idx) else {
                continue;
            };

            matched.indices.sort_unstable();
            ranges_by_item[item_idx].extend(char_indices_to_byte_offsets(path, &matched.indices));
        }
    }

    // One part already emits sorted, non-overlapping ranges. Extra parts still merge.
    if parts.len() > 1 {
        for ranges in &mut ranges_by_item {
            *ranges = merge_byte_offsets(std::mem::take(ranges));
        }
    }

    path_strs.clear();
    PATH_STRS.with(|slot| {
        let buf: Vec<&'static str> = unsafe { std::mem::transmute(path_strs) };
        let slot = &mut *slot.borrow_mut();
        if buf.capacity() > slot.capacity() {
            *slot = buf;
        }
    });
    ranges_by_item
}

fn char_indices_to_byte_offsets(line: &str, char_indices: &[u32]) -> SmallVec<[(u32, u32); 4]> {
    // Caller sorts. ASCII index `i` is the byte range `(i, i + 1)`; adjacent hits merge.
    // A non-ASCII line walks chars up to the last hit.
    if char_indices.is_empty() {
        return SmallVec::new();
    }

    if line.is_ascii() {
        let mut result: SmallVec<[(u32, u32); 4]> = SmallVec::with_capacity(char_indices.len());
        let len = line.len();
        for &idx in char_indices {
            if idx as usize >= len {
                break;
            }
            let end = idx + 1;
            if let Some(last) = result.last_mut()
                && last.1 == idx
            {
                last.1 = end;
            } else {
                result.push((idx, end));
            }
        }
        return result;
    }

    let mut result: SmallVec<[(u32, u32); 4]> = SmallVec::with_capacity(char_indices.len());
    let mut next_char = 0u32;
    let mut wi = 0usize;
    #[allow(clippy::explicit_counter_loop)]
    for (byte_pos, ch) in line.char_indices() {
        if wi >= char_indices.len() {
            break;
        }
        if next_char == char_indices[wi] {
            let start = byte_pos as u32;
            let end = (byte_pos + ch.len_utf8()) as u32;
            if let Some(last) = result.last_mut()
                && last.1 == start
            {
                last.1 = end;
            } else {
                result.push((start, end));
            }
            wi += 1;
            while wi < char_indices.len() && char_indices[wi] == next_char {
                wi += 1;
            }
        }
        next_char += 1;
    }

    result
}

fn merge_byte_offsets(mut ranges: SmallVec<[(u32, u32); 4]>) -> SmallVec<[(u32, u32); 4]> {
    if ranges.len() <= 1 {
        return ranges;
    }

    ranges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    // Merge in place so the caller's buffer is reused, not reallocated.
    let mut w = 0usize;
    for r in 0..ranges.len() {
        let (start, end) = ranges[r];
        if end <= start {
            continue;
        }
        if w > 0 && start <= ranges[w - 1].1 {
            ranges[w - 1].1 = ranges[w - 1].1.max(end);
            continue;
        }
        ranges[w] = (start, end);
        w += 1;
    }
    ranges.truncate(w);

    ranges
}

// Resolve a DirItem's chunked path into frizbee's pointer buffer.
#[inline]
fn resolve_dir_chunks(
    dir: &DirItem,
    arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    buf: &mut [*const u8; MAX_FRIZBEE_CHUNKS],
) -> Option<(usize, u16)> {
    let arena = if dir.is_overflow() {
        overflow_arena
    } else {
        arena
    };
    crate::matcher::resolve_frizbee_bytes(&dir.path, arena, buf)
}

// Frizbee, not the file gate: a directory is not a `FileItem` window.
// Part scores are one mean: the sum, divided by the part count.
fn match_fuzzy_parts_dirs(
    fuzzy_parts: &[&str],
    working_dirs: &[&DirItem],
    options: &neo_frizbee::Config,
    max_threads: usize,
    arena: ArenaPtr,
    overflow_arena: ArenaPtr,
) -> Vec<neo_frizbee::Match> {
    // 1-2 part queries fit on the stack.
    let valid_parts: SmallVec<[&str; 4]> = fuzzy_parts
        .iter()
        .copied()
        .filter(|p| p.len() >= 2)
        .collect();

    if valid_parts.is_empty() {
        return vec![];
    }

    // Dir haystack bound sampled, not estimated: dirs are few.
    let max_dir_len = working_dirs
        .iter()
        .map(|d| d.path.byte_len as usize)
        .max()
        .unwrap_or(0);
    let dir_threads = effective_match_threads(
        working_dirs.len(),
        valid_parts[0].len(),
        max_dir_len,
        options.max_typos,
        max_threads.max(1),
    );
    let first_part_matches = with_cached_matcher(valid_parts[0], options, |matcher| {
        matcher.match_range_parallel_resolved(
            working_dirs.len(),
            &|index, buf: &mut [*const u8; MAX_FRIZBEE_CHUNKS]| {
                resolve_dir_chunks(working_dirs[index as usize], arena, overflow_arena, buf)
            },
            dir_threads,
        )
    });

    if valid_parts.len() == 1 {
        return first_part_matches;
    }

    let mut matches = first_part_matches;
    let mut sums: Vec<u32> = matches.iter().map(|m| m.score as u32).collect();
    for (ord, part) in valid_parts[1..].iter().enumerate() {
        let parts_done = (ord + 2) as u32;
        let mut part_options = *options;
        part_options.max_typos = options.max_typos.map(|t| t.min(part.len() as u16));

        // Match only surviving dirs without a subset; bound by the corpus max (O(1)).
        let survivors = &matches;
        let sub_dir_threads = effective_match_threads(
            survivors.len(),
            part.len(),
            max_dir_len,
            part_options.max_typos,
            max_threads.max(1),
        );
        let sub_matches = with_cached_matcher(part, &part_options, |matcher| {
            matcher.match_range_parallel_resolved(
                survivors.len(),
                &|index, buf: &mut [*const u8; MAX_FRIZBEE_CHUNKS]| {
                    let dir = working_dirs[survivors[index as usize].index as usize];
                    resolve_dir_chunks(dir, arena, overflow_arena, buf)
                },
                sub_dir_threads,
            )
        });

        if sub_matches.is_empty() {
            return vec![];
        }

        let prev_matches = std::mem::take(&mut matches);
        let prev_sums = std::mem::take(&mut sums);
        let mut next_sums = Vec::with_capacity(sub_matches.len());
        matches = sub_matches
            .into_iter()
            .map(|sm| {
                let i = sm.index as usize;
                let prev = &prev_matches[i];
                let sum = prev_sums[i].saturating_add(sm.score as u32);
                next_sums.push(sum);
                neo_frizbee::Match {
                    index: prev.index,
                    score: (sum / parts_done).min(u16::MAX as u32) as u16,
                    end_col: prev.end_col,
                    exact: prev.exact && sm.exact,
                }
            })
            .collect();
        sums = next_sums;
    }

    matches
}

// Match + score directories: base, frecency boost, distance penalty, dirname bonus.
#[tracing::instrument(skip_all, level = tracing::Level::DEBUG)]
pub(crate) fn fuzzy_match_and_score_dirs<'a>(
    dirs: &'a [DirItem],
    context: &ScoringContext,
    arena: ArenaPtr,
    overflow_arena: ArenaPtr,
) -> (Vec<&'a DirItem>, Vec<Score>, usize) {
    if dirs.is_empty() {
        return (vec![], vec![], 0);
    }

    let parsed_query = context.query;
    // Ghost dirs (all files tombstoned) never surface in search results.
    let working_dirs: Vec<&DirItem> = if parsed_query.constraints.is_empty() {
        dirs.iter().filter(|d| !d.is_deleted()).collect()
    } else {
        match apply_constraints(dirs, &parsed_query.constraints, arena, overflow_arena) {
            Some(filtered) if !filtered.is_empty() => {
                filtered.into_iter().filter(|d| !d.is_deleted()).collect()
            }
            Some(_) => return (vec![], vec![], 0),
            None => dirs.iter().filter(|d| !d.is_deleted()).collect(),
        }
    };

    let fuzzy_parts: &[&str] = match &parsed_query.fuzzy_query {
        FuzzyQuery::Text(t) if t.len() >= 2 => std::slice::from_ref(t),
        FuzzyQuery::Parts(parts) if !parts.is_empty() => parts.as_slice(),
        _ => {
            return score_dirs_by_frecency(&working_dirs, context);
        }
    };

    // 1-2 part queries fit on the stack.
    let valid_parts: SmallVec<[&str; 4]> = fuzzy_parts
        .iter()
        .copied()
        .filter(|p| p.len() >= 2)
        .collect();

    if valid_parts.is_empty() {
        return score_dirs_by_frecency(&working_dirs, context);
    }

    let has_uppercase = valid_parts
        .iter()
        .any(|p| p.chars().any(|c| c.is_uppercase()));

    let options = neo_frizbee::Config {
        max_typos: Some(context.max_typos),
        sort: neo_frizbee::SortStrategy::Unsorted,
        scoring: Scoring {
            capitalization_bonus: if has_uppercase { 8 } else { 0 },
            matching_case_bonus: if has_uppercase { 4 } else { 0 },
            ..Default::default()
        },
        ..Default::default()
    };

    let path_matches = match_fuzzy_parts_dirs(
        fuzzy_parts,
        &working_dirs,
        &options,
        context.max_threads,
        arena,
        overflow_arena,
    );

    let main_needle = valid_parts[0].as_bytes();
    let main_needle_len = main_needle.len() as u16;
    let mut dir_lower_buf = [0u8; 128];
    let dir_folded = fill_ascii_lower(main_needle, &mut dir_lower_buf);
    let dir_lower = &dir_lower_buf[..main_needle.len().min(dir_lower_buf.len())];

    let cur_dir = CurrentDir::new(context.current_file);
    let mut dirname_buf = String::new();

    let results: Vec<(&DirItem, Score)> = path_matches
        .into_iter()
        .map(|path_match| {
            let dir = working_dirs[path_match.index as usize];
            let dir_arena = if dir.is_overflow() {
                overflow_arena
            } else {
                arena
            };
            let base_score = path_match.score as i32;
            let frecency_boost = base_score.saturating_mul(dir.max_access_frecency()) / 100;

            let distance_penalty = cur_dir.penalty_for_path(&dir.path, dir_arena);

            // Dirname bonus: if the match is in the last path segment.
            let last_seg_offset = dir.last_segment_offset();
            let match_start_approx = path_match.end_col.saturating_sub(main_needle_len - 1);
            let is_dirname_match = match_start_approx >= last_seg_offset;

            // The last segment is already addressed. Copy it only past PATH_BUF_SIZE.
            let total = dir.path.byte_len as usize;
            let offset = last_seg_offset as usize;
            let is_exact_dirname = if !is_dirname_match || offset >= total {
                false
            } else if total <= crate::simd_path::PATH_BUF_SIZE {
                main_needle_len as usize == total - offset
                    && if dir_folded {
                        dir.path
                            .window_eq_lower(dir_arena, offset as u32, dir_lower)
                    } else {
                        dir.path
                            .window_eq_ignore_ascii(dir_arena, offset as u32, main_needle)
                    }
            } else {
                dir.write_dir_name(dir_arena, &mut dirname_buf);
                main_needle_len as usize == dirname_buf.len()
                    && eq_query_needle(dirname_buf.as_bytes(), main_needle, dir_lower, dir_folded)
            };

            let filename_bonus = if is_exact_dirname {
                base_score / 5 * 2 // 40% bonus for exact dirname match
            } else if is_dirname_match {
                base_score / 6 // ~16% bonus for fuzzy dirname match
            } else {
                0
            };

            let total = base_score
                .saturating_add(frecency_boost)
                .saturating_add(distance_penalty)
                .saturating_add(filename_bonus);

            let score = Score {
                total,
                base_score,
                filename_bonus,
                special_filename_bonus: 0,
                frecency_boost,
                git_status_boost: 0,
                git_recency_boost: 0,
                distance_penalty,
                current_file_penalty: 0,
                combo_match_boost: 0,
                path_alignment_bonus: 0,
                evidence_bonus: 0,
                exact_match: is_exact_dirname || path_match.exact,
                match_type: if is_exact_dirname {
                    "exact_dirname"
                } else if is_dirname_match {
                    "fuzzy_dirname"
                } else if path_match.exact {
                    "exact_path"
                } else {
                    "fuzzy_path"
                },
            };

            (dir, score)
        })
        .collect();

    sort_and_paginate_dirs(results, context)
}

fn score_dirs_by_frecency<'a>(
    dirs: &[&'a DirItem],
    context: &ScoringContext,
) -> (Vec<&'a DirItem>, Vec<Score>, usize) {
    let results: Vec<(&DirItem, Score)> = dirs
        .iter()
        .map(|&dir| {
            let score = Score {
                total: dir.max_access_frecency(),
                frecency_boost: dir.max_access_frecency(),
                match_type: "frecency",
                ..Default::default()
            };

            (dir, score)
        })
        .collect();

    sort_and_paginate_dirs(results, context)
}

// Sort dir results by total score (descending) and apply pagination.
fn sort_and_paginate_dirs<'a>(
    mut results: Vec<(&'a DirItem, Score)>,
    context: &ScoringContext,
) -> (Vec<&'a DirItem>, Vec<Score>, usize) {
    let total_matched = results.len();
    if total_matched == 0 {
        return (vec![], vec![], 0);
    }

    let offset = context.pagination.offset;
    let limit = if context.pagination.limit > 0 {
        context.pagination.limit
    } else {
        total_matched
    };

    if offset >= total_matched {
        return (vec![], vec![], total_matched);
    }

    let items_needed = offset.saturating_add(limit).min(total_matched);
    let compare = |a: &(&DirItem, Score), b: &(&DirItem, Score)| {
        b.1.total
            .cmp(&a.1.total)
            .then_with(|| std::ptr::from_ref(a.0).cmp(&std::ptr::from_ref(b.0)))
    };
    if items_needed < results.len() {
        sort_prefix_with_buffer(&mut results, items_needed, compare);
        results.truncate(items_needed);
    } else {
        sort_with_buffer(&mut results, compare);
    }

    if offset > 0 {
        results.drain(..offset.min(results.len()));
    }
    if results.len() > limit {
        results.truncate(limit);
    }

    let (items, scores): (Vec<&DirItem>, Vec<Score>) = results.into_iter().unzip();
    (items, scores, total_matched)
}

fn match_and_score_in_arena<'a>(
    files: FileItems<'a>,
    context: &ScoringContext,
    arena: ArenaPtr,
    max_path_len: usize,
    abort: crate::parallelism::AbortToken<'_>,
    capture_offset: u32,
    capture_append: bool,
    cache: Option<MatchCacheRef<'_>>,
) -> (Vec<(&'a FileItem, Score)>, usize) {
    let (results, total, _) = match_and_score_scanned(
        files,
        &[],
        context,
        arena,
        max_path_len,
        abort,
        capture_offset,
        capture_append,
        cache,
    );
    (results, total)
}

#[derive(Clone, Copy)]
enum FallbackName<'a> {
    Borrowed(&'a str),
    Copied(u32, u32),
}

// Cleared before it is stored again. Hit rows are index, exact, and score.
thread_local! {
    static FALLBACK_INDICES: std::cell::RefCell<Vec<u32>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static FALLBACK_NAME_BUF: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static FALLBACK_NAMES: std::cell::RefCell<Vec<FallbackName<'static>>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static FALLBACK_STRS: std::cell::RefCell<Vec<&'static str>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static FALLBACK_HIT: std::cell::RefCell<Vec<(u32, bool, i32)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn take_tls_vec<T>(key: &'static std::thread::LocalKey<std::cell::RefCell<Vec<T>>>) -> Vec<T> {
    let mut v = key.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    v.clear();
    v
}

fn reclaim_tls_vec<T>(
    key: &'static std::thread::LocalKey<std::cell::RefCell<Vec<T>>>,
    mut v: Vec<T>,
) {
    v.clear();
    key.with(|slot| {
        let slot = &mut *slot.borrow_mut();
        if v.capacity() > slot.capacity() {
            *slot = v;
        }
    });
}

fn take_fallback_names<'a>() -> Vec<FallbackName<'a>> {
    let buf = take_tls_vec(&FALLBACK_NAMES);
    // SAFETY: cleared above, so the buffer holds no name pointer.
    unsafe { std::mem::transmute::<Vec<FallbackName<'static>>, Vec<FallbackName<'a>>>(buf) }
}

fn reclaim_fallback_names<'a>(mut names: Vec<FallbackName<'a>>) {
    names.clear();
    // SAFETY: cleared above, so the buffer holds no name pointer.
    let buf: Vec<FallbackName<'static>> = unsafe {
        std::mem::transmute::<Vec<FallbackName<'a>>, Vec<FallbackName<'static>>>(names)
    };
    reclaim_tls_vec(&FALLBACK_NAMES, buf);
}

fn take_fallback_strs<'a>() -> Vec<&'a str> {
    let buf = take_tls_vec(&FALLBACK_STRS);
    // SAFETY: cleared above, so the buffer holds no filename pointer.
    unsafe { std::mem::transmute::<Vec<&'static str>, Vec<&'a str>>(buf) }
}

fn reclaim_fallback_strs<'a>(mut strs: Vec<&'a str>) {
    strs.clear();
    // SAFETY: cleared above, so the buffer holds no filename pointer.
    let buf: Vec<&'static str> =
        unsafe { std::mem::transmute::<Vec<&'a str>, Vec<&'static str>>(strs) };
    reclaim_tls_vec(&FALLBACK_STRS, buf);
}

// Drop returns the hit buffer after scoring.
struct FallbackHitBuf(Vec<(u32, bool, i32)>);

impl Drop for FallbackHitBuf {
    fn drop(&mut self) {
        reclaim_tls_vec(&FALLBACK_HIT, std::mem::take(&mut self.0));
    }
}

impl std::ops::Deref for FallbackHitBuf {
    type Target = Vec<(u32, bool, i32)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for FallbackHitBuf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn match_and_score_scanned<'a>(
    files: FileItems<'a>,
    presence: &[u64],
    context: &ScoringContext,
    arena: ArenaPtr,
    max_path_len: usize,
    abort: crate::parallelism::AbortToken<'_>,
    capture_offset: u32,
    capture_append: bool,
    cache: Option<MatchCacheRef<'_>>,
) -> (Vec<(&'a FileItem, Score)>, usize, bool) {
    if files.len() == 0 {
        return (vec![], 0, true);
    }

    let parsed = context.query;
    // `max_path_len` is the caller's upper bound, including for a narrowed list.

    let working_files: FileItems<'a> = if parsed.constraints.is_empty() {
        files
    } else {
        // apply_constraints needs a slice; only All can be filtered further
        match files {
            FileItems::All(all) => {
                match apply_constraints(all, &parsed.constraints, arena, arena) {
                    Some(filtered) if !filtered.is_empty() => {
                        // Map refs back to indices into `all` (same allocation).
                        let idx: Vec<u32> = filtered
                            .iter()
                            .map(|f| {
                                // SAFETY: `filtered` borrows from `all`; both ends lie in one allocation.
                                let off =
                                    unsafe { (*f as *const FileItem).offset_from(all.as_ptr()) };
                                debug_assert!(off >= 0);
                                off as u32
                            })
                            .collect();
                        FileItems::Filtered {
                            base: all,
                            idx: crate::matcher::CandList::Owned(idx),
                        }
                    }
                    Some(_) => {
                        return (vec![], 0, true);
                    }
                    None => FileItems::All(all),
                }
            }
            other => other,
        }
    };

    let fuzzy_parts: &[&str] = match &parsed.fuzzy_query {
        FuzzyQuery::Text(t) if t.len() >= 2 => std::slice::from_ref(t),
        FuzzyQuery::Parts(parts) if !parts.is_empty() => parts.as_slice(),
        _ => {
            let (scored, n) = score_filtered_by_frecency(&working_files, context, arena);
            return (scored, n, false);
        }
    };

    debug_assert!(!fuzzy_parts.is_empty());
    let has_uppercase = fuzzy_parts
        .iter()
        .any(|p| p.chars().any(|c| c.is_uppercase()));
    // Users type `/` on every platform; the OS separator alone misses Windows queries.
    let query_contains_path_separator = fuzzy_parts
        .iter()
        .any(|p| p.contains('/') || p.contains(MAIN_SEPARATOR));

    let options = neo_frizbee::Config {
        max_typos: Some(context.max_typos),
        sort: neo_frizbee::SortStrategy::Unsorted,
        scoring: Scoring {
            capitalization_bonus: if has_uppercase { 8 } else { 0 },
            matching_case_bonus: if has_uppercase { 4 } else { 0 },
            ..Default::default()
        },
        ..Default::default()
    };

    let phase_timing = phase_timing();
    let phase_log = |msg: String| {
        if !phase_timing {
            return;
        }
        let phase_sink = std::env::var("FFF_PHASE_TIMING").unwrap_or_default();
        if phase_sink.is_empty() || phase_sink == "1" {
            eprintln!("{msg}");
        } else {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&phase_sink)
            {
                let _ = writeln!(f, "{msg}");
            }
        }
    };
    let t_match = phase_timing.then(std::time::Instant::now);
    let cache_idx = match &working_files {
        FileItems::Filtered { idx, .. } => Some(idx),
        FileItems::All(_) => None,
    };
    let cached_matches = match (cache, cache_idx) {
        (Some(cr), Some(idx)) => cr.cache.lock().get_list(
            cr.generation,
            fuzzy_parts,
            context.max_typos,
            cr.side,
            idx,
        ),
        _ => None,
    };
    let match_cached = cached_matches.is_some();
    let before_filename = std::sync::atomic::AtomicBool::new(false);
    let held_matches = match cached_matches {
        Some(matches) => HeldMatches::Shared(matches),
        None => {
            let matches = match_fuzzy_scanned(
                fuzzy_parts,
                &working_files,
                presence,
                &options,
                context.max_threads,
                arena,
                max_path_len,
                abort,
                Some(&before_filename),
                context.check_deleted,
            );
            if let (Some(cr), Some(idx)) = (cache, cache_idx) {
                // An aborted search can return a partial match set; never
                // memoize it, or a later repeat would silently lose hits.
                if !abort.aborted()
                    && idx.len() <= MATCH_CACHE_MAX_IDX
                    && matches.len() <= MATCH_CACHE_MAX_MATCHES
                {
                    let shared = std::sync::Arc::new(matches);
                    let idx_owned = match idx {
                        crate::matcher::CandList::Shared(arc) => std::sync::Arc::clone(arc),
                        crate::matcher::CandList::Owned(v) => std::sync::Arc::new(v.clone()),
                    };
                    let shared_slot = std::sync::Arc::clone(&shared);
                    cr.cache.lock().put(
                        cr.generation,
                        fuzzy_parts,
                        context.max_typos,
                        cr.side,
                        Some(idx_owned),
                        Some(shared_slot),
                    );
                    HeldMatches::Shared(shared)
                } else {
                    if !abort.aborted() {
                        cr.cache.lock().put(
                            cr.generation,
                            fuzzy_parts,
                            context.max_typos,
                            cr.side,
                            None,
                            None,
                        );
                    }
                    HeldMatches::Owned(matches)
                }
            } else {
                HeldMatches::Owned(matches)
            }
        }
    };
    let path_matches = held_matches.as_slice();
    if let Some(t_match) = t_match {
        phase_log(format!(
            "PHASE match_us={} n_matches={} n_files={} cached={}",
            t_match.elapsed().as_micros(),
            path_matches.len(),
            working_files.len(),
            match_cached as u8
        ));
    }

    // Capture indices for live-narrow: Match(longer,T) is a subset of Match(shorter,T).
    let t_capture = phase_timing.then(std::time::Instant::now);
    match &working_files {
        crate::matcher::FileItems::All(_) => crate::score::capture_hit_indices(
            path_matches.iter().map(|m| m.index as u32),
            capture_offset,
            capture_append,
        ),
        crate::matcher::FileItems::Filtered { idx, .. } => crate::score::capture_hit_indices(
            path_matches.iter().map(|m| {
                let i = m.index as usize;
                debug_assert!(i < idx.len());
                unsafe { *idx.get_unchecked(i) }
            }),
            capture_offset,
            capture_append,
        ),
    }

    let capture_us = t_capture.map(|t| t.elapsed().as_micros()).unwrap_or(0);
    let t_fallback = phase_timing.then(std::time::Instant::now);
    let main_needle = fuzzy_parts[0].as_bytes(); // safe
    let main_needle_len = main_needle.len() as u16;
    // Collapsed queries zero filename/evidence bonuses, so skipping fallback keeps ranking.
    let rank_collapsed = {
        let n_files = working_files.len().max(1);
        path_matches.len() >= 4096
            && path_matches.len() * 100 >= n_files * 85
            && main_needle_len <= 8
    };

    // Fresh scans record this while the file is already loaded. Cache hits scan once.
    let needs_name_fallback = if query_contains_path_separator
        || path_matches.len() > 15_000
        || rank_collapsed
    {
        false
    } else if match_cached {
        match &working_files {
            crate::matcher::FileItems::All(files) => path_matches.iter().any(|path_match| {
                let file = crate::matcher::FileItems::at_all(files, path_match.index as usize);
                let filename_start = file.filename_offset_in_relative_path() as u16;
                path_match.end_col.saturating_sub(main_needle_len - 1) < filename_start
            }),
            crate::matcher::FileItems::Filtered { base, idx } => {
                path_matches.iter().any(|path_match| {
                    let file =
                        crate::matcher::FileItems::at_filtered(base, idx, path_match.index as usize);
                    let filename_start = file.filename_offset_in_relative_path() as u16;
                    path_match.end_col.saturating_sub(main_needle_len - 1) < filename_start
                })
            }
        }
    } else {
        before_filename.load(std::sync::atomic::Ordering::Relaxed)
    };

    let mut fallback_hit = FallbackHitBuf(take_tls_vec(&FALLBACK_HIT));
    let mut held_indices: Option<Vec<u32>> = None;
    let mut held_matches: Option<Vec<neo_frizbee::Match>> = None;
    if needs_name_fallback {
        // Borrow a name that sits in one chunk; copy only a name that crosses a boundary.
        let mut fallback_indices = take_tls_vec(&FALLBACK_INDICES);
        let mut name_buf = take_tls_vec(&FALLBACK_NAME_BUF);
        let mut names = take_fallback_names();

        macro_rules! collect_names {
            ($at:ident, $($src:expr),+) => {
                for (i, path_match) in path_matches.iter().enumerate() {
                    let file = crate::matcher::FileItems::$at($($src),+, path_match.index as usize);
                    let (byte_len_u, name_off) = file.path.span_lens();
                    let off = name_off as usize;
                    let total = byte_len_u as usize;
                    let filename_start = off as u16;
                    let match_start_approx = path_match.end_col.saturating_sub(main_needle_len - 1);

                    if match_start_approx < filename_start {
                        fallback_indices.push(i as u32);
                        if let Some(name) = file.path.filename_in_chunk_at(arena, off, total) {
                            names.push(FallbackName::Borrowed(name));
                        } else {
                            let end = total as u32;
                            let start = name_buf.len();
                            let len = end.saturating_sub(off as u32) as usize;
                            // The copy writes every new byte, so the spare is not zeroed first.
                            name_buf.reserve(len);
                            unsafe {
                                file.path.copy_trimmed_ptr(
                                    arena,
                                    off as u32,
                                    end,
                                    name_buf.as_mut_ptr().add(start),
                                );
                                name_buf.set_len(start + len);
                            }
                            names.push(FallbackName::Copied(start as u32, len as u32));
                        }
                    }
                }
            };
        }
        match &working_files {
            crate::matcher::FileItems::All(files) => collect_names!(at_all, files),
            crate::matcher::FileItems::Filtered { base, idx } => {
                collect_names!(at_filtered, base, idx)
            }
        }

        if !names.is_empty() {
            let mut filename_strs = take_fallback_strs();
            filename_strs.reserve(names.len());
            for name in &names {
                filename_strs.push(match *name {
                    FallbackName::Borrowed(s) => s,
                    FallbackName::Copied(s, l) => {
                        let s = s as usize;
                        // SAFETY: arena path bytes are valid UTF-8, as `read_to_buf` assumes.
                        unsafe { std::str::from_utf8_unchecked(&name_buf[s..s + l as usize]) }
                    }
                });
            }
            // The gate window is one accepting span, not the filename match,
            // so this second score is still frizbee's.
            let fb_threads = effective_match_threads(
                filename_strs.len(),
                fuzzy_parts[0].len(),
                max_path_len,
                options.max_typos,
                context.max_threads.max(1),
            );
            held_matches = Some(with_cached_matcher(fuzzy_parts[0], &options, |matcher| {
                matcher.match_list_parallel(&filename_strs, fb_threads)
            }));
            reclaim_fallback_strs(filename_strs);
        }
        reclaim_tls_vec(&FALLBACK_NAME_BUF, name_buf);
        reclaim_fallback_names(names);
        held_indices = Some(fallback_indices);
    }

    let fallback_us = t_fallback.map(|t| t.elapsed().as_micros()).unwrap_or(0);
    if phase_timing {
        phase_log(format!(
            "PHASE capture_us={capture_us} fallback_us={fallback_us} n={}",
            path_matches.len()
        ));
    }

    if let (Some(matches), Some(indices)) = (&held_matches, &held_indices) {
        // Only the names that fell back are stored, in path-match order.
        fallback_hit.reserve(matches.len());
        for m in matches {
            let mi = indices[m.index as usize];
            fallback_hit.push((mi, m.exact, m.score as i32));
        }
        fallback_hit.sort_unstable_by_key(|row| row.0);
    }
    if let Some(indices) = held_indices {
        reclaim_tls_vec(&FALLBACK_INDICES, indices);
    }

    // Demote third-party dirs: outranks vendored bonuses without displacing better first-party matches.

    // Separate filename scratch: one buffer cannot serve both borrows at once.
    type ScoreScratch = (
        std::mem::MaybeUninit<[u8; crate::simd_path::PATH_BUF_SIZE]>,
        DirPenaltyMemo,
    );
    let make_scratch = || {
        (
            std::mem::MaybeUninit::<[u8; crate::simd_path::PATH_BUF_SIZE]>::uninit(),
            DirPenaltyMemo::new(0),
        )
    };

    // Ranking-feature collapse: when the query hits nearly every path, per-hit
    // filename reads and intent evidence cannot discriminate the top page, so skip them.
    let main_needle_as_str = context.effective_query();
    let query_intent =
        crate::intent::detect_intent(main_needle_as_str, context.current_file, true);
    // Query-constant evidence state, built once.
    let prepared_query = crate::intent::PreparedQuery::new(main_needle_as_str);
    // Combo target is query-constant; hoist it out of the per-candidate closure.
    let combo_last: Option<&[u8]> = context
        .last_same_query_match
        .as_ref()
        .map(|m| m.file_path.as_os_str().as_encoded_bytes());
    // Memo is exact only when evidence ignores the path: no open file, no Path intent.
    // Epoch retires the table per query.
    let memo_epoch = MEMO_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let memo_active = std::env::var_os("FFF_NO_EVIDENCE_MEMO").is_none()
        && query_intent != crate::intent::QueryIntent::Path
        && context.current_file.is_none();

    // Split the current-file dir once; penalties memoized per parent dir.
    let cur_dir = CurrentDir::new(context.current_file);
    let path_align_on = query_contains_path_separator && main_needle.len() > 10;
    // Folded once. Exact and suffix compares must not refold the needle.
    let mut main_lower_buf = [0u8; 128];
    let main_folded = fill_ascii_lower(main_needle, &mut main_lower_buf);
    let main_lower = &main_lower_buf[..main_needle.len().min(main_lower_buf.len())];
    let (cur_slash_dir, cur_back_dir) = current_file_parents(context.current_file);
    let mut slash_buf = Vec::new();
    let mut back_buf = Vec::new();
    let slash_l = parent_bytes(cur_slash_dir, &mut slash_buf);
    let back_l = parent_bytes(cur_back_dir, &mut back_buf);
    let cur_fname = context
        .current_file
        .map(|c| c.rsplit(['/', '\\']).next().unwrap_or(c))
        .unwrap_or("");
    let related = crate::intent::related_key(cur_fname);

    let score_one = |scratch: &mut ScoreScratch,
                     memo: Option<&mut EvidenceMemo>,
                     path_match: &neo_frizbee::Match,
                     file: &'a FileItem,
                     frecency_boost: i32,
                     git_recency: i32,
                     vendored: bool,
                     git_status_boost: i32,
                     simd_filename_match: Option<(bool, i32)>|
     -> (&'a FileItem, Score) {
        let (fname_buf, dir_memo) = scratch;
        // SAFETY: `filename_ref` writes the returned prefix before it is read.
        let fname_buf = unsafe { &mut *fname_buf.as_mut_ptr() };

        let base_score = path_match.score as i32;
        let git_recency_boost = git_recency;

        let distance_penalty = if context.current_file.is_some() {
            let dir_idx = file.parent_dir_index as usize;
            match dir_memo.get(dir_idx) {
                Some(p) => p,
                None => dir_memo.penalty_for_grow_path(&cur_dir, dir_idx, &file.path, arena),
            }
        } else {
            0
        };

        let (byte_len_u, name_off) = file.path.span_lens();
        let fname_off = name_off as usize;
        let path_len = byte_len_u as usize;
        let filename_start = fname_off as u16;
        let match_start_approx = path_match.end_col.saturating_sub(main_needle_len - 1);

        let end_col_filename_match = match_start_approx >= filename_start;
        let simd_filename_match = if end_col_filename_match {
            None
        } else {
            simd_filename_match
        };

        let is_filename_match = end_col_filename_match || simd_filename_match.is_some();
        let fname_len = path_len - fname_off;
        let len_eq = main_needle_len as usize == fname_len;
        // The full path serves the suffix test, combo test, and path-shaped evidence.
        // Collapsed ranking reads the filename only when an exact name is still possible.
        let need_fname = !rank_collapsed || (end_col_filename_match && len_eq);
        let fname: Option<&str> = if !need_fname || rank_collapsed {
            None
        } else {
            Some(file.path.filename_at(arena, fname_buf, fname_off, path_len))
        };

        let is_exact_filename = simd_filename_match.is_some_and(|(exact, _)| exact)
            || (end_col_filename_match
                && len_eq
                && (match fname {
                    Some(f) => eq_query_needle(f.as_bytes(), main_needle, main_lower, main_folded),
                    None => {
                        if main_folded {
                            file.path.window_eq_lower_ready(arena, fname_off as u32, main_lower)
                        } else {
                            file.path.window_eq_ignore_ascii_ready(
                                arena,
                                fname_off as u32,
                                main_needle,
                            )
                        }
                    }
                }));

        let mut has_special_filename_bonus = false;
        let filename_bonus = if is_exact_filename {
            base_score / 5 * 2 // 40% bonus for exact filename match
        } else if rank_collapsed {
            // Cheap filename relevance for broad queries: no fname read,
            // but a hit that lands in the filename still beats one that
            // ended in a directory segment.
            if end_col_filename_match {
                (base_score / 6).min(30)
            } else {
                0
            }
        } else if is_filename_match {
            let max_bonus = (base_score / 6).min(30);
            if let Some((_, fm_score)) = simd_filename_match {
                let max_possible = main_needle_len as i32 * 16;
                let quality = fm_score.min(max_possible);
                max_bonus * quality / max_possible
            } else {
                max_bonus
            }
        } else if !is_filename_match && (5..=11).contains(&fname_len) {
            if is_special_entry_point_bytes(fname.as_ref().unwrap().as_bytes()) {
                has_special_filename_bonus = true;
                base_score * 5 / 100
            } else {
                0
            }
        } else {
            0
        };

        let current_file_penalty =
            calculate_current_file_penalty(file, base_score / 4, context, arena);
        let combo_match_boost = {
            let last_same_query_match = context.last_same_query_match.as_ref().filter(|_| {
                combo_last.is_some_and(|last| file.path.is_suffix_of(arena, last))
            });

            match last_same_query_match {
                Some(_) if context.min_combo_count == 0 => 1000,
                Some(combo_match) if combo_match.open_count >= context.min_combo_count => {
                    combo_match.open_count as i32 * context.combo_boost_score_multiplier
                }
                Some(combo_match) => combo_match.open_count as i32 * 5,
                _ => 0,
            }
        };

        // Path alignment: reward suffix overlap with the typed query;
        // a full prefix match is the 100% case, needing no separate branch.
        let path_alignment_bonus = if path_align_on {
            let common_suffix = if main_folded {
                file.path.common_suffix_len_lower(arena, main_lower)
            } else {
                file.path
                    .common_suffix_len_ignore_ascii_case(arena, main_needle)
            };

            let needle_len = main_needle.len();
            if common_suffix > 10 && needle_len > 0 {
                let coverage = common_suffix * 100 / needle_len;
                if coverage >= 30 {
                    base_score * coverage as i32 / 100
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            0
        };

        // Intent evidence mixed per job; memoized by filename when path terms are dead.
        let evidence_bonus = if rank_collapsed {
            0
        } else if let Some(memo) = memo {
            let fname_ref = fname.as_ref().unwrap();
            let ev = memoized_evidence(
                memo,
                memo_epoch,
                fname_ref,
                &prepared_query,
                query_intent,
                phase_timing,
            );
            crate::intent::evidence_bonus(query_intent, &ev, base_score)
        } else {
            let fname_ref = fname.as_ref().unwrap();
            // Same-dir and related are applied below from the hoisted current file.
            let mut ev = crate::intent::evidence_prepared(
                &prepared_query,
                "",
                fname_ref,
                query_intent,
                None,
            );
            if matches!(
                query_intent,
                crate::intent::QueryIntent::Concept
                    | crate::intent::QueryIntent::Typo
                    | crate::intent::QueryIntent::WorkingSet
            ) && context.current_file.is_some()
            {
                ev.same_dir = file.path.same_parent_dir_lower(arena, slash_l, back_l);
                ev.related = crate::intent::is_related_key(fname_ref, &related);
            }
            if query_intent == crate::intent::QueryIntent::Path {
                let mut seg_buf = [0u8; 256];
                fill_path_segments(&mut ev, file, arena, &mut seg_buf, &prepared_query);
            }
            crate::intent::evidence_bonus(query_intent, &ev, base_score)
        };

        let vendored_penalty = if vendored { VENDORED_PATH_PENALTY } else { 0 };

        let total = base_score
            .saturating_add(frecency_boost)
            .saturating_add(git_status_boost)
            .saturating_add(git_recency_boost)
            .saturating_add(distance_penalty)
            .saturating_add(filename_bonus)
            .saturating_add(evidence_bonus)
            .saturating_add(current_file_penalty)
            .saturating_add(combo_match_boost)
            .saturating_add(path_alignment_bonus)
            .saturating_sub(vendored_penalty);

        let score = Score {
            total,
            base_score,
            current_file_penalty,
            filename_bonus,
            special_filename_bonus: if has_special_filename_bonus {
                filename_bonus
            } else {
                0
            },
            frecency_boost,
            git_status_boost,
            git_recency_boost,
            distance_penalty,
            combo_match_boost,
            path_alignment_bonus,
            evidence_bonus,
            exact_match: is_exact_filename || path_match.exact,
            match_type: if is_exact_filename {
                "exact_filename"
            } else if is_filename_match {
                "fuzzy_filename"
            } else if path_match.exact {
                "exact_path"
            } else {
                "fuzzy_path"
            },
        };

        // Census stays out of the scorer. It runs only when phase timing is on.
        if phase_timing {
            note_score_profile(
                is_filename_match,
                is_exact_filename,
                combo_match_boost != 0,
            );
        }

        (file, score)
    };
    let score_loaded = |scratch: &mut ScoreScratch,
                        memo: Option<&mut EvidenceMemo>,
                        path_match: &neo_frizbee::Match,
                        file: &'a FileItem,
                        simd_filename_match: Option<(bool, i32)>| {
        let (freq, git_r, vendored, git_dirty) = score_inputs(file);
        score_one(
            scratch,
            memo,
            path_match,
            file,
            freq_term_of(path_match.score as i32, freq),
            git_r,
            vendored,
            dirty_bonus_of(path_match.score as i32, git_dirty),
            simd_filename_match,
        )
    };

    // Min-heap keeps only offset+limit rows; `total_matched` still counts every match.
    let page_cap = context
        .pagination
        .offset
        .saturating_add(if context.pagination.limit > 0 {
            context.pagination.limit
        } else {
            usize::MAX / 4
        })
        .saturating_add(1);
    // Heap on work alone (O(n log k) vs O(n log n)); serial vs parallel from dispatch.
    let use_topk = page_cap < path_matches.len();
    // Per-match cost is path bytes plus struct; the requested width caps the stage.
    let score_bytes =
        crate::dispatch::sampled_path_avg(&working_files) + std::mem::size_of::<Score>();
    let score_cap = context
        .max_threads
        .max(1)
        .min(crate::parallelism::search_pool_threads());
    let mut score_plan = crate::dispatch::plan_matches(path_matches.len(), score_bytes, score_cap);
    // Candidate scoring is branch/string-heavy; the byte-cost planner leaves
    // ~1k-match top-k pages serial even though two workers win there. Keep
    // larger plans unchanged and leave sub-1k/full-result pages serial.
    if score_plan.threads == 1 && use_topk && path_matches.len() >= 1024 && score_cap >= 2 {
        score_plan.threads = 2;
        score_plan.chunk = path_matches.len().div_ceil(2);
    }
    let t_score = phase_timing.then(std::time::Instant::now);
    let score_entrance_count = path_matches.len();

    let combo_max: i32 = match context.last_same_query_match.as_ref() {
        None => 0,
        Some(_) if context.min_combo_count == 0 => 1000,
        Some(m) if m.open_count >= context.min_combo_count => {
            (m.open_count as i64 * context.combo_boost_score_multiplier as i64).min(i32::MAX as i64)
                as i32
        }
        Some(m) => (m.open_count as i32).saturating_mul(5),
    };
    let q_intent = query_intent;
    let q_collapsed = rank_collapsed;
    // Filename, path, and evidence ceilings depend only on the match score.
    let mut fixed_ub = [0i32; 256];
    for (b, slot) in fixed_ub.iter_mut().enumerate() {
        let base = b as i32;
        let filename = if q_collapsed {
            (base / 6).min(30)
        } else {
            base / 5 * 2
        };
        let path = if path_align_on && !q_collapsed { base } else { 0 };
        *slot = filename + path + evidence_ub_for(base, q_intent, q_collapsed);
    }
    // Read per query; both settings rank identically.
    let no_prune = std::env::var_os("FFF_NO_PRUNE").is_some();
    let mut skipped_topk = 0u64;

    if use_topk {
        let mut heap: Heap<'a> = Heap::new();
        let consider_count = |heap: &mut Heap<'a>, item: (&'a FileItem, Score)| {
                match consider(heap, page_cap, item) {
                    Admit::Push if phase_timing => note_heap_admit(true),
                    Admit::Replace if phase_timing => note_heap_admit(false),
                    _ => {}
                }
            };
        if score_plan.threads < 2 {
            let mut scratch = make_scratch();
            let mut floor = i32::MIN;
            {
            let mut run = |mut memo: Option<&mut EvidenceMemo>| {
                macro_rules! rank_serial {
                    ($at:ident, $($src:expr),+) => {{
                        let mut fb_at = 0usize;
                        for (match_idx, path_match) in path_matches.iter().enumerate() {
                            if let Some(next) = path_matches.get(match_idx + 8) {
                                prefetch_ranked_file(crate::matcher::FileItems::$at(
                                    $($src),+,
                                    next.index as usize,
                                ));
                            }
                            let file = crate::matcher::FileItems::$at(
                                $($src),+,
                                path_match.index as usize,
                            );
                    let (bits, freq, git_r) = file.rank_freq_git();
                    let base_score = path_match.score as i32;
                    let (freq_term, freq_pos) = freq_terms(base_score, freq);
                    let full = !no_prune && heap.len() >= page_cap;
                    if full
                        && ub_cannot_enter(
                            base_score,
                            freq_pos,
                            git_r,
                            combo_max,
                            floor,
                        )
                    {
                        skipped_topk += 1;
                        continue;
                    }
                    let (vendored, git_dirty) = FileItem::rank_flags(bits);
                    let dirty_bonus = dirty_bonus_of(base_score, git_dirty);
                    if full {
                        let ub = ub_total_for(
                            base_score,
                            freq_term,
                            git_r,
                            dirty_bonus,
                            q_intent,
                            q_collapsed,
                            path_align_on,
                            combo_max,
                            &fixed_ub,
                        );
                        // `floor` is the worst total. A smaller upper bound cannot enter.
                        if ub < floor {
                            skipped_topk += 1;
                            continue;
                        }
                    }
                    let item = score_one(
                        &mut scratch,
                        memo.as_deref_mut(),
                        path_match,
                        file,
                        freq_term,
                        git_r,
                        vendored,
                        dirty_bonus,
                        take_fallback_hit(&fallback_hit[..], &mut fb_at, match_idx),
                    );
                    if heap.len() >= page_cap && item.1.total < floor {
                        continue;
                    }
                            consider_count(&mut heap, item);
                            sync_floor(&heap, page_cap, &mut floor);
                        }
                    }};
                }
                match &working_files {
                    crate::matcher::FileItems::All(files) => rank_serial!(at_all, files),
                    crate::matcher::FileItems::Filtered { base, idx } => {
                        rank_serial!(at_filtered, base, idx)
                    }
                }
            };
            if memo_active {
                with_evidence_memo(|memo| run(Some(memo)));
            } else {
                run(None);
            }
            }
        } else {
            // Per-worker heaps sized page_cap, reduced after; ranking stays exact.
            // Scratch stays per-thread via map_init, never in the fold accumulator.
            let (par_heap, par_skipped) = crate::parallelism::run_on_search_pool(|| {
                use rayon::prelude::*;
                let chunk = score_plan.chunk.max(1);
                let n = path_matches.len();
                let n_chunks = n.div_ceil(chunk);
                (0..n_chunks)
                    .into_par_iter()
                    .map_init(make_scratch, |scratch, chunk_idx| {
                        let mut h: Heap<'_> = Heap::new();
                        let mut skipped = 0u64;
                        let mut floor = i32::MIN;
                        let lo = chunk_idx * chunk;
                        let hi = (lo + chunk).min(n);
                        {
                        let mut run = |mut memo: Option<&mut EvidenceMemo>| {
                            macro_rules! rank_parallel {
                                ($at:ident, $($src:expr),+) => {{
                                    let mut fb_at = fallback_hit
                                        .partition_point(|row| (row.0 as usize) < lo);
                                    for match_idx in lo..hi {
                                        let path_match = &path_matches[match_idx];
                                        if match_idx + 8 < hi {
                                            let next = &path_matches[match_idx + 8];
                                            prefetch_ranked_file(crate::matcher::FileItems::$at(
                                                $($src),+,
                                                next.index as usize,
                                            ));
                                        }
                                        let file = crate::matcher::FileItems::$at(
                                            $($src),+,
                                            path_match.index as usize,
                                        );
                                let (bits, freq, git_r) = file.rank_freq_git();
                                let base_score = path_match.score as i32;
                                let (freq_term, freq_pos) = freq_terms(base_score, freq);
                                let full = !no_prune && h.len() >= page_cap;
                                if full
                                    && ub_cannot_enter(
                                        base_score,
                                        freq_pos,
                                        git_r,
                                        combo_max,
                                        floor,
                                    )
                                {
                                    skipped += 1;
                                    continue;
                                }
                                let (vendored, git_dirty) = FileItem::rank_flags(bits);
                                let dirty_bonus = dirty_bonus_of(base_score, git_dirty);
                                if full {
                                    let ub = ub_total_for(
                                        base_score,
                                        freq_term,
                                        git_r,
                                        dirty_bonus,
                                        q_intent,
                                        q_collapsed,
                                        path_align_on,
                                        combo_max,
                                        &fixed_ub,
                                    );
                                    if ub < floor {
                                        skipped += 1;
                                        continue;
                                    }
                                }
                                let item = score_one(
                                    scratch,
                                    memo.as_deref_mut(),
                                    path_match,
                                    file,
                                    freq_term,
                                    git_r,
                                    vendored,
                                    dirty_bonus,
                                    take_fallback_hit(&fallback_hit[..], &mut fb_at, match_idx),
                                );
                                if h.len() >= page_cap && item.1.total < floor {
                                    continue;
                                }
                                match consider(&mut h, page_cap, item) {
                                    Admit::Push if phase_timing => note_heap_admit(true),
                                    Admit::Replace if phase_timing => note_heap_admit(false),
                                    _ => {}
                                }
                                        sync_floor(&h, page_cap, &mut floor);
                                    }
                                }};
                            }
                            match &working_files {
                                crate::matcher::FileItems::All(files) => {
                                    rank_parallel!(at_all, files)
                                }
                                crate::matcher::FileItems::Filtered { base, idx } => {
                                    rank_parallel!(at_filtered, base, idx)
                                }
                            }
                        };
                        if memo_active {
                            with_evidence_memo(|memo| run(Some(memo)));
                        } else {
                            run(None);
                        }
                        }
                        (h, skipped)
                    })
                    .reduce(
                        || (Heap::new(), 0u64),
                        |a, b| (merge(a.0, b.0, page_cap), a.1 + b.1),
                    )
            });
            heap = par_heap;
            skipped_topk = par_skipped;
        }
        let results: Vec<(&'a FileItem, Score)> = heap.into_page();
        if phase_timing {
            eprintln!(
                "PHASE score_us={} (topk) n={} skipped={} threads={}",
                t_score.map(|t| t.elapsed().as_micros()).unwrap_or(0),
                score_entrance_count,
                skipped_topk,
                score_plan.threads,
            );
            let hit = MEMO_HIT.swap(0, std::sync::atomic::Ordering::Relaxed);
            let miss = MEMO_MISS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let push = HEAP_PUSH.swap(0, std::sync::atomic::Ordering::Relaxed);
            let repl = HEAP_REPLACE.swap(0, std::sync::atomic::Ordering::Relaxed);
            let nrel = PROF_NREL.swap(0, std::sync::atomic::Ordering::Relaxed);
            let fmatch = PROF_FMATCH.swap(0, std::sync::atomic::Ordering::Relaxed);
            let exact = PROF_EXACT.swap(0, std::sync::atomic::Ordering::Relaxed);
            let combo = PROF_COMBO.swap(0, std::sync::atomic::Ordering::Relaxed);
            let scored = PROF_SCORED.swap(0, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "PHASE scoreprof memo_hit={hit} memo_miss={miss} heap_push={push} heap_replace={repl} nrel={nrel} fmatch={fmatch} exact={exact} combo={combo} scored={scored}"
            );
        }
        return (results, path_matches.len(), true);
    }

    let results: Vec<_> = if score_plan.threads < 2 {
        let mut scratch = make_scratch();
        let mut out = Vec::with_capacity(path_matches.len());
        {
        let mut run = |mut memo: Option<&mut EvidenceMemo>| {
            macro_rules! push_scored {
                ($at:ident, $($src:expr),+) => {{
                    let mut fb_at = 0usize;
                    for (match_idx, path_match) in path_matches.iter().enumerate() {
                        if let Some(next) = path_matches.get(match_idx + 8) {
                            prefetch_ranked_file(crate::matcher::FileItems::$at(
                                $($src),+,
                                next.index as usize,
                            ));
                        }
                        let file = crate::matcher::FileItems::$at(
                            $($src),+,
                            path_match.index as usize,
                        );
                        out.push(score_loaded(
                            &mut scratch,
                            memo.as_deref_mut(),
                            path_match,
                            file,
                            take_fallback_hit(&fallback_hit[..], &mut fb_at, match_idx),
                        ));
                    }
                }};
            }
            match &working_files {
                crate::matcher::FileItems::All(files) => push_scored!(at_all, files),
                crate::matcher::FileItems::Filtered { base, idx } => {
                    push_scored!(at_filtered, base, idx)
                }
            }
        };
        if memo_active {
            with_evidence_memo(|memo| run(Some(memo)));
        } else {
            run(None);
        }
        }
        out
    } else if context.max_threads < crate::parallelism::search_pool_threads() {
        // Binding width: one planned chunk per task, at most max_threads tasks.
        crate::parallelism::run_on_search_pool(|| {
            let span = score_plan.chunk.max(1);
            let n_chunks = path_matches.len().div_ceil(span);
            let parts: Vec<Vec<_>> = (0..n_chunks)
                .into_par_iter()
                .map_init(make_scratch, |scratch, c| {
                    let lo = c * span;
                    let hi = ((c + 1) * span).min(path_matches.len());
                    let mut out = Vec::with_capacity(hi - lo);
                    {
                    let mut run = |mut memo: Option<&mut EvidenceMemo>| {
                            macro_rules! push_span {
                            ($at:ident, $($src:expr),+) => {{
                                let mut fb_at = fallback_hit
                                    .partition_point(|row| (row.0 as usize) < lo);
                                for match_idx in lo..hi {
                                    let path_match = &path_matches[match_idx];
                                    if match_idx + 8 < hi {
                                        let next = &path_matches[match_idx + 8];
                                        prefetch_ranked_file(crate::matcher::FileItems::$at(
                                            $($src),+,
                                            next.index as usize,
                                        ));
                                    }
                                    let file = crate::matcher::FileItems::$at(
                                        $($src),+,
                                        path_match.index as usize,
                                    );
                                    out.push(score_loaded(
                                        scratch,
                                        memo.as_deref_mut(),
                                        path_match,
                                        file,
                                        take_fallback_hit(&fallback_hit[..], &mut fb_at, match_idx),
                                    ));
                                }
                            }};
                        }
                        match &working_files {
                            crate::matcher::FileItems::All(files) => push_span!(at_all, files),
                            crate::matcher::FileItems::Filtered { base, idx } => {
                                push_span!(at_filtered, base, idx)
                            }
                        }
                    };
                    if memo_active {
                        with_evidence_memo(|memo| run(Some(memo)));
                    } else {
                        run(None);
                    }
                    }
                    out
                })
                .collect();
            parts.into_iter().flatten().collect()
        })
    } else {
        crate::parallelism::run_on_search_pool(|| {
            macro_rules! score_each {
                ($at:ident, $($src:expr),+) => {
                    path_matches
                        .par_iter()
                        .enumerate()
                        .map_init(make_scratch, |scratch, (match_idx, path_match)| {
                            let file = crate::matcher::FileItems::$at(
                                $($src),+,
                                path_match.index as usize,
                            );
                            let mut scored = None;
                            if memo_active {
                                with_evidence_memo(|memo| {
                                    scored = Some(score_loaded(
                                        scratch,
                                        Some(memo),
                                        path_match,
                                        file,
                                        fallback_hit.binary_search_by_key(&(match_idx as u32), |row| row.0).ok().map(|i| {
                                            (fallback_hit[i].1, fallback_hit[i].2)
                                        }),
                                    ));
                                });
                            } else {
                                scored = Some(score_loaded(
                                    scratch,
                                    None,
                                    path_match,
                                    file,
                                    fallback_hit.binary_search_by_key(&(match_idx as u32), |row| row.0).ok().map(|i| {
                                        (fallback_hit[i].1, fallback_hit[i].2)
                                    }),
                                ));
                            }
                            scored.expect("score")
                        })
                };
            }
            match &working_files {
                crate::matcher::FileItems::All(files) => {
                    score_each!(at_all, files).collect::<Vec<_>>()
                }
                crate::matcher::FileItems::Filtered { base, idx } => {
                    score_each!(at_filtered, base, idx).collect::<Vec<_>>()
                }
            }
        })
    };
    let n = results.len();
    if phase_timing {
        eprintln!(
            "PHASE score_us={} (full) n={}",
            t_score.map(|t| t.elapsed().as_micros()).unwrap_or(0),
            score_entrance_count
        );
        let hit = MEMO_HIT.swap(0, std::sync::atomic::Ordering::Relaxed);
        let miss = MEMO_MISS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let push = HEAP_PUSH.swap(0, std::sync::atomic::Ordering::Relaxed);
        let repl = HEAP_REPLACE.swap(0, std::sync::atomic::Ordering::Relaxed);
        let nrel = PROF_NREL.swap(0, std::sync::atomic::Ordering::Relaxed);
        let fmatch = PROF_FMATCH.swap(0, std::sync::atomic::Ordering::Relaxed);
        let exact = PROF_EXACT.swap(0, std::sync::atomic::Ordering::Relaxed);
        let combo = PROF_COMBO.swap(0, std::sync::atomic::Ordering::Relaxed);
        let scored = PROF_SCORED.swap(0, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "PHASE scoreprof memo_hit={hit} memo_miss={miss} heap_push={push} heap_replace={repl} nrel={nrel} fmatch={fmatch} exact={exact} combo={combo} scored={scored}"
        );
    }
    (results, n, false)
}

// Heap priority is the page order inverted; equal keys must evict the worse stamp.
// `modified` is the mtime loaded at admit. The score lives in `PageHeap::scores[slot]`.
struct Ranked<'a> {
    total: i32,
    modified: u64,
    stamp: u64,
    file: &'a FileItem,
    slot: u32,
}
impl PartialEq for Ranked<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.total == other.total
            && self.stamp == other.stamp
            && self.modified == other.modified
            && std::ptr::eq(self.file, other.file)
    }
}
impl Eq for Ranked<'_> {}
impl PartialOrd for Ranked<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.total
            .cmp(&other.total)
            .then_with(|| self.modified.cmp(&other.modified))
            .then_with(|| other.stamp.cmp(&self.stamp))
            .then_with(|| std::ptr::from_ref(other.file).cmp(&std::ptr::from_ref(self.file)))
    }
}

type Heap<'a> = PageHeap<'a>;

// Sifts move the order key. Scores stay in `scores` and a replaced row reuses its slot.
struct PageHeap<'a> {
    heap: BinaryHeap<Reverse<Ranked<'a>>>,
    scores: Vec<Score>,
}

impl<'a> PageHeap<'a> {
    fn new() -> Self {
        // The buffer is empty here, so no file reference is live.
        let heap = PAGE_HEAP.with(|c| {
            let mut h = c.borrow_mut();
            h.clear();
            std::mem::take(&mut *h)
        });
        let scores = PAGE_SCORES.with(|c| {
            let mut s = c.borrow_mut();
            s.clear();
            std::mem::take(&mut *s)
        });
        Self {
            heap: unsafe { std::mem::transmute(heap) },
            scores,
        }
    }

    fn reclaim(self) {
        let Self { mut heap, mut scores } = self;
        heap.clear();
        scores.clear();
        if scores.capacity() > 4096 {
            scores.shrink_to(4096);
        }
        if heap.capacity() > 4096 {
            heap.shrink_to(4096);
        }
        let heap: BinaryHeap<Reverse<Ranked<'static>>> = unsafe { std::mem::transmute(heap) };
        PAGE_HEAP.with(|c| *c.borrow_mut() = heap);
        PAGE_SCORES.with(|c| *c.borrow_mut() = scores);
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn worst_total(&self) -> Option<i32> {
        self.heap.peek().map(|Reverse(worst)| worst.total)
    }

    fn into_pairs(self) -> Vec<(&'a FileItem, Score)> {
        let Self { mut heap, mut scores } = self;
        let rows = heap
            .drain()
            .map(|Reverse(ranked)| {
                let score = std::mem::take(&mut scores[ranked.slot as usize]);
                (ranked.file, score)
            })
            .collect();
        Self { heap, scores }.reclaim();
        rows
    }

    // Page order from the admit-time key: no second mtime load, no second path rank.
    fn into_page(self) -> Vec<(&'a FileItem, Score)> {
        let Self { mut heap, mut scores } = self;
        let mut rows = PAGE_SORT.with(|slot| {
            let mut rows = std::mem::take(&mut *slot.borrow_mut());
            rows.clear();
            rows
        });
        rows.extend(heap.drain().map(|Reverse(ranked)| {
            let score = std::mem::take(&mut scores[ranked.slot as usize]);
            PageSortRow {
                file: std::ptr::from_ref(ranked.file),
                score,
                modified: ranked.modified,
                stamp: ranked.stamp,
            }
        }));
        Self { heap, scores }.reclaim();
        rows.sort_unstable_by(|a, b| {
            b.score
                .total
                .cmp(&a.score.total)
                .then_with(|| b.modified.cmp(&a.modified))
                .then_with(|| a.stamp.cmp(&b.stamp))
                .then_with(|| a.file.cmp(&b.file))
        });
        let out = rows
            .drain(..)
            .map(|row| {
                // The pointer is the file admitted into this page.
                let file = unsafe { &*row.file };
                (file, row.score)
            })
            .collect();
        PAGE_SORT.with(|slot| {
            let slot = &mut *slot.borrow_mut();
            if rows.capacity() > slot.capacity() {
                *slot = rows;
            }
        });
        out
    }
}

// Sort keys for one page. The next page on this thread reuses the buffer.
struct PageSortRow {
    file: *const FileItem,
    score: Score,
    modified: u64,
    stamp: u64,
}

thread_local! {
    static PAGE_SORT: std::cell::RefCell<Vec<PageSortRow>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// One page heap per thread. Cleared before it is stored again.
thread_local! {
    static PAGE_HEAP: std::cell::RefCell<BinaryHeap<Reverse<Ranked<'static>>>> =
        const { std::cell::RefCell::new(BinaryHeap::new()) };
    static PAGE_SCORES: std::cell::RefCell<Vec<Score>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[inline]
fn score_inputs(file: &FileItem) -> (i32, i32, bool, bool) {
    let (bits, freq, git) = file.rank_freq_git();
    let (vendored, git_dirty) = FileItem::rank_flags(bits);
    (freq, git, vendored, git_dirty)
}

// `floor` tracks the peeked worst total so a strict loss skips the heap.
#[inline]
fn sync_floor(heap: &Heap<'_>, page_cap: usize, floor: &mut i32) {
    if heap.len() >= page_cap {
        if let Some(total) = heap.worst_total() {
            *floor = total;
        }
    }
}

enum Admit {
    Push,
    Replace,
    Drop,
}

// `None` loses to `Ranked::cmp`. `Some((mtime, path_rank))` enters the heap.
fn winning_stamp(file: &FileItem, total: i32, worst: &Ranked<'_>) -> Option<(u64, u64)> {
    if total < worst.total {
        return None;
    }
    if total == worst.total {
        let modified = file.modified();
        if modified < worst.modified {
            return None;
        }
        if modified == worst.modified {
            let stamp = path_rank(file);
            if stamp > worst.stamp
                || (stamp == worst.stamp
                    && std::ptr::from_ref(file) >= std::ptr::from_ref(worst.file))
            {
                return None;
            }
            return Some((modified, stamp));
        }
        return Some((modified, path_rank(file)));
    }
    Some((file.modified(), path_rank(file)))
}

fn consider<'a>(heap: &mut Heap<'a>, page_cap: usize, item: (&'a FileItem, Score)) -> Admit {
    let admitted = if heap.len() >= page_cap {
        match heap.heap.peek() {
            Some(Reverse(worst)) => winning_stamp(item.0, item.1.total, worst),
            None => None,
        }
    } else {
        Some((item.0.modified(), path_rank(item.0)))
    };
    let Some((modified, stamp)) = admitted else {
        return Admit::Drop;
    };
    let total = item.1.total;
    let replacing = heap.len() >= page_cap;
    let slot = if replacing {
        let Reverse(old) = heap.heap.pop().expect("page heap");
        let slot = old.slot;
        heap.scores[slot as usize] = item.1;
        slot
    } else {
        let slot = heap.scores.len() as u32;
        heap.scores.push(item.1);
        slot
    };
    heap.heap.push(Reverse(Ranked {
        total,
        modified,
        stamp,
        file: item.0,
        slot,
    }));
    if replacing {
        Admit::Replace
    } else {
        Admit::Push
    }
}

fn merge<'a>(mut a: Heap<'a>, b: Heap<'a>, page_cap: usize) -> Heap<'a> {
    let Heap {
        heap: mut bh,
        scores: mut bs,
    } = b;
    while let Some(Reverse(ranked)) = bh.pop() {
        let score = std::mem::take(&mut bs[ranked.slot as usize]);
        if a.len() < page_cap {
            let slot = a.scores.len() as u32;
            a.scores.push(score);
            a.heap.push(Reverse(Ranked {
                total: ranked.total,
                modified: ranked.modified,
                stamp: ranked.stamp,
                file: ranked.file,
                slot,
            }));
        } else if a.heap.peek().is_some_and(|Reverse(worst)| ranked > *worst) {
            let Reverse(old) = a.heap.pop().expect("page heap");
            a.scores[old.slot as usize] = score;
            a.heap.push(Reverse(Ranked {
                total: ranked.total,
                modified: ranked.modified,
                stamp: ranked.stamp,
                file: ranked.file,
                slot: old.slot,
            }));
        }
    }
    Heap { heap: bh, scores: bs }.reclaim();
    a
}

fn take_tls<T: Default>(cell: &std::cell::RefCell<Vec<T>>, min_cap: usize) -> Vec<T> {
    let mut b = cell.borrow_mut();
    b.clear();
    if b.capacity() < min_cap {
        b.reserve(min_cap);
    }
    std::mem::take(&mut *b)
}

fn take_tls_init<T: Copy + Default>(cell: &std::cell::RefCell<Vec<T>>, len: usize) -> Vec<T> {
    let mut b = cell.borrow_mut();
    let cap = b.capacity();
    if cap < len {
        b.reserve(len - cap);
    }
    if b.len() < len {
        b.resize(len, T::default());
    } else {
        b.truncate(len);
    }
    std::mem::take(&mut *b)
}

// Stable tie-break for equal scores; `sort_unstable_by` alone orders runs differently.
#[inline]
fn path_rank(file: &FileItem) -> u64 {
    file.path.rank_key()
}

#[inline]
fn evidence_ub_for(base: i32, intent: crate::intent::QueryIntent, collapsed: bool) -> i32 {
    if collapsed {
        return 0;
    }
    // evidence_bonus floors base at 1; mirror it or base=0 undercounts by 1.
    let base = base.max(1);
    match intent {
        crate::intent::QueryIntent::FileName => {
            // Mirrors evidence_bonus, where the exact branch dominates the substring one.
            base + base / 5 + base / 4
        }
        crate::intent::QueryIntent::Path => base / 2 + base / 5 + base / 10 + base / 2,
        crate::intent::QueryIntent::Concept | crate::intent::QueryIntent::Typo => {
            // Exact, full word coverage, role, substring, related, same dir, extension.
            base / 2 + base * 100 / 250 + base / 4 + base / 6 + base / 10 + base / 10 + base / 20
        }
        crate::intent::QueryIntent::WorkingSet => base / 5 + base / 8,
    }
}

// Fallback rows are sorted by match index. The scorer walks that same order.
#[inline(always)]
fn take_fallback_hit(
    hits: &[(u32, bool, i32)],
    at: &mut usize,
    match_idx: usize,
) -> Option<(bool, i32)> {
    if hits.is_empty() {
        return None;
    }
    while *at < hits.len() && (hits[*at].0 as usize) < match_idx {
        *at += 1;
    }
    if *at < hits.len() && hits[*at].0 as usize == match_idx {
        let row = hits[*at];
        *at += 1;
        Some((row.1, row.2))
    } else {
        None
    }
}

// `freq == 0` is the common file. A negative frecency lowers the exact bound only.
#[inline(always)]
fn dirty_bonus_of(base: i32, git_dirty: bool) -> i32 {
    if git_dirty { base * 15 / 100 } else { 0 }
}

// Signed term, then the non-negative term the cheap bound uses.
#[inline(always)]
fn freq_terms(base: i32, freq: i32) -> (i32, i32) {
    if freq > 0 {
        let term = base.saturating_mul(freq) / 100;
        (term, term)
    } else if freq < 0 {
        (base.saturating_mul(freq) / 100, 0)
    } else {
        (0, 0)
    }
}

#[inline(always)]
fn freq_term_of(base: i32, freq: i32) -> i32 {
    freq_terms(base, freq).0
}

// True when even a generous bound loses to `floor`. The exact bound is then unnecessary.
#[inline(always)]
fn ub_cannot_enter(base: i32, freq_pos: i32, git_recency: i32, combo: i32, floor: i32) -> bool {
    // Six times the match score covers every bonus that scales with it, plus the base-0 evidence floor.
    // The match score fits in a u16, so the scale itself does not overflow.
    debug_assert!(base <= u16::MAX as i32);
    let cheap = (base * 6 + 2)
        .saturating_add(freq_pos)
        .saturating_add(git_recency.max(0))
        .saturating_add(combo);
    cheap < floor
}

#[inline]
fn ub_total_for(
    base: i32,
    freq_term: i32,
    git_recency: i32,
    dirty: i32,
    intent: crate::intent::QueryIntent,
    collapsed: bool,
    path_on: bool,
    combo: i32,
    fixed_ub: &[i32],
) -> i32 {
    let fixed = if base >= 0 && (base as usize) < fixed_ub.len() {
        fixed_ub[base as usize]
    } else {
        let filename = if collapsed {
            (base / 6).min(30)
        } else {
            // Exact filename bonus is `base / 5 * 2`. Fuzzy filename bonuses are smaller.
            base / 5 * 2
        };
        let path = if path_on && !collapsed { base } else { 0 };
        filename + path + evidence_ub_for(base, intent, collapsed)
    };
    base.saturating_add(freq_term)
        .saturating_add(fixed)
        .saturating_add(dirty)
        .saturating_add(git_recency)
        .saturating_add(combo)
}

// `lower` is the query needle, ASCII-lowercased once.
#[inline]
fn scalar_flip<'a>(needle: &[u8], buf: &'a mut [u8; 256]) -> &'a [u8] {
    if crate::interseq_sw::prepare_scalar_flip(needle, buf) {
        &buf[..needle.len()]
    } else {
        &[]
    }
}

#[inline(always)]
fn score_scalar_window_direct(
    direct_short: bool,
    needle: &[u8],
    flip: &[u8],
    hay: &[u8],
    p: &crate::interseq_sw::SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    if direct_short {
        crate::interseq_sw::score_one_scalar_short(needle, flip, hay, p, include_prefix)
    } else {
        score_scalar_window(needle, flip, hay, p, include_prefix)
    }
}

fn score_scalar_window(
    needle: &[u8],
    flip: &[u8],
    hay: &[u8],
    p: &crate::interseq_sw::SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    if flip.len() == needle.len() {
        crate::interseq_sw::score_one_scalar_flipped(needle, flip, hay, p, include_prefix)
    } else {
        crate::interseq_sw::score_one_scalar_full(needle, hay, p, include_prefix)
    }
}

fn fill_ascii_lower(needle: &[u8], buf: &mut [u8]) -> bool {
    if needle.len() > buf.len() {
        return false;
    }
    crate::simd_path::fold_ascii_needle(needle, buf);
    true
}

// Same as `needle.eq_ignore_ascii_case(hay)` when `folded` and `lower` match it.
#[inline]
fn eq_query_needle(hay: &[u8], needle: &[u8], lower: &[u8], folded: bool) -> bool {
    if folded {
        eq_ascii_lower(hay, lower)
    } else {
        hay.len() == needle.len() && crate::simd_path::eq_ignore_ascii_bytes(hay, needle)
    }
}

#[inline]
fn eq_ascii_lower(hay: &[u8], lower: &[u8]) -> bool {
    // `lower` is ASCII-lowercased. Equal lengths stay inside `hay`.
    hay.len() == lower.len()
        && unsafe { crate::simd_string_utils::case::eq_lowered_case(hay.as_ptr(), lower) }
}

fn fill_path_segments(
    ev: &mut crate::intent::Evidence,
    file: &FileItem,
    arena: ArenaPtr,
    scratch: &mut [u8],
    pq: &crate::intent::PreparedQuery<'_>,
) {
    let Some(pt) = pq.path_tokens.as_ref() else {
        return;
    };
    if pt.segments.is_empty() {
        return;
    }
    let last_q = crate::intent::query_part_lower(pq.query, &pq.lower, pt.segments[pt.segments.len() - 1]);
    let mut qi = 0usize;
    let mut last_match = false;
    file.path.for_each_segment(arena, scratch, |seg| {
        last_match = eq_ascii_lower(seg, last_q);
        if qi < pt.segments.len() {
            let part = crate::intent::query_part_lower(pq.query, &pq.lower, pt.segments[qi]);
            if eq_ascii_lower(seg, part) {
                qi += 1;
            }
        }
        true
    });
    ev.last_segment_match = last_match;
    ev.segment_sequence = qi == pt.segments.len();
    ev.segment_boundary_pct = (qi * 100 / pt.segments.len()) as u8;
}

// Byte compare avoids materializing a `String` per scored match.
fn is_special_entry_point_bytes(filename: &[u8]) -> bool {
    let Some(&first) = filename.first() else {
        return false;
    };
    match first {
        b'm' => matches!(
            filename,
            b"mod.rs" | b"main.rs" | b"main.go" | b"main.c" | b"main.rb"
        ),
        b'l' => filename == b"lib.rs",
        b'i' => matches!(
            filename,
            b"index.js"
                | b"index.jsx"
                | b"index.ts"
                | b"index.tsx"
                | b"index.mjs"
                | b"index.cjs"
                | b"index.vue"
                | b"index.php"
                | b"index.rb"
        ),
        b'_' => matches!(filename, b"__init__.py" | b"__main__.py"),
        _ => false,
    }
}

fn current_file_parents(current: Option<&str>) -> (&str, &str) {
    let Some(current) = current else {
        return ("", "");
    };
    (
        current.rsplit_once('/').map(|(dir, _)| dir).unwrap_or(""),
        current
            .rsplit_once('\\')
            .map(|(dir, _)| dir)
            .unwrap_or(""),
    )
}

// One lowercase copy per search when the parent has `A..=Z`.
fn parent_bytes<'a>(dir: &'a str, buf: &'a mut Vec<u8>) -> &'a [u8] {
    let b = dir.as_bytes();
    if !crate::simd_path::needle_has_ascii_upper(b) {
        return b;
    }
    buf.resize(b.len(), 0);
    crate::simd_path::fold_ascii_needle(b, buf);
    buf
}

fn score_filtered_by_frecency<'a>(
    files: &FileItems<'a>,
    context: &ScoringContext,
    arena: ArenaPtr,
) -> (Vec<(&'a FileItem, Score)>, usize) {
    // Stored frecency pair, loaded once; the bound and the score share it.
    #[inline(always)]
    fn load_frecency(file: &FileItem) -> Option<(i32, crate::types::ScoreHot)> {
        let hot = file.score_hot_if_live()?;
        let freq = hot.access as i32 + (hot.modification as i32).saturating_mul(4);
        Some((freq, hot))
    }
    // Working-set queries are constant (""); build the state once.
    let pq = crate::intent::PreparedQuery::new("");
    let (cur_slash_dir, cur_back_dir) = current_file_parents(context.current_file);
    let mut slash_buf = Vec::new();
    let mut back_buf = Vec::new();
    let slash_l = parent_bytes(cur_slash_dir, &mut slash_buf);
    let back_l = parent_bytes(cur_back_dir, &mut back_buf);
    let cur_fname = context
        .current_file
        .map(|c| c.rsplit(['/', '\\']).next().unwrap_or(c))
        .unwrap_or("");
    let related = crate::intent::related_key(cur_fname);
    let score_file = |file: &'a FileItem, total_frecency_score: i32, hot: crate::types::ScoreHot| {
        // Give modified/dirty files a boost even in frecency-only mode
        let git_status_boost = if hot.git_dirty {
            total_frecency_score * 15 / 100
        } else {
            0
        };
        let git_recency_boost = hot.git_recency as i32;

        let current_file_penalty =
            calculate_current_file_penalty(file, total_frecency_score, context, arena);

        // Working set: same dir / related siblings of the current file.
        let mut evidence_bonus = 0i32;
        if context.current_file.is_some() {
            let mut name_buf =
                std::mem::MaybeUninit::<[u8; crate::simd_path::PATH_BUF_SIZE]>::uninit();
            let (byte_len_u, name_off) = file.path.span_lens();
            let off = name_off as usize;
            let total = byte_len_u as usize;
            let fname = if let Some(name) = file.path.filename_in_chunk_at(arena, off, total) {
                name
            } else {
                // SAFETY: `filename_at` writes the returned bytes before they are read.
                file.path.filename_at(
                    arena,
                    unsafe { &mut *name_buf.as_mut_ptr() },
                    off,
                    total,
                )
            };
            let mut ev = crate::intent::evidence_prepared(
                &pq,
                "",
                fname,
                crate::intent::QueryIntent::WorkingSet,
                None,
            );
            ev.related = crate::intent::is_related_key(fname, &related);
            ev.same_dir = file.path.same_parent_dir_lower(arena, slash_l, back_l);
            let base = total_frecency_score.max(8);
            evidence_bonus =
                crate::intent::evidence_bonus(crate::intent::QueryIntent::WorkingSet, &ev, base);
        }

        let total = total_frecency_score
            .saturating_add(git_status_boost)
            .saturating_add(git_recency_boost)
            .saturating_add(evidence_bonus)
            .saturating_add(current_file_penalty);

        let score = Score {
            total,
            base_score: 0,
            filename_bonus: 0,
            distance_penalty: 0,
            special_filename_bonus: 0,
            combo_match_boost: 0,
            path_alignment_bonus: 0,
            evidence_bonus,
            current_file_penalty,
            frecency_boost: total_frecency_score,
            git_status_boost,
            git_recency_boost,
            exact_match: false,
            match_type: "frecency",
        };

        (file, score)
    };

    // Sound upper bound from register fields only; ignored penalties are <= 0,
    // so the bound stays above every actual total.
    let has_current = context.current_file.is_some();
    // Working-set evidence depends only on the floored frecency.
    let mut ws_ub = [0i32; 1024];
    if has_current {
        for (b, slot) in ws_ub.iter_mut().enumerate() {
            *slot = evidence_ub_for(b as i32, crate::intent::QueryIntent::WorkingSet, false);
        }
    }
    let ub_file = |hot: &crate::types::ScoreHot, freq: i32| {
        let status = if hot.git_dirty {
            freq * 15 / 100
        } else {
            0
        };
        let evidence = if has_current {
            let b = freq.max(8) as usize;
            if b < ws_ub.len() {
                ws_ub[b]
            } else {
                evidence_ub_for(b as i32, crate::intent::QueryIntent::WorkingSet, false)
            }
        } else {
            0
        };
        freq.saturating_add(status)
            .saturating_add(hot.git_recency as i32)
            .saturating_add(evidence)
    };

    // Page-bounded top-k as in the fuzzy path; `total` still counts every live file.
    let page_cap = context
        .pagination
        .offset
        .saturating_add(if context.pagination.limit > 0 {
            context.pagination.limit
        } else {
            usize::MAX / 4
        })
        .saturating_add(1);
    let use_topk = page_cap < files.len();
    let no_prune = std::env::var_os("FFF_NO_PRUNE").is_some();
    // Dispatch plan shared by both frecency loops below.
    let fre_bytes = crate::dispatch::sampled_path_avg(files) + std::mem::size_of::<Score>();
    let fre_cap = context
        .max_threads
        .max(1)
        .min(crate::parallelism::search_pool_threads());
    let fre_plan = crate::dispatch::plan_matches(files.len(), fre_bytes, fre_cap);

    if !use_topk {
        let scored: Vec<(&'a FileItem, Score)> = match files {
            FileItems::All(s)
                if fre_plan.threads >= 2
                    && context.max_threads < crate::parallelism::search_pool_threads() =>
            {
                // Binding width: one planned chunk per task, at most max_threads tasks.
                let span = fre_plan.chunk.max(1);
                let n_chunks = s.len().div_ceil(span);
                let parts: Vec<Vec<(&'a FileItem, Score)>> = (0..n_chunks)
                    .into_par_iter()
                    .map(|c| {
                        let lo = c * span;
                        let hi = ((c + 1) * span).min(s.len());
                        s[lo..hi]
                            .iter()
                            .filter_map(|f| {
                                let (freq, hot) = load_frecency(f)?;
                                Some(score_file(f, freq, hot))
                            })
                            .collect()
                    })
                    .collect();
                parts.into_iter().flatten().collect()
            }
            FileItems::All(s) if fre_plan.threads >= 2 => s
                .par_iter()
                .filter_map(|f| {
                    let (freq, hot) = load_frecency(f)?;
                    Some(score_file(f, freq, hot))
                })
                .collect(),
            FileItems::All(s) => s
                .iter()
                .filter_map(|f| {
                    let (freq, hot) = load_frecency(f)?;
                    Some(score_file(f, freq, hot))
                })
                .collect(),
            FileItems::Filtered { base, idx } => idx
                .iter()
                .filter_map(|&i| {
                    let f = &base[i as usize];
                    let (freq, hot) = load_frecency(f)?;
                    Some(score_file(f, freq, hot))
                })
                .collect(),
        };
        let n = scored.len();
        return (scored, n);
    }

    fn insert_ranked<'a>(heap: &mut Heap<'a>, page_cap: usize, item: (&'a FileItem, Score)) {
        let _ = consider(heap, page_cap, item);
    }

    // Filtered views are constrained subsets: serial heap, same prune.
    if let FileItems::Filtered { base, idx } = files {
        let mut heap: Heap<'_> = Heap::new();
        let mut total = 0usize;
        for &i in idx.iter() {
            let f = &base[i as usize];
            let Some((freq, hot)) = load_frecency(f) else {
                continue;
            };
            total += 1;
            if !no_prune
                && has_current
                && heap.len() >= page_cap
                && heap.worst_total().is_some_and(|total| ub_file(&hot, freq) < total)
            {
                continue;
            }
            insert_ranked(&mut heap, page_cap, score_file(f, freq, hot));
        }
        return (
            heap.into_pairs(),
            total,
        );
    }

    let FileItems::All(s) = files else {
        unreachable!("Filtered handled above")
    };
    if fre_plan.threads < 2 {
        let mut heap: Heap<'_> = Heap::new();
        let mut total = 0usize;
        for f in s.iter() {
            let Some((freq, hot)) = load_frecency(f) else {
                continue;
            };
            total += 1;
            if !no_prune
                && has_current
                && heap.len() >= page_cap
                && heap.worst_total().is_some_and(|total| ub_file(&hot, freq) < total)
            {
                continue;
            }
            insert_ranked(&mut heap, page_cap, score_file(f, freq, hot));
        }
        return (
            heap.into_pairs(),
            total,
        );
    }

    let chunk = fre_plan.chunk.max(1);
    let n = s.len();
    let n_chunks = n.div_ceil(chunk);
    let (heap, total) = crate::parallelism::run_on_search_pool(|| {
        use rayon::prelude::*;
        (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let mut h: Heap<'_> = Heap::new();
                let mut t = 0usize;
                let hi = ((chunk_idx + 1) * chunk).min(n);
                for f in &s[chunk_idx * chunk..hi] {
                    let Some((freq, hot)) = load_frecency(f) else {
                        continue;
                    };
                    t += 1;
                    if !no_prune
                        && has_current
                        && h.len() >= page_cap
                        && h.worst_total().is_some_and(|total| ub_file(&hot, freq) < total)
                    {
                        continue;
                    }
                    insert_ranked(&mut h, page_cap, score_file(f, freq, hot));
                }
                (h, t)
            })
            .reduce(
                || (Heap::new(), 0usize),
                |(ha, ta), (hb, tb)| (merge(ha, hb, page_cap), ta + tb),
            )
    });
    (
        heap.into_pairs(),
        total,
    )
}

#[inline]
fn calculate_current_file_penalty(
    file: &FileItem,
    base_score: i32,
    context: &ScoringContext,
    arena: ArenaPtr,
) -> i32 {
    let mut penalty = 0i32;

    if let Some(current) = context.current_file
        && file.path.byte_len as usize == current.len()
        && file.relative_path_eq(arena, current)
    {
        penalty -= base_score;
    }

    penalty
}

// `results` is already D-040 order. Slice the page; do not reload mtimes.
fn paginate_ranked<'a>(
    mut results: Vec<(&'a FileItem, Score)>,
    context: &ScoringContext,
    total_matched: usize,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    if total_matched == 0 || context.pagination.offset >= total_matched {
        return (vec![], vec![], total_matched);
    }
    let limit = if context.pagination.limit > 0 {
        context.pagination.limit
    } else {
        total_matched
    };
    let offset = context.pagination.offset.min(results.len());
    if offset > 0 {
        results.drain(..offset);
    }
    if results.len() > limit {
        results.truncate(limit);
    }
    let mut items = take_page_items();
    let mut scores = take_page_scores();
    items.reserve(results.len());
    scores.reserve(results.len());
    for (file, score) in results {
        items.push(file);
        scores.push(score);
    }
    (items, scores, total_matched)
}

fn finish_page<'a>(
    results: Vec<(&'a FileItem, Score)>,
    context: &ScoringContext,
    total_matched: usize,
    ranked: bool,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    if ranked {
        paginate_ranked(results, context, total_matched)
    } else {
        sort_and_paginate(results, context, total_matched)
    }
}

// Descending page; the UI handles prompt-position rendering order.
#[tracing::instrument(skip_all, level = tracing::Level::DEBUG)]
fn sort_and_paginate<'a>(
    mut results: Vec<(&'a FileItem, Score)>,
    context: &ScoringContext,
    total_matched: usize,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    if total_matched == 0 {
        return (vec![], vec![], 0);
    }

    let offset = context.pagination.offset;
    let limit = if context.pagination.limit > 0 {
        context.pagination.limit
    } else {
        total_matched
    };

    if offset >= total_matched {
        tracing::warn!(
            offset = offset,
            total_matched = total_matched,
            "Pagination: offset >= total_matched, returning empty"
        );

        return (vec![], vec![], total_matched);
    }

    let items_needed = offset
        .saturating_add(limit)
        .min(total_matched)
        .min(results.len());
    if results.len() > 1 && items_needed > 0 {
        // One mtime load per row. Comparisons read this, not the atomic.
        thread_local! {
            static SORT_KEYS: std::cell::RefCell<Vec<(i32, u64, u64, *const FileItem)>> =
                const { std::cell::RefCell::new(Vec::new()) };
            static ORDER: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let n = results.len();
        let mut keys = SORT_KEYS.with(|cell| {
            let mut keys = cell.borrow_mut();
            keys.clear();
            let cap = keys.capacity();
            if cap < n {
                keys.reserve(n - cap);
            }
            std::mem::take(&mut *keys)
        });
        for (file, score) in &results {
            keys.push((
                score.total,
                file.modified(),
                path_rank(file),
                std::ptr::from_ref(*file),
            ));
        }
        let cmp_at = |i: usize, j: usize| {
            keys[j]
                .0
                .cmp(&keys[i].0)
                .then_with(|| keys[j].1.cmp(&keys[i].1))
                .then_with(|| keys[i].2.cmp(&keys[j].2))
                .then_with(|| keys[i].3.cmp(&keys[j].3))
        };
        if items_needed < n && items_needed <= 2048 && n <= u32::MAX as usize {
            // Prefix sort compares in place, then swaps. The slot is still the source row.
            let base = results.as_ptr() as usize;
            let stride = std::mem::size_of_val(&results[0]);
            sort_prefix_with_buffer(&mut results, items_needed, |a, b| {
                let ia = (a as *const _ as usize - base) / stride;
                let ib = (b as *const _ as usize - base) / stride;
                keys[ib]
                    .0
                    .cmp(&keys[ia].0)
                    .then_with(|| keys[ib].1.cmp(&keys[ia].1))
                    .then_with(|| keys[ia].2.cmp(&keys[ib].2))
                    .then_with(|| keys[ia].3.cmp(&keys[ib].3))
            });
            results.truncate(items_needed);
        } else {
            ORDER.with(|cell| {
                let mut order = cell.borrow_mut();
                let order_cap = order.capacity();
                if order_cap < n {
                    order.reserve(n - order_cap);
                }
                order.clear();
                order.extend(0..n as u32);
                sort_with_buffer(&mut order, |a, b| cmp_at(*a as usize, *b as usize));
                apply_source_order(&mut results, &mut order);
            });
        }
        SORT_KEYS.with(|cell| *cell.borrow_mut() = keys);
    }

    if offset > 0 {
        results.drain(..offset.min(results.len()));
    }
    if results.len() > limit {
        results.truncate(limit);
    }

    let mut items = take_page_items();
    let mut scores = take_page_scores();
    items.reserve(results.len());
    scores.reserve(results.len());
    for (file, score) in results {
        items.push(file);
        scores.push(score);
    }
    (items, scores, total_matched)
}

// The page returned to the caller. Dropping it gives the file list back.
thread_local! {
    static PAGE_ITEMS: std::cell::RefCell<Vec<&'static FileItem>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static RESULT_SCORES: std::cell::RefCell<Vec<Score>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn take_page_items<'a>() -> Vec<&'a FileItem> {
    let items = PAGE_ITEMS.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    let mut items: Vec<&FileItem> = unsafe { std::mem::transmute(items) };
    items.clear();
    items
}

pub(crate) fn park_page_items(mut items: Vec<&FileItem>) {
    items.clear();
    let items: Vec<&'static FileItem> = unsafe { std::mem::transmute(items) };
    PAGE_ITEMS.with(|slot| {
        let slot = &mut *slot.borrow_mut();
        if items.capacity() > slot.capacity() {
            *slot = items;
        }
    });
}

fn take_page_scores() -> Vec<Score> {
    let mut scores = RESULT_SCORES.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    scores.clear();
    scores
}

pub(crate) fn park_page_scores(scores: std::sync::Arc<Vec<Score>>) {
    if let Ok(mut buf) = std::sync::Arc::try_unwrap(scores) {
        buf.clear();
        RESULT_SCORES.with(|slot| {
            let slot = &mut *slot.borrow_mut();
            if buf.capacity() > slot.capacity() {
                *slot = buf;
            }
        });
    }
}

impl Drop for crate::types::SearchResult<'_> {
    fn drop(&mut self) {
        park_page_items(std::mem::take(&mut self.items));
    }
}

const MEMO_SLOTS: usize = 1024;
const MEMO_KEY_MAX: usize = 64;

#[derive(Clone, Copy)]
#[repr(C)]
struct MemoHead {
    tag: u64,
    hash: u64,
}

const MEMO_HEAD_EMPTY: MemoHead = MemoHead { tag: 0, hash: 0 };

// Length is the top byte. The epoch counter stays below it.
#[inline(always)]
fn memo_tag(epoch: u64, len: usize) -> u64 {
    epoch | ((len as u64) << 56)
}

#[derive(Clone, Copy)]
struct MemoBody {
    key: [u8; MEMO_KEY_MAX],
    ev: crate::intent::Evidence,
}

struct EvidenceMemo {
    heads: Vec<MemoHead>,
    bodies: Vec<MemoBody>,
}

thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static EVIDENCE_MEMO: std::cell::RefCell<EvidenceMemo> =
        const { std::cell::RefCell::new(EvidenceMemo { heads: Vec::new(), bodies: Vec::new() }) };
}
static MEMO_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static MEMO_HIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MEMO_MISS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HEAP_PUSH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HEAP_REPLACE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_NREL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_FMATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_EXACT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_COMBO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_SCORED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cold]
#[inline(never)]
fn run_timed_scan(scan: impl FnOnce()) {
    scan();
}

#[cold]
#[inline(never)]
fn note_score_profile(filename: bool, exact: bool, combo: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if filename {
        PROF_FMATCH.fetch_add(1, Relaxed);
    }
    if exact {
        PROF_EXACT.fetch_add(1, Relaxed);
    }
    if combo {
        PROF_COMBO.fetch_add(1, Relaxed);
    }
    PROF_SCORED.fetch_add(1, Relaxed);
}

#[cold]
#[inline(never)]
fn note_heap_admit(push: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if push {
        HEAP_PUSH.fetch_add(1, Relaxed);
    } else {
        HEAP_REPLACE.fetch_add(1, Relaxed);
    }
}

// One TLS borrow per chunk. The table stays warm; the epoch still retires slots.
fn with_evidence_memo(f: impl FnOnce(&mut EvidenceMemo)) {
    EVIDENCE_MEMO.with(|cell| {
        let mut memo = cell.borrow_mut();
        if memo.heads.is_empty() {
            memo.heads.resize(MEMO_SLOTS, MEMO_HEAD_EMPTY);
            memo.bodies
                .resize(MEMO_SLOTS, MemoBody { key: [0; MEMO_KEY_MAX], ev: crate::intent::Evidence::default() });
        }
        f(&mut memo);
    });
}

// Eight-byte chunks. A hit still checks the stored filename bytes.
#[inline(always)]
fn fname_hash(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x100_0000_01b3;
    let mut h = 0x51ab_3c9d_u64 ^ bytes.len() as u64;
    let p = bytes.as_ptr();
    let n = bytes.len();
    let mut i = 0usize;
    while i + 16 <= n {
        let v0 = unsafe { core::ptr::read_unaligned(p.add(i) as *const u64) };
        let v1 = unsafe { core::ptr::read_unaligned(p.add(i + 8) as *const u64) };
        h = (h ^ v0).wrapping_mul(PRIME);
        h = (h ^ v1).wrapping_mul(PRIME);
        i += 16;
    }
    while i + 8 <= n {
        let v = unsafe { core::ptr::read_unaligned(p.add(i) as *const u64) };
        h = (h ^ v).wrapping_mul(PRIME);
        i += 8;
    }
    if i + 4 <= n {
        let v = unsafe { core::ptr::read_unaligned(p.add(i) as *const u32) } as u64;
        h = (h ^ v).wrapping_mul(PRIME);
        i += 4;
    }
    while i < n {
        h = (h ^ unsafe { *p.add(i) } as u64).wrapping_mul(PRIME);
        i += 1;
    }
    h
}

// Filename-keyed evidence cache; sound only for pure (query, filename) evidence.
// Collisions replace the slot; the stored key is always verified.
fn memoized_evidence(
    memo: &mut EvidenceMemo,
    epoch: u64,
    fname: &str,
    pq: &crate::intent::PreparedQuery<'_>,
    intent: crate::intent::QueryIntent,
    record: bool,
) -> crate::intent::Evidence {
    if memo.heads.len() != MEMO_SLOTS {
        memo.heads.resize(MEMO_SLOTS, MEMO_HEAD_EMPTY);
        memo.bodies
            .resize(MEMO_SLOTS, MemoBody { key: [0; MEMO_KEY_MAX], ev: crate::intent::Evidence::default() });
    }
    let fb = fname.as_bytes();
    let h = fname_hash(fb);
    let idx = h as usize & (MEMO_SLOTS - 1);
    let head = memo.heads[idx];
    if fb.len() <= MEMO_KEY_MAX {
        let tag = memo_tag(epoch, fb.len());
        if head.tag == tag && head.hash == h {
            let body = &memo.bodies[idx];
            if memo_key_eq(&body.key, fb) {
                if record {
                    MEMO_HIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return body.ev;
            }
        }
    }
    let ev = crate::intent::evidence_prepared(pq, "", fname, intent, None);
    if record {
        MEMO_MISS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if fb.len() <= MEMO_KEY_MAX {
        let mut key = [0u8; MEMO_KEY_MAX];
        key[..fb.len()].copy_from_slice(fb);
        memo.heads[idx] = MemoHead {
            tag: memo_tag(epoch, fb.len()),
            hash: h,
        };
        memo.bodies[idx] = MemoBody { key, ev };
    }
    ev
}

#[inline(always)]
fn memo_key_eq(key: &[u8; MEMO_KEY_MAX], bytes: &[u8]) -> bool {
    let n = bytes.len();
    let mut i = 0usize;
    #[cfg(target_arch = "x86_64")]
    {
        while i + 32 <= n {
            let eq0 = unsafe {
                let a = core::arch::x86_64::_mm_loadu_si128(key.as_ptr().add(i) as *const _);
                let b = core::arch::x86_64::_mm_loadu_si128(bytes.as_ptr().add(i) as *const _);
                core::arch::x86_64::_mm_movemask_epi8(core::arch::x86_64::_mm_cmpeq_epi8(a, b))
            };
            let eq1 = unsafe {
                let a = core::arch::x86_64::_mm_loadu_si128(key.as_ptr().add(i + 16) as *const _);
                let b = core::arch::x86_64::_mm_loadu_si128(bytes.as_ptr().add(i + 16) as *const _);
                core::arch::x86_64::_mm_movemask_epi8(core::arch::x86_64::_mm_cmpeq_epi8(a, b))
            };
            if eq0 != 0xFFFF || eq1 != 0xFFFF {
                return false;
            }
            i += 32;
        }
        while i + 16 <= n {
            // SAFETY: both spans cover `i..i+16`. SSE2 is the x86_64 baseline.
            let eq = unsafe {
                let a = core::arch::x86_64::_mm_loadu_si128(key.as_ptr().add(i) as *const _);
                let b = core::arch::x86_64::_mm_loadu_si128(bytes.as_ptr().add(i) as *const _);
                core::arch::x86_64::_mm_movemask_epi8(core::arch::x86_64::_mm_cmpeq_epi8(a, b))
            };
            if eq != 0xFFFF {
                return false;
            }
            i += 16;
        }
    }
    while i + 8 <= n {
        let a = unsafe { core::ptr::read_unaligned(key.as_ptr().add(i) as *const u64) };
        let b = unsafe { core::ptr::read_unaligned(bytes.as_ptr().add(i) as *const u64) };
        if a != b {
            return false;
        }
        i += 8;
    }
    if i + 4 <= n {
        let a = unsafe { core::ptr::read_unaligned(key.as_ptr().add(i) as *const u32) };
        let b = unsafe { core::ptr::read_unaligned(bytes.as_ptr().add(i) as *const u32) };
        if a != b {
            return false;
        }
        i += 4;
    }
    while i < n {
        if key[i] != bytes[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PaginationArgs;
    use fff_query_parser::QueryParser;

    // Every part is the same scorer, so the combination is the mean of the
    // separate runs, including a third part where a stepwise divide would drift.
    #[test]
    fn longer_second_part_preserves_matches_and_first_part_positions() {
        let paths: Vec<_> = [
            "src/controller.rs",
            "src/contrller.rs",
            "test/src.rs",
            "日本語/controller.tsx",
            "src/utils.rs",
            "src/controller/controller.rs",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let mut files: Vec<_> = paths
            .iter()
            .map(|path| FileItem::new_raw(path.rfind('/').unwrap() as u16 + 1, 0, 0, None, false))
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&paths, &files);
        for (file, path) in files.iter_mut().zip(strings) {
            file.set_path(path);
        }
        let arena = store.as_arena_ptr();
        let working = FileItems::All(&files);
        for parts in [
            ["src", "controller"],
            ["rs", "contrller"],
            ["rs", "日本語"],
            ["no", "absent"],
            ["src", "src/controller.rs"],
        ] {
            for max_typos in [0, 2, 4] {
                let options = neo_frizbee::Config {
                    max_typos: Some(max_typos),
                    sort: neo_frizbee::SortStrategy::Unsorted,
                    ..Default::default()
                };
                let max_path = crate::matcher::max_path_len_of(&files);
                let first = match_fuzzy_parts(
                    &parts[..1],
                    &working,
                    &options,
                    1,
                    arena,
                    max_path,
                    crate::parallelism::AbortToken::disabled(),
                );
                let mut second_options = options;
                second_options.max_typos = Some(max_typos.min(parts[1].len() as u16));
                let second = match_fuzzy_parts(
                    &parts[1..],
                    &working,
                    &second_options,
                    1,
                    arena,
                    max_path,
                    crate::parallelism::AbortToken::disabled(),
                );
                let mut expected: Vec<_> = first
                    .iter()
                    .filter_map(|a| {
                        let b = second.iter().find(|b| b.index == a.index)?;
                        Some((
                            a.index,
                            ((a.score as u32 + b.score as u32) / 2) as u16,
                            a.end_col,
                            a.exact && b.exact,
                        ))
                    })
                    .collect();
                let mut actual: Vec<_> = match_fuzzy_parts(
                    &parts,
                    &working,
                    &options,
                    4,
                    arena,
                    max_path,
                    crate::parallelism::AbortToken::disabled(),
                )
                .iter()
                .map(|m| (m.index, m.score, m.end_col, m.exact))
                .collect();
                expected.sort_unstable();
                actual.sort_unstable();
                assert_eq!(actual, expected, "{parts:?}, typos={max_typos}");
            }
        }
        for parts in [["src", "controller", "rs"], ["no", "absent", "zz"]] {
            for max_typos in [0u16, 2] {
                let options = neo_frizbee::Config {
                    max_typos: Some(max_typos),
                    sort: neo_frizbee::SortStrategy::Unsorted,
                    ..Default::default()
                };
                let max_path = crate::matcher::max_path_len_of(&files);
                let alone: Vec<Vec<neo_frizbee::Match>> = parts
                    .iter()
                    .map(|part| {
                        let one = [*part];
                        match_fuzzy_parts(
                            &one,
                            &working,
                            &options,
                            1,
                            arena,
                            max_path,
                            crate::parallelism::AbortToken::disabled(),
                        )
                    })
                    .collect();
                let mut expected = Vec::new();
                for a in &alone[0] {
                    let Some(b) = alone[1].iter().find(|m| m.index == a.index) else {
                        continue;
                    };
                    let Some(c) = alone[2].iter().find(|m| m.index == a.index) else {
                        continue;
                    };
                    let sum = (a.score as u32) + (b.score as u32) + (c.score as u32);
                    expected.push((
                        a.index,
                        (sum / 3) as u16,
                        a.end_col,
                        a.exact && b.exact && c.exact,
                    ));
                }
                let mut actual: Vec<_> = match_fuzzy_parts(
                    &parts,
                    &working,
                    &options,
                    4,
                    arena,
                    max_path,
                    crate::parallelism::AbortToken::disabled(),
                )
                .iter()
                .map(|m| (m.index, m.score, m.end_col, m.exact))
                .collect();
                expected.sort_unstable();
                actual.sort_unstable();
                assert_eq!(actual, expected, "3 {parts:?}, typos={max_typos}");
            }
        }
    }

    #[test]
    fn fuzzy_parallel_and_sequential_scores_match() {
        let paths: Vec<_> = (0..36_000)
            .map(|index| {
                format!(
                    "src/components/group_{}/controller_é_{index}.tsx",
                    index / 100
                )
            })
            .collect();
        let mut files: Vec<_> = paths
            .iter()
            .enumerate()
            .map(|(index, path)| {
                let mut file = FileItem::new_raw(
                    path.rfind('/').unwrap() as u16 + 1,
                    0,
                    index as u64,
                    Some(git2::Status::WT_MODIFIED),
                    false,
                );
                file.parent_dir_index = (index / 100) as u32;
                file.set_access_frecency_score((index % 97) as i16);
                file.set_modification_frecency_score((index % 29) as i16);
                file.set_git_recency_score((index % 13) as i16);
                file.set_deleted(index % 97 == 0);
                file
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&paths, &files);
        for (file, path) in files.iter_mut().zip(strings) {
            file.set_path(path);
        }
        let arena = store.as_arena_ptr();
        let parser = QueryParser::default();
        for text in [
            "mo",
            "controller",
            "src controller",
            "group_0/controller_é_1.tsx",
            "*.tsx",
        ] {
            let query = parser.parse(text);
            for current_file in [None, Some(paths[1].as_str())] {
                let mut context = ScoringContext {
                    query: &query,
                    max_threads: 4,
                    max_typos: 2,
                    project_path: None,
                    current_file,
                    last_same_query_match: None,
                    combo_boost_score_multiplier: 100,
                    min_combo_count: 3,
            check_deleted: true,
                    pagination: PaginationArgs::default(),
                };
                let max_path = crate::matcher::max_path_len_of(&files);
                let (mut parallel, _) = match_and_score_in_arena(
                    FileItems::All(&files),
                    &context,
                    arena,
                    max_path,
                    crate::parallelism::AbortToken::disabled(),
                    0,
                    false,
                    None,
                );
                context.max_threads = 1;
                let (mut sequential, _) = match_and_score_in_arena(
                    FileItems::All(&files),
                    &context,
                    arena,
                    max_path,
                    crate::parallelism::AbortToken::disabled(),
                    0,
                    false,
                    None,
                );
                parallel.sort_unstable_by_key(|(file, _)| file.modified());
                sequential.sort_unstable_by_key(|(file, _)| file.modified());
                assert_eq!(parallel.len(), sequential.len(), "{text}");
                for ((a, a_score), (b, b_score)) in parallel.iter().zip(&sequential) {
                    assert!(std::ptr::eq(*a, *b));
                    assert_eq!(format!("{a_score:?}"), format!("{b_score:?}"), "{text}");
                }
                let (pn, sn) = (parallel.len(), sequential.len());
                let (parallel_items, parallel_scores, _) =
                    sort_and_paginate(parallel, &context, pn);
                let (sequential_items, sequential_scores, _) =
                    sort_and_paginate(sequential, &context, sn);
                assert_eq!(
                    parallel_items
                        .iter()
                        .map(|f| f.modified())
                        .collect::<Vec<_>>(),
                    sequential_items
                        .iter()
                        .map(|f| f.modified())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    format!("{parallel_scores:?}"),
                    format!("{sequential_scores:?}")
                );
            }
        }
    }

    #[test]
    fn width_bounded_stages_match_serial_results() {
        let paths: Vec<_> = (0..20_000)
            .map(|index| format!("src/components/group_{}/controller_{index}.ts", index / 100))
            .collect();
        let mut files: Vec<_> = paths
            .iter()
            .enumerate()
            .map(|(index, path)| {
                let file = FileItem::new_raw(
                    path.rfind('/').unwrap() as u16 + 1,
                    0,
                    index as u64,
                    None,
                    false,
                );
                file.set_access_frecency_score((index % 41) as i16);
                file.set_modification_frecency_score((index % 17) as i16);
                file
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&paths, &files);
        for (file, path) in files.iter_mut().zip(strings) {
            file.set_path(path);
        }
        let arena = store.as_arena_ptr();
        let parser = QueryParser::default();
        let max_path = crate::matcher::max_path_len_of(&files);
        for text in ["controller", "mo", "src controller", "group_7"] {
            let query = parser.parse(text);
            let mut context = ScoringContext {
                query: &query,
                max_threads: 1,
                max_typos: 2,
                project_path: None,
                current_file: None,
                last_same_query_match: None,
                combo_boost_score_multiplier: 0,
                min_combo_count: 0,
                check_deleted: true,
                pagination: PaginationArgs::default(),
            };
            let (mut base, base_total) = match_and_score_in_arena(
                FileItems::All(&files),
                &context,
                arena,
                max_path,
                crate::parallelism::AbortToken::disabled(),
                0,
                false,
                None,
            );
            base.sort_unstable_by_key(|(file, _)| file.modified());
            for width in [2usize, 3, 5, 8] {
                context.max_threads = width;
                let (mut got, got_total) = match_and_score_in_arena(
                    FileItems::All(&files),
                    &context,
                    arena,
                    max_path,
                    crate::parallelism::AbortToken::disabled(),
                    0,
                    false,
                    None,
                );
                assert_eq!(got_total, base_total, "{text} width={width}");
                got.sort_unstable_by_key(|(file, _)| file.modified());
                assert_eq!(got.len(), base.len(), "{text} width={width}");
                for ((a, a_score), (b, b_score)) in got.iter().zip(&base) {
                    assert!(
                        std::ptr::eq(*a, *b),
                        "{text} width={width} files {} vs {}",
                        a.modified(),
                        b.modified()
                    );
                    assert_eq!(
                        format!("{a_score:?}"),
                        format!("{b_score:?}"),
                        "{text} width={width}"
                    );
                }
            }
        }
    }

    #[test]
    fn one_part_highlight_keeps_sorted_ranges() {
        let parser = QueryParser::default();
        let query = parser.parse("src");
        let bytes = b"src/lib.rs".to_vec();
        let row = crate::types::CachedFileRow {
            start: 0,
            len: bytes.len() as u32,
            name_off: 4,
            size: 0,
            modified: 0,
            access_frecency_score: 0,
            modification_frecency_score: 0,
            total_frecency_score: 0,
            git_status: "",
            git_recency_score: 0,
            is_binary: false,
        };
        let offsets = fuzzy_match_byte_offsets_for_page(&query, &bytes, &[row], 2);
        assert_eq!(offsets.len(), 1);
        assert!(!offsets[0].is_empty());
        assert_eq!(offsets[0][0].0, 0);
        let mut prev = 0u32;
        for (start, end) in &offsets[0] {
            assert!(*start >= prev);
            assert!(*end > *start);
            prev = *end;
        }
    }

    #[test]
    fn directory_exact_name_matches_without_copying_it() {
        let paths = ["src/components", "src/handler", "other/components_extra"];
        let mut builder = crate::simd_path::ChunkedPathStoreBuilder::new(paths.len());
        let dirs: Vec<DirItem> = paths
            .iter()
            .map(|dir_rel| {
                let chunked = builder.add_dir_immediate(dir_rel);
                let last_seg = dir_rel
                    .rfind(std::path::is_separator)
                    .map(|i| i + 1)
                    .unwrap_or(0) as u16;
                DirItem::new(chunked, last_seg)
            })
            .collect();
        let store = builder.finish();
        let arena = store.as_arena_ptr();
        let parser = QueryParser::default();
        let query = parser.parse("components");
        let context = ScoringContext {
            query: &query,
            project_path: None,
            current_file: None,
            max_typos: 2,
            max_threads: 1,
            last_same_query_match: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            check_deleted: true,
            pagination: PaginationArgs::default(),
        };
        let (items, scores, total) = fuzzy_match_and_score_dirs(&dirs, &context, arena, arena);
        assert!(total >= 1);
        assert_eq!(scores.len(), items.len());
        let exact_idx = scores
            .iter()
            .position(|s| s.match_type == "exact_dirname")
            .expect("exact dirname");
        assert!(std::ptr::eq(items[exact_idx], &dirs[0]));
        assert_eq!(
            scores[exact_idx].total,
            scores.iter().map(|s| s.total).max().unwrap()
        );
    }

    #[test]
    fn pagination_matches_full_ranking() {
        let files: Vec<_> = (0..257)
            .map(|i| FileItem::new_raw(0, 0, i as u64, None, false))
            .collect();
        let dirs: Vec<_> = (0..257)
            .map(|_| DirItem::new(crate::simd_path::ChunkedString::empty(), 0))
            .collect();
        let parser = QueryParser::default();
        let query = parser.parse("");
        for count in [0, 1, 3, 100, 101, 257] {
            let results: Vec<_> = files[..count]
                .iter()
                .enumerate()
                .map(|(index, file)| {
                    (
                        file,
                        Score {
                            total: (index * 73 % 257) as i32,
                            ..Default::default()
                        },
                    )
                })
                .collect();
            let mut expected = results.clone();
            expected.sort_by(|a, b| {
                b.1.total
                    .cmp(&a.1.total)
                    .then_with(|| b.0.modified().cmp(&a.0.modified()))
            });
            for offset in [0, 1, count / 2, count, usize::MAX] {
                for limit in [0, 1, 2, 50, count, usize::MAX] {
                    let context = ScoringContext {
                        query: &query,
                        max_threads: 1,
                        max_typos: 0,
                        project_path: None,
                        current_file: None,
                        last_same_query_match: None,
                        combo_boost_score_multiplier: 0,
                        min_combo_count: 0,
                check_deleted: true,
                        pagination: PaginationArgs { offset, limit },
                    };
                    let expected_page: Vec<_> = expected
                        .iter()
                        .skip(offset)
                        .take(if limit == 0 { count } else { limit })
                        .map(|(_, score)| score.total)
                        .collect();
                    let (items, scores, total) =
                        sort_and_paginate(results.clone(), &context, count);
                    assert_eq!(total, count);
                    assert_eq!(items.len(), expected_page.len());
                    assert_eq!(
                        scores.iter().map(|score| score.total).collect::<Vec<_>>(),
                        expected_page
                    );
                    let dir_results = results
                        .iter()
                        .enumerate()
                        .map(|(index, (_, score))| (&dirs[index], score.clone()))
                        .collect();
                    let (items, scores, total) = sort_and_paginate_dirs(dir_results, &context);
                    assert_eq!(total, count);
                    assert_eq!(items.len(), expected_page.len());
                    assert_eq!(
                        scores.iter().map(|score| score.total).collect::<Vec<_>>(),
                        expected_page
                    );
                }
            }
        }
    }

    #[test]
    fn tied_scores_paginate_deterministically() {
        let files: Vec<_> = (0..3000)
            .map(|_| FileItem::new_raw(0, 0, 1, None, false))
            .collect();
        let parser = QueryParser::default();
        let query = parser.parse("");
        let page = |shuffle: usize, offset: usize| {
            let mut results: Vec<_> = files.iter().map(|file| (file, Score::default())).collect();
            results.rotate_left(shuffle);
            results.reverse();
            let context = ScoringContext {
                query: &query,
                max_threads: 1,
                max_typos: 0,
                project_path: None,
                current_file: None,
                last_same_query_match: None,
                combo_boost_score_multiplier: 0,
                min_combo_count: 0,
                check_deleted: true,
                pagination: PaginationArgs { offset, limit: 8 },
            };
            let n = results.len();
            sort_and_paginate(results, &context, n).0
        };
        for offset in [0, 8, 16] {
            let expected: Vec<_> = files[offset..offset + 8].iter().collect();
            for shuffle in [0, 7, 1500, 2999] {
                assert!(
                    page(shuffle, offset)
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| std::ptr::eq(*a, *b)),
                    "offset={offset} shuffle={shuffle}"
                );
            }
        }
    }

    #[test]
    fn merge_offsets_reuses_storage() {
        let mut ranges: SmallVec<[(u32, u32); 4]> = smallvec::smallvec![
            (20, 25),
            (2, 6),
            (0, 3),
            (6, 10),
            (11, 11),
            (24, 30),
            (40, 45),
        ];
        let storage = ranges.as_mut_ptr();
        let merged = merge_byte_offsets(ranges);
        assert_eq!(merged.as_slice(), &[(0, 10), (20, 30), (40, 45)]);
        assert_eq!(merged.as_ptr(), storage);
    }

    fn make_test_files(specs: &[(&str, i32, u64)]) -> (Vec<(FileItem, Score)>, ArenaPtr) {
        let path_strings: Vec<String> = specs.iter().map(|(p, _, _)| p.to_string()).collect();
        let items: Vec<FileItem> = specs
            .iter()
            .map(|(p, _, _)| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&path_strings, &items);
        let arena = store.as_arena_ptr();
        let result: Vec<(FileItem, Score)> = specs
            .iter()
            .enumerate()
            .map(|(i, &(_, score, modified))| {
                let filename_start = path_strings[i]
                    .rfind(std::path::is_separator)
                    .map(|j| j + 1)
                    .unwrap_or(0) as u16;
                let mut file = FileItem::new_raw(filename_start, 0, modified, None, false);
                file.set_path(strings[i].clone());
                file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
                let score_obj = Score {
                    total: score,
                    base_score: score,
                    filename_bonus: 0,
                    distance_penalty: 0,
                    special_filename_bonus: 0,
                    current_file_penalty: 0,
                    frecency_boost: 0,
                    git_status_boost: 0,
                    git_recency_boost: 0,
                    exact_match: false,
                    match_type: "test",
                    combo_match_boost: 0,
                    path_alignment_bonus: 0,
                    evidence_bonus: 0,
                };
                (file, score_obj)
            })
            .collect();
        std::mem::forget(store);
        (result, arena)
    }

    #[test]
    fn test_partial_sort_descending() {
        let (test_data, arena) = make_test_files(&[
            ("file1.rs", 100, 1000),
            ("file2.rs", 200, 2000),
            ("file3.rs", 50, 3000),
            ("file4.rs", 300, 4000),
            ("file5.rs", 150, 5000),
            ("file6.rs", 250, 6000),
            ("file7.rs", 80, 7000),
            ("file8.rs", 180, 8000),
            ("file9.rs", 120, 9000),
            ("file10.rs", 90, 10000),
        ]);

        let results: Vec<(&FileItem, Score)> = test_data
            .iter()
            .map(|(file, score)| (file, score.clone()))
            .collect();

        let query_str = "test";
        let parser = QueryParser::default();
        let query = parser.parse(query_str);
        let context = ScoringContext {
            query: &query,
            max_threads: 1,
            max_typos: 2,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs {
                offset: 0,
                limit: 0,
            },
        };

        let total = results.len();
        let (items, scores, total) = sort_and_paginate(results.clone(), &context, total);

        assert_eq!(total, 10);
        assert_eq!(scores.len(), 10);
        assert_eq!(scores[0].total, 300, "First should be highest score");
        assert_eq!(scores[1].total, 250, "Second should be second highest");
        assert_eq!(scores[2].total, 200, "Third should be third highest");

        assert_eq!(items[0].relative_path(arena), "file4.rs");
        assert_eq!(items[1].relative_path(arena), "file6.rs");
        assert_eq!(items[2].relative_path(arena), "file2.rs");
    }

    #[test]
    fn test_partial_sort_with_same_scores() {
        // Test tiebreaker with modified time
        let (test_data, _arena) = make_test_files(&[
            ("file1.rs", 100, 5000), // Same score, older
            ("file2.rs", 100, 8000), // Same score, newer
            ("file3.rs", 100, 3000), // Same score, oldest
            ("file4.rs", 200, 1000),
            ("file5.rs", 200, 9000), // Higher score, newest
        ]);

        let results: Vec<(&FileItem, Score)> = test_data
            .iter()
            .map(|(file, score)| (file, score.clone()))
            .collect();

        let query_str = "test";
        let parser = QueryParser::default();
        let query = parser.parse(query_str);
        let context = ScoringContext {
            query: &query,
            max_threads: 1,
            max_typos: 2,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs {
                offset: 0,
                limit: 0,
            },
        };

        let n = results.len();
        let (items, scores, _) = sort_and_paginate(results, &context, n);

        assert_eq!(scores.len(), 5);
        assert_eq!(scores[0].total, 200);
        assert_eq!(items[0].modified(), 9000, "First 200 should be newest");
        assert_eq!(scores[1].total, 200);
        assert_eq!(items[1].modified(), 1000, "Second 200 should be older");
        assert_eq!(scores[2].total, 100);
        assert_eq!(items[2].modified(), 8000, "First 100 should be newest");
        assert_eq!(scores[3].total, 100);
        assert_eq!(items[3].modified(), 5000);
        assert_eq!(scores[4].total, 100);
        assert_eq!(items[4].modified(), 3000, "Last 100 should be oldest");
    }

    #[test]
    fn test_no_partial_sort_for_small_results() {
        let (test_data, arena) = make_test_files(&[
            ("file1.rs", 100, 1000),
            ("file2.rs", 200, 2000),
            ("file3.rs", 50, 3000),
        ]);

        let results: Vec<(&FileItem, Score)> = test_data
            .iter()
            .map(|(file, score)| (file, score.clone()))
            .collect();

        let query_str = "test";
        let parser = QueryParser::default();
        let query = parser.parse(query_str);
        let context = ScoringContext {
            query: &query,
            max_threads: 1,
            max_typos: 2,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs {
                offset: 0,
                limit: 0,
            },
        };

        let n = results.len();
        let (items, scores, _) = sort_and_paginate(results, &context, n);

        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].total, 200);
        assert_eq!(scores[1].total, 100);
        assert_eq!(scores[2].total, 50);
        assert_eq!(items[0].relative_path(arena), "file2.rs");
        assert_eq!(items[1].relative_path(arena), "file1.rs");
        assert_eq!(items[2].relative_path(arena), "file3.rs");
    }

    #[test]
    fn two_part_narrowing_order_independent_sets() {
        let (test_data, arena) = make_test_files(&[
            ("src/models/user_service.rs", 10, 1),
            ("src/models/mod.rs", 10, 2),
            ("src/auth/login.rs", 10, 3),
            ("src/api/user_handler.rs", 10, 4),
            ("tests/user_test.rs", 10, 5),
            ("README.md", 10, 6),
        ]);
        let files: Vec<FileItem> = test_data.iter().map(|(file, _)| file.clone()).collect();
        let options = neo_frizbee::Config {
            max_typos: Some(2),
            sort: neo_frizbee::SortStrategy::Unsorted,
            ..Default::default()
        };
        // Pairs exclude end_col: it stays query-order-first by design.
        let run = |parts: &[&str]| {
            let mut pairs: Vec<(u32, u16)> = match_fuzzy_parts(
                parts,
                &crate::matcher::FileItems::All(&files),
                &options,
                1,
                arena,
                crate::matcher::max_path_len_of(&files),
                crate::parallelism::AbortToken::disabled(),
            )
            .iter()
            .map(|m| (m.index, m.score))
            .collect();
            pairs.sort_unstable();
            pairs
        };
        let short_first = run(&["mo", "models"]);
        let long_first = run(&["models", "mo"]);
        assert!(
            !short_first.is_empty(),
            "sample must match at least one file"
        );
        assert_eq!(
            short_first, long_first,
            "narrowing order must not change the match set"
        );
    }

    #[test]
    fn sort_and_paginate_matches_legacy_path() {
        let owned: Vec<(String, i32, u64)> = (0..150)
            .map(|i| {
                (
                    format!("src/file{i:03}.rs"),
                    (i % 10) as i32,
                    (i % 4) as u64,
                )
            })
            .collect();
        let refs: Vec<(&str, i32, u64)> =
            owned.iter().map(|(p, s, m)| (p.as_str(), *s, *m)).collect();
        let (test_data, arena) = make_test_files(&refs);
        let results: Vec<(&FileItem, Score)> = test_data
            .iter()
            .map(|(file, score)| (file, score.clone()))
            .collect();

        let query_str = "test";
        let parser = QueryParser::default();
        let query = parser.parse(query_str);
        for (offset, limit) in [
            (0, 10),
            (5, 20),
            (0, 100),
            (30, 100),
            (0, 0),
            (140, 50),
            (150, 10),
            (200, 10),
        ] {
            let context = ScoringContext {
                query: &query,
                max_threads: 1,
                max_typos: 2,
                current_file: None,
                last_same_query_match: None,
                project_path: None,
                combo_boost_score_multiplier: 100,
                min_combo_count: 3,
            check_deleted: true,
                pagination: PaginationArgs { offset, limit },
            };
            let n = results.len();
            let (items, scores, total) = sort_and_paginate(results.clone(), &context, n);
            let (exp_items, exp_scores, exp_total) =
                legacy_sort_and_paginate(results.clone(), &context, n);
            assert_eq!(
                total, exp_total,
                "total_matched offset={offset} limit={limit}"
            );
            let paths: Vec<_> = items.iter().map(|f| f.relative_path(arena)).collect();
            let exp_paths: Vec<_> = exp_items.iter().map(|f| f.relative_path(arena)).collect();
            assert_eq!(paths, exp_paths, "ordering offset={offset} limit={limit}");
            assert_eq!(
                format!("{scores:?}"),
                format!("{exp_scores:?}"),
                "scores offset={offset} limit={limit}"
            );
        }
    }

    fn legacy_sort_and_paginate<'a>(
        mut results: Vec<(&'a FileItem, Score)>,
        context: &ScoringContext,
        total_matched: usize,
    ) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
        if total_matched == 0 {
            return (vec![], vec![], 0);
        }
        let offset = context.pagination.offset;
        let limit = if context.pagination.limit > 0 {
            context.pagination.limit
        } else {
            total_matched
        };
        if offset >= total_matched {
            return (vec![], vec![], total_matched);
        }
        let items_needed = offset.saturating_add(limit).min(total_matched);
        let use_partial_sort = items_needed < total_matched / 2 && total_matched > 100;
        if use_partial_sort {
            results.select_nth_unstable_by(items_needed - 1, |a, b| {
                b.1.total
                    .cmp(&a.1.total)
                    .then_with(|| b.0.modified().cmp(&a.0.modified()))
                    .then_with(|| std::ptr::from_ref(a.0).cmp(&std::ptr::from_ref(b.0)))
            });
            results.truncate(items_needed);
        }
        sort_with_buffer(&mut results, |a, b| {
            b.1.total
                .cmp(&a.1.total)
                .then_with(|| b.0.modified().cmp(&a.0.modified()))
                .then_with(|| std::ptr::from_ref(a.0).cmp(&std::ptr::from_ref(b.0)))
        });
        if results.len() > limit {
            let page_end = std::cmp::min(offset + limit, results.len());
            let page_size = page_end - offset;
            results.drain(0..offset);
            results.truncate(page_size);
        }
        let (items, scores): (Vec<&FileItem>, Vec<Score>) = results.into_iter().unzip();
        (items, scores, total_matched)
    }
}

#[cfg(test)]
mod filename_bonus_tests {
    use super::*;
    use crate::types::PaginationArgs;
    use fff_query_parser::QueryParser;
    fn make_files(paths: &[&str]) -> (Vec<FileItem>, ArenaPtr) {
        let path_strings: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
        let items: Vec<FileItem> = paths
            .iter()
            .map(|p| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&path_strings, &items);
        let arena = store.as_arena_ptr();
        let mut result: Vec<FileItem> = items;
        for (i, file) in result.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
        }
        std::mem::forget(store);
        (result, arena)
    }

    fn make_files_with_frecency(specs: &[(&str, i16)]) -> (Vec<FileItem>, ArenaPtr) {
        let path_strings: Vec<String> = specs.iter().map(|(p, _)| p.to_string()).collect();
        let items: Vec<FileItem> = specs
            .iter()
            .map(|(p, _)| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&path_strings, &items);
        let arena = store.as_arena_ptr();
        let mut result: Vec<FileItem> = items;
        for (i, file) in result.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
            file.set_access_frecency_score(specs[i].1);
        }
        std::mem::forget(store);
        (result, arena)
    }

    // Run `fuzzy_match_and_score_files` with production-like max_typos scaling.
    fn search(files: &[FileItem], query: &str, arena: ArenaPtr) -> Vec<(String, Score)> {
        let parser = QueryParser::default();
        let parsed = parser.parse(query);

        let effective_query = match &parsed.fuzzy_query {
            FuzzyQuery::Text(t) => *t,
            FuzzyQuery::Parts(parts) if !parts.is_empty() => parts[0],
            _ => query,
        };
        let max_typos = (effective_query.len() as u16 / 4).clamp(2, 6);

        let ctx = ScoringContext {
            query: &parsed,
            max_threads: 1,
            max_typos,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs {
                offset: 0,
                limit: 100,
            },
        };
        let (items, scores, _) = fuzzy_match_and_score_files(
            files,
            &ctx,
            files.len(),
            arena,
            arena,
            max_path_len_of(&files),
            crate::parallelism::AbortToken::disabled(),
        );
        items
            .iter()
            .zip(scores.iter())
            .map(|(f, s)| (f.relative_path(arena).to_string(), s.clone()))
            .collect()
    }

    #[test]
    fn match_cache_rejects_stale_generation_and_candidates() {
        let idx = vec![1u32, 3, 5];
        let matches = vec![neo_frizbee::Match {
            score: 10,
            index: 0,
            exact: false,
            end_col: 2,
        }];
        let mut cache = MatchCache::default();
        cache.put(
            7,
            &["pick"],
            2,
            MatchSide::Base,
            Some(std::sync::Arc::new(idx.clone())),
            Some(std::sync::Arc::new(matches)),
        );
        assert!(cache.get(7, &["pick"], 2, MatchSide::Base, &idx).is_some());
        assert!(cache.get(8, &["pick"], 2, MatchSide::Base, &idx).is_none());
        assert!(cache.get(7, &["pick2"], 2, MatchSide::Base, &idx).is_none());
        assert!(cache.get(7, &["pick"], 3, MatchSide::Base, &idx).is_none());
        assert!(
            cache
                .get(7, &["pick"], 2, MatchSide::Base, &[1, 3, 6])
                .is_none()
        );
        assert!(
            cache
                .get(7, &["pick"], 2, MatchSide::Overflow, &idx)
                .is_none()
        );
    }

    #[test]
    fn match_cache_exact_repeat_matches_uncached_subset() {
        let (files, arena) = make_files(&[
            "src/file_picker.rs",
            "src/other.rs",
            "tests/file_picker_test.rs",
            "src/picker/file.rs",
        ]);
        let parser = QueryParser::default();
        let parsed = parser.parse("filepick");
        let ctx = ScoringContext {
            query: &parsed,
            max_threads: 1,
            max_typos: 2,
            project_path: None,
            current_file: None,
            last_same_query_match: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs::default(),
        };
        let idx: Vec<u32> = (0..files.len() as u32).collect();
        let max_path = crate::matcher::max_path_len_of(&files);
        let presence = crate::types::PresenceTable::from_files(&files);
        let run = |cache: Option<&parking_lot::Mutex<MatchCache>>| {
            fuzzy_match_and_score_subset(
                &files,
                presence.as_slice(),
                crate::matcher::CandList::Owned(idx.clone()),
                &ctx,
                files.len(),
                arena,
                arena,
                max_path,
                crate::parallelism::AbortToken::disabled(),
                cache,
                11,
            )
        };

        let (items_a, scores_a, n_a) = run(None);
        let cache = parking_lot::Mutex::new(MatchCache::default());
        let (items_b, scores_b, n_b) = run(Some(&cache));
        let (items_c, scores_c, n_c) = run(Some(&cache));

        assert!(n_a > 0, "query should match at least one file");
        assert_eq!(n_a, n_b);
        assert_eq!(n_a, n_c);
        let summarize = |items: &[&FileItem], scores: &[Score]| {
            items
                .iter()
                .zip(scores)
                .map(|(file, score)| (file.relative_path(arena).to_string(), format!("{score:?}")))
                .collect::<Vec<_>>()
        };
        let baseline = summarize(&items_a, &scores_a);
        assert_eq!(baseline, summarize(&items_b, &scores_b));
        assert_eq!(baseline, summarize(&items_c, &scores_c));
        // First cached call misses and stores; the second must hit.
        assert_eq!(cache.lock().hits.get(), 1);
    }

    #[test]
    fn test_filename_match_ranks_above_path_only_match() {
        let (files, arena) = make_files(&["src/username/handler.rs", "src/username/username.rs"]);

        let results = search(&files, "usrnmea", arena);

        assert!(
            results.len() >= 2,
            "both files should match, got {}",
            results.len()
        );
        assert_eq!(
            results[0].0, "src/username/username.rs",
            "filename match should rank first"
        );
        assert!(
            results[0].1.filename_bonus > 0,
            "username.rs should have filename bonus"
        );
        assert_eq!(
            results[1].1.filename_bonus, 0,
            "handler.rs should have no filename bonus"
        );
    }

    #[test]
    fn test_exact_filename_beats_fuzzy_filename() {
        let (files, arena) = make_files(&["src/user_name_handler.rs", "src/username.rs"]);

        let results = search(&files, "username.rs", arena);

        assert!(results.len() >= 2);
        assert_eq!(
            results[0].0, "src/username.rs",
            "exact filename should rank first"
        );
        assert_eq!(results[0].1.match_type, "exact_filename");
        assert!(results[0].1.filename_bonus > results[1].1.filename_bonus);
    }

    #[test]
    fn test_same_length_filename_no_false_exact() {
        let (files, arena) = make_files(&["src/item_sync/file.rs", "src/models/item.rs"]);

        let results = search(&files, "item.rs", arena);

        assert!(results.len() >= 2);
        assert_eq!(results[0].0, "src/models/item.rs");
        assert_eq!(results[0].1.match_type, "exact_filename");
        assert_ne!(
            results[1].1.match_type, "exact_filename",
            "file.rs should not get exact_filename"
        );
    }

    #[test]
    fn test_path_separator_disables_filename_bonus() {
        let (files, arena) = make_files(&["src/controllers/user.rs"]);

        let results = search(&files, "src/user", arena);

        assert!(!results.is_empty());
        assert_eq!(
            results[0].1.filename_bonus, 0,
            "path-like query should not get filename bonus"
        );
    }

    #[test]
    fn test_full_path_query_prefers_closer_filename_match() {
        let (files, arena) = make_files_with_frecency(&[
            (
                "test-utils/completion/condition-key/yaml_partial-svc-colon.yml",
                0,
            ),
            (
                "test-utils/test-cases/completion/condition-key/yaml_partial-svc.yml",
                0,
            ),
            (
                "test-utils/action-value/yaml_inline_partial-svc-colon.yml",
                0,
            ),
            (
                "test-utils/completion/action-value/yaml_array_partial-svc-colon.yml",
                0,
            ),
            (
                "test-utils/completion/action-value/yaml_array_partial-svc.yml",
                0,
            ),
            (
                "test-utils/completion/condition-key/yaml_global-tag-keys.yml",
                0,
            ),
            (
                "test-utils/test-cases/completion/condition-key/yaml_partial.yml",
                10,
            ),
        ]);

        let results = search(
            &files,
            "t-utils/test-cases/completion/condition-key/yaml_partial-svc.yml",
            arena,
        );

        assert!(!results.is_empty(), "query should match at least one file");

        assert_eq!(
            results[0].0, "test-utils/test-cases/completion/condition-key/yaml_partial-svc.yml",
            "near-exact full-path match should rank first, but got: {} \
             (total={}, base={}, frecency={})",
            results[0].0, results[0].1.total, results[0].1.base_score, results[0].1.frecency_boost,
        );
    }

    // Paths past 512B overflowed the fixed 32 chunk-pointer buffer;
    // must not panic and must still match.
    #[test]
    fn test_path_longer_than_512_bytes_does_not_panic_and_matches() {
        let mut long_path = String::new();
        while long_path.len() < 600 {
            long_path.push_str("deeply_nested_directory_segment/");
        }
        long_path.push_str("needle_file.rs");
        assert!(long_path.len() > 512 && long_path.len() < crate::simd_path::PATH_BUF_SIZE);

        let (files, arena) = make_files(&[long_path.as_str(), "src/other.rs"]);

        let results = search(&files, "needle", arena);

        assert!(
            results.iter().any(|(p, _)| p == &long_path),
            "filename at the tail of a >512-byte path must still match, got: {:?}",
            results.iter().map(|(p, _)| p).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_single_path_matching() {
        let path = "core_workflow_service/kafka_event_consumer/src/ai_part_extraction_request/ai_part_extraction_request_handler.rs";

        let options = neo_frizbee::Config {
            max_typos: Some(2),
            sort: neo_frizbee::SortStrategy::Unsorted,
            ..Default::default()
        };

        let matches = neo_frizbee::Matcher::new("aipart", &options).match_list(&[path]);
        assert!(!matches.is_empty(), "'aipart' should match the path");

        let matches = neo_frizbee::Matcher::new("core", &options).match_list(&[path]);
        assert!(!matches.is_empty(), "'core' should match the path");

        let co_options = neo_frizbee::Config {
            max_typos: Some(2),
            ..options
        };
        let matches = neo_frizbee::Matcher::new("co", &co_options).match_list(&[path]);
        assert!(!matches.is_empty(), "'co' should match the path");
    }

    #[test]
    fn test_lowercase_path_matching() {
        let path = "core_workflow_service/kafka_event_consumer/src/ai_part_extraction_request/ai_part_extraction_request_handler.rs".to_lowercase();

        let options = neo_frizbee::Config {
            max_typos: Some(2),
            sort: neo_frizbee::SortStrategy::Unsorted,
            ..Default::default()
        };

        let matches = neo_frizbee::Matcher::new("co", &options).match_list(&[path.as_str()]);
        assert!(!matches.is_empty(), "'co' should match the lowercase path");

        let matches = neo_frizbee::Matcher::new("core", &options).match_list(&[path.as_str()]);
        assert!(
            !matches.is_empty(),
            "'core' should match the lowercase path"
        );
    }
}

#[cfg(test)]
mod git_recency_scoring_tests {
    use super::*;
    use crate::types::PaginationArgs;
    use fff_query_parser::QueryParser;

    fn make_files(specs: &[(&str, i16)]) -> (Vec<FileItem>, ArenaPtr) {
        let path_strings: Vec<String> = specs.iter().map(|(p, _)| p.to_string()).collect();
        let items: Vec<FileItem> = specs
            .iter()
            .map(|(p, _)| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&path_strings, &items);
        let arena = store.as_arena_ptr();
        let mut result: Vec<FileItem> = items;
        for (i, file) in result.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
            file.set_git_recency_score(specs[i].1);
        }
        std::mem::forget(store);
        (result, arena)
    }

    fn search(files: &[FileItem], query: &str, arena: ArenaPtr) -> Vec<(String, Score)> {
        let parser = QueryParser::default();
        let parsed = parser.parse(query);
        let ctx = ScoringContext {
            query: &parsed,
            max_threads: 1,
            max_typos: 2,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs {
                offset: 0,
                limit: 100,
            },
        };
        let (items, scores, _) = fuzzy_match_and_score_files(
            files,
            &ctx,
            files.len(),
            arena,
            arena,
            max_path_len_of(&files),
            crate::parallelism::AbortToken::disabled(),
        );
        items
            .iter()
            .zip(scores)
            .map(|(f, s)| (f.relative_path(arena), s))
            .collect()
    }

    #[test]
    fn recency_boost_breaks_ties_between_equal_fuzzy_matches() {
        // Dir names share no query letters, so only recency separates them.
        let (files, arena) = make_files(&[("src/xxx/handler.rs", 0), ("src/yyy/handler.rs", 5)]);

        let results = search(&files, "handler", arena);

        assert!(results.len() >= 2);
        assert_eq!(results[0].0, "src/yyy/handler.rs");
        assert_eq!(results[0].1.git_recency_boost, 5);
        assert_eq!(results[1].1.git_recency_boost, 0);
        assert_eq!(
            results[0].1.total - results[1].1.total,
            5,
            "boost is additive: exactly +1 point per participating commit"
        );
    }

    #[test]
    fn recency_boost_ranks_files_in_frecency_only_mode() {
        let (files, arena) = make_files(&[("cold.rs", 0), ("committed.rs", 7)]);

        let results = search(&files, "", arena);

        assert_eq!(results[0].0, "committed.rs");
        assert_eq!(results[0].1.git_recency_boost, 7);
        assert_eq!(results[0].1.total, 7);
        assert_eq!(results[0].1.match_type, "frecency");
    }
}

#[cfg(test)]
mod typo_resistance_tests {
    use super::*;
    use crate::types::PaginationArgs;
    use fff_query_parser::QueryParser;

    fn make_files(paths: &[&str]) -> (Vec<FileItem>, ArenaPtr) {
        let path_strings: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
        let items: Vec<FileItem> = paths
            .iter()
            .map(|p| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&path_strings, &items);
        let arena = store.as_arena_ptr();
        let mut result: Vec<FileItem> = items;
        for (i, file) in result.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
        }
        std::mem::forget(store);
        (result, arena)
    }

    fn search_with_typos(
        files: &[FileItem],
        query: &str,
        arena: ArenaPtr,
        max_typos: u16,
    ) -> Vec<String> {
        let parser = QueryParser::default();
        let parsed = parser.parse(query);
        let ctx = ScoringContext {
            query: &parsed,
            max_threads: 1,
            max_typos,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs {
                offset: 0,
                limit: 100,
            },
        };
        let (items, _, _) = fuzzy_match_and_score_files(
            files,
            &ctx,
            files.len(),
            arena,
            arena,
            max_path_len_of(&files),
            crate::parallelism::AbortToken::disabled(),
        );
        items.iter().map(|f| f.relative_path(arena)).collect()
    }

    #[test]
    fn test_typo_resistant_long_query() {
        let (files, arena) = make_files(&[
            "src/pricing/bid_comparison_supplier_part_cost_modifiers.rs",
            "src/pricing/bid_evaluation_handler.rs",
            "src/pricing/supplier_contract_terms.rs",
            "src/models/purchase_order.rs",
            "src/models/inventory_item.rs",
            "src/controllers/user_controller.rs",
            "src/controllers/admin_controller.rs",
            "src/services/notification_service.rs",
            "src/services/email_dispatcher.rs",
            "src/utils/string_helpers.rs",
            "src/utils/date_formatter.rs",
            "src/db/migration_runner.rs",
            "src/db/connection_pool.rs",
            "src/config/app_settings.rs",
            "src/config/feature_flags.rs",
        ]);

        let results = search_with_typos(&files, "bid_comparison", arena, 6);
        assert!(
            results
                .iter()
                .any(|p| p.contains("bid_comparison_supplier_part_cost_modifiers")),
            "exact substring 'bid_comparison' should match, got: {results:?}"
        );

        let results = search_with_typos(&files, "bidcomparsionsupplierpartcostmodfiers", arena, 6);
        assert!(
            results
                .iter()
                .any(|p| p.contains("bid_comparison_supplier_part_cost_modifiers")),
            "typo query 'bidcomparsionsupplierpartcostmodfiers' should match, got: {results:?}"
        );

        let results = search_with_typos(&files, "bidcomp", arena, 4);
        assert!(
            results.iter().any(|p| p.contains("bid_comparison")),
            "'bidcomp' should match bid_comparison file, got: {results:?}"
        );

        let results = search_with_typos(&files, "supplierpartcost", arena, 6);
        assert!(
            results
                .iter()
                .any(|p| p.contains("bid_comparison_supplier_part_cost_modifiers")),
            "'supplierpartcost' should match, got: {results:?}"
        );
    }
}

#[cfg(test)]
mod constraint_only_query_tests {
    use super::*;
    use crate::types::PaginationArgs;
    use fff_query_parser::QueryParser;

    #[test]
    fn constraint_only_git_status_query_returns_filtered_files() {
        let paths = ["modified.rs", "clean.rs"];
        let path_strings: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
        let items: Vec<FileItem> = paths
            .iter()
            .map(|p| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&path_strings, &items);
        let arena = store.as_arena_ptr();
        let mut files: Vec<FileItem> = items;
        for (i, file) in files.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
        }
        files[0].set_git_status(Some(git2::Status::WT_MODIFIED));
        files[1].set_git_status(Some(git2::Status::CURRENT));
        std::mem::forget(store);

        let parser = QueryParser::default();
        let parsed = parser.parse("git:modified");

        let ctx = ScoringContext {
            query: &parsed,
            max_threads: 1,
            max_typos: 2,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
        };

        let (items, _scores, total_matched) = fuzzy_match_and_score_files(
            &files,
            &ctx,
            files.len(),
            arena,
            arena,
            max_path_len_of(&files),
            crate::parallelism::AbortToken::disabled(),
        );

        assert_eq!(
            total_matched, 1,
            "git:modified constraint-only query should match exactly 1 modified file"
        );
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].relative_path(arena), "modified.rs");
    }

    #[test]
    fn constraint_only_status_modified_alias_returns_filtered_files() {
        let paths = ["a.rs", "b.rs"];
        let path_strings: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
        let items: Vec<FileItem> = paths
            .iter()
            .map(|p| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&path_strings, &items);
        let arena = store.as_arena_ptr();
        let mut files: Vec<FileItem> = items;
        for (i, file) in files.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
        }
        files[0].set_git_status(Some(git2::Status::WT_MODIFIED));
        files[1].set_git_status(Some(git2::Status::CURRENT));
        std::mem::forget(store);

        let parser = QueryParser::default();
        let parsed = parser.parse("status:modified");

        let ctx = ScoringContext {
            query: &parsed,
            max_threads: 1,
            max_typos: 2,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
        };

        let (items, _scores, total_matched) = fuzzy_match_and_score_files(
            &files,
            &ctx,
            files.len(),
            arena,
            arena,
            max_path_len_of(&files),
            crate::parallelism::AbortToken::disabled(),
        );

        assert_eq!(total_matched, 1);
        assert_eq!(items[0].relative_path(arena), "a.rs");
    }
}

// Ranking policy (base = SW score): exact filename > fuzzy filename > path-only;
// at equal quality frecency wins; entry points beat path-only but lose to filename hits.
// If these fail, update this spec deliberately, not as a perf side effect.
#[cfg(test)]
mod ranking_policy_tests {
    use super::*;
    use crate::types::PaginationArgs;
    use fff_query_parser::QueryParser;

    fn make(specs: &[(&str, i16)]) -> (Vec<FileItem>, ArenaPtr) {
        let path_strings: Vec<String> = specs.iter().map(|(p, _)| p.to_string()).collect();
        let items: Vec<FileItem> = specs
            .iter()
            .map(|(p, _)| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&path_strings, &items);
        let arena = store.as_arena_ptr();
        let mut result = items;
        for (i, file) in result.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
            file.set_access_frecency_score(specs[i].1);
        }
        std::mem::forget(store);
        (result, arena)
    }

    fn rank(files: &[FileItem], query: &str, arena: ArenaPtr) -> Vec<String> {
        let parser = QueryParser::default();
        let parsed = parser.parse(query);
        let effective = match &parsed.fuzzy_query {
            FuzzyQuery::Text(t) => *t,
            FuzzyQuery::Parts(parts) if !parts.is_empty() => parts[0],
            _ => query,
        };
        let max_typos = (effective.len() as u16 / 4).clamp(2, 6);
        let ctx = ScoringContext {
            query: &parsed,
            max_threads: 1,
            max_typos,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs {
                offset: 0,
                limit: 100,
            },
        };
        let (items, _scores, _) = fuzzy_match_and_score_files(
            files,
            &ctx,
            files.len(),
            arena,
            arena,
            max_path_len_of(&files),
            crate::parallelism::AbortToken::disabled(),
        );
        items.iter().map(|f| f.relative_path(arena)).collect()
    }

    #[test]
    fn exact_filename_beats_fuzzy_filename_beats_path_only() {
        let (files, arena) = make(&[
            ("src/username/handler.rs", 0),  // path-only
            ("src/username/username.rs", 0), // fuzzy/exact filename
            ("other/username", 0),           // exact filename
        ]);
        let order = rank(&files, "username", arena);
        assert_eq!(order[0], "other/username", "exact filename first");
        assert!(
            order.iter().position(|p| p == "src/username/username.rs")
                < order.iter().position(|p| p == "src/username/handler.rs"),
            "filename hit above path-only, got {order:?}"
        );
    }

    #[test]
    fn frecency_breaks_ties_at_equal_match_quality() {
        let (files, arena) = make(&[("src/a/user_service.rs", 10), ("src/b/user_service.rs", 90)]);
        let order = rank(&files, "user_service", arena);
        assert_eq!(order[0], "src/b/user_service.rs", "higher frecency first");
    }

    #[test]
    fn special_entry_point_beats_ordinary_path_only() {
        let (files, arena) = make(&[("src/deep/mod.rs", 0), ("src/deep/helper_mod.rs", 0)]);
        let order = rank(&files, "mod", arena);
        assert_eq!(
            order[0], "src/deep/mod.rs",
            "special entry point above ordinary path hit, got {order:?}"
        );
    }

    // Job: "I know the name + ext" — exact stem+ext wins over wrong ext.
    #[test]
    fn filename_intent_prefers_matching_extension() {
        let (files, arena) = make(&[
            ("src/user_service.py", 0),
            ("src/user_service.rs", 0),
            ("src/other.rs", 0),
        ]);
        let order = rank(&files, "user_service.rs", arena);
        assert_eq!(
            order[0], "src/user_service.rs",
            "matching ext first, got {order:?}"
        );
    }

    // Job: "I know the path" — segment hits beat scattered letters.
    #[test]
    fn path_intent_prefers_segment_sequence() {
        let (files, arena) = make(&[("xsrc/xauth/xlogin.rs", 0), ("src/auth/login.rs", 0)]);
        let order = rank(&files, "src/auth", arena);
        assert_eq!(
            order[0], "src/auth/login.rs",
            "clean segment sequence first, got {order:?}"
        );
    }

    // Job: "I know the concept" — filename words beat path-only scatter.
    #[test]
    fn concept_intent_prefers_filename_words() {
        let (files, arena) = make(&[("src/hidden/handler_util.rs", 0), ("src/api/handler.rs", 0)]);
        let order = rank(&files, "handler", arena);
        assert_eq!(
            order[0], "src/api/handler.rs",
            "filename word hit first, got {order:?}"
        );
    }

    // Job: "I want the test for this role" — role keyword ranks test files up.
    #[test]
    fn concept_test_role_ranks_test_files() {
        let (files, arena) = make(&[("src/user.rs", 0), ("src/user_test.rs", 0)]);
        let order = rank(&files, "test", arena);
        assert_eq!(
            order[0], "src/user_test.rs",
            "role=test should prefer test file, got {order:?}"
        );
    }
    // MRR-style: first result must be the intended file for each job.
    #[test]
    fn mrr_jobs_first_result_is_intended() {
        let (files, arena) = make(&[
            ("src/api/user_service.rs", 0),
            ("src/api/user_service.py", 0),
            ("src/api/handler.rs", 0),
            ("src/auth/login.rs", 0),
            ("src/auth/login_test.rs", 0),
            ("tests/unit/login_test.rs", 0),
            ("mod.rs", 0),
            ("src/models/mod.rs", 0),
        ]);
        let o = rank(&files, "user_service.rs", arena);
        assert_eq!(o[0], "src/api/user_service.rs", "name.ext {o:?}");
        let o = rank(&files, "handler", arena);
        assert_eq!(o[0], "src/api/handler.rs", "concept {o:?}");
        let o = rank(&files, "src/auth", arena);
        assert_eq!(o[0], "src/auth/login.rs", "path {o:?}");
        let o = rank(&files, "test", arena);
        assert!(o[0].contains("test"), "role=test {o:?}");
    }
}

#[cfg(test)]
mod topk_prune_tests {
    use super::*;
    use crate::types::PaginationArgs;
    use fff_query_parser::QueryParser;

    fn make_paths(paths: &[&str]) -> (Vec<FileItem>, ArenaPtr) {
        let path_strings: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
        let items: Vec<FileItem> = paths
            .iter()
            .map(|p| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&path_strings, &items);
        let arena = store.as_arena_ptr();
        let mut result = items;
        for (i, file) in result.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
            file.parent_dir_index = 0;
        }
        std::mem::forget(store);
        (result, arena)
    }

    // Frecency corpus with distinct totals; empty query ranks by score.
    fn make_frecency(n: usize) -> (Vec<FileItem>, ArenaPtr) {
        let paths: Vec<String> = (0..n).map(|i| format!("src/g{i}/data_{i}.log")).collect();
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (mut files, arena) = make_paths(&refs);
        for (i, file) in files.iter_mut().enumerate() {
            file.set_access_frecency_score(i as i16);
        }
        (files, arena)
    }

    fn search_with(
        files: &[FileItem],
        query: &str,
        arena: ArenaPtr,
        threads: usize,
        current: Option<&str>,
    ) -> (Vec<String>, Vec<Score>) {
        let (items, scores, _) = search_limit(files, query, arena, threads, current, 50);
        (items, scores)
    }

    fn search_limit(
        files: &[FileItem],
        query: &str,
        arena: ArenaPtr,
        threads: usize,
        current: Option<&str>,
        limit: usize,
    ) -> (Vec<String>, Vec<Score>, usize) {
        let parser = QueryParser::default();
        let parsed = parser.parse(query);
        let ctx = ScoringContext {
            query: &parsed,
            max_threads: threads,
            max_typos: 2,
            current_file: current,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs { offset: 0, limit },
        };
        let (items, scores, total) = fuzzy_match_and_score_files(
            files,
            &ctx,
            files.len(),
            arena,
            arena,
            max_path_len_of(files),
            crate::parallelism::AbortToken::disabled(),
        );
        (
            items.iter().map(|f| f.relative_path(arena)).collect(),
            scores,
            total,
        )
    }

    #[test]
    fn ub_covers_actual_total() {
        use crate::intent::{Evidence, QueryIntent};
        let mut state = 0x12345678u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u32
        };
        let intents = [
            QueryIntent::FileName,
            QueryIntent::Path,
            QueryIntent::Concept,
            QueryIntent::Typo,
            QueryIntent::WorkingSet,
        ];
        for _ in 0..20_000 {
            let base = (next() % 400) as i32;
            let freq = (next() % 200) as i32;
            let rec = (next() % 50) as i32;
            let intent = intents[(next() as usize) % intents.len()];
            let collapsed = next() % 2 == 0;
            let path_on = next() % 2 == 0;
            let combo = (next() % 1000) as i32;
            // The bound must cover every intent, including all-positive.
            let bits = next();
            let ev = Evidence {
                exact_filename: bits & 1 != 0,
                filename_substring: bits & 2 != 0,
                ext_match: bits & 4 != 0,
                ext_mismatch: bits & 8 != 0,
                last_segment_match: bits & 16 != 0,
                segment_sequence: bits & 32 != 0,
                segment_boundary_pct: (next() % 101) as u8,
                word_coverage: (next() % 101) as u8,
                same_dir: bits & 64 != 0,
                related: bits & 128 != 0,
                role_match: bits & 256 != 0,
            };
            let real_ev = crate::intent::evidence_bonus(intent, &ev, base);
            if !collapsed {
                assert!(
                    real_ev <= evidence_ub_for(base, intent, false),
                    "evidence bound unsound: intent={intent:?} base={base} real={real_ev} bound={} ev={ev:?}",
                    evidence_ub_for(base, intent, false),
                );
            }
            let ub = ub_total_for(
                base,
                freq_term_of(base, freq),
                rec,
                dirty_bonus_of(base, bits & 512 != 0),
                intent,
                collapsed,
                path_on,
                combo,
                &[],
            );
            assert!(ub >= base, "UB must cover base alone");
        }
    }

    #[test]
    fn parallel_topk_matches_serial_with_current_file() {
        let mut paths = Vec::new();
        for i in 0..6000 {
            paths.push(format!("src/mod{i}/handler_{}.rs", i % 50));
        }
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (files, arena) = make_paths(&refs);
        for q in ["handler", "mod handler", "src/mod"] {
            let (s_items, s_scores) = search_with(&files, q, arena, 1, Some("src/mod7/handler.rs"));
            let (p_items, p_scores) = search_with(&files, q, arena, 4, Some("src/mod7/handler.rs"));
            assert_eq!(s_items, p_items, "page items differ for {q:?}");
            assert_eq!(
                s_scores.iter().map(|s| s.total).collect::<Vec<_>>(),
                p_scores.iter().map(|s| s.total).collect::<Vec<_>>(),
                "page totals differ for {q:?}"
            );
        }
    }

    #[test]
    fn wide_windows_split_to_u16_and_agree_across_threads() {
        // Windows past 256B split into exact u16 halves; identical lanes must score
        // identically under any threading.
        let long: String = format!("src/{}/file.rs", "a".repeat(280));
        let paths: Vec<String> = (0..20).map(|_| long.clone()).collect();
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (files, arena) = make_paths(&refs);
        // "aa" with max_typos=2 has need==0: full-path windows.
        let (a1, s1, t1) = search_limit(&files, "aa", arena, 1, None, 50);
        assert_eq!(t1, 20, "all long files must match");
        assert_eq!(a1.len(), 20);
        assert!(s1.iter().all(|s| s.total > 0), "real scores, not zeroed");
        assert!(
            s1.windows(2).all(|w| w[0].total == w[1].total),
            "identical lanes diverge: {s1:?}"
        );
        let (a4, s4, t4) = search_limit(&files, "aa", arena, 4, None, 50);
        assert_eq!(t4, 20);
        assert_eq!(a1, a4, "threading changed ranking");
        assert!(
            s1.iter().zip(s4.iter()).all(|(a, b)| a.total == b.total),
            "threading changed scores"
        );
    }

    #[test]
    fn collapsed_skips_filename_bonus() {
        // Collapsed queries zero filename/evidence bonuses, so the fallback skip is exact.
        let mut paths = Vec::new();
        for i in 0..5000 {
            paths.push(format!("src/ab/file_{i}.rs"));
        }
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (files, arena) = make_paths(&refs);
        let (items, scores) = {
            let parser = QueryParser::default();
            let parsed = parser.parse("ab");
            let ctx = ScoringContext {
                query: &parsed,
                max_threads: 1,
                max_typos: 2,
                current_file: None,
                last_same_query_match: None,
                project_path: None,
                combo_boost_score_multiplier: 100,
                min_combo_count: 3,
            check_deleted: true,
                pagination: PaginationArgs {
                    offset: 0,
                    limit: 50,
                },
            };
            let (items, scores, total) = fuzzy_match_and_score_files(
                &files,
                &ctx,
                files.len(),
                arena,
                arena,
                max_path_len_of(&files),
                crate::parallelism::AbortToken::disabled(),
            );
            assert!(total >= 4096, "fixture must trigger collapse, got {total}");
            (
                items
                    .iter()
                    .map(|f| f.relative_path(arena))
                    .collect::<Vec<_>>(),
                scores,
            )
        };
        assert_eq!(items.len(), 50);
        for s in &scores {
            assert_eq!(s.filename_bonus, 0, "collapsed must zero filename bonus");
            assert_eq!(s.evidence_bonus, 0, "collapsed must zero evidence bonus");
        }
    }

    #[test]
    fn no_prune_hatch_ranks_identically() {
        // Both settings rank identically, so toggling mid-process is safe.
        let mut paths = Vec::new();
        for i in 0..6000 {
            paths.push(format!("src/mod{i}/handler_{}.rs", i % 50));
        }
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (files, arena) = make_paths(&refs);
        for q in ["handler", "mod handler"] {
            for threads in [1usize, 4] {
                let (on_items, on_scores) = search_with(&files, q, arena, threads, None);
                unsafe {
                    std::env::set_var("FFF_NO_PRUNE", "1");
                }
                let (off_items, off_scores) = search_with(&files, q, arena, threads, None);
                unsafe {
                    std::env::remove_var("FFF_NO_PRUNE");
                }
                assert_eq!(on_items, off_items, "items differ for {q:?} t={threads}");
                let totals = |s: Vec<Score>| s.into_iter().map(|x| x.total).collect::<Vec<_>>();
                assert_eq!(totals(on_scores), totals(off_scores), "totals differ");
            }
        }
    }

    #[test]
    fn evidence_memo_hatch_ranks_identically() {
        // Memo must equal fresh computation, including the memo-inactive legs.
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _held = ENV_LOCK.lock().unwrap();
        let mut paths = Vec::new();
        for i in 0..3000 {
            paths.push(format!("src/mod{}/Handler_{}.rs", i % 37, i % 50));
        }
        paths.push("src/a/mod.rs".to_string());
        paths.push("src/b/lib.rs".to_string());
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (files, arena) = make_paths(&refs);
        for q in ["handler", "mod handler", "src/mod"] {
            for threads in [1usize, 4] {
                for current in [None, Some("src/mod7/other.rs")] {
                    let run = || {
                        let parser = QueryParser::default();
                        let parsed = parser.parse(q);
                        let ctx = ScoringContext {
                            query: &parsed,
                            max_threads: threads,
                            max_typos: 2,
                            current_file: current,
                            last_same_query_match: None,
                            project_path: None,
                            combo_boost_score_multiplier: 100,
                            min_combo_count: 3,
            check_deleted: true,
                            pagination: PaginationArgs {
                                offset: 0,
                                limit: 50,
                            },
                        };
                        fuzzy_match_and_score_files(
                            &files,
                            &ctx,
                            files.len(),
                            arena,
                            arena,
                            max_path_len_of(&files),
                            crate::parallelism::AbortToken::disabled(),
                        )
                    };
                    let (base_items, base_scores, base_total) = run();
                    unsafe {
                        std::env::set_var("FFF_NO_EVIDENCE_MEMO", "1");
                    }
                    let (hatch_items, hatch_scores, hatch_total) = run();
                    unsafe {
                        std::env::remove_var("FFF_NO_EVIDENCE_MEMO");
                    }
                    let names = |items: Vec<&FileItem>| {
                        items
                            .iter()
                            .map(|f| f.relative_path(arena))
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(
                        names(base_items),
                        names(hatch_items),
                        "items differ for {q:?} t={threads} cur={current:?}"
                    );
                    assert_eq!(base_total, hatch_total, "total differs");
                    let totals = |s: Vec<Score>| s.into_iter().map(|x| x.total).collect::<Vec<_>>();
                    assert_eq!(totals(base_scores), totals(hatch_scores), "totals differ");
                }
            }
        }
    }

    #[test]
    fn fused_groups_match_regardless_of_size() {
        // Group outputs are per-index, so group size cannot change results.
        let paths: Vec<String> = (0..1500)
            .map(|i| format!("src/mod{}/Handler_{}.rs", i % 37, i % 50))
            .collect();
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (files, arena) = make_paths(&refs);
        for q in ["handler", "mod handler", "handler_"] {
            for threads in [1usize, 4] {
                let (base_items, base_scores) = search_with(&files, q, arena, threads, None);
                unsafe {
                    std::env::set_var("FFF_SW_CHUNK", "8");
                }
                let (tiny_items, tiny_scores) = search_with(&files, q, arena, threads, None);
                unsafe {
                    std::env::remove_var("FFF_SW_CHUNK");
                }
                assert_eq!(base_items, tiny_items, "items differ for {q:?} t={threads}");
                let totals = |s: Vec<Score>| s.into_iter().map(|x| x.total).collect::<Vec<_>>();
                assert_eq!(totals(base_scores), totals(tiny_scores), "totals differ");
            }
        }
    }

    #[test]
    fn presence_recheck_decision_follows_measurement() {
        assert!(!presence_rechecks_pay(1.0, 5.0, 100.0));
        assert!(!presence_rechecks_pay(0.99, 5.0, 100.0));
        assert!(presence_rechecks_pay(0.1, 5.0, 100.0));
        assert!(presence_rechecks_pay(0.5, 5.0, 100.0));
    }

    #[test]
    fn gate_prefilter_hatch_ranks_identically() {
        // Pretest is sound, so both prefilter settings rank identically.
        let paths: Vec<String> = (0..1500)
            .map(|i| format!("src/mod{}/Handler_{}.rs", i % 37, i % 50))
            .collect();
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (files, arena) = make_paths(&refs);
        for q in ["handler", "mod7 Handler_7"] {
            for threads in [1usize, 4] {
                let (base_items, base_scores) = search_with(&files, q, arena, threads, None);
                unsafe {
                    std::env::set_var("FFF_NO_GATE_PREFILTER", "1");
                }
                let (hatch_items, hatch_scores) = search_with(&files, q, arena, threads, None);
                unsafe {
                    std::env::remove_var("FFF_NO_GATE_PREFILTER");
                }
                assert_eq!(
                    base_items, hatch_items,
                    "items differ for {q:?} t={threads}"
                );
                let totals = |s: Vec<Score>| s.into_iter().map(|x| x.total).collect::<Vec<_>>();
                assert_eq!(totals(base_scores), totals(hatch_scores), "totals differ");
            }
        }
    }

    #[test]
    fn frecency_topk_matches_full_ranking() {
        // Heap page must equal the full-ranking prefix, on both thread counts, with and without a current file.
        let (files, arena) = make_frecency(6000);
        for threads in [1usize, 4] {
            let (items, scores, total) = search_limit(&files, "", arena, threads, None, 50);
            let (full_items, full_scores, full_total) =
                search_limit(&files, "", arena, threads, None, 0);
            assert_eq!(total, 6000);
            assert_eq!(total, full_total);
            assert_eq!(items.len(), 50);
            assert_eq!(items, &full_items[..50]);
            assert_eq!(
                scores.iter().map(|s| s.total).collect::<Vec<_>>(),
                full_scores
                    .iter()
                    .take(50)
                    .map(|s| s.total)
                    .collect::<Vec<_>>(),
            );
        }
        // With a current file, evidence can tie totals; compare as sets.
        for threads in [1usize, 4] {
            let (items, scores, total) =
                search_limit(&files, "", arena, threads, Some("src/g7/other.rs"), 50);
            let (full_items, full_scores, full_total) =
                search_limit(&files, "", arena, threads, Some("src/g7/other.rs"), 0);
            assert_eq!(total, 6000);
            assert_eq!(total, full_total);
            let mut page: Vec<(&str, i32)> = items
                .iter()
                .zip(scores.iter())
                .map(|(p, s)| (p.as_str(), s.total))
                .collect();
            let mut full: Vec<(&str, i32)> = full_items
                .iter()
                .zip(full_scores.iter())
                .take(50)
                .map(|(p, s)| (p.as_str(), s.total))
                .collect();
            page.sort_unstable();
            full.sort_unstable();
            assert_eq!(page, full);
        }
    }

    // Heap page must equal the full-ranking page across shapes and configs.
    #[test]
    fn ranking_parity_across_shapes_and_configs() {
        fn page_as_set(
            files: &[FileItem],
            query: &str,
            arena: ArenaPtr,
            threads: usize,
            current: Option<&str>,
            offset: usize,
            limit: usize,
        ) -> (Vec<(String, i32, u64)>, usize) {
            let parser = QueryParser::default();
            let parsed = parser.parse(query);
            let ctx = ScoringContext {
                query: &parsed,
                max_threads: threads,
                max_typos: 2,
                current_file: current,
                last_same_query_match: None,
                project_path: None,
                combo_boost_score_multiplier: 100,
                min_combo_count: 3,
            check_deleted: true,
                pagination: PaginationArgs { offset, limit },
            };
            let (items, scores, total) = fuzzy_match_and_score_files(
                files,
                &ctx,
                files.len(),
                arena,
                arena,
                max_path_len_of(files),
                crate::parallelism::AbortToken::disabled(),
            );
            let mut pairs: Vec<(String, i32, u64)> = items
                .iter()
                .zip(scores.iter())
                .map(|(f, s)| (f.relative_path(arena), s.total, f.modified()))
                .collect();
            // Order by (total, modified); path breaks full ties.
            pairs.sort_by(|a, b| {
                b.1.cmp(&a.1)
                    .then_with(|| b.2.cmp(&a.2))
                    .then_with(|| a.0.cmp(&b.0))
            });
            (pairs, total)
        }
        for n in [200usize, 1500] {
            let paths: Vec<String> = (0..n)
                .map(|i| format!("src/mod{}/Handler_{}.rs", i % 37, i % 50))
                .collect();
            let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
            let (mut files, arena) = make_paths(&refs);
            // Unique mtimes make (total, modified) a strict order.
            for (i, file) in files.iter_mut().enumerate() {
                file.set_modified(1_700_000_000 + i as u64);
            }
            for query in [
                "handler",
                "Handler",
                "src/mod",
                "MOD Handler",
                "handler_",
                "",
            ] {
                for threads in [0usize, 1, 4] {
                    for current in [None, Some("src/mod7/other.rs")] {
                        let (full, full_total) =
                            page_as_set(&files, query, arena, threads, current, 0, 0);
                        for (offset, limit) in [(0usize, 50usize), (20, 30)] {
                            let (page, total) =
                                page_as_set(&files, query, arena, threads, current, offset, limit);
                            assert_eq!(total, full_total, "totals n={n} q={query:?}");
                            // The page must be the full list's slice under a strict key.
                            let want = full
                                .iter()
                                .skip(offset)
                                .take(page.len())
                                .cloned()
                                .collect::<Vec<_>>();
                            assert_eq!(
                                page, want,
                                "page n={n} q={query:?} t={threads} cur={current:?} off={offset} lim={limit}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn topk_pages_are_prefixes_under_score_ties() {
        let paths: Vec<String> = (0..200)
            .map(|i| format!("src/mod{:02}/Handler_{:02}.rs", i % 37, i / 37))
            .collect();
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (mut files, arena) = make_paths(&refs);
        for file in files.iter_mut() {
            file.set_modified(1_700_000_000);
        }
        let parser = QueryParser::default();
        let parsed = parser.parse("handler");
        let page = |limit: usize| {
            let ctx = ScoringContext {
                query: &parsed,
                max_threads: 1,
                max_typos: 2,
                current_file: None,
                last_same_query_match: None,
                project_path: None,
                combo_boost_score_multiplier: 100,
                min_combo_count: 3,
            check_deleted: true,
                pagination: PaginationArgs { offset: 0, limit },
            };
            let (items, scores, _) = fuzzy_match_and_score_files(
                &files,
                &ctx,
                files.len(),
                arena,
                arena,
                max_path_len_of(&files),
                crate::parallelism::AbortToken::disabled(),
            );
            let paths = items
                .iter()
                .map(|f| f.relative_path(arena))
                .collect::<Vec<_>>();
            let totals = scores.iter().map(|s| s.total).collect::<Vec<_>>();
            (paths, totals)
        };
        let (full, full_totals) = page(0);
        assert!(full.len() >= 128);
        assert!(
            full_totals[..128].iter().all(|&t| t == full_totals[0]),
            "test needs a wide score tie"
        );
        for limit in [1, 2, 3, 5, 8, 16, 32, 64] {
            let (got, _) = page(limit);
            assert_eq!(got, full[..limit], "limit={limit}");
        }
    }

    #[test]
    fn same_config_repeats_identically() {
        // Unique-key input; identical totals ordered by modified alone.
        let paths: Vec<String> = (0..200)
            .map(|i| format!("src/mod{}/f{i}.rs", i % 37))
            .collect();
        let items: Vec<FileItem> = paths
            .iter()
            .map(|p| {
                let fname = p.rfind(std::path::is_separator).map(|i| i + 1).unwrap_or(0) as u16;
                FileItem::new_raw(fname, 0, 0, None, false)
            })
            .collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&paths, &items);
        let arena = store.as_arena_ptr();
        let mut files = items;
        for (i, file) in files.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(paths[i].as_bytes()));
            file.set_modified(1_700_000_000 + i as u64);
        }
        std::mem::forget(store);
        let results: Vec<(&FileItem, Score)> = files
            .iter()
            .map(|f| {
                (
                    f,
                    Score {
                        total: 151,
                        ..Default::default()
                    },
                )
            })
            .collect();
        let parser = QueryParser::default();
        let parsed = parser.parse("handler");
        let ctx = |limit: usize| ScoringContext {
            query: &parsed,
            max_threads: 1,
            max_typos: 2,
            current_file: None,
            last_same_query_match: None,
            project_path: None,
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            check_deleted: true,
            pagination: PaginationArgs { offset: 0, limit },
        };
        let (items50, _, total50) = sort_and_paginate(results.clone(), &ctx(50), results.len());
        let (items0, _, total0) = sort_and_paginate(results.clone(), &ctx(0), results.len());
        let names = |items: Vec<&FileItem>| {
            items
                .iter()
                .map(|f| f.relative_path(arena))
                .collect::<Vec<_>>()
        };
        let n50 = names(items50);
        let n0 = names(items0);
        assert_eq!(total50, total0);
        assert_eq!(n50, &n0[..50]);
    }

    #[test]
    fn parallel_topk_deterministic_across_runs() {
        let mut paths = Vec::new();
        for i in 0..2000 {
            paths.push(format!("src/a/file_{}.rs", i % 100));
        }
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (files, arena) = make_paths(&refs);
        let first = search_with(&files, "file", arena, 4, None).0;
        for _ in 0..10 {
            let again = search_with(&files, "file", arena, 4, None).0;
            assert_eq!(first, again, "parallel top-k must be deterministic");
        }
    }
}

#[cfg(test)]
mod scratch_reuse_tests {
    use super::*;

    #[test]
    fn take_tls_init_reuses_allocation() {
        let cell = std::cell::RefCell::new(Vec::<u16>::new());
        // Odd sizes around lane/batch boundaries, plus empty.
        for n in [0usize, 1, 7, 8, 9, 31, 32, 33, 129, 1000] {
            let mut v = take_tls_init(&cell, n);
            assert!(v.capacity() >= n);
            assert_eq!(v.len(), n);
            // Mirror the production contract: overwrite every lane.
            for (i, s) in v.iter_mut().enumerate() {
                *s = (i % 251) as u16;
            }
            for (i, s) in v.iter().enumerate() {
                assert_eq!(*s, (i % 251) as u16);
            }
            let cap = v.capacity();
            *cell.borrow_mut() = v;
            // Re-take below capacity reuses the allocation; above may grow.
            let want = n / 2 + 1;
            let w = take_tls_init(&cell, want);
            assert_eq!(w.len(), want);
            if cap >= want {
                assert_eq!(w.capacity(), cap);
            } else {
                assert!(w.capacity() >= want);
            }
            *cell.borrow_mut() = w;
        }
    }

    #[test]
    fn helpers_cover_all_scratch_types() {
        // Every T in take_tls_init's call sites must satisfy the bound.
        let trim_cell = std::cell::RefCell::new(Vec::<crate::matcher::Trim>::new());
        let t = take_tls_init(&trim_cell, 4);
        assert_eq!(t.len(), 4);
        let u32_cell = std::cell::RefCell::new(Vec::<u32>::new());
        let u = take_tls_init(&u32_cell, 4);
        assert_eq!(u.len(), 4);
        let u8_cell = std::cell::RefCell::new(Vec::<u8>::new());
        let b = take_tls_init(&u8_cell, 4);
        assert_eq!(b.len(), 4);
    }
}
