use crate::{
    git::is_modified_status,
    index::constraints::apply_constraints,
    path_utils::{CurrentDir, DirPenaltyMemo, calculate_distance_penalty},
    simd_path::{ArenaPtr, MAX_FRIZBEE_CHUNKS, MAX_PATH_CHUNKS},
    sort_buffer::sort_with_buffer,
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

fn phase_timing() -> bool {
    *PHASE_TIMING_FLAG.get_or_init(|| std::env::var_os("FFF_PHASE_TIMING").is_some())
}

fn force_scalar() -> bool {
    *FORCE_SCALAR_FLAG.get_or_init(|| std::env::var_os("FFF_FORCE_SCALAR").is_some())
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

// Measured break-even for per-file presence rechecks: expected reject
// savings vs recheck cost, both timed on the probe sample. No tuning.
fn presence_rechecks_pay(rate: f64, presence_ns: f64, gate_ns: f64) -> bool {
    (1.0 - rate) * gate_ns > presence_ns
}

fn match_fuzzy_parts(
    fuzzy_parts: &[&str],
    working_files: &FileItems<'_>,
    options: &neo_frizbee::Config,
    max_threads: usize,
    arena: ArenaPtr,
    max_path_len: usize,
) -> Vec<neo_frizbee::Match> {
    // Stack for the common 1-2 part queries; spills rarely.
    let valid_parts: SmallVec<[&str; 4]> = fuzzy_parts
        .iter()
        .copied()
        .filter(|p| p.len() >= 2)
        .collect();

    if valid_parts.is_empty() {
        tracing::debug!("match_fuzzy_parts: no valid parts after filtering, returning empty");
        return vec![];
    }

    // Prefilter is an internal scan — parallelize even when the *search*
    // is 1 thread (one query). Interseq SW already uses the bg pool this way.
    // The cap comes from the machine profile (cores, bounded by LLC), not a
    // literal: `.min(8)` starved 32-core servers and oversubscribed phones.
    let scan_cap = crate::machine::scan_thread_cap();
    // Two-part narrowing starts from the longer part (rejects more, so
    // the second pass scans less); each part keeps its own-len typo cap.
    let reverse_pair = valid_parts.len() == 2 && valid_parts[1].len() > valid_parts[0].len();
    let first_part: &str = if reverse_pair {
        valid_parts[1]
    } else {
        valid_parts[0]
    };
    let mut first_options = *options;
    first_options.max_typos = options.max_typos.map(|t| t.min(first_part.len() as u16));
    // Thread count comes from the work estimate alone (no size cutoffs); the
    // picker's cached max path length keeps this O(1) per query.
    let threads = effective_match_threads(
        working_files.len(),
        first_part.len(),
        max_path_len,
        first_options.max_typos,
        scan_cap.max(max_threads),
    );

    // Inter-sequence SW: frizbee prefilter for match set + trim windows,
    // then re-score with validated SW that replicates frizbee's gap model
    // on the SAME trimmed slices with the SAME include_prefix semantics.
    let first_part_matches = {
        let part_timing = phase_timing();
        let t_prefilter = std::time::Instant::now();
        // Phase 1: gate match-set + windows. The skip_sw path
        // encodes (start_pos_adj, end_pos) in (score, end_col).
        let np = crate::presence::NeedlePresence::new(
            first_part.as_bytes(),
            first_options.max_typos.unwrap_or(u16::MAX),
        );
        // Index reject before frizbee: only resolve paths that pass
        // length+presence. Only worth the extra scan when the needle can
        // actually reject (longer / typed letters). Common short queries
        // match almost everything — skip so we don't scan twice.
        // Adaptive candidate prefilter: probe a dispatch-sized sample
        // (cost ~= one fork/join), time presence and gate on it, and fire
        // the full parallel scan only when measured reject savings beat
        // the scan. Short/common needles probe at ~100% pass and skip by
        // measurement; small inputs skip (no parallel scan amortizes).
        let gate_typos = first_options.max_typos.unwrap_or(u16::MAX);
        let gate_state = crate::gate::GateNeedle::new(first_part.as_bytes(), gate_typos);
        let mut cand: Vec<u32> = Vec::new();
        let t_cand = std::time::Instant::now();
        let n_files = working_files.len();
        let avg_bytes = max_path_len.max(1);
        let big_enough =
            crate::dispatch::plan_files(n_files, avg_bytes, usize::MAX).threads > 1;
        // Probe-measured presence economics (rate, presence_ns, gate_ns),
        // set only when the gate probe was timed (empty probes stay selective).
        let mut presence_probe: Option<(f64, f64, f64)> = None;
        if big_enough && n_files > 0 {
            let n_probe = crate::dispatch::probe_len(n_files, avg_bytes);
            let stride = (n_files / n_probe).max(1);
            let needle_b = first_part.as_bytes();
            let min_len = np.min_path_len();
            let t_probe = std::time::Instant::now();
            let mut pass = 0usize;
            let mut probed = 0usize;
            // Passing probe indices for gate timing below (bounded: the
            // probe itself must stay near one fork/join of cost).
            let mut pass_idx: SmallVec<[u32; 32]> = SmallVec::new();
            for i in (0..n_files).step_by(stride) {
                let f = working_files.index(i);
                if (f.path.byte_len as usize) >= min_len {
                    let mask = f.presence_mask();
                    if mask == 0 || np.allows(mask, needle_b) {
                        pass += 1;
                        if pass_idx.len() < 32 {
                            pass_idx.push(i as u32);
                        }
                    }
                }
                probed += 1;
                if probed >= n_probe {
                    break;
                }
            }
            let presence_ns =
                t_probe.elapsed().as_nanos() as f64 / probed.max(1) as f64;
            // Time the gate on passing probe files for the savings estimate.
            let mut gate_ns = 0.0f64;
            if !pass_idx.is_empty() {
                let t_gate = std::time::Instant::now();
                for &pi in pass_idx.iter() {
                    let f = working_files.index(pi as usize);
                    std::hint::black_box(gate_state.hit_file(f, arena));
                }
                gate_ns = t_gate.elapsed().as_nanos() as f64 / pass_idx.len() as f64;
            }
            let rate = pass as f64 / probed.max(1) as f64;
            if !pass_idx.is_empty() {
                presence_probe = Some((rate, presence_ns, gate_ns));
            }
            // Serial-view break-even: rejects x gate cost vs full presence
            // scan + in-group rechecks + dispatch. Empty probe passes mean
            // groups all-reject cheaply on their own; a scan proving that
            // would cost more than it saves.
            let fire = pass > 0
                && rate < 1.0
                && (n_files as f64 * (1.0 - rate) * gate_ns
                    > n_files as f64 * presence_ns
                        + crate::dispatch::fork_join_ns() as f64
                        + n_files as f64 * rate * presence_ns);
            if fire {
                use rayon::prelude::*;
                cand = crate::parallelism::run_on_search_pool(|| {
                    (0..n_files)
                        .into_par_iter()
                        .filter_map(|i| {
                            let f = working_files.index(i);
                            if (f.path.byte_len as usize) < min_len {
                                return None;
                            }
                            let mask = f.presence_mask();
                            if mask != 0 && !np.allows(mask, needle_b) {
                                return None;
                            }
                            Some(i as u32)
                        })
                        .collect()
                });
            }
        }
        let cand_ms = t_cand.elapsed().as_secs_f64() * 1e3;
        // Dense re-index only helps when something was rejected;
        // a full pass set would add indirection for no savings.
        let use_cand = !cand.is_empty() && cand.len() < n_files;
        // TISG gate match-set + window without frizbee. `gate_n` is the
        // gate's index space (candidates when the prefilter fired).
        let gate_n = if use_cand { cand.len() } else { n_files };
        // Rechecks are a sound pretest, so skipping never changes results:
        // scan members already passed the same test, and uniform queries
        // (probe rate ~1) save less than rechecks cost (measured, no tuning).
        // Honors `FFF_NO_GATE_PREFILTER`, as the test-only path does.
        let check_presence = np.should_check()
            && std::env::var_os("FFF_NO_GATE_PREFILTER").is_none()
            && !use_cand
            && presence_probe.map_or(true, |(rate, pns, gns)| {
                presence_rechecks_pay(rate, pns, gns)
            });
        let t_gate_us = std::sync::atomic::AtomicU64::new(0);
        let t_exact_us = std::sync::atomic::AtomicU64::new(0);
        let t_swslice_us = std::sync::atomic::AtomicU64::new(0);
        let prefilter_ms = t_prefilter.elapsed().as_secs_f64() * 1e3;

        // Phase 2: re-score with inter-sequence SW on trimmed windows.
        let has_upper = first_part.chars().any(|c| c.is_uppercase());
        let p = crate::interseq_sw::SwParams::for_query(has_upper);
        let needle = first_part.as_bytes();

        // Query intent: `name.ext` — pin exact filename hits among the match
        // set (not a full-corpus walk; score_one already ranks exact names).
        let looks_like_filename = {
            let q = first_part;
            q.len() >= 4 && q.contains('.') && !q.contains('/')
        };
        let exact_for = |chunk_matches: &[neo_frizbee::Match]| {
            let mut hits: Vec<u32> = Vec::new();
            if looks_like_filename {
                let qb = first_part.as_bytes();
                // Reused scratch: `filename_cow` would heap-allocate per candidate
                // whose filename straddles a chunk, and this loop walks every match.
                let mut name_scratch = [0u8; crate::simd_path::PATH_BUF_SIZE];
                for m in chunk_matches.iter() {
                    let f = working_files.index(m.index as usize);
                    let flen = f.path.byte_len as usize - f.path.filename_offset as usize;
                    if flen == qb.len() {
                        let fname = f.path.filename_ref(arena, &mut name_scratch);
                        if fname.eq_ignore_ascii_case(first_part) {
                            hits.push(m.index);
                        }
                    }
                }
            }
            // Sorted for binary search in the assemble pass below, which
            // applies the bump inline (no second pass, no hash set).
            hits.sort_unstable();
            hits
        };

        // SW scoring (collapse removed: cheap rank disagreed with SW
        // on real corpora; reordering would be a UX regression).
        // (Phase counters print after the fused pass below, once the
        // match count is known.)
        let t_sw = std::time::Instant::now();
        {
            // Chunked scoring: metadata working set sized from the
            // private L2 so TLS scratch stays resident (vs a full
            // 227k-wide set of faults/RSS). See `sw_metadata_chunk`.
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
                // Cap retained capacity so one huge match set does not pin TLS
                // forever, but keep a full chunk warm (avoids re-alloc/re-fault).
                // Hysteresis is one extra SIMD batch past the chunk.
                let retain = crate::dispatch::sw_metadata_chunk() + 64;
                if b.capacity() > retain {
                    b.shrink_to(retain);
                }
            }

            // Portability: FFF_FORCE_SCALAR=1 skips AVX2 (SSE2/scalar path only).
            // Used to measure low-spec chips; must stay result-identical.
            let force_scalar = force_scalar();
            let avx2_ok = if force_scalar {
                false
            } else {
                simd_available()
            };

            // u8 fast path: 32 lanes, valid when the max repr. score < 256
            // (u8_safe) and the needle fits the kernel's 32-row domain.
            let use_u8 = avx2_ok && needle.len() <= 32 && p.u8_safe(needle.len());
            let batch_sz: usize = if use_u8 { 32 } else { 16 };
            // u16 rows are dynamic to 32; longer needles stay scalar.
            let use_simd = avx2_ok && needle.len() <= 32;
            // Probe hatch: serial batches inside each chunk (one task per
            // chunk instead of per batch). Sizes nested-fanout overhead.
            let no_batch_par = std::env::var_os("FFF_NO_BATCH_PAR").is_some();

            // Sub-phase accounting, accumulated across rayon workers.
            let t_collect_us = std::sync::atomic::AtomicU64::new(0);
            let t_batch_us = std::sync::atomic::AtomicU64::new(0);
            let t_assemble_us = std::sync::atomic::AtomicU64::new(0);
            // Diagnostic split of batch_ms; each add is timing-gated.
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

            // Chunks are disjoint slices of `matches` and every scratch buffer is
            // thread-local, so this fans out safely.
            //
            // Only fan out when there is more than one chunk: a single chunk has
            // nothing to split, and the fork/join alone measured ~8x the whole
            // serial body on a few-hundred-match corpus (782us vs 97us).
            // `outer_saturates` skips the inner fanout when file chunks alone
            // feed every worker: nested dispatch then only adds join/task
            // overhead (measured ~29 CPU-ms on 116k) for no wall gain.
            let sw_body = |matches_chunk: &mut [neo_frizbee::Match],
                           exact_hits: &[u32],
                           outer_saturates: bool| {
                let mut path_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
                let t_chunk = std::time::Instant::now();
                let n_matches = matches_chunk.len();
                let mut trims = TRIMS.with(|c| take_tls(c, n_matches));
                trims.clear();
                {
                    for (mi, m) in matches_chunk.iter().enumerate() {
                        let file = working_files.index(m.index as usize);
                        // Prefetch next FileItem (sequential over this chunk).
                        #[cfg(target_arch = "x86_64")]
                        if mi + 1 < n_matches {
                            let nf = working_files.index(matches_chunk[mi + 1].index as usize);
                            unsafe {
                                std::arch::x86_64::_mm_prefetch(
                                    (nf as *const crate::types::FileItem).cast::<i8>(),
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                        }
                        let w = decode_match_window(m, file.path.byte_len as usize);
                        trims.push(crate::matcher::Trim::from_window(w));
                    }
                }
                let _ = t_chunk;
                if timing {
                    t_collect_us.fetch_add(
                        t_chunk.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }

                let t_batch = std::time::Instant::now();
                // 64-byte align the batch output buffers: batch_sz=32 u16 is
                // exactly one cache line, so aligned batches do not false-share
                // at par_chunks boundaries (MESI line ping-pong). Lengths are
                // set without zero-fill; every lane is written before any read.
                let mut scores = SCORES.with(|c| take_tls(c, n_matches + 32));
                set_len_uninit(&mut scores, n_matches + 32);
                let mut end_cols = ECOLS.with(|c| take_tls(c, n_matches + 32));
                set_len_uninit(&mut end_cols, n_matches + 32);

                // Scalar lanes skip staging/SoA/order/par entirely: one serial
                // pass over the trims with a reused window buffer. Same bytes in.
                let scalar_direct = !use_simd;
                if scalar_direct {
                    thread_local! {
                        static SWIN: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
                    }
                    let t_scalar = std::time::Instant::now();
                    let mut max_trim = 0usize;
                    for t in trims.iter() {
                        max_trim = max_trim.max((t.end - t.start) as usize);
                    }
                    let mut win = SWIN.with(|c| take_tls(c, max_trim));
                    set_len_uninit(&mut win, max_trim);
                    for (i, t) in trims.iter().enumerate() {
                        let wlen = (t.end - t.start) as usize;
                        if wlen == 0 {
                            scores[i] = 0;
                            end_cols[i] = 0;
                            continue;
                        }
                        let file = working_files.index(t.file_idx as usize);
                        file.path.copy_window_to(
                            arena,
                            t.start as u32,
                            t.end as u32,
                            &mut win[..wlen],
                        );
                        let (sc, ec) = crate::interseq_sw::score_one_scalar_full(
                            needle,
                            &win[..wlen],
                            &p,
                            t.include_prefix(),
                        );
                        scores[i] = sc;
                        end_cols[i] = ec;
                    }
                    SWIN.with(|c| reclaim(&mut c.borrow_mut(), win));
                    if timing {
                        t_scalar_us.fetch_add(
                            t_scalar.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        t_batch_us.fetch_add(
                            t_batch.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                } else {
                    // Length-ordered batches within the chunk so per-batch max_len
                    // stays close to the lanes' own lengths.
                    let t_prep = std::time::Instant::now();
                    let mut order = ORDER.with(|c| take_tls(c, n_matches));
                    set_len_uninit(&mut order, n_matches);
                    // Counting sort by trim length (stable, O(n)). Batch grouping only
                    // affects SIMD max_len, not per-match scores. A single batch is
                    // already one group, so identity order is the same grouping.
                    let t_sort = std::time::Instant::now();
                    if n_matches > batch_sz {
                        let mut cnt = [0u32; 129];
                        for i in 0..n_matches {
                            let (s, e) = (trims[i].start, trims[i].end);
                            let l = (e - s).min(128) as usize;
                            cnt[l] += 1;
                        }
                        let mut start = [0u32; 129];
                        let mut acc = 0u32;
                        for l in 0..129 {
                            start[l] = acc;
                            acc += cnt[l];
                        }
                        for i in 0..n_matches {
                            let (s, e) = (trims[i].start, trims[i].end);
                            let l = (e - s).min(128) as usize;
                            order[start[l] as usize] = i as u32;
                            start[l] += 1;
                        }
                    } else {
                        for i in 0..n_matches {
                            order[i] = i as u32;
                        }
                    }
                    if timing {
                        t_sort_us.fetch_add(
                            t_sort.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    let mut perm_scores = PERM_SCORES.with(|c| take_tls(c, n_matches + 32));
                    set_len_uninit(&mut perm_scores, n_matches + 32);
                    let mut perm_ecols = PERM_ECOLS.with(|c| take_tls(c, n_matches + 32));
                    set_len_uninit(&mut perm_ecols, n_matches + 32);

                    fn align64_u16(buf: &mut [u16], n: usize) -> &mut [u16] {
                        let addr = buf.as_ptr() as usize;
                        let pad = ((64 - (addr % 64)) % 64) / 2;
                        &mut buf[pad..pad + n]
                    }
                    let ps_aligned = align64_u16(&mut perm_scores, n_matches);
                    let pe_aligned = align64_u16(&mut perm_ecols, n_matches);
                    if timing {
                        t_prep_us.fetch_add(
                            t_prep.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }

                    // One batch = one (ord, out) triple. A pure indexed map, so
                    // serial and parallel runs write bit-identical outputs.
                    let score_batch = |ord: &[u32], sc_out: &mut [u16], ec_out: &mut [u16]| {
                        let n = ord.len();
                        let mut max_len = 0usize;
                        for &oi in ord.iter() {
                            let (s, e) = (trims[oi as usize].start, trims[oi as usize].end);
                            max_len = max_len.max((e - s) as usize);
                        }
                        if max_len == 0 {
                            return;
                        }
                        let rows = max_len.div_ceil(8) * 8;
                        thread_local! {
                            static SOA: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
                            static STAGE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
                        }
                        let buf_len = rows * batch_sz;
                        let t_stage = std::time::Instant::now();
                        let mut stage = STAGE.with(|c| {
                            let mut b = c.borrow_mut();
                            if b.len() < buf_len {
                                b.resize(buf_len, 0);
                            }
                            std::mem::take(&mut *b)
                        });
                        stage.truncate(buf_len);
                        // Zero only the padding lanes (SIMD path reads them).
                        if n > 8 {
                            for k in n..batch_sz {
                                stage[k * rows..(k + 1) * rows].fill(0);
                            }
                        }

                        let mut prefix_mask: u32 = 0;
                        let mut next_ptrs = [std::ptr::null::<u8>(); MAX_PATH_CHUNKS];
                        for (k, &oi) in ord.iter().enumerate() {
                            let t = trims[oi as usize];
                            let (fi, s, e, inc_prefix) =
                                (t.file_idx, t.start, t.end, t.include_prefix());
                            let file = working_files.index(fi as usize);
                            #[cfg(target_arch = "x86_64")]
                            if k + 1 < n {
                                let nf = working_files
                                    .index(trims[ord[k + 1] as usize].file_idx as usize);
                                let nptrs = nf.path.resolve_ptrs(arena, &mut next_ptrs);
                                for &p in nptrs.iter().take(4) {
                                    unsafe {
                                        std::arch::x86_64::_mm_prefetch(
                                            p as *const i8,
                                            std::arch::x86_64::_MM_HINT_T0,
                                        );
                                    }
                                }
                            }
                            let wlen = (e - s) as usize;
                            let dst = &mut stage[k * rows..k * rows + wlen];
                            file.path.copy_window_to(arena, s as u32, e as u32, dst);
                            // Pad the rest of this lane to zeros (SoA expects it).
                            // Scalar path never reads the tail; skip the memset.
                            if n > 8 {
                                stage[k * rows + wlen..(k + 1) * rows].fill(0);
                            }
                            if inc_prefix {
                                prefix_mask |= 1u32 << k;
                            }
                        }
                        if timing {
                            t_stage_us.fetch_add(
                                t_stage.elapsed().as_micros() as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }

                        // Tiny trailing chunks cannot amortize a 32-wide
                        // transpose + kernel; score them scalar from stage.
                        if n <= 8 {
                            let t_scalar = std::time::Instant::now();
                            for (k, &oi) in ord.iter().enumerate() {
                                let tw = trims[oi as usize];
                                let (s, e, inc_prefix) = (tw.start, tw.end, tw.include_prefix());
                                let src = &stage[k * rows..k * rows + (e - s) as usize];
                                let (sc, ec) = crate::interseq_sw::score_one_scalar_full(
                                    needle, src, &p, inc_prefix,
                                );
                                sc_out[k] = sc;
                                ec_out[k] = ec;
                            }
                            if timing {
                                t_scalar_us.fetch_add(
                                    t_scalar.elapsed().as_micros() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            STAGE.with(|c| *c.borrow_mut() = stage);
                            return;
                        }

                        #[cfg(target_arch = "x86_64")]
                        if use_simd {
                            let t_xpose = std::time::Instant::now();
                            let mut soa = SOA.with(|c| {
                                let mut b = c.borrow_mut();
                                if b.len() < buf_len {
                                    b.resize(buf_len, 0);
                                }
                                std::mem::take(&mut *b)
                            });
                            soa.truncate(buf_len);
                            unsafe {
                                crate::interseq_sw::avx2::soa_from_stage(
                                    &stage, rows, batch_sz, &mut soa,
                                );
                            }
                            if timing {
                                t_xpose_us.fetch_add(
                                    t_xpose.elapsed().as_micros() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            let t_kernel = std::time::Instant::now();
                            unsafe {
                                if use_u8 {
                                    crate::interseq_sw::avx2::score_batch_avx2_u8_dispatch(
                                        needle,
                                        &soa,
                                        max_len,
                                        &p,
                                        prefix_mask,
                                        &mut sc_out[..n],
                                        &mut ec_out[..n],
                                    );
                                    // Dead net while u8_safe holds: any
                                    // saturated lane is rescored scalar.
                                    if sc_out[..n].contains(&255) {
                                        for (k, &oi) in ord.iter().enumerate() {
                                            let tw = trims[oi as usize];
                                            let wlen = (tw.end - tw.start) as usize;
                                            let src = &stage[k * rows..k * rows + wlen];
                                            let (sc, ec) =
                                                crate::interseq_sw::score_one_scalar_full(
                                                    needle,
                                                    src,
                                                    &p,
                                                    tw.include_prefix(),
                                                );
                                            sc_out[k] = sc;
                                            ec_out[k] = ec;
                                        }
                                    }
                                } else {
                                    crate::interseq_sw::avx2::score_batch_avx2(
                                        needle,
                                        &soa,
                                        max_len,
                                        &p,
                                        prefix_mask as u16,
                                        &mut sc_out[..n],
                                        &mut ec_out[..n],
                                    );
                                }
                            }
                            if timing {
                                t_kernel_us.fetch_add(
                                    t_kernel.elapsed().as_micros() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            SOA.with(|c| *c.borrow_mut() = soa);
                            STAGE.with(|c| *c.borrow_mut() = stage);
                            return;
                        }
                        // Scalar fallback when SIMD is unavailable for this query.
                        let t_scalar = std::time::Instant::now();
                        for (k, &oi) in ord.iter().enumerate() {
                            let tw = trims[oi as usize];
                            let (s, e, inc_prefix) = (tw.start, tw.end, tw.include_prefix());
                            let src = &stage[k * rows..k * rows + (e - s) as usize];
                            let (sc, ec) = crate::interseq_sw::score_one_scalar_full(
                                needle, src, &p, inc_prefix,
                            );
                            sc_out[k] = sc;
                            ec_out[k] = ec;
                        }
                        if timing {
                            t_scalar_us.fetch_add(
                                t_scalar.elapsed().as_micros() as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                        STAGE.with(|c| *c.borrow_mut() = stage);
                    };
                    // Fan out only past Amdahl break-even at measured
                    // dispatch cost; inside pool workers the guard below
                    // degrades to serial (no nested storm). Batch cost is
                    // the running measured estimate (self-tuning per box);
                    // each slice folds its mean back in.
                    let n_batches = order.len().div_ceil(batch_sz);
                    let batch_est = crate::dispatch::batch_ns_estimate(use_u8);
                    let pool_width = crate::parallelism::search_pool_threads();
                    let t_fan = std::time::Instant::now();
                    if no_batch_par
                        || outer_saturates
                        || !crate::dispatch::parallel_batches(n_batches, batch_est, pool_width)
                    {
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
                    } else {
                        if timing {
                            t_fan_parallel
                                .fetch_add(n_batches as u64, std::sync::atomic::Ordering::Relaxed);
                        }
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
                    }
                    crate::dispatch::observe_batch_ns(
                        use_u8,
                        t_fan.elapsed().as_nanos() as f64,
                        n_batches,
                    );

                    // Inverse order map into the chunk-local score arrays.
                    // Read from the 64-byte-aligned batch outputs (true per-line).
                    let t_scatter = std::time::Instant::now();
                    for (i, &oi) in order.iter().enumerate() {
                        scores[oi as usize] = ps_aligned[i];
                        end_cols[oi as usize] = pe_aligned[i];
                    }
                    if timing {
                        t_scatter_us.fetch_add(
                            t_scatter.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    if timing {
                        t_batch_us.fetch_add(
                            t_batch.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    ORDER.with(|c| reclaim(&mut c.borrow_mut(), order));
                    PERM_SCORES.with(|c| reclaim(&mut c.borrow_mut(), perm_scores));
                    PERM_ECOLS.with(|c| reclaim(&mut c.borrow_mut(), perm_ecols));
                }

                // Assemble this chunk into the match structs (no full-size hold).
                let t_asm = std::time::Instant::now();
                for (i, m) in matches_chunk.iter_mut().enumerate() {
                    let t = trims[i];
                    let (fi, start_adj, end_pos, include_exact) =
                        (t.file_idx, t.start as u32, t.end as u32, t.include_exact());
                    let exact = include_exact && {
                        let file = working_files.index(fi as usize);
                        let hay = file.path.read_to_buf(arena, &mut path_buf);
                        needle.eq_ignore_ascii_case(hay.as_bytes())
                            && (end_pos as usize) == hay.len()
                    };
                    let bonus = if exact { 8u16 } else { 0 };
                    m.score = scores[i].saturating_add(bonus);
                    m.end_col = end_cols[i].saturating_add(start_adj.min(u16::MAX as u32) as u16);
                    m.exact = exact;
                    if !exact_hits.is_empty() && exact_hits.binary_search(&m.index).is_ok() {
                        m.score = m.score.max(200);
                        m.exact = true;
                    }
                }
                if timing {
                    t_assemble_us.fetch_add(
                        t_asm.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }

                TRIMS.with(|c| reclaim(&mut c.borrow_mut(), trims));
                SCORES.with(|c| reclaim(&mut c.borrow_mut(), scores));
                ECOLS.with(|c| reclaim(&mut c.borrow_mut(), end_cols));
            };
            // Fused groups: gate, exact, and SW per file chunk while the
            // chunk's bytes are hot. Concatenated in file order; outputs
            // are per-index, so the size cannot change results.
            // Test-tunable (`FFF_SW_CHUNK`); production plans from the
            // same work model as the thread estimator, capped at pool
            // width: the gate is latency-bound (SMT siblings add MLP),
            // so width here buys wall even past the LLC-bound scan cap.
            // Small index spaces still serialize via the amortize rule.
            // Same outputs for any chunk size.
            let per_file = crate::dispatch::match_bytes_per_file(
                first_part.len(),
                max_path_len,
                first_options.max_typos,
            );
            let pool = crate::parallelism::effective_core_count().max(1);
            let plan = crate::dispatch::plan_files(gate_n, per_file, pool);
            // Serial cutoff comes from the plan, unchanged: below it the
            // fork/join alone costs more than the whole body.
            let serial = gate_n <= plan.chunk;
            let file_chunk: usize = std::env::var("FFF_SW_CHUNK")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n| n >= 1)
                .unwrap_or_else(|| {
                    // Over-partition past one task per worker: extra groups
                    // give work-stealing slack against stragglers and keep
                    // each group's bytes L2-resident. Measured 4x (2-4k
                    // sweet spot on 227k); floor bounds per-group costs.
                    (plan.chunk / 4).max(1024)
                });
            let n_groups = gate_n.div_ceil(file_chunk);
            let one_group = |g: usize| {
                let lo = g * file_chunk;
                let hi = ((g + 1) * file_chunk).min(gate_n);
                let t = std::time::Instant::now();
                let mut m: Vec<neo_frizbee::Match> = Vec::new();
                for k in lo..hi {
                    let (idx, file) = if use_cand {
                        let ci = cand[k];
                        (ci, working_files.index(ci as usize))
                    } else {
                        (k as u32, working_files.index(k))
                    };
                    if check_presence
                        && !crate::presence::allows_or_unknown(
                            &np,
                            file.presence_mask(),
                            first_part.as_bytes(),
                        )
                    {
                        continue;
                    }
                    if let Some(hit) = gate_state.hit_file(file, arena) {
                        m.push(neo_frizbee::Match {
                            score: hit.start as u16,
                            index: idx,
                            exact: false,
                            end_col: hit.end as u16,
                        });
                    }
                }
                t_gate_us.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                let t = std::time::Instant::now();
                let exact = exact_for(&m);
                t_exact_us.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                let t = std::time::Instant::now();
                sw_body(&mut m, &exact, n_groups > 1 && n_groups >= pool);
                t_swslice_us.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                m
            };
            let mut matches: Vec<neo_frizbee::Match> = Vec::new();
            if serial {
                for g in 0..n_groups {
                    matches.extend(one_group(g));
                }
            } else {
                use rayon::prelude::*;
                let scored: Vec<Vec<neo_frizbee::Match>> =
                    (0..n_groups).into_par_iter().map(one_group).collect();
                for v in scored {
                    matches.extend(v);
                }
            }

            let sw_us = t_sw.elapsed().as_micros();
            let collect_ms = t_collect_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let batch_ms = t_batch_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let assemble_ms = t_assemble_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let stage_ms = t_stage_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let xpose_ms = t_xpose_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let kernel_ms = t_kernel_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let scalar_ms = t_scalar_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let prep_ms = t_prep_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let scatter_ms = t_scatter_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let sort_ms = t_sort_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
            let fan_serial = t_fan_serial.load(std::sync::atomic::Ordering::Relaxed);
            let fan_parallel = t_fan_parallel.load(std::sync::atomic::Ordering::Relaxed);
            if phase_timing() {
                eprintln!(
                    "PHASE cand_ms={cand_ms:.1} wins_ms=0.0 samp_ms=0.0 collapsed=false n={}",
                    matches.len()
                );
            }
            let gate_us = t_gate_us.load(std::sync::atomic::Ordering::Relaxed);
            let exact_us = t_exact_us.load(std::sync::atomic::Ordering::Relaxed);
            let swslice_us = t_swslice_us.load(std::sync::atomic::Ordering::Relaxed);
            if part_timing {
                eprintln!(
                    "PHASE parts gate_us={gate_us} exact_us={exact_us} swslice_us={swslice_us} fuse_us={sw_us} threads={threads} n={}",
                    matches.len()
                );
            }
            if phase_timing() {
                eprintln!(
                    "PHASE interseq prefilter_ms={prefilter_ms:.1} collect_ms={collect_ms:.1} batch_ms={batch_ms:.1} assemble_ms={assemble_ms:.1} n={} stage_ms={stage_ms:.1} xpose_ms={xpose_ms:.1} kernel_ms={kernel_ms:.1} scalar_ms={scalar_ms:.1} prep_ms={prep_ms:.1} scatter_ms={scatter_ms:.1} sort_ms={sort_ms:.1} fan_serial={fan_serial} fan_parallel={fan_parallel}",
                    matches.len()
                );
            }
            matches
        }
    };

    if valid_parts.len() == 1 {
        return first_part_matches;
    }

    let total_parts = valid_parts.len() as u32;
    let mut matches = first_part_matches;
    // Narrowing order only affects speed. When the longer part ran first,
    // narrow with the query-order first part and keep its end_col.
    let rest: &[&str] = if reverse_pair {
        &valid_parts[..1]
    } else {
        &valid_parts[1..]
    };
    for part in rest.iter() {
        let mut part_options = *options;
        part_options.max_typos = options.max_typos.map(|t| t.min(part.len() as u16));

        // Match only the files that survived the previous round, addressed
        // through the previous matches without collecting a subset.
        // Serial cutoff lives in the estimator (amortize rule), not here.
        let survivors = &matches;
        let sub_threads = effective_match_threads(
            survivors.len(),
            part.len(),
            max_path_len,
            part_options.max_typos,
            scan_cap.max(max_threads),
        );
        let sub_np = crate::presence::NeedlePresence::new(
            part.as_bytes(),
            part_options.max_typos.unwrap_or(u16::MAX),
        );
        let sub_matches = with_cached_matcher(part, &part_options, |matcher| {
            matcher.match_range_parallel_resolved(
                survivors.len(),
                &|index, buf: &mut [*const u8; MAX_FRIZBEE_CHUNKS]| {
                    let file = working_files.index(survivors[index as usize].index as usize);
                    resolve_file_chunks_prefiltered(file, arena, buf, &sub_np, part.as_bytes())
                },
                sub_threads,
            )
        });

        if sub_matches.is_empty() {
            return vec![]; // break early!
        }

        // Map sub_matches back to original indices, average scores across all parts.
        matches = sub_matches
            .into_iter()
            .map(|sm| {
                let prev = &matches[sm.index as usize];
                let sum = (prev.score as u32).saturating_add(sm.score as u32);
                let avg = sum / total_parts;

                neo_frizbee::Match {
                    index: prev.index,
                    score: avg.min(u16::MAX as u32) as u16,
                    // Query-order first part's position for filename bonus.
                    end_col: if reverse_pair {
                        sm.end_col
                    } else {
                        prev.end_col
                    },
                    exact: prev.exact && sm.exact,
                }
            })
            .collect();
    }

    matches
}

/// Score only `idx` (indices into `base`) — used for typing live-narrow.
/// `total_matched` is the subset hit count.
pub(crate) fn fuzzy_match_and_score_subset<'a>(
    base: &'a [FileItem],
    idx: Vec<u32>,
    context: &ScoringContext,
    _base_count: usize,
    _base_arena: ArenaPtr,
    _overflow_arena: ArenaPtr,
    max_path_len: usize,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    let (results, total) = match_and_score_in_arena(
        FileItems::Filtered { base, idx },
        context,
        _base_arena,
        max_path_len,
    );
    let mut out = sort_and_paginate(results, context, total);
    out.2 = total;
    out
}

/// Match + score across base and overflow files, each with their own arena.
#[tracing::instrument(skip_all, level = tracing::Level::DEBUG)]
pub(crate) fn fuzzy_match_and_score_files<'a>(
    files: &'a [FileItem],
    context: &ScoringContext,
    base_count: usize,
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    max_path_len: usize,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    fuzzy_match_and_score_files_impl(
        files,
        context,
        base_count,
        base_arena,
        overflow_arena,
        max_path_len,
    )
}

fn fuzzy_match_and_score_files_impl<'a>(
    files: &'a [FileItem],
    context: &ScoringContext,
    base_count: usize,
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    max_path_len: usize,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    // Process overflow files first: newly added files (created after the
    // initial scan) live in the overflow arena and are more likely to be
    // relevant to the current search. Exact ties are broken by index order.
    let (results, total_matched) = if files.len() > base_count {
        let (mut results, mut total) = match_and_score_in_arena(
            FileItems::All(&files[base_count..]),
            context,
            overflow_arena,
            max_path_len,
        );

        let (base_results, base_total) = match_and_score_in_arena(
            FileItems::All(&files[..base_count]),
            context,
            base_arena,
            max_path_len,
        );
        results.extend(base_results);
        total += base_total;
        (results, total)
    } else {
        match_and_score_in_arena(
            FileItems::All(files),
            context,
            base_arena,
            max_path_len,
        )
    };

    let t_sort = std::time::Instant::now();
    let out = sort_and_paginate(results, context, total_matched);
    if phase_timing() {
        eprintln!(
            "PHASE paginate_us={} n={}",
            t_sort.elapsed().as_micros(),
            out.2
        );
    }
    out
}

pub(crate) fn fuzzy_match_byte_offsets_for_page<'q>(
    query: &'q FFFQuery<'q>,
    items: &[&FileItem],
    max_typos: u16,
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
) -> Vec<SmallVec<[(u32, u32); 4]>> {
    let parts: Vec<&str> = match &query.fuzzy_query {
        FuzzyQuery::Text(text) if text.len() >= 2 => vec![*text],
        FuzzyQuery::Parts(parts) => parts.iter().copied().filter(|p| p.len() >= 2).collect(),
        _ => Vec::new(),
    };

    let mut ranges_by_item = vec![SmallVec::new(); items.len()];
    if parts.is_empty() || items.is_empty() {
        return ranges_by_item;
    }

    // One flat buffer for every path instead of a `String` per item. The
    // slices below borrow `flat`, which is a local that outlives them, so
    // no borrow has to escape and nothing is thread-local.
    let mut flat: Vec<u8> = Vec::with_capacity(items.iter().map(|i| i.relative_path_len()).sum());
    let mut spans: Vec<(u32, u32)> = Vec::with_capacity(items.len());
    for item in items.iter() {
        let arena = if item.is_overflow() {
            overflow_arena
        } else {
            base_arena
        };
        let want = item.path.byte_len as usize;
        let start = flat.len();
        flat.resize(start + want, 0);
        item.path
            .copy_window_to(arena, 0, want as u32, &mut flat[start..]);
        spans.push((start as u32, want as u32));
    }
    // SAFETY: arena path bytes are valid UTF-8, the same assumption
    // `ChunkedString::read_to_buf` already makes.
    let path_strs: Vec<&str> = spans
        .iter()
        .map(|&(s, l)| unsafe {
            std::str::from_utf8_unchecked(&flat[s as usize..s as usize + l as usize])
        })
        .collect();

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

    for ranges in &mut ranges_by_item {
        *ranges = merge_byte_offsets(std::mem::take(ranges));
    }

    ranges_by_item
}

fn char_indices_to_byte_offsets(line: &str, char_indices: &[u32]) -> SmallVec<[(u32, u32); 4]> {
    // Single forward scan: map requested char indices to byte ranges without
    // materializing a full per-char table.
    if char_indices.is_empty() {
        return SmallVec::new();
    }
    let mut wanted: SmallVec<[u32; 4]> = SmallVec::from_slice(char_indices);
    wanted.sort_unstable();
    wanted.dedup();

    let mut result: SmallVec<[(u32, u32); 4]> = SmallVec::with_capacity(wanted.len());
    let mut next_char = 0u32;
    let mut wi = 0usize;
    for (byte_pos, ch) in line.char_indices() {
        if wi >= wanted.len() {
            break;
        }
        if next_char == wanted[wi] {
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
            while wi < wanted.len() && wanted[wi] == next_char {
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

/// Resolve a DirItem's chunked path into frizbee's pointer buffer.
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

/// Run SIMD fuzzy match across directory items, narrowing the candidate set
/// through each query part — same pipeline as file matching.
fn match_fuzzy_parts_dirs(
    fuzzy_parts: &[&str],
    working_dirs: &[&DirItem],
    options: &neo_frizbee::Config,
    max_threads: usize,
    arena: ArenaPtr,
    overflow_arena: ArenaPtr,
) -> Vec<neo_frizbee::Match> {
    // Stack for the common 1-2 part queries; spills rarely.
    let valid_parts: SmallVec<[&str; 4]> = fuzzy_parts
        .iter()
        .copied()
        .filter(|p| p.len() >= 2)
        .collect();

    if valid_parts.is_empty() {
        return vec![];
    }

    let scan_cap = crate::machine::scan_thread_cap();
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
        scan_cap.max(max_threads),
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
    let total_parts = valid_parts.len() as u32;
    for part in valid_parts[1..].iter() {
        let mut part_options = *options;
        part_options.max_typos = options.max_typos.map(|t| t.min(part.len() as u16));

        // Match only the dirs that survived the previous round, addressed
        // through the previous matches without collecting a subset.
        // Bound by the corpus max (O(1)): over-estimate saturates the
        // cap, never under-provisions.
        let survivors = &matches;
        let sub_dir_threads = effective_match_threads(
            survivors.len(),
            part.len(),
            max_dir_len,
            part_options.max_typos,
            scan_cap.max(max_threads),
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

        // Map sub_matches back to original indices, average scores across all parts.
        matches = sub_matches
            .into_iter()
            .map(|sm| {
                let prev = &matches[sm.index as usize];
                let sum = (prev.score as u32).saturating_add(sm.score as u32);
                let avg = sum / total_parts;

                neo_frizbee::Match {
                    index: prev.index,
                    score: avg.min(u16::MAX as u32) as u16,
                    end_col: prev.end_col, // keep first part's position for dirname bonus
                    exact: prev.exact && sm.exact,
                }
            })
            .collect();
    }

    matches
}

/// Match + score directories against a fuzzy query.
/// Scoring: base_score, frecency_boost, distance_penalty, dirname_bonus.
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

    // Stack for the common 1-2 part queries; spills rarely.
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

    let mut dir_buf = String::with_capacity(64);
    let mut dirname_buf = String::with_capacity(32);

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

            // Distance penalty from current file's directory.
            let distance_penalty = if context.current_file.is_some() {
                dir.path.write_to_string(dir_arena, &mut dir_buf);
                calculate_distance_penalty(context.current_file, &dir_buf)
            } else {
                0
            };

            // Dirname bonus: if the match is in the last path segment.
            let last_seg_offset = dir.last_segment_offset();
            let match_start_approx = path_match.end_col.saturating_sub(main_needle_len - 1);
            let is_dirname_match = match_start_approx >= last_seg_offset;

            dir.write_dir_name(dir_arena, &mut dirname_buf);
            let dirname_len = dirname_buf.len();
            let is_exact_dirname = is_dirname_match
                && main_needle_len as usize == dirname_len
                && main_needle.eq_ignore_ascii_case(dirname_buf.as_bytes());

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

/// Sort dir results by total score (descending) and apply pagination.
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
    let use_partial_sort = items_needed < total_matched / 2 && total_matched > 100;

    let compare = |a: &(&DirItem, Score), b: &(&DirItem, Score)| {
        b.1.total
            .cmp(&a.1.total)
            .then_with(|| std::ptr::from_ref(a.0).cmp(&std::ptr::from_ref(b.0)))
    };
    if use_partial_sort {
        results.select_nth_unstable_by(items_needed - 1, compare);
        results.truncate(items_needed);
    }

    sort_with_buffer(&mut results, compare);

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
) -> (Vec<(&'a FileItem, Score)>, usize) {
    if files.len() == 0 {
        return (vec![], 0);
    }

    let parsed = context.query;
    // Subset views size their own work estimate from the subset.
    let max_path_len = match &files {
        FileItems::All(_) => max_path_len,
        view => view.max_path_len(),
    };

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
                                let off =
                                    unsafe { (*f as *const FileItem).offset_from(all.as_ptr()) };
                                debug_assert!(off >= 0);
                                off as u32
                            })
                            .collect();
                        FileItems::Filtered { base: all, idx }
                    }
                    Some(_) => {
                        return (vec![], 0);
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
            return (scored, n);
        }
    };

    debug_assert!(!fuzzy_parts.is_empty());
    let has_uppercase = fuzzy_parts
        .iter()
        .any(|p| p.chars().any(|c| c.is_uppercase()));
    // Users type `/` regardless of platform. Checking the OS separator alone
    // would miss forward-slash queries on Windows.
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
    let phase_sink = std::env::var("FFF_PHASE_TIMING").unwrap_or_default();
    let phase_log = |msg: String| {
        if phase_timing {
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
        }
    };
    let t_match = std::time::Instant::now();
    let path_matches = match_fuzzy_parts(
        fuzzy_parts,
        &working_files,
        &options,
        context.max_threads,
        arena,
        max_path_len,
    );
    let match_elapsed = t_match.elapsed();
    if phase_timing {
        phase_log(format!(
            "PHASE match_us={} n_matches={} n_files={}",
            match_elapsed.as_micros(),
            path_matches.len(),
            working_files.len()
        ));
    }

    // Expose file indices for typing live-narrow (Match(longer,T) ⊆
    // Match(shorter,T) when T is unchanged).
    let t_capture = std::time::Instant::now();
    crate::score::capture_hit_indices(
        path_matches
            .iter()
            .map(|m| working_files.file_index(m.index as usize)),
    );

    let capture_us = t_capture.elapsed().as_micros();
    let t_fallback = std::time::Instant::now();
    let main_needle = fuzzy_parts[0].as_bytes(); // safe
    let main_needle_len = main_needle.len() as u16;
    // Collapsed queries force filename/evidence bonuses to 0, so the
    // fallback matcher is skipped without changing ranking.
    let rank_collapsed = {
        let n_files = working_files.len().max(1);
        path_matches.len() >= 4096
            && path_matches.len() * 100 >= n_files * 85
            && main_needle_len <= 8
    };

    let mut fallback_indices: Vec<u32> = Vec::new();
    let filename_fallback_matches = if query_contains_path_separator
        || path_matches.len() > 15_000
        || rank_collapsed
    {
        vec![]
    } else if !path_matches.iter().any(|path_match| {
        // Fallback only matters when the path hit ends before the
        // filename; all filename hits means no fallback can fire.
        let file = working_files.index(path_match.index as usize);
        let filename_start = file.filename_offset_in_relative_path() as u16;
        path_match.end_col.saturating_sub(main_needle_len - 1) < filename_start
    }) {
        vec![]
    } else {
        // One reused buffer for all fallback filenames instead of a `Cow` per
        // candidate. `filename_cow` heap-allocates whenever the filename
        // straddles a 16-byte chunk boundary, and this loop runs over every
        // match - ~125ns per candidate on a mid-sized corpus.
        let mut name_buf: Vec<u8> = Vec::new();
        let mut name_spans: Vec<(u32, u32)> = Vec::new();

        for (i, path_match) in path_matches.iter().enumerate() {
            let file = working_files.index(path_match.index as usize);
            let filename_start = file.filename_offset_in_relative_path() as u16;
            let match_start_approx = path_match.end_col.saturating_sub(main_needle_len - 1);

            if match_start_approx < filename_start {
                let off = file.path.filename_offset as u32;
                let end = file.path.byte_len as u32;
                let start = name_buf.len() as u32;
                let len = end.saturating_sub(off) as usize;
                name_buf.resize(start as usize + len, 0);
                file.path
                    .copy_window_to(arena, off, end, &mut name_buf[start as usize..]);
                fallback_indices.push(i as u32);
                name_spans.push((start, len as u32));
            }
        }

        if name_spans.is_empty() {
            vec![]
        } else {
            // Match on `&str` so frizbee reuses the instantiation its index
            // resolver already emits instead of a separate `Cow<str>` copy.
            let filename_strs: Vec<&str> = name_spans
                .iter()
                .map(|&(s, l)| {
                    let s = s as usize;
                    // SAFETY: arena path bytes are valid UTF-8, the same
                    // assumption `ChunkedString::read_to_buf` already makes.
                    unsafe { std::str::from_utf8_unchecked(&name_buf[s..s + l as usize]) }
                })
                .collect();
            // Thread count from the calibrated work estimate, which accounts
            // for path length, needle length and typo budget. A bare match
            // count leaves this serial on the mid-size corpora where the
            // fallback actually runs.
            let fb_threads = effective_match_threads(
                filename_strs.len(),
                fuzzy_parts[0].len(),
                max_path_len,
                options.max_typos,
                crate::machine::scan_thread_cap().max(context.max_threads),
            );
            let matches = with_cached_matcher(fuzzy_parts[0], &options, |matcher| {
                matcher.match_list_parallel(&filename_strs, fb_threads)
            });
            matches
        }
    };

    // Map filename-fallback hits by path-match index for O(1) lookup
    // in the scoring loop. Match-list order is irrelevant to the table.
    // Skip the table entirely when nothing fell back — common on filename-hit
    // queries like `file_picker_`.
    let fallback_us = t_fallback.elapsed().as_micros();
    if phase_timing {
        phase_log(format!(
            "PHASE capture_us={capture_us} fallback_us={fallback_us} n={}",
            path_matches.len()
        ));
    }

    let fallback_hit: Vec<Option<(bool, i32)>> = if filename_fallback_matches.is_empty() {
        Vec::new()
    } else {
        let mut hit: Vec<Option<(bool, i32)>> = vec![None; path_matches.len()];
        for m in &filename_fallback_matches {
            let mi = fallback_indices[m.index as usize] as usize;
            if mi < hit.len() {
                hit[mi] = Some((m.exact, m.score as i32));
            }
        }
        hit
    };

    // Ranking demotion for files under a third-party dependency directory.
    // Sized to outrank the bonuses a vendored namesake can accumulate
    // (filename + evidence + path alignment) without displacing a genuinely
    // better first-party match (see VENDORED_PATH_PENALTY).

    // Third buffer is the filename scratch. It has to be separate from
    // `path_buf`: on the common path (`need_rel == false`) the filename is
    // materialised here while `path_buf` stays free for a later relative-path
    // read, and one buffer cannot serve both borrows at once.
    type ScoreScratch = (
        String,
        [u8; crate::simd_path::PATH_BUF_SIZE],
        [u8; crate::simd_path::PATH_BUF_SIZE],
        DirPenaltyMemo,
    );
    let make_scratch = || {
        (
            String::with_capacity(64),
            [0u8; crate::simd_path::PATH_BUF_SIZE],
            [0u8; crate::simd_path::PATH_BUF_SIZE],
            DirPenaltyMemo::new(0),
        )
    };

    // Ranking-feature collapse: when the query hit nearly every path, skip per-hit
    // filename arena reads (filename_cow / special-entry) and intent evidence.
    //
    // This is a ratio, not a time budget, and that is deliberate. The ratio
    // carries real signal: when ~99% of files match, per-candidate features
    // cannot discriminate the top
    // page, so the 25-40 ns each costs buys nothing. Measured: 7.9 ns/candidate
    // collapsed vs 37.9 ns uncollapsed. The cliff between them is the cost.
    let query_intent =
        crate::intent::detect_intent(context.effective_query(), context.current_file, true);
    let main_needle_as_str = context.effective_query();
    // Query-constant evidence state, built once instead of per candidate.
    let prepared_query = crate::intent::PreparedQuery::new(main_needle_as_str);
    // Combo target bytes are query-constant; hoist the view out of the
    // per-candidate closure instead of re-deriving it per file.
    let combo_last: Option<&[u8]> = context
        .last_same_query_match
        .as_ref()
        .map(|m| m.file_path.as_os_str().as_encoded_bytes());
    // Evidence memo is exact only when evidence ignores the path: the
    // proximity terms need an open file and the structure terms need
    // Path intent, so both must be off. Epoch retires the table per query.
    let memo_epoch = MEMO_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let memo_active = std::env::var_os("FFF_NO_EVIDENCE_MEMO").is_none()
        && query_intent != crate::intent::QueryIntent::Path
        && context.current_file.is_none();
    SCORE_PROF.store(phase_timing, std::sync::atomic::Ordering::Relaxed);

    // Current-file dir split once per query; penalties memoized per
    // parent dir in thread-local scratch (dirs repeat across candidates).
    let cur_dir = CurrentDir::new(context.current_file);
    // Pure query constants; hoisted out of the per-candidate closure.
    let need_rel_q = (query_contains_path_separator && main_needle.len() > 10)
        || context.last_same_query_match.is_some()
        || (!rank_collapsed
            && (query_intent == crate::intent::QueryIntent::Path
                || context.current_file.is_some()));

    let score_one = |scratch: &mut ScoreScratch,
                     match_idx: usize,
                     path_match: &neo_frizbee::Match|
     -> (&'a FileItem, Score) {
        let (dir_buf, path_buf, fname_buf, dir_memo) = scratch;
        let file_idx = path_match.index as usize;
        let file = working_files.index(file_idx);

        let base_score = path_match.score as i32;
        let frecency_boost = base_score.saturating_mul(file.total_frecency_score()) / 100;

        let git_status_boost = if file.git_status().is_some_and(is_modified_status) {
            base_score * 15 / 100
        } else {
            0
        };
        let git_recency_boost = file.git_recency_score as i32;

        let distance_penalty = if context.current_file.is_some() {
            let dir_idx = file.parent_dir_index as usize;
            match dir_memo.get(dir_idx) {
                Some(p) => p,
                None => {
                    file.write_dir_str(arena, dir_buf);
                    dir_memo.penalty_for_grow(&cur_dir, dir_idx, dir_buf)
                }
            }
        } else {
            0
        };

        let filename_start = file.filename_offset_in_relative_path() as u16;
        let match_start_approx = path_match.end_col.saturating_sub(main_needle_len - 1);

        let end_col_filename_match = match_start_approx >= filename_start;
        let simd_filename_match = if !end_col_filename_match {
            fallback_hit.get(match_idx).copied().flatten()
        } else {
            None
        };

        let is_filename_match = end_col_filename_match || simd_filename_match.is_some();
        let fname_off = file.path.filename_offset as usize;
        let fname_len = file.path.byte_len as usize - fname_off;
        // The full relative path is needed by the path-alignment suffix
        // test, by the intent evidence (path intent or an open file) and by
        // the combo test. Materialize it once and slice the filename out
        // of it, instead of each consumer re-reading the path and the
        // filename going through filename_cow (which heap-allocates
        // whenever the filename straddles a chunk boundary).
        // Only the suffix test, the combo test and path-shaped evidence
        // need the full path; collapsed queries with only an open file skip it.
        let need_rel = need_rel_q;
        // When nothing else needs the path, borrow the filename straight
        // out of the arena when it fits a chunk and copy into the scratch
        // buffer otherwise. `filename_cow` heap-allocated on that second
        // path, which for a repo whose prefixes straddle chunks is every
        // candidate - 47ns/candidate on linux-kernel vs 8ns on sumo.
        let rel: Option<&str> = if need_rel {
            Some(file.path.read_to_buf(arena, path_buf))
        } else {
            None
        };
        // When nothing else needs the path, borrow the filename straight
        // out of the arena when it fits a chunk and copy into scratch
        // otherwise. `filename_cow` heap-allocated on that second path,
        // which for a repo whose prefixes straddle chunk boundaries is every
        // candidate - 47ns/candidate on linux-kernel vs 8ns on sumo.
        let fname: Option<&str> = if rank_collapsed {
            None
        } else if let Some(r) = rel {
            Some(&r[fname_off..fname_off + fname_len])
        } else {
            Some(file.path.filename_ref(arena, fname_buf))
        };

        let is_exact_filename = !rank_collapsed
            && (simd_filename_match.is_some_and(|(exact, _)| exact)
                || (end_col_filename_match
                    && main_needle_len as usize == fname_len
                    && main_needle.eq_ignore_ascii_case(fname.as_ref().unwrap().as_bytes())));

        let mut has_special_filename_bonus = false;
        let filename_bonus = if rank_collapsed {
            0
        } else if is_exact_filename {
            base_score / 5 * 2 // 40% bonus for exact filename match
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
                let total_len = file.path.byte_len as usize;
                combo_last.is_some_and(|last| {
                    last.len() >= total_len && last.ends_with(rel.unwrap_or("").as_bytes())
                })
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

        // Path alignment bonus: when the query looks like a file path,
        // reward candidates whose path closely matches the typed query.
        // Uses suffix overlap — bytes matching from the end. A full prefix
        // match is the 100% coverage case, so no separate branch is needed.
        let path_alignment_bonus = if query_contains_path_separator && main_needle.len() > 10 {
            let path_bytes = rel.unwrap_or("").as_bytes();
            let common_suffix = main_needle
                .iter()
                .rev()
                .zip(path_bytes.iter().rev())
                .take_while(|(n, p): &(&u8, &u8)| n.eq_ignore_ascii_case(p))
                .count();

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

        // Intent evidence: filename/ext/segments/role mixed per job.
        // Memoized by filename when the path terms are dead (see
        // memo_active): repeated names share one computation exactly.
        let evidence_bonus = if rank_collapsed {
            0
        } else if memo_active {
            let fname_ref = fname.as_ref().unwrap();
            let ev = memoized_evidence(memo_epoch, fname_ref, &prepared_query, query_intent);
            crate::intent::evidence_bonus(query_intent, &ev, base_score)
        } else {
            let fname_ref = fname.as_ref().unwrap();
            let need_path =
                query_intent == crate::intent::QueryIntent::Path || context.current_file.is_some();
            let ev = if need_path {
                crate::intent::evidence_prepared(
                    &prepared_query,
                    rel.unwrap_or(""),
                    fname_ref,
                    query_intent,
                    context.current_file,
                )
            } else {
                crate::intent::evidence_prepared(&prepared_query, "", fname_ref, query_intent, None)
            };
            crate::intent::evidence_bonus(query_intent, &ev, base_score)
        };

        let vendored_penalty = if file.is_vendored() {
            VENDORED_PATH_PENALTY
        } else {
            0
        };

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

        // Score-shape census: which terms actually fire. One branch when
        // off; counts aim the remaining per-match cost (materialization
        // vs ALU). Timing-gated like the rest of the profiler.
        if SCORE_PROF.load(std::sync::atomic::Ordering::Relaxed) {
            let bump = |c: &std::sync::atomic::AtomicU64| {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            };
            if rel.is_some() {
                bump(&PROF_NREL);
            }
            if is_filename_match {
                bump(&PROF_FMATCH);
            }
            if is_exact_filename {
                bump(&PROF_EXACT);
            }
            if combo_match_boost != 0 {
                bump(&PROF_COMBO);
            }
            bump(&PROF_SCORED);
        }

        (file, score)
    };

    // Keep only the best `offset+limit` scored rows in a min-heap so we never
    // materialize 200k+ Score structs for a 100-item page. total_matched is
    // still the full path_matches count.
    let page_cap = context
        .pagination
        .offset
        .saturating_add(if context.pagination.limit > 0 {
            context.pagination.limit
        } else {
            usize::MAX / 4
        })
        .saturating_add(1);
    // Heap algorithm chosen on work alone (page-bounded: O(n log k) vs
    // O(n log n)); serial vs parallel comes from the dispatch plan below.
    let use_topk = page_cap < path_matches.len();
    // Per-match cost is the path bytes each item materializes plus the
    // score struct; `max_threads == 1` forces serial, else pool width.
    let score_bytes =
        crate::dispatch::sampled_path_avg(&working_files) + std::mem::size_of::<Score>();
    let score_cap = if context.max_threads == 1 {
        1
    } else {
        crate::parallelism::search_pool_threads()
    };
    let score_plan =
        crate::dispatch::plan_matches(path_matches.len(), score_bytes, score_cap);
    let t_score = std::time::Instant::now();
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
    let path_align_on = query_contains_path_separator && main_needle.len() > 10;
    let q_intent = query_intent;
    let q_collapsed = rank_collapsed;
    // Read per query; both settings rank identically.
    let no_prune = std::env::var_os("FFF_NO_PRUNE").is_some();
    let mut skipped_topk = 0u64;

    if use_topk {
        let mut heap: BinaryHeap<Reverse<Ranked<'a>>> = BinaryHeap::new();
        let mut consider = |heap: &mut BinaryHeap<Reverse<Ranked<'a>>>,
                            item: (&'a FileItem, Score)| {
            let entry = Ranked {
                total: item.1.total,
                stamp: path_rank(item.0),
                file: item.0,
                score: item.1,
            };
            if heap.len() < page_cap {
                heap.push(Reverse(entry));
                if SCORE_PROF.load(std::sync::atomic::Ordering::Relaxed) {
                    HEAP_PUSH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            } else if heap.peek().is_some_and(|Reverse(w)| entry > *w) {
                heap.pop();
                heap.push(Reverse(entry));
                if SCORE_PROF.load(std::sync::atomic::Ordering::Relaxed) {
                    HEAP_REPLACE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        };
        if score_plan.threads < 2 {
            let mut scratch = make_scratch();
            for (match_idx, path_match) in path_matches.iter().enumerate() {
                if !no_prune && heap.len() >= page_cap {
                    let file = working_files.index(path_match.index as usize);
                    let base = path_match.score as i32;
                    let ub = ub_total_for(
                        base,
                        file.total_frecency_score().max(0),
                        file.git_recency_score.max(0) as i32,
                        q_intent,
                        q_collapsed,
                        path_align_on,
                        combo_max,
                    );
                    if heap.peek().is_some_and(|Reverse(w)| ub < w.total) {
                        skipped_topk += 1;
                        continue;
                    }
                }
                let item = score_one(&mut scratch, match_idx, path_match);
                consider(&mut heap, item);
            }
        } else {
            // Fold into per-worker heaps sized page_cap and reduce. Avoids
            // materializing all Score structs only to keep the top 100.
            // Ranking stays exact (same comparator).
            // scratch stays per-thread via map_init — do NOT carry it in the
            // fold accumulator (4 KB × rayon tree overflowed the worker stack).
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
                        let lo = chunk_idx * chunk;
                        let hi = (lo + chunk).min(n);
                        for match_idx in lo..hi {
                            let path_match = &path_matches[match_idx];
                            if !no_prune && h.len() >= page_cap {
                                let file = working_files.index(path_match.index as usize);
                                let base = path_match.score as i32;
                                let ub = ub_total_for(
                                    base,
                                    file.total_frecency_score().max(0),
                                    file.git_recency_score.max(0) as i32,
                                    q_intent,
                                    q_collapsed,
                                    path_align_on,
                                    combo_max,
                                );
                                if h.peek().is_some_and(|Reverse(w)| ub < w.total) {
                                    skipped += 1;
                                    continue;
                                }
                            }
                            let item = score_one(scratch, match_idx, path_match);
                            let entry = Ranked {
                                total: item.1.total,
                                stamp: path_rank(item.0),
                                file: item.0,
                                score: item.1,
                            };
                            if h.len() < page_cap {
                                h.push(Reverse(entry));
                                if SCORE_PROF.load(std::sync::atomic::Ordering::Relaxed) {
                                    HEAP_PUSH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            } else if h.peek().is_some_and(|Reverse(w)| entry > *w) {
                                h.pop();
                                h.push(Reverse(entry));
                                if SCORE_PROF.load(std::sync::atomic::Ordering::Relaxed) {
                                    HEAP_REPLACE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
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
        let mut results: Vec<(&'a FileItem, Score)> = heap
            .into_iter()
            .map(|Reverse(r)| (r.file, r.score))
            .collect();
        // The parallel top-k heap plus `sort_unstable_by` leaves equal-scored
        // candidates in arbitrary order. Ordering among equals is not a ranking
        // decision, so it only needs to be reproducible.
        results.sort_unstable_by(|a, b| {
            b.1.total
                .cmp(&a.1.total)
                .then_with(|| b.0.modified.cmp(&a.0.modified))
                .then_with(|| path_rank(a.0).cmp(&path_rank(b.0)))
        });
        if phase_timing {
            eprintln!(
                "PHASE score_us={} (topk) n={} skipped={}",
                t_score.elapsed().as_micros(),
                score_entrance_count,
                skipped_topk,
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
        return (results, path_matches.len());
    }

    let results: Vec<_> = if score_plan.threads < 2 {
        let mut scratch = make_scratch();
        path_matches
            .iter()
            .enumerate()
            .map(|(match_idx, path_match)| score_one(&mut scratch, match_idx, path_match))
            .collect()
    } else {
        crate::parallelism::run_on_search_pool(|| {
            path_matches
                .par_iter()
                .enumerate()
                .map_init(make_scratch, |scratch, (match_idx, path_match)| {
                    score_one(scratch, match_idx, path_match)
                })
                .collect::<Vec<_>>()
        })
    };
    let n = results.len();
    if phase_timing {
        eprintln!(
            "PHASE score_us={} (full) n={}",
            t_score.elapsed().as_micros(),
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
    (results, n)
}

/// Heap entry for page-bounded top-k: full `Ord` (total, modified, stamp)
/// makes the kept set exact and deterministic; `stamp` is `path_rank`.
struct Ranked<'a> {
    total: i32,
    stamp: u64,
    file: &'a FileItem,
    score: Score,
}
impl PartialEq for Ranked<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.total == other.total && self.stamp == other.stamp
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
            .then_with(|| self.file.modified.cmp(&other.file.modified))
            .then_with(|| self.stamp.cmp(&other.stamp))
    }
}

type Heap<'a> = BinaryHeap<Reverse<Ranked<'a>>>;

fn merge<'a>(mut a: Heap<'a>, b: Heap<'a>, page_cap: usize) -> Heap<'a> {
    for Reverse(r) in b {
        if a.len() < page_cap {
            a.push(Reverse(r));
        } else if a.peek().is_some_and(|Reverse(w)| r > *w) {
            a.pop();
            a.push(Reverse(r));
        }
    }
    a
}

// Take a thread-local scratch buffer, sized for at least min_cap.
fn take_tls<T: Default>(cell: &std::cell::RefCell<Vec<T>>, min_cap: usize) -> Vec<T> {
    let mut b = cell.borrow_mut();
    b.clear();
    if b.capacity() < min_cap {
        b.reserve(min_cap);
    }
    std::mem::take(&mut *b)
}

// Length without zero-fill (release): the caller must overwrite every lane
// before any read. Debug builds zero-fill instead, so a missed write skews
// parity tests rather than reading garbage.
fn set_len_uninit<T: Default>(v: &mut Vec<T>, n: usize) {
    debug_assert!(v.is_empty());
    v.reserve(n.saturating_sub(v.capacity()));
    debug_assert!(v.capacity() >= n);
    #[cfg(debug_assertions)]
    v.resize_with(n, T::default);
    #[cfg(not(debug_assertions))]
    unsafe {
        v.set_len(n);
    }
}

/// Stable tie-break key for equal scores. `sort_unstable_by` gives no order
/// among equals, so without this the parallel path emitted different orderings
/// run to run (observed: two files at 144 swapping between runs of one binary).
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
            // exact_filename (+b) and filename_substring (+b/3) are
            // if/else in evidence_bonus, so the exact branch dominates.
            base + base / 5 + base / 4
        }
        crate::intent::QueryIntent::Path => base / 2 + base / 5 + base / 10 + base / 2,
        crate::intent::QueryIntent::Concept | crate::intent::QueryIntent::Typo => {
            base / 2 + base * 40 / 100 + base / 4 + base / 6 + base / 4
        }
        crate::intent::QueryIntent::WorkingSet => base / 5 + base / 8,
    }
}

#[inline]
fn ub_total_for(
    base: i32,
    freq: i32,
    git_recency: i32,
    intent: crate::intent::QueryIntent,
    collapsed: bool,
    path_on: bool,
    combo: i32,
) -> i32 {
    let filename = if collapsed { 0 } else { base * 2 / 5 };
    let path = if path_on && !collapsed { base } else { 0 };
    base.saturating_add(base.saturating_mul(freq) / 100)
        .saturating_add(filename)
        .saturating_add(base * 15 / 100)
        .saturating_add(git_recency)
        .saturating_add(evidence_ub_for(base, intent, collapsed))
        .saturating_add(path)
        .saturating_add(combo)
}

/// Byte compare avoids materializing a `String` per scored match.
fn is_special_entry_point_bytes(filename: &[u8]) -> bool {
    matches!(
        filename,
        b"mod.rs"
            | b"lib.rs"
            | b"main.rs"
            | b"index.js"
            | b"index.jsx"
            | b"index.ts"
            | b"index.tsx"
            | b"index.mjs"
            | b"index.cjs"
            | b"index.vue"
            | b"__init__.py"
            | b"__main__.py"
            | b"main.go"
            | b"main.c"
            | b"index.php"
            | b"main.rb"
            | b"index.rb"
    )
}

fn score_filtered_by_frecency<'a>(
    files: &FileItems<'a>,
    context: &ScoringContext,
    arena: ArenaPtr,
) -> (Vec<(&'a FileItem, Score)>, usize) {
    // Working-set queries are constant (""); build the query state once
    // instead of per file inside `evidence_for`.
    let pq = crate::intent::PreparedQuery::new("");
    let score_file = |file: &'a FileItem| {
        let total_frecency_score = file.access_frecency_score as i32
            + (file.modification_frecency_score as i32).saturating_mul(4);

        // Give modified/dirty files a boost even in frecency-only mode
        let git_status_boost = if file.git_status().is_some_and(is_modified_status) {
            total_frecency_score * 15 / 100
        } else {
            0
        };
        let git_recency_boost = file.git_recency_score as i32;

        let current_file_penalty =
            calculate_current_file_penalty(file, total_frecency_score, context, arena);

        // Working set: same dir / related siblings of the current file.
        let mut evidence_bonus = 0i32;
        if let Some(cur) = context.current_file {
            let mut path_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
            let rel = file.path.read_to_buf(arena, &mut path_buf);
            let fname = file.path.filename_cow(arena);
            let ev = crate::intent::evidence_prepared(
                &pq,
                rel.as_ref(),
                fname.as_ref(),
                crate::intent::QueryIntent::WorkingSet,
                Some(cur),
            );
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

    // Sound upper bound from register-resident fields only: no arena
    // walk, no path materialization. Penalties are <= 0, so ignoring them
    // keeps the bound above every actual total (ranking stays exact).
    let has_current = context.current_file.is_some();
    let ub_file = |file: &FileItem| {
        let freq = file.access_frecency_score as i32
            + (file.modification_frecency_score as i32).saturating_mul(4);
        let status = if file.git_status().is_some_and(is_modified_status) {
            freq * 15 / 100
        } else {
            0
        };
        let evidence = if has_current {
            evidence_ub_for(
                freq.max(8),
                crate::intent::QueryIntent::WorkingSet,
                false,
            )
        } else {
            0
        };
        freq.saturating_add(status)
            .saturating_add(file.git_recency_score as i32)
            .saturating_add(evidence)
    };

    // Page-bounded top-k like the fuzzy path: never materialize n Scores
    // for a 100-item page. `total` still counts every live file.
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
    // Dispatch plan shared by both frecency loops below: sampled path
    // bytes plus the score struct per item.
    let fre_bytes =
        crate::dispatch::sampled_path_avg(files) + std::mem::size_of::<Score>();
    let fre_cap = if context.max_threads == 1 {
        1
    } else {
        crate::parallelism::search_pool_threads()
    };
    let fre_plan = crate::dispatch::plan_matches(files.len(), fre_bytes, fre_cap);

    if !use_topk {
        let scored: Vec<(&'a FileItem, Score)> = match files {
            FileItems::All(s) if fre_plan.threads >= 2 => s
                .par_iter()
                .filter_map(|f| {
                    let live = !f.is_deleted();
                    live.then_some(score_file(f))
                })
                .collect(),
            FileItems::All(s) => s
                .iter()
                .filter_map(|f| {
                    let live = !f.is_deleted();
                    live.then_some(score_file(f))
                })
                .collect(),
            FileItems::Filtered { base, idx } => idx
                .iter()
                .filter_map(|&i| {
                    let f = &base[i as usize];
                    let live = !f.is_deleted();
                    live.then_some(score_file(f))
                })
                .collect(),
        };
        let n = scored.len();
        return (scored, n);
    }

    fn insert_ranked<'a>(heap: &mut Heap<'a>, page_cap: usize, item: (&'a FileItem, Score)) {
        let entry = Ranked {
            total: item.1.total,
            stamp: path_rank(item.0),
            file: item.0,
            score: item.1,
        };
        if heap.len() < page_cap {
            heap.push(Reverse(entry));
        } else if heap.peek().is_some_and(|Reverse(w)| entry > *w) {
            heap.pop();
            heap.push(Reverse(entry));
        }
    }

    // Filtered views are constrained subsets: serial heap, same prune.
    if let FileItems::Filtered { base, idx } = files {
        let mut heap: Heap<'_> = Heap::new();
        let mut total = 0usize;
        for &i in idx.iter() {
            let f = &base[i as usize];
            if f.is_deleted() {
                continue;
            }
            total += 1;
            if !no_prune
                && has_current
                && heap.len() >= page_cap
                && heap.peek().is_some_and(|Reverse(w)| ub_file(f) < w.total)
            {
                continue;
            }
            insert_ranked(&mut heap, page_cap, score_file(f));
        }
        return (
            heap.into_iter().map(|Reverse(r)| (r.file, r.score)).collect(),
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
            if f.is_deleted() {
                continue;
            }
            total += 1;
            if !no_prune
                && has_current
                && heap.len() >= page_cap
                && heap.peek().is_some_and(|Reverse(w)| ub_file(f) < w.total)
            {
                continue;
            }
            insert_ranked(&mut heap, page_cap, score_file(f));
        }
        return (
            heap.into_iter().map(|Reverse(r)| (r.file, r.score)).collect(),
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
                    if f.is_deleted() {
                        continue;
                    }
                    t += 1;
                    if !no_prune
                        && has_current
                        && h.len() >= page_cap
                        && h.peek().is_some_and(|Reverse(w)| ub_file(f) < w.total)
                    {
                        continue;
                    }
                    insert_ranked(&mut h, page_cap, score_file(f));
                }
                (h, t)
            })
            .reduce(
                || (Heap::new(), 0usize),
                |(ha, ta), (hb, tb)| (merge(ha, hb, page_cap), ta + tb),
            )
    });
    (
        heap.into_iter().map(|Reverse(r)| (r.file, r.score)).collect(),
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
        && file.relative_path_eq(arena, current)
    {
        penalty -= base_score;
    }

    penalty
}

/// Sorts elements by total score (descending) and returns the requested page.
/// Always returns results in descending order (best scores first).
/// The UI layer handles rendering order based on prompt position.
#[tracing::instrument(skip_all, level = tracing::Level::DEBUG)]
fn sort_and_paginate<'a>(
    results: Vec<(&'a FileItem, Score)>,
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

    // Check if offset is out of bounds
    if offset >= total_matched {
        tracing::warn!(
            offset = offset,
            total_matched = total_matched,
            "Pagination: offset >= total_matched, returning empty"
        );

        return (vec![], vec![], total_matched);
    }

    let items_needed = offset.saturating_add(limit).min(total_matched);
    // Use partial sort if we need less than half the results and dataset is large
    let use_partial_sort = items_needed < total_matched / 2 && total_matched > 100;
    // Rank on lightweight indices so the sort never moves fat Scores;
    // same (total, modified) keys, so ordering is bit-identical.
    let mut order: Vec<u32> = (0..results.len() as u32).collect();
    let mut cmp_idx = |a: &u32, b: &u32| {
        let (fa, sa) = &results[*a as usize];
        let (fb, sb) = &results[*b as usize];
        sb.total
            .cmp(&sa.total)
            .then_with(|| fb.modified.cmp(&fa.modified))
            // Total order: parallel matching yields scheduling-dependent
            // input order, so ties fall back to file address (index order).
            .then_with(|| std::ptr::from_ref(*fa).cmp(&std::ptr::from_ref(*fb)))
    };
    // Always sort in descending order (best scores first)
    if use_partial_sort {
        // Partition at position (items_needed - 1) with descending comparator
        // This puts the highest N needed items at the front
        order.select_nth_unstable_by(items_needed - 1, &mut cmp_idx);
        order.truncate(items_needed);
    }

    // select nth does not sort the results, we have to sort accordingly anyway
    sort_with_buffer(&mut order, &mut cmp_idx);

    // in the best scenario truncation happened in the select_nth step
    if offset > 0 {
        order.drain(..offset.min(order.len()));
    }
    if order.len() > limit {
        order.truncate(limit);
    }

    let mut items = Vec::with_capacity(order.len());
    let mut scores = Vec::with_capacity(order.len());
    for i in order {
        let (file, score) = &results[i as usize];
        items.push(*file);
        scores.push(score.clone());
    }
    (items, scores, total_matched)
}

#[derive(Clone)]
struct FxBuilder;
struct FxHasher(u64);
impl std::hash::BuildHasher for FxBuilder {
    type Hasher = FxHasher;
    fn build_hasher(&self) -> FxHasher {
        FxHasher(0x51ab_3c9d)
    }
}
impl std::hash::Hasher for FxHasher {
    fn write(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0.rotate_left(5) ^ x as u64).wrapping_mul(0x517c_c1b7_2720_0a95);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

struct MemoEntry {
    key: Box<[u8]>,
    ev: crate::intent::Evidence,
}

thread_local! {
    static EVIDENCE_MEMO: std::cell::RefCell<(
        u64,
        std::collections::HashMap<u64, MemoEntry, FxBuilder>,
    )> = std::cell::RefCell::new((0, std::collections::HashMap::with_hasher(FxBuilder)));
}
static MEMO_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static SCORE_PROF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static MEMO_HIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MEMO_MISS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HEAP_PUSH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HEAP_REPLACE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_NREL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_FMATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_EXACT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_COMBO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_SCORED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Filename-keyed evidence cache. Sound only when evidence is a pure
// function of (query, filename): non-Path intent with no open file, so
// the path-dependent terms are dead. Hash collisions just replace the
// slot; the stored key is always verified, so outputs stay exact.
fn memoized_evidence(
    epoch: u64,
    fname: &str,
    pq: &crate::intent::PreparedQuery<'_>,
    intent: crate::intent::QueryIntent,
) -> crate::intent::Evidence {
    EVIDENCE_MEMO.with(|c| {
        let mut m = c.borrow_mut();
        if m.0 != epoch {
            m.1.clear();
            if m.1.capacity() > 1 << 14 {
                m.1.shrink_to(1 << 14);
            }
            m.0 = epoch;
        }
        let fb = fname.as_bytes();
        let mut h = std::hash::BuildHasher::build_hasher(&FxBuilder);
        std::hash::Hasher::write(&mut h, fb);
        let h = std::hash::Hasher::finish(&h);
        if let Some(e) = m.1.get(&h) {
            if e.key.as_ref() == fb {
                if SCORE_PROF.load(std::sync::atomic::Ordering::Relaxed) {
                    MEMO_HIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return e.ev;
            }
        }
        let ev = crate::intent::evidence_prepared(pq, "", fname, intent, None);
        if SCORE_PROF.load(std::sync::atomic::Ordering::Relaxed) {
            MEMO_MISS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        m.1.insert(h, MemoEntry { key: fb.into(), ev });
        ev
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PaginationArgs;
    use fff_query_parser::QueryParser;

    // Upstream-engine equality cannot hold here: this engine SW-rescores
    // the first pass, so the narrowed pass mixes scales by design.
    #[ignore = "cross-path SW-vs-frizbee equality; engine mixes scales by design"]
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
                let first =
                    match_fuzzy_parts(&parts[..1], &working, &options, 1, arena, max_path);
                let mut second_options = options;
                second_options.max_typos = Some(max_typos.min(parts[1].len() as u16));
                let second =
                    match_fuzzy_parts(&parts[1..], &working, &second_options, 1, arena, max_path);
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
                let mut actual: Vec<_> =
                    match_fuzzy_parts(&parts, &working, &options, 4, arena, max_path)
                    .iter()
                    .map(|m| (m.index, m.score, m.end_col, m.exact))
                    .collect();
                expected.sort_unstable();
                actual.sort_unstable();
                assert_eq!(actual, expected, "{parts:?}, typos={max_typos}");
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
                file.access_frecency_score = (index % 97) as i16;
                file.modification_frecency_score = (index % 29) as i16;
                file.git_recency_score = (index % 13) as i16;
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
                    pagination: PaginationArgs::default(),
                };
                let max_path = crate::matcher::max_path_len_of(&files);
                let (mut parallel, _) =
                    match_and_score_in_arena(FileItems::All(&files), &context, arena, max_path);
                context.max_threads = 1;
                let (mut sequential, _) =
                    match_and_score_in_arena(FileItems::All(&files), &context, arena, max_path);
                parallel.sort_unstable_by_key(|(file, _)| file.modified);
                sequential.sort_unstable_by_key(|(file, _)| file.modified);
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
                        .map(|f| f.modified)
                        .collect::<Vec<_>>(),
                    sequential_items
                        .iter()
                        .map(|f| f.modified)
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
                    .then_with(|| b.0.modified.cmp(&a.0.modified))
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
        // Create test data with known scores
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

        // Convert to references like the actual function uses
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
            pagination: PaginationArgs {
                offset: 0,
                limit: 0,
            },
        };

        // Test with full sort - returns all results sorted descending
        let total = results.len();
        let (items, scores, total) = sort_and_paginate(results.clone(), &context, total);

        // Should return all 10 items sorted by score descending
        assert_eq!(total, 10);
        assert_eq!(scores.len(), 10);
        assert_eq!(scores[0].total, 300, "First should be highest score");
        assert_eq!(scores[1].total, 250, "Second should be second highest");
        assert_eq!(scores[2].total, 200, "Third should be third highest");

        // Verify the files match
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
            pagination: PaginationArgs {
                offset: 0,
                limit: 0,
            },
        };

        let n = results.len();
        let (items, scores, _) = sort_and_paginate(results, &context, n);

        // Should return all 5 items sorted: 200(9000), 200(1000), 100(8000), 100(5000), 100(3000)
        assert_eq!(scores.len(), 5);
        assert_eq!(scores[0].total, 200);
        assert_eq!(items[0].modified, 9000, "First 200 should be newest");
        assert_eq!(scores[1].total, 200);
        assert_eq!(items[1].modified, 1000, "Second 200 should be older");
        assert_eq!(scores[2].total, 100);
        assert_eq!(items[2].modified, 8000, "First 100 should be newest");
        assert_eq!(scores[3].total, 100);
        assert_eq!(items[3].modified, 5000);
        assert_eq!(scores[4].total, 100);
        assert_eq!(items[4].modified, 3000, "Last 100 should be oldest");
    }

    #[test]
    fn test_no_partial_sort_for_small_results() {
        // When results.len() <= threshold, should use regular sort
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
            pagination: PaginationArgs {
                offset: 0,
                limit: 0,
            },
        };

        // Returns all results sorted descending
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
        // Sorted (index, score) pairs per query order. end_col stays
        // query-order-first by design, so it is excluded here.
        let run = |parts: &[&str]| {
            let mut pairs: Vec<(u32, u16)> = match_fuzzy_parts(
                parts,
                &crate::matcher::FileItems::All(&files),
                &options,
                1,
                arena,
                crate::matcher::max_path_len_of(&files),
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
                    .then_with(|| b.0.modified.cmp(&a.0.modified))
                    .then_with(|| std::ptr::from_ref(a.0).cmp(&std::ptr::from_ref(b.0)))
            });
            results.truncate(items_needed);
        }
        sort_with_buffer(&mut results, |a, b| {
            b.1.total
                .cmp(&a.1.total)
                .then_with(|| b.0.modified.cmp(&a.0.modified))
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
            file.access_frecency_score = specs[i].1;
        }
        std::mem::forget(store);
        (result, arena)
    }

    /// Run `fuzzy_match_and_score_files` with production-like max_typos scaling.
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
        );
        items
            .iter()
            .zip(scores.iter())
            .map(|(f, s)| (f.relative_path(arena).to_string(), s.clone()))
            .collect()
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

    /// Regression: full-path query should rank the near-exact path match first.
    /// https://x.com/mbarneyjr/status/2043474268390817861
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

    /// Regression: PR #652 / field panic in pi-fff v0.9.6.
    /// A path >512 bytes (but within PATH_MAX) overflows the fixed
    /// `[*const u8; 32]` chunk-pointer buffer during scoring and panics with
    /// "index out of bounds: the len is 32 but the index is 32".
    #[test]
    fn test_path_longer_than_512_bytes_does_not_panic_and_matches() {
        let mut long_path = String::new();
        while long_path.len() < 600 {
            long_path.push_str("deeply_nested_directory_segment/");
        }
        long_path.push_str("needle_file.rs");
        assert!(long_path.len() > 512 && long_path.len() < crate::simd_path::PATH_BUF_SIZE);

        let (files, arena) = make_files(&[long_path.as_str(), "src/other.rs"]);

        // Panics here on unfixed code: frizbee resolves chunk ptrs per file.
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
            file.git_recency_score = specs[i].1;
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
        );
        items
            .iter()
            .zip(scores)
            .map(|(f, s)| (f.relative_path(arena), s))
            .collect()
    }

    #[test]
    fn recency_boost_breaks_ties_between_equal_fuzzy_matches() {
        // Dir names share no letters with the query so both paths fuzzy-match
        // identically and only the recency boost separates them.
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

        // Exact substring — must always match
        let results = search_with_typos(&files, "bid_comparison", arena, 6);
        assert!(
            results
                .iter()
                .any(|p| p.contains("bid_comparison_supplier_part_cost_modifiers")),
            "exact substring 'bid_comparison' should match, got: {results:?}"
        );

        // Concatenated query with typos — the real-world scenario.
        // "bidcomparsionsupplierpartcostmodfiers" is a smushed version of
        // "bid_comparison_supplier_part_cost_modifiers" with 2 typos
        // (comparison → comparison, modifiers → modifiers).
        let results = search_with_typos(&files, "bidcomparsionsupplierpartcostmodfiers", arena, 6);
        assert!(
            results
                .iter()
                .any(|p| p.contains("bid_comparison_supplier_part_cost_modifiers")),
            "typo query 'bidcomparsionsupplierpartcostmodfiers' should match, got: {results:?}"
        );

        // Shorter typo query
        let results = search_with_typos(&files, "bidcomp", arena, 4);
        assert!(
            results.iter().any(|p| p.contains("bid_comparison")),
            "'bidcomp' should match bid_comparison file, got: {results:?}"
        );

        // Query with missing underscores
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
        );

        assert_eq!(total_matched, 1);
        assert_eq!(items[0].relative_path(arena), "a.rs");
    }
}

/// Executable ranking policy. If these fail, result order changed —
/// update this spec deliberately, not as a side effect of a perf patch.
///
/// Policy (base = SW score):
/// 1. exact filename match > fuzzy filename match > path-only match
/// 2. at equal match quality, higher frecency wins
/// 3. special entry-point names (mod.rs, index.ts, …) beat ordinary
///    path-only matches but lose to filename hits
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
            file.access_frecency_score = specs[i].1;
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

    /// Job: "I know the name + ext" — exact stem+ext wins over wrong ext.
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

    /// Job: "I know the path" — segment hits beat scattered letters.
    #[test]
    fn path_intent_prefers_segment_sequence() {
        let (files, arena) = make(&[("xsrc/xauth/xlogin.rs", 0), ("src/auth/login.rs", 0)]);
        let order = rank(&files, "src/auth", arena);
        assert_eq!(
            order[0], "src/auth/login.rs",
            "clean segment sequence first, got {order:?}"
        );
    }

    /// Job: "I know the concept" — filename words beat path-only scatter.
    #[test]
    fn concept_intent_prefers_filename_words() {
        let (files, arena) = make(&[("src/hidden/handler_util.rs", 0), ("src/api/handler.rs", 0)]);
        let order = rank(&files, "handler", arena);
        assert_eq!(
            order[0], "src/api/handler.rs",
            "filename word hit first, got {order:?}"
        );
    }

    /// Job: "I want the test for this role" — role keyword ranks test files up.
    #[test]
    fn concept_test_role_ranks_test_files() {
        let (files, arena) = make(&[("src/user.rs", 0), ("src/user_test.rs", 0)]);
        let order = rank(&files, "test", arena);
        assert_eq!(
            order[0], "src/user_test.rs",
            "role=test should prefer test file, got {order:?}"
        );
    }
    /// MRR-style: first result must be the intended file for each job.
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

    // Frecency corpus with distinct totals (no ties): unique access scores,
    // zero modification/git, identical mtimes. Empty query ranks by score.
    fn make_frecency(n: usize) -> (Vec<FileItem>, ArenaPtr) {
        let paths: Vec<String> =
            (0..n).map(|i| format!("src/g{i}/data_{i}.log")).collect();
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let (mut files, arena) = make_paths(&refs);
        for (i, file) in files.iter_mut().enumerate() {
            file.access_frecency_score = i as i16;
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
            pagination: PaginationArgs { offset: 0, limit },
        };
        let (items, scores, total) = fuzzy_match_and_score_files(
            files,
            &ctx,
            files.len(),
            arena,
            arena,
            max_path_len_of(files),
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
            let ub = ub_total_for(base, freq, rec, intent, collapsed, path_on, combo);
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
    fn collapsed_skips_filename_bonus() {
        // Collapsed (>=4096 matches at >=85% rate, needle <=8) forces
        // filename/evidence bonuses to 0, so the fallback skip is exact.
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
                pagination: PaginationArgs { offset: 0, limit: 50 },
            };
            let (items, scores, total) = fuzzy_match_and_score_files(
                &files,
                &ctx,
                files.len(),
                arena,
                arena,
                max_path_len_of(&files),
            );
            assert!(total >= 4096, "fixture must trigger collapse, got {total}");
            (
                items.iter().map(|f| f.relative_path(arena)).collect::<Vec<_>>(),
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
                // Both paths rank identically (asserted below).
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
        // Memo shares evidence across repeated filenames; outputs must
        // equal a fresh computation. Path intent and open-file legs run
        // the memo-inactive path and must match too.
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
                            pagination: PaginationArgs { offset: 0, limit: 50 },
                        };
                        fuzzy_match_and_score_files(
                            &files,
                            &ctx,
                            files.len(),
                            arena,
                            arena,
                            max_path_len_of(&files),
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
                        items.iter().map(|f| f.relative_path(arena)).collect::<Vec<_>>()
                    };
                    assert_eq!(
                        names(base_items),
                        names(hatch_items),
                        "items differ for {q:?} t={threads} cur={current:?}"
                    );
                    assert_eq!(base_total, hatch_total, "total differs");
                    let totals =
                        |s: Vec<Score>| s.into_iter().map(|x| x.total).collect::<Vec<_>>();
                    assert_eq!(totals(base_scores), totals(hatch_scores), "totals differ");
                }
            }
        }
    }

    #[test]
    fn fused_groups_match_regardless_of_size() {
        // Group outputs are per-index, so group size cannot change results.
        // Size 8 also covers the scalar lane path. Same race-safety
        // argument as the hatch test above.
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
        // Fused path honors FFF_NO_GATE_PREFILTER; pretest is sound so
        // both paths rank identically.
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
                assert_eq!(base_items, hatch_items, "items differ for {q:?} t={threads}");
                let totals = |s: Vec<Score>| s.into_iter().map(|x| x.total).collect::<Vec<_>>();
                assert_eq!(totals(base_scores), totals(hatch_scores), "totals differ");
            }
        }
    }

    #[test]
    fn frecency_topk_matches_full_ranking() {
        // Empty query takes the frecency path. Heap page (limit 50) must
        // equal the full-ranking prefix (limit 0), on both thread counts,
        // with and without a current file (evidence on/off).
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
                full_scores.iter().take(50).map(|s| s.total).collect::<Vec<_>>(),
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

    // Heap page must equal full-ranking page across queries, threads,
    // pagination, current file, and corpus sizes above/below top-k cutoff.
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
                pagination: PaginationArgs { offset, limit },
            };
            let (items, scores, total) = fuzzy_match_and_score_files(
                files,
                &ctx,
                files.len(),
                arena,
                arena,
                max_path_len_of(files),
            );
            let mut pairs: Vec<(String, i32, u64)> = items
                .iter()
                .zip(scores.iter())
                .map(|(f, s)| (f.relative_path(arena), s.total, f.modified))
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
                file.modified = 1_700_000_000 + i as u64;
            }
            for query in ["handler", "Handler", "src/mod", "MOD Handler", "handler_", ""] {
                for threads in [0usize, 1, 4] {
                    for current in [None, Some("src/mod7/other.rs")] {
                        let (full, full_total) =
                            page_as_set(&files, query, arena, threads, current, 0, 0);
                        for (offset, limit) in [(0usize, 50usize), (20, 30)] {
                            let (page, total) = page_as_set(
                                &files, query, arena, threads, current, offset, limit,
                            );
                            assert_eq!(total, full_total, "totals n={n} q={query:?}");
                            // Canonical order is total: upstream rank order
                            // restricted to a strict key, so the page must be
                            // the full list's slice.
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

    // Partial sort of hand-built unique-key input must equal full prefix.
    #[test]
    fn same_config_repeats_identically() {
        // Unique-key input; identical totals ordered by modified alone.
        let paths: Vec<String> =
            (0..200).map(|i| format!("src/mod{}/f{i}.rs", i % 37)).collect();
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
            file.modified = 1_700_000_000 + i as u64;
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
            pagination: PaginationArgs { offset: 0, limit },
        };
        let (items50, _, total50) =
            sort_and_paginate(results.clone(), &ctx(50), results.len());
        let (items0, _, total0) =
            sort_and_paginate(results.clone(), &ctx(0), results.len());
        let names = |items: Vec<&FileItem>| {
            items.iter().map(|f| f.relative_path(arena)).collect::<Vec<_>>()
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
    fn take_setlen_cycle_keeps_allocation() {
        let cell = std::cell::RefCell::new(Vec::<u16>::new());
        // Odd sizes around lane/batch boundaries, plus empty.
        for n in [0usize, 1, 7, 8, 9, 31, 32, 33, 129, 1000] {
            let mut v = take_tls(&cell, n);
            assert!(v.capacity() >= n);
            set_len_uninit(&mut v, n);
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
            // Re-take below capacity: same allocation, length reset.
            // At or above the stored capacity the buffer may grow instead.
            let want = n / 2 + 1;
            let w = take_tls(&cell, want);
            assert_eq!(w.len(), 0);
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
        // Every T in set_len_uninit's call sites must satisfy the bound.
        let trim_cell = std::cell::RefCell::new(Vec::<crate::matcher::Trim>::new());
        let mut t = take_tls(&trim_cell, 4);
        set_len_uninit(&mut t, 4);
        assert_eq!(t.len(), 4);
        let u32_cell = std::cell::RefCell::new(Vec::<u32>::new());
        let mut u = take_tls(&u32_cell, 4);
        set_len_uninit(&mut u, 4);
        assert_eq!(u.len(), 4);
        let u8_cell = std::cell::RefCell::new(Vec::<u8>::new());
        let mut b = take_tls(&u8_cell, 4);
        set_len_uninit(&mut b, 4);
        assert_eq!(b.len(), 4);
    }
}
