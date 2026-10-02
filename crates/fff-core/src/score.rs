use crate::{
    git::is_modified_status,
    index::constraints::apply_constraints,
    path_utils::calculate_distance_penalty,
    simd_path::{ArenaPtr, MAX_PATH_CHUNKS},
    sort_buffer::{sort_by_key_with_buffer, sort_with_buffer},
    types::{DirItem, FileItem, Score, ScoringContext},
};
use fff_query_parser::{FFFQuery, FuzzyQuery};
use neo_frizbee::{Matcher, Scoring};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::Arc;
use std::{borrow::Cow, path::MAIN_SEPARATOR};

/// Raw frizbee match lists are expensive (full corpus × needle). Cache them
/// across identical searches on an unchanged index generation — typing and
/// the profile-driver warm loop otherwise re-pay matching every op.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RawMatchKey {
    generation: u64,
    partition: u8,
    query: String,
    max_typos: u16,
    file_count: usize,
}

const RAW_MATCH_CACHE_MAX_BYTES: usize = 4 * 1024 * 1024;
const RAW_MATCH_CACHE_MAX_ENTRIES: usize = 8;

#[derive(Default)]
pub(crate) struct RawMatchCache {
    map: ahash::AHashMap<RawMatchKey, Arc<Vec<neo_frizbee::Match>>>,
    bytes: usize,
}

impl RawMatchCache {
    fn get(&self, key: &RawMatchKey) -> Option<Arc<Vec<neo_frizbee::Match>>> {
        self.map.get(key).cloned()
    }

    fn insert(&mut self, key: RawMatchKey, matches: Arc<Vec<neo_frizbee::Match>>) {
        let bytes = matches.len().saturating_mul(std::mem::size_of::<neo_frizbee::Match>());
        if bytes > RAW_MATCH_CACHE_MAX_BYTES {
            return;
        }
        if let Some(old) = self.map.remove(&key) {
            self.bytes = self
                .bytes
                .saturating_sub(old.len() * std::mem::size_of::<neo_frizbee::Match>());
        }
        while self.map.len() >= RAW_MATCH_CACHE_MAX_ENTRIES
            || self.bytes.saturating_add(bytes) > RAW_MATCH_CACHE_MAX_BYTES
        {
            let Some(first_key) = self.map.keys().next().cloned() else {
                break;
            };
            if let Some(old) = self.map.remove(&first_key) {
                self.bytes = self
                    .bytes
                    .saturating_sub(old.len() * std::mem::size_of::<neo_frizbee::Match>());
            }
        }
        self.bytes = self.bytes.saturating_add(bytes);
        self.map.insert(key, matches);
    }

    pub(crate) fn clear(&mut self) {
        self.map.clear();
        self.bytes = 0;
    }
}

pub(crate) use crate::matcher::{capture_hit_indices, with_hit_capture};
use crate::matcher::*;

fn match_fuzzy_parts(
    fuzzy_parts: &[&str],
    working_files: &FileItems<'_>,
    options: &neo_frizbee::Config,
    max_threads: usize,
    arena: ArenaPtr,
    max_path_len: usize,
) -> Vec<neo_frizbee::Match> {
    let valid_parts: Vec<&str> = fuzzy_parts
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
    let scan_cap = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(8);
    // Thread count comes from the work estimate alone (no size cutoffs); the
    // picker's cached max path length keeps this O(1) per query.
    let threads = effective_match_threads(
        working_files.len(),
        valid_parts[0].len(),
        max_path_len,
        options.max_typos,
        scan_cap.max(max_threads),
    );

    // Inter-sequence SW: frizbee prefilter for match set + trim windows,
    // then re-score with validated SW that replicates frizbee's gap model
    // on the SAME trimmed slices with the SAME include_prefix semantics.
    let first_part_matches = {
        let part_timing = std::env::var_os("FFF_PHASE_TIMING").is_some();
        let t_prefilter = std::time::Instant::now();
        // Phase 1: frizbee prefilter only (skip its SW). The skip_sw path
        // encodes (start_pos_adj, end_pos) in (score, end_col).
        let np = crate::presence::NeedlePresence::new(
            valid_parts[0].as_bytes(),
            options.max_typos.unwrap_or(u16::MAX),
        );
        // Index reject before frizbee: only resolve paths that pass
        // length+presence. Only worth the extra scan when the needle can
        // actually reject (longer / typed letters). Common short queries
        // match almost everything — skip so we don't scan twice.
        let mut cand: Vec<u32> = Vec::new();
        let t_cand = std::time::Instant::now();
        let try_cand = valid_parts[0].len() >= 6 && working_files.len() >= 8_192;
        if try_cand {
            // Probe presence selectivity on a small sample first. A full
            // 227k scan is wasted when the needle is too common to reject.
            let n_probe = working_files.len().min(4_096);
            let stride = (working_files.len() / n_probe).max(1);
            let mut pass = 0usize;
            let mut probed = 0usize;
            let needle_b = valid_parts[0].as_bytes();
            for i in (0..working_files.len()).step_by(stride) {
                let f = working_files.index(i);
                let mask = f.presence_mask();
                let ok = (f.path.byte_len as usize) >= np.min_path_len()
                    && (mask == 0 || np.allows(mask, needle_b));
                pass += ok as usize;
                probed += 1;
                if probed >= n_probe {
                    break;
                }
            }
            // Need the probe to eliminate at least half the corpus.
            if probed > 0 && pass * 2 <= probed {
                use rayon::prelude::*;
                let min_len = np.min_path_len();
                cand = crate::parallelism::run_on_bg_pool(|| {
                    (0..working_files.len())
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
        let use_cand = try_cand && cand.len() * 2 < working_files.len() && !cand.is_empty();
        // TISG gate (span + skip-DP) — match-set + window without frizbee.
        // Emits the same skip_sw Match encoding the pipeline already decodes.
        let t_gate = std::time::Instant::now();
        let mut matches = if use_cand {
            let cand = cand.as_slice();
            crate::gate::gate_match_range(
                valid_parts[0].as_bytes(),
                cand.len(),
                options.max_typos.unwrap_or(u16::MAX),
                threads,
                |i| cand[i as usize],
                |i| working_files.index(cand[i as usize] as usize),
                arena,
            )
        } else {
            crate::gate::gate_match_range(
                valid_parts[0].as_bytes(),
                working_files.len(),
                options.max_typos.unwrap_or(u16::MAX),
                threads,
                |i| i,
                |i| working_files.index(i as usize),
                arena,
            )
        };
        let gate_us = t_gate.elapsed().as_micros();
        let prefilter_ms = t_prefilter.elapsed().as_secs_f64() * 1e3;

        // Phase 2: re-score with inter-sequence SW on trimmed windows.
        let has_upper = valid_parts[0].chars().any(|c| c.is_uppercase());
        let p = crate::interseq_sw::SwParams::for_query(has_upper);
        let needle = valid_parts[0].as_bytes();

        // Query intent: `name.ext` — pin exact filename hits among the match
        // set (not a full-corpus walk; score_one already ranks exact names).
        let looks_like_filename = {
            let q = valid_parts[0];
            q.len() >= 4 && q.contains('.') && !q.contains('/')
        };
        let mut exact_name_hits: Vec<u32> = Vec::new();
        let t_exact = std::time::Instant::now();
        if looks_like_filename {
            let q = valid_parts[0];
            let qb = q.as_bytes();
            for m in matches.iter() {
                let f = working_files.index(m.index as usize);
                let flen = f.path.byte_len as usize - f.path.filename_offset as usize;
                if flen == qb.len() {
                    let fname = f.path.filename_cow(arena);
                    if fname.eq_ignore_ascii_case(q) {
                        exact_name_hits.push(m.index);
                    }
                }
            }
        }

        let exact_us = t_exact.elapsed().as_micros();

        // SW scoring (collapse removed: cheap rank disagreed with SW
        // on real corpora; reordering would be a UX regression).
        let t_collect = std::time::Instant::now();
        let wins_ms = 0.0f64;
        let samp_ms = 0.0f64;
        if std::env::var_os("FFF_PHASE_TIMING").is_some() {
            eprintln!("PHASE cand_ms={cand_ms:.1} wins_ms={wins_ms:.1} samp_ms={samp_ms:.1} collapsed=false n={}", matches.len());
        }
        let t_sw = std::time::Instant::now();
        {

        // Chunked scoring: process at most CHUNK matches of metadata at a
        // time so peak working set stays small (faults/RSS vs upstream).
        // 32k × ~16 B × a few arrays ≈ 1 MB peak, vs a full 227k-wide set.
        const CHUNK: usize = 32 * 1024;
        thread_local! {
            static TRIMS: std::cell::RefCell<Vec<crate::matcher::Trim>> =
                const { std::cell::RefCell::new(Vec::new()) };
            static SCORES: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
            static ECOLS: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
            static ORDER: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
            static PERM_SCORES: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
            static PERM_ECOLS: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        fn take_tls<T: Default>(cell: &std::cell::RefCell<Vec<T>>, min_cap: usize) -> Vec<T> {
            let mut b = cell.borrow_mut();
            b.clear();
            if b.capacity() < min_cap {
                b.reserve(min_cap);
            }
            std::mem::take(&mut *b)
        }
        fn reclaim<T>(b: &mut Vec<T>, v: Vec<T>) {
            *b = v;
            // Cap retained capacity so one huge match set does not pin TLS
            // forever, but keep a full chunk warm (avoids re-alloc/re-fault).
            const TLS_RETAIN: usize = CHUNK + 1024;
            if b.capacity() > TLS_RETAIN {
                b.shrink_to(TLS_RETAIN);
            }
        }

        #[cfg(target_arch = "x86_64")]
        // Portability: FFF_FORCE_SCALAR=1 skips AVX2 (SSE2/scalar path only).
        // Used to measure low-spec chips; must stay result-identical.
        let force_scalar = std::env::var_os("FFF_FORCE_SCALAR").is_some();
        #[cfg(target_arch = "x86_64")]
        let avx2_ok = if force_scalar {
            false
        } else {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("sse4.1")
        };
        #[cfg(not(target_arch = "x86_64"))]
        let avx2_ok = false;

        // u8 fast path: 32 lanes, valid when max score < 256.
        let use_u8 = avx2_ok && needle.len() <= 14;
        let batch_sz: usize = if use_u8 { 32 } else { 16 };
        let use_simd = avx2_ok && needle.len() <= 16;

        // Sub-phase accounting, accumulated across rayon workers.
        let t_collect_us = std::sync::atomic::AtomicU64::new(0);
        let t_batch_us = std::sync::atomic::AtomicU64::new(0);
        let t_assemble_us = std::sync::atomic::AtomicU64::new(0);
        let collect_ms = 0.0f64;
        let batch_ms = 0.0f64;
        let assemble_ms = 0.0f64;
        let timing = part_timing;

        // Chunks are disjoint slices of `matches` and every scratch buffer is
        // thread-local, so this fans out safely. It used to be a serial `for`,
        // which pinned ~6 ms of the query to one core no matter how many
        // threads were available.
        {
        use rayon::prelude::*;
        matches.par_chunks_mut(CHUNK).for_each(|matches_chunk| {
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
            if timing { t_collect_us.fetch_add(t_chunk.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed); }

            let t_batch = std::time::Instant::now();
            // 64-byte align the batch output buffers: batch_sz=32 u16 is
            // exactly one cache line, so aligned batches do not false-share
            // at par_chunks boundaries (MESI line ping-pong).
            let mut scores = SCORES.with(|c| take_tls(c, n_matches + 32));
            scores.clear();
            scores.resize(n_matches + 32, 0);
            let mut end_cols = ECOLS.with(|c| take_tls(c, n_matches + 32));
            end_cols.clear();
            end_cols.resize(n_matches + 32, 0);

            // Length-ordered batches within the chunk so per-batch max_len
            // stays close to the lanes' own lengths.
            let mut order = ORDER.with(|c| take_tls(c, n_matches));
            order.clear();
            // Counting sort by trim length (stable, O(n)). Batch grouping only
            // affects SIMD max_len, not per-match scores.
            {
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
                order.resize(n_matches, 0);
                for i in 0..n_matches {
                    let (s, e) = (trims[i].start, trims[i].end);
                    let l = (e - s).min(128) as usize;
                    order[start[l] as usize] = i as u32;
                    start[l] += 1;
                }
            }
            let mut perm_scores = PERM_SCORES.with(|c| take_tls(c, n_matches + 32));
            perm_scores.clear();
            perm_scores.resize(n_matches + 32, 0);
            let mut perm_ecols = PERM_ECOLS.with(|c| take_tls(c, n_matches + 32));
            perm_ecols.clear();
            perm_ecols.resize(n_matches + 32, 0);

            fn align64_u16(buf: &mut [u16], n: usize) -> &mut [u16] {
                let addr = buf.as_ptr() as usize;
                let pad = ((64 - (addr % 64)) % 64) / 2;
                &mut buf[pad..pad + n]
            }
            let ps_aligned = align64_u16(&mut perm_scores, n_matches);
            let pe_aligned = align64_u16(&mut perm_ecols, n_matches);

            crate::parallelism::run_on_bg_pool(|| {
                use rayon::prelude::*;
                order
                    .par_chunks(batch_sz)
                    .zip(ps_aligned.par_chunks_mut(batch_sz))
                    .zip(pe_aligned.par_chunks_mut(batch_sz))
                    .for_each(|((ord, sc_out), ec_out)| {
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
                        let mut stage = STAGE.with(|c| {
                            let mut b = c.borrow_mut();
                            if b.len() < buf_len {
                                b.resize(buf_len, 0);
                            }
                            std::mem::take(&mut *b)
                        });
                        stage.truncate(buf_len);
                        // Zero only padding (was fill(0) of the whole buffer,
                        // then overwrite every window — double-touch).
                        for k in n..batch_sz {
                            stage[k * rows..(k + 1) * rows].fill(0);
                        }

                        let mut prefix_mask: u32 = 0;
                        let mut next_ptrs = [std::ptr::null::<u8>(); MAX_PATH_CHUNKS];
                        for (k, &oi) in ord.iter().enumerate() {
                            let t = trims[oi as usize];
                            let (fi, s, e, inc_prefix) = (t.file_idx, t.start, t.end, t.include_prefix());
                            let file = working_files.index(fi as usize);
                            #[cfg(target_arch = "x86_64")]
                            if k + 1 < n {
                                let nf = working_files.index(
                                    trims[ord[k + 1] as usize].file_idx as usize,
                                );
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
                            stage[k * rows + wlen..(k + 1) * rows].fill(0);
                            if inc_prefix {
                                prefix_mask |= 1u32 << k;
                            }
                        }

                        #[cfg(target_arch = "x86_64")]
                        if use_simd {
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
                                    &stage,
                                    rows,
                                    batch_sz,
                                    &mut soa,
                                );
                                if use_u8 {
                                    crate::interseq_sw::avx2::score_batch_avx2_u8(
                                        needle,
                                        &soa,
                                        max_len,
                                        &p,
                                        prefix_mask,
                                        &mut sc_out[..n],
                                        &mut ec_out[..n],
                                    );
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
                            SOA.with(|c| *c.borrow_mut() = soa);
                            STAGE.with(|c| *c.borrow_mut() = stage);
                            return;
                        }
                        for (k, &oi) in ord.iter().enumerate() {
                            let tw = trims[oi as usize]; let (s, e, inc_prefix) = (tw.start, tw.end, tw.include_prefix());
                            let src = &stage[k * rows..k * rows + (e - s) as usize];
                            let (sc, ec) =
                                crate::interseq_sw::score_one_scalar_full(needle, src, &p, inc_prefix);
                            sc_out[k] = sc;
                            ec_out[k] = ec;
                        }
                        STAGE.with(|c| *c.borrow_mut() = stage);
                    });
            });

            // Inverse order map into the chunk-local score arrays.
            // Read from the 64-byte-aligned batch outputs (true per-line).
            for (i, &oi) in order.iter().enumerate() {
                scores[oi as usize] = ps_aligned[i];
                end_cols[oi as usize] = pe_aligned[i];
            }
            if timing { t_batch_us.fetch_add(t_batch.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed); }

            // Assemble this chunk into the match structs (no full-size hold).
            let t_asm = std::time::Instant::now();
            for (i, m) in matches_chunk.iter_mut().enumerate() {
                let t = trims[i];
                let (fi, start_adj, end_pos, include_exact) = (t.file_idx, t.start as u32, t.end as u32, t.include_exact());
                let exact = include_exact && {
                    let file = working_files.index(fi as usize);
                    let hay = file.path.read_to_buf(arena, &mut path_buf);
                    needle.eq_ignore_ascii_case(hay.as_bytes())
                        && (end_pos as usize) == hay.len()
                };
                let bonus = if exact { 8u16 } else { 0 };
                m.score = scores[i].saturating_add(bonus);
                m.end_col = end_cols[i]
                    .saturating_add(start_adj.min(u16::MAX as u32) as u16);
                m.exact = exact;
            }
            if timing { t_assemble_us.fetch_add(t_asm.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed); }

            TRIMS.with(|c| reclaim(&mut c.borrow_mut(), trims));
            SCORES.with(|c| reclaim(&mut c.borrow_mut(), scores));
            ECOLS.with(|c| reclaim(&mut c.borrow_mut(), end_cols));
            ORDER.with(|c| reclaim(&mut c.borrow_mut(), order));
            PERM_SCORES.with(|c| reclaim(&mut c.borrow_mut(), perm_scores));
            PERM_ECOLS.with(|c| reclaim(&mut c.borrow_mut(), perm_ecols));
        });
        }

        let sw_us = t_sw.elapsed().as_micros();
        let collect_ms = t_collect_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
        let batch_ms = t_batch_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
        let assemble_ms = t_assemble_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3;
        if part_timing {
            eprintln!(
                "PHASE parts gate_us={gate_us} exact_us={exact_us} sw_us={sw_us} threads={threads} n={}",
                matches.len()
            );
        }
        if std::env::var_os("FFF_PHASE_TIMING").is_some() {
            eprintln!(
                "PHASE interseq prefilter_ms={prefilter_ms:.1} collect_ms={collect_ms:.1} batch_ms={batch_ms:.1} assemble_ms={assemble_ms:.1} n={}",
                matches.len()
            );
        }
            if !exact_name_hits.is_empty() {
                let pinned = exact_name_hits.as_slice();
                for m in matches.iter_mut() {
                    if pinned.contains(&m.index) {
                        m.score = m.score.max(200);
                        m.exact = true;
                    }
                }
            }
            matches
        }
    };

    if valid_parts.len() == 1 {
        return first_part_matches;
    }

    let total_parts = valid_parts.len() as u32;
    let mut matches = first_part_matches;
    for part in valid_parts[1..].iter() {
        let mut part_options = *options;
        part_options.max_typos = options.max_typos.map(|t| t.min(part.len() as u16));

        // Match only the files that survived the previous round, addressed
        // through the previous matches without collecting a subset.
        let survivors = &matches;
        let sub_threads = if survivors.len() < 1024 {
            1
        } else {
            let max_path = max_path_len;
            effective_match_threads(
                survivors.len(),
                part.len(),
                max_path,
                part_options.max_typos,
                scan_cap.max(max_threads),
            )
        };
        let sub_np = crate::presence::NeedlePresence::new(
            part.as_bytes(),
            part_options.max_typos.unwrap_or(u16::MAX),
        );
        let sub_matches = with_cached_matcher(part, &part_options, |matcher| {
            matcher.match_range_parallel_resolved(
                survivors.len(),
                &|index, buf: &mut [*const u8; MAX_PATH_CHUNKS]| {
                    let file = working_files.index(survivors[index as usize].index as usize);
                    resolve_file_chunks_prefiltered(
                        file,
                        arena,
                        buf,
                        &sub_np,
                        part.as_bytes(),
                    )
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
                    end_col: prev.end_col, // keep first part's position for filename bonus
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
        None,
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
        None,
        0,
        max_path_len,
    )
}

/// Same as [`fuzzy_match_and_score_files`] but stores/loads raw frizbee
/// matches in `cache` keyed by (generation, partition, query, typos, count).
pub(crate) fn fuzzy_match_and_score_files_cached<'a>(
    files: &'a [FileItem],
    context: &ScoringContext,
    base_count: usize,
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    cache: &parking_lot::Mutex<RawMatchCache>,
    generation: u64,
    max_path_len: usize,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    fuzzy_match_and_score_files_impl(
        files,
        context,
        base_count,
        base_arena,
        overflow_arena,
        Some(cache),
        generation,
        max_path_len,
    )
}

fn fuzzy_match_and_score_files_impl<'a>(
    files: &'a [FileItem],
    context: &ScoringContext,
    base_count: usize,
    base_arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    cache: Option<&parking_lot::Mutex<RawMatchCache>>,
    generation: u64,
    max_path_len: usize,
) -> (Vec<&'a FileItem>, Vec<Score>, usize) {
    // Process overflow files first: newly added files (created after the
    // initial scan) live in the overflow arena and are more likely to be
    // relevant to the current search.
    //
    // putting them first in the list makes sorting more efficient and gives
    // them tiebreaker advantage in case sorting is the same
    let (results, total_matched) = if files.len() > base_count {
        let (mut results, mut total) = match_and_score_in_arena(
            FileItems::All(&files[base_count..]),
            context,
            overflow_arena,
            cache.map(|c| (c, generation, 1u8)),
            max_path_len,
        );

        let (base_results, base_total) = match_and_score_in_arena(
            FileItems::All(&files[..base_count]),
            context,
            base_arena,
            cache.map(|c| (c, generation, 0u8)),
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
            cache.map(|c| (c, generation, 0u8)),
            max_path_len,
        )
    };

    sort_and_paginate(results, context, total_matched)
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

    let paths: Vec<String> = items
        .iter()
        .map(|item| {
            let arena = if item.is_overflow() {
                overflow_arena
            } else {
                base_arena
            };
            let mut path = String::with_capacity(item.relative_path_len());
            item.write_relative_path_from_arena(arena, &mut path);
            path
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

    // Match on `&str` so this shares frizbee's instantiation with fuzzy grep.
    let path_strs: Vec<&str> = paths.iter().map(String::as_str).collect();
    for (idx, part) in parts.iter().copied().enumerate() {
        let mut part_config = config;
        if idx > 0 {
            part_config.max_typos = config.max_typos.map(|t| t.min(part.len() as u16));
        }

        let matched_list = with_cached_matcher(part, &part_config, |m| {
            m.match_list_indices(&path_strs)
        });
        for mut matched in matched_list {
            let item_idx = matched.index as usize;
            let Some(path) = paths.get(item_idx) else {
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
    let mut merged: SmallVec<[(u32, u32); 4]> = SmallVec::with_capacity(ranges.len());

    for (start, end) in ranges {
        if end <= start {
            continue;
        }

        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
            continue;
        }

        merged.push((start, end));
    }

    merged
}

/// Resolve a DirItem's chunked path into frizbee's pointer buffer.
#[inline]
fn resolve_dir_chunks(
    dir: &DirItem,
    arena: ArenaPtr,
    overflow_arena: ArenaPtr,
    buf: &mut [*const u8; MAX_PATH_CHUNKS],
) -> Option<(usize, u16)> {
    let arena = if dir.is_overflow() {
        overflow_arena
    } else {
        arena
    };
    let ptrs = dir.path.resolve_ptrs(arena, buf);
    Some((ptrs.len(), dir.path.byte_len))
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
    let valid_parts: Vec<&str> = fuzzy_parts
        .iter()
        .copied()
        .filter(|p| p.len() >= 2)
        .collect();

    if valid_parts.is_empty() {
        return vec![];
    }

    let scan_cap = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(8);
    let dir_threads = effective_match_threads(
        working_dirs.len(),
        valid_parts[0].len(),
        256,
        options.max_typos,
        scan_cap.max(max_threads),
    );
    let first_part_matches = with_cached_matcher(valid_parts[0], options, |matcher| {
        matcher.match_range_parallel_resolved(
            working_dirs.len(),
            &|index, buf: &mut [*const u8; MAX_PATH_CHUNKS]| {
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
        let survivors = &matches;
        let sub_dir_threads = effective_match_threads(
            survivors.len(),
            part.len(),
            256,
            part_options.max_typos,
            scan_cap.max(max_threads),
        );
        let sub_matches = with_cached_matcher(part, &part_options, |matcher| {
            matcher.match_range_parallel_resolved(
                survivors.len(),
                &|index, buf: &mut [*const u8; MAX_PATH_CHUNKS]| {
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

    let valid_parts: Vec<&str> = fuzzy_parts
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

    if use_partial_sort {
        results.select_nth_unstable_by(items_needed - 1, |a, b| b.1.total.cmp(&a.1.total));
        results.truncate(items_needed);
    }

    sort_with_buffer(&mut results, |a, b| b.1.total.cmp(&a.1.total));

    if results.len() > limit {
        let page_end = std::cmp::min(offset + limit, results.len());
        let page_size = page_end - offset;
        results.drain(0..offset);
        results.truncate(page_size);
    }

    let (items, scores): (Vec<&DirItem>, Vec<Score>) = results.into_iter().unzip();
    (items, scores, total_matched)
}

fn match_and_score_in_arena<'a>(
    files: FileItems<'a>,
    context: &ScoringContext,
    arena: ArenaPtr,
    raw_cache: Option<(&parking_lot::Mutex<RawMatchCache>, u64, u8)>,
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
                                let off = unsafe {
                                    (*f as *const FileItem).offset_from(all.as_ptr())
                                };
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
            let scored = score_filtered_by_frecency(&working_files, context, arena);
            let n = scored.len();
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

    let cache_key = raw_cache.map(|(_, generation, partition)| RawMatchKey {
        generation,
        partition,
        query: context.query.raw_query.to_string(),
        max_typos: context.max_typos,
        file_count: working_files.len(),
    });
    let phase_timing = std::env::var_os("FFF_PHASE_TIMING").is_some();
    let phase_sink = std::env::var("FFF_PHASE_TIMING").unwrap_or_default();
    let phase_log = |msg: String| {
        if phase_timing {
            if phase_sink.is_empty() || phase_sink == "1" {
                eprintln!("{msg}");
            } else {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&phase_sink)
                {
                    let _ = writeln!(f, "{msg}");
                }
            }
        }
    };
    let t_fn = std::time::Instant::now();
    let t_match = std::time::Instant::now();
    let path_matches: Arc<Vec<neo_frizbee::Match>> = raw_cache
        .zip(cache_key.as_ref())
        .and_then(|((cache, _, _), key)| cache.try_lock().and_then(|g| g.get(key)))
        .unwrap_or_else(|| {
            let matches = match_fuzzy_parts(
                fuzzy_parts,
                &working_files,
                &options,
                context.max_threads,
                arena,
                max_path_len,
            );
            let matches_arc = Arc::new(matches);
            if let Some(((cache, _, _), key)) = raw_cache.zip(cache_key) {
                if let Some(mut guard) = cache.try_lock() {
                    guard.insert(key, Arc::clone(&matches_arc));
                }
            }
            matches_arc
        });
    let match_elapsed = t_match.elapsed();
    phase_log(format!(
        "PHASE match_us={} n_matches={} n_files={}",
        match_elapsed.as_micros(),
        path_matches.len(),
        working_files.len()
    ));

    // Expose file indices for typing live-narrow (Match(longer,T) ⊆
    // Match(shorter,T) when T is unchanged).
    crate::score::capture_hit_indices(
        path_matches
            .iter()
            .map(|m| working_files.file_index(m.index as usize)),
    );

    let main_needle = fuzzy_parts[0].as_bytes(); // safe
    let main_needle_len = main_needle.len() as u16;

    let mut fallback_indices: Vec<u32> = Vec::new();
    let filename_fallback_matches = if query_contains_path_separator || path_matches.len() > 15_000
    {
        vec![]
    } else {
        let mut fallback_filenames: Vec<Cow<'_, str>> = Vec::new();

        for (i, path_match) in path_matches.iter().enumerate() {
            let file = working_files.index(path_match.index as usize);
            let filename_start = file.filename_offset_in_relative_path() as u16;
            let match_start_approx = path_match.end_col.saturating_sub(main_needle_len - 1);

            if match_start_approx < filename_start {
                fallback_indices.push(i as u32);
                fallback_filenames.push(file.path.filename_cow(arena));
            }
        }

        if fallback_filenames.is_empty() {
            vec![]
        } else {
            // Match on `&str` so frizbee reuses the instantiation its index
            // resolver already emits instead of a separate `Cow<str>` copy.
            let filename_strs: Vec<&str> = fallback_filenames.iter().map(Cow::as_ref).collect();
            let mut matches = with_cached_matcher(fuzzy_parts[0], &options, |matcher| {
                matcher.match_list_parallel(
                    &filename_strs,
                    if path_matches.len() > 4096 {
                        // Was max_threads.div_ceil(2048), which is 1 for any
                        // realistic pool and left 4T filename fallback serial.
                        context.max_threads.clamp(1, 8)
                    } else {
                        1
                    },
                )
            });

            sort_by_key_with_buffer(&mut matches, |m| fallback_indices[m.index as usize]);
            matches
        }
    };

    // Map filename-fallback hits by path-match index so the scoring loop can
    // run without a sequential cursor. `fallback_indices` is strictly
    // increasing (build order); the fallback match list is sorted by it.
    // Skip the table entirely when nothing fell back — common on filename-hit
    // queries like `file_picker_`.
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

    // Rayon dispatch + per-worker 4KB scratch costs more than scoring itself
    // below a few thousand matches (file_picker_-class queries). Short
    // high-match queries still fan out so the 200k-match path keeps its win.
    const PARALLEL_SCORE_MIN: usize = 4096;

    type ScoreScratch = (String, [u8; crate::simd_path::PATH_BUF_SIZE]);
    let make_scratch = || {
        (
            String::with_capacity(64),
            [0u8; crate::simd_path::PATH_BUF_SIZE],
        )
    };

    // Collapse-aware rank: when the query hit ~every path, skip per-hit
    // filename arena reads (filename_cow / special-entry).
    let rank_collapsed = {
        let n_files = working_files.len().max(1);
        path_matches.len() >= 4096
            && path_matches.len() * 100 >= n_files * 85
            && main_needle_len <= 8
    };
    let query_intent = crate::intent::detect_intent(
        context.effective_query(),
        context.current_file,
        true,
    );
    let main_needle_as_str = context.effective_query();

    let score_one = |scratch: &mut ScoreScratch,
                     match_idx: usize,
                     path_match: &neo_frizbee::Match|
     -> (&'a FileItem, Score) {
        let (dir_buf, path_buf) = scratch;
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

            if context.current_file.is_some() {
                file.write_dir_str(arena, dir_buf);
            }

            let distance_penalty = if context.current_file.is_some() {
                calculate_distance_penalty(context.current_file, dir_buf)
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
            let need_rel = query_contains_path_separator
                || context.current_file.is_some()
                || query_intent == crate::intent::QueryIntent::Path
                || context.last_same_query_match.is_some();
            let rel: Option<&str> = if need_rel {
                Some(file.path.read_to_buf(arena, path_buf))
            } else {
                None
            };
            // When nothing else needs the path, keep the zero-copy borrow.
            let fname_cow = if rel.is_none() && !rank_collapsed {
                Some(file.path.filename_cow(arena))
            } else {
                None
            };
            let fname: Option<&str> = if rank_collapsed {
                None
            } else if let Some(r) = rel {
                Some(&r[fname_off..fname_off + fname_len])
            } else {
                Some(fname_cow.as_ref().unwrap().as_ref())
            };

            let is_exact_filename = !rank_collapsed
                && (simd_filename_match.is_some_and(|(exact, _)| exact)
                    || (end_col_filename_match
                        && main_needle_len as usize == fname_len
                        && main_needle.eq_ignore_ascii_case(
                            fname.as_ref().unwrap().as_bytes(),
                        )));

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
                let last_same_query_match = context.last_same_query_match.as_ref().filter(|m| {
                    // to_string_lossy once per scoring pass via outer cache would
                    // need a lifetime; compare via bytes on the Path instead.
                    let last_bytes = m.file_path.as_os_str().as_encoded_bytes();
                    let total_len = file.path.byte_len as usize;
                    if last_bytes.len() < total_len {
                        return false;
                    }
                    last_bytes.ends_with(rel.unwrap_or("").as_bytes())
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
            // match is just the 100% coverage case, so no separate branch needed.
            let path_alignment_bonus = if query_contains_path_separator {
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
            let evidence_bonus = if rank_collapsed {
                0
            } else {
                let fname_ref = fname.as_ref().unwrap();
                let need_path = query_intent == crate::intent::QueryIntent::Path
                    || context.current_file.is_some();
                let ev = if need_path {
                    crate::intent::evidence_for(
                        rel.unwrap_or(""),
                        fname_ref,
                        main_needle_as_str,
                        query_intent,
                        context.current_file,
                    )
                } else {
                    crate::intent::evidence_filename_only(
                        fname_ref,
                        main_needle_as_str,
                        query_intent,
                    )
                };
                crate::intent::evidence_bonus(query_intent, &ev, base_score)
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
                .saturating_add(path_alignment_bonus);

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
    let use_topk = page_cap < path_matches.len() && path_matches.len() > 256;
    let t_score = std::time::Instant::now();
    let score_entrance_count = path_matches.len();

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

    // Cheap upper bound on `Score.total` when combo/current_file/path-align
    // are all inert. filename ≤ 40% base, git_status ≤ 15% base, frecency is
    // `base * freq / 100`. Skip `score_one` when this bound cannot beat the
    // current kth best. Ranking stays exact.
    let prunable = context.current_file.is_none()
        && context.last_same_query_match.is_none()
        && !query_contains_path_separator;

    if use_topk {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<Reverse<Ranked<'a>>> = BinaryHeap::new();
        let mut stamp = 0u64;
        let mut consider = |heap: &mut BinaryHeap<Reverse<Ranked<'a>>>,
                            item: (&'a FileItem, Score)| {
            let entry = Ranked {
                total: item.1.total,
                stamp,
                file: item.0,
                score: item.1,
            };
            stamp = stamp.wrapping_add(1);
            if heap.len() < page_cap {
                heap.push(Reverse(entry));
            } else if heap.peek().is_some_and(|Reverse(w)| entry.total > w.total) {
                heap.pop();
                heap.push(Reverse(entry));
            }
        };
        if path_matches.len() < PARALLEL_SCORE_MIN {
            let mut scratch = make_scratch();
            for (match_idx, path_match) in path_matches.iter().enumerate() {
                if prunable && heap.len() >= page_cap {
                    let file = working_files.index(path_match.index as usize);
                    let base = path_match.score as i32;
                    let freq = file.total_frecency_score().max(0);
                    let ub = base
                        .saturating_add(base.saturating_mul(freq) / 100)
                        .saturating_add(base * 2 / 5)
                        .saturating_add(base * 15 / 100)
                        .saturating_add(file.git_recency_score.max(0) as i32);
                    if heap.peek().is_some_and(|Reverse(w)| ub < w.total) {
                        continue;
                    }
                }
                let item = score_one(&mut scratch, match_idx, path_match);
                consider(&mut heap, item);
            }
        } else {
            // Fold into per-worker heaps sized page_cap and reduce. Avoids
            // materializing 226k Score structs (~10 MB of faults/RSS) just to
            // keep the top 100. Ranking stays exact (same comparator).
            // scratch stays per-thread via map_init — do NOT carry it in the
            // fold accumulator (4 KB × rayon tree overflowed the worker stack).
            type Heap<'a> = BinaryHeap<Reverse<Ranked<'a>>>;
            fn merge<'a>(mut a: Heap<'a>, b: Heap<'a>, page_cap: usize) -> Heap<'a> {
                for Reverse(r) in b {
                    if a.len() < page_cap {
                        a.push(Reverse(r));
                    } else if a.peek().is_some_and(|Reverse(w)| r.total > w.total) {
                        a.pop();
                        a.push(Reverse(r));
                    }
                }
                a
            }
            heap = crate::parallelism::run_on_bg_pool(|| {
                path_matches
                    .par_iter()
                    .enumerate()
                    .map_init(make_scratch, |scratch, (match_idx, path_match)| {
                        score_one(scratch, match_idx, path_match)
                    })
                    .fold(
                        || Heap::new(),
                        |mut h, item| {
                            let entry = Ranked {
                                total: item.1.total,
                                stamp: 0,
                                file: item.0,
                                score: item.1,
                            };
                            if h.len() < page_cap {
                                h.push(Reverse(entry));
                            } else if h
                                .peek()
                                .is_some_and(|Reverse(w)| entry.total > w.total)
                            {
                                h.pop();
                                h.push(Reverse(entry));
                            }
                            h
                        },
                    )
                    .reduce(|| Heap::new(), |a, b| merge(a, b, page_cap))
            });
        }
        let mut results: Vec<(&'a FileItem, Score)> = heap
            .into_iter()
            .map(|Reverse(r)| (r.file, r.score))
            .collect();
        results.sort_unstable_by(|a, b| {
            b.1.total
                .cmp(&a.1.total)
                .then_with(|| b.0.modified.cmp(&a.0.modified))
        });
        if phase_timing {
            eprintln!(
                "PHASE score_us={} (topk) n={}",
                t_score.elapsed().as_micros(),
                score_entrance_count
            );
        }
        return (results, path_matches.len());
    }

    let results: Vec<_> = if path_matches.len() < PARALLEL_SCORE_MIN {
        let mut scratch = make_scratch();
        path_matches
            .iter()
            .enumerate()
            .map(|(match_idx, path_match)| score_one(&mut scratch, match_idx, path_match))
            .collect()
    } else {
        crate::parallelism::run_on_bg_pool(|| {
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
    }
    (results, n)
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
) -> Vec<(&'a FileItem, Score)> {
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
            let ev = crate::intent::evidence_for(
                rel.as_ref(),
                fname.as_ref(),
                "",
                crate::intent::QueryIntent::WorkingSet,
                Some(cur),
            );
            let base = total_frecency_score.max(8);
            evidence_bonus = crate::intent::evidence_bonus(
                crate::intent::QueryIntent::WorkingSet,
                &ev,
                base,
            );
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

    match files {
        // Small indexes cannot amortize Rayon scheduling for this cheap pass
        // (upstream parity: stay serial under 32k or an explicit 1T search).
        FileItems::All(s) if s.len() >= 32_768 && context.max_threads > 1 => s
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
    }
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
    // Always sort in descending order (best scores first)
    if use_partial_sort {
        // Partition at position (items_needed - 1) with descending comparator
        // This puts the highest N needed items at the front
        results.select_nth_unstable_by(items_needed - 1, |a, b| {
            b.1.total
                .cmp(&a.1.total)
                .then_with(|| b.0.modified.cmp(&a.0.modified))
        });
        results.truncate(items_needed);
    }

    // select nth does not sort the results, we have to sort accordingly anyway
    sort_with_buffer(&mut results, |a, b| {
        b.1.total
            .cmp(&a.1.total)
            .then_with(|| b.0.modified.cmp(&a.0.modified))
    });

    // in the best scenario truncation happened in the select_nth step
    if results.len() > limit {
        let page_end = std::cmp::min(offset + limit, results.len());
        let page_size = page_end - offset;

        results.drain(0..offset);
        results.truncate(page_size);
    }

    let (items, scores): (Vec<&FileItem>, Vec<Score>) = results.into_iter().unzip();
    (items, scores, total_matched)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PaginationArgs;
    use fff_query_parser::QueryParser;

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
        let (items, scores, _) =
            fuzzy_match_and_score_files(files, &ctx, files.len(), arena, arena, max_path_len_of(&files));
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
        let (items, scores, _) =
            fuzzy_match_and_score_files(files, &ctx, files.len(), arena, arena, max_path_len_of(&files));
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
        let (items, _, _) = fuzzy_match_and_score_files(files, &ctx, files.len(), arena, arena, max_path_len_of(&files));
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

        let (items, _scores, total_matched) =
            fuzzy_match_and_score_files(&files, &ctx, files.len(), arena, arena, max_path_len_of(&files));

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

        let (items, _scores, total_matched) =
            fuzzy_match_and_score_files(&files, &ctx, files.len(), arena, arena, max_path_len_of(&files));

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
        let (items, _scores, _) =
            fuzzy_match_and_score_files(files, &ctx, files.len(), arena, arena, max_path_len_of(&files));
        items.iter().map(|f| f.relative_path(arena)).collect()
    }

    #[test]
    fn exact_filename_beats_fuzzy_filename_beats_path_only() {
        let (files, arena) = make(&[
            ("src/username/handler.rs", 0), // path-only
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
        let (files, arena) = make(&[
            ("src/a/user_service.rs", 10),
            ("src/b/user_service.rs", 90),
        ]);
        let order = rank(&files, "user_service", arena);
        assert_eq!(order[0], "src/b/user_service.rs", "higher frecency first");
    }

    #[test]
    fn special_entry_point_beats_ordinary_path_only() {
        let (files, arena) = make(&[
            ("src/deep/mod.rs", 0),
            ("src/deep/helper_mod.rs", 0),
        ]);
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
        assert_eq!(order[0], "src/user_service.rs", "matching ext first, got {order:?}");
    }

    /// Job: "I know the path" — segment hits beat scattered letters.
    #[test]
    fn path_intent_prefers_segment_sequence() {
        let (files, arena) = make(&[
            ("xsrc/xauth/xlogin.rs", 0),
            ("src/auth/login.rs", 0),
        ]);
        let order = rank(&files, "src/auth", arena);
        assert_eq!(
            order[0], "src/auth/login.rs",
            "clean segment sequence first, got {order:?}"
        );
    }

    /// Job: "I know the concept" — filename words beat path-only scatter.
    #[test]
    fn concept_intent_prefers_filename_words() {
        let (files, arena) = make(&[
            ("src/hidden/handler_util.rs", 0),
            ("src/api/handler.rs", 0),
        ]);
        let order = rank(&files, "handler", arena);
        assert_eq!(order[0], "src/api/handler.rs", "filename word hit first, got {order:?}");
    }

    /// Job: "I want the test for this role" — role keyword ranks test files up.
    #[test]
    fn concept_test_role_ranks_test_files() {
        let (files, arena) = make(&[
            ("src/user.rs", 0),
            ("src/user_test.rs", 0),
        ]);
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
