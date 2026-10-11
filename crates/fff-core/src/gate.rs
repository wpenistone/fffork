//! Tolerant subsequence gate — match-set + window without frizbee.
//!
//! Novel pipeline per haystack (and batched across paths):
//! 1. **Span gate (SIMD-friendly)**: min/max positions of any *needle* byte.
//!    Everything outside the span cannot belong to an LCS. If
//!    `span_len < m - max_typos`, reject without touching the rest.
//! 2. **Skip-DP (Allison-style)**: `best[s]` = furthest needle cursor after
//!    `s` deletions. Accept iff some `s <= T` reaches `m`.
//! 3. **Window**: the needle-byte span (sound superset of the minimal
//!    alignment window). Includes_prefix = span starts at 0.
//!
//! Soundness: never rejects when `LCS(needle, hay) >= m - T`.
//! False accepts are impossible — skip-DP is exact for the typo model
//! (deletions in the needle = unmatched needle chars).

use crate::simd_path::ArenaPtr;
use crate::types::FileItem;

/// Result of a successful gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateHit {
    pub start: u32,
    pub end: u32,
    pub include_prefix: bool,
}

const MAX_SKIPS: usize = 7;

#[inline]
fn fold_byte(b: u8) -> u8 {
    if b.is_ascii_uppercase() { b + 32 } else { b }
}

/// Bitmask of needle bytes (case-folded) — 256 bits as 4×u64.
#[derive(Clone, Copy)]
pub struct NeedleMask {
    w: [u64; 4],
}

impl NeedleMask {
    #[inline]
    pub fn new(needle: &[u8]) -> Self {
        let mut w = [0u64; 4];
        for &b in needle {
            let x = fold_byte(b) as usize;
            w[x >> 6] |= 1u64 << (x & 63);
        }
        Self { w }
    }

    #[inline]
    pub fn contains(&self, b: u8) -> bool {
        let x = fold_byte(b) as usize;
        self.w[x >> 6] & (1u64 << (x & 63)) != 0
    }
}

/// Span of needle-bytes in `hay`. None if fewer than `need` such bytes.
fn needle_span(mask: &NeedleMask, hay: &[u8], need: usize) -> Option<(usize, usize)> {
    if hay.len() < need {
        return None;
    }
    let mut first = None;
    let mut last = 0usize;
    let mut n = 0usize;
    for (i, &b) in hay.iter().enumerate() {
        if mask.contains(b) {
            if first.is_none() {
                first = Some(i);
            }
            last = i;
            n += 1;
            if n >= need {
                // keep scanning only for last…
            }
        }
    }
    if n < need {
        return None;
    }
    Some((first.unwrap(), last))
}

/// Exact skip-DP over the whole hay: true iff
/// `LCS(needle, hay) >= needle.len() - max_typos`.
/// Fallback for needles longer than one 64-bit word.
fn skip_dp_accept_slow(needle: &[u8], hay: &[u8], max_typos: usize) -> bool {
    let m = needle.len();
    if m == 0 {
        return false;
    }
    let need = m.saturating_sub(max_typos);
    if need == 0 {
        return true;
    }
    if hay.len() < need {
        return false;
    }
    const INF: usize = usize::MAX;
    let t = max_typos;
    let mut best = [INF; MAX_SKIPS + 1];
    let mut nxt: [usize; MAX_SKIPS + 1];
    best[0] = 0;
    for &c in needle {
        nxt = [INF; MAX_SKIPS + 1];
        for s in 0..=t {
            let cur = best[s];
            if cur == INF {
                continue;
            }
            // Delete this needle char (spend a typo).
            if s < t && cur < nxt[s + 1] {
                nxt[s + 1] = cur;
            }
            // Match this needle char at/after the cursor.
            if let Some(p) = find_byte_fold(hay, cur, c) {
                let adv = p + 1;
                if adv < nxt[s] {
                    nxt[s] = adv;
                }
            }
        }
        best = nxt;
    }
    best[..=t].iter().any(|&p| p != INF)
}

/// Hyyrö bit-parallel LCS length: one linear pass, no per-needle rescans.
/// `pm[c]` must hold the folded-byte position masks and `len_mask` the
/// low `needle.len()` bits. Exact for `needle.len() <= 64`.
#[inline]
fn lcs_len_bitparallel(pm: &[u64; 256], len_mask: u64, hay: &[u8]) -> usize {
    let mut s = 0u64;
    for &b in hay {
        let x = s | pm[fold_byte(b) as usize];
        let y = (s << 1) | 1;
        s = x & (x ^ x.wrapping_sub(y));
    }
    (s & len_mask).count_ones() as usize
}

#[inline]
fn find_byte_fold(hay: &[u8], from: usize, needle_b: u8) -> Option<usize> {
    let target = fold_byte(needle_b);
    let slice = &hay[from..];
    // Fast path: needle already lowercase (or hay) — memchr.
    if target == needle_b {
        return memchr::memchr(needle_b, slice).map(|i| from + i);
    }
    for (i, &b) in slice.iter().enumerate() {
        if fold_byte(b) == target {
            return Some(from + i);
        }
    }
    None
}

/// Exact subsequence (T=0) — faster than skip-DP.
fn is_subseq_fold(needle: &[u8], hay: &[u8]) -> bool {
    let mut hi = 0usize;
    for &nb in needle {
        match find_byte_fold(hay, hi, nb) {
            Some(p) => hi = p + 1,
            None => return false,
        }
    }
    true
}

/// Bit i set when `chunk[i]` folds to a needle byte. The arena stores paths in
/// 16-byte chunks, so one SSE2 register covers exactly one chunk and SSE2 is
/// baseline on x86_64 (no runtime dispatch). Short tails and non-x86 use the
/// 256-bit span test, which is O(1) like the vector path.
#[inline]
fn needle_byte_mask(chunk: &[u8], distinct: &[u8], span: &NeedleMask) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        // One vector compare per distinct byte; past 8 the scalar span
        // test is cheaper than the compare+or chain. Same bits either way.
        if chunk.len() == 16 && !distinct.is_empty() && distinct.len() <= 8 {
            // SAFETY: SSE2 is baseline on x86_64; `chunk` is exactly 16 bytes.
            return unsafe { needle_byte_mask_sse2(chunk, distinct) };
        }
    }
    let mut m = 0u16;
    for (i, &b) in chunk.iter().enumerate() {
        if span.contains(b) {
            m |= 1 << i;
        }
    }
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn needle_byte_mask_sse2(chunk: &[u8], distinct: &[u8]) -> u16 {
    use core::arch::x86_64::*;
    let v = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
    // Branchless ASCII fold: add 32 where the byte is in 'A'..='Z'.
    let upper = _mm_cmpgt_epi8(v, _mm_set1_epi8(b'A' as i8 - 1));
    let below_z = _mm_cmpgt_epi8(_mm_set1_epi8(b'Z' as i8 + 1), v);
    let is_upper = _mm_and_si128(upper, below_z);
    let folded = _mm_add_epi8(v, _mm_and_si128(is_upper, _mm_set1_epi8(32)));
    let mut acc = _mm_setzero_si128();
    for &c in distinct {
        acc = _mm_or_si128(acc, _mm_cmpeq_epi8(folded, _mm_set1_epi8(c as i8)));
    }
    _mm_movemask_epi8(acc) as u16
}

/// Per-query gate state: span mask + bit-parallel LCS position masks.
/// Built once per search so per-path work is a single linear pass.
pub struct GateNeedle {
    needle: Box<[u8]>,
    /// Typo budget actually used (capped at `MAX_SKIPS`).
    typos: usize,
    /// `needle.len() - typos`.
    need: usize,
    span: NeedleMask,
    /// `pm[c]` = bitmask of needle positions holding folded byte `c`.
    pm: Box<[u64; 256]>,
    /// Distinct folded needle bytes, for the vectorized membership test.
    distinct: Box<[u8]>,
    len_mask: u64,
}

impl GateNeedle {
    pub fn new(needle: &[u8], max_typos: u16) -> Self {
        let typos = (max_typos as usize).min(MAX_SKIPS);
        let m = needle.len();
        let mut pm = Box::new([0u64; 256]);
        let mut seen = [false; 256];
        let mut distinct = Vec::with_capacity(m.min(16));
        for (i, &b) in needle.iter().enumerate().take(64) {
            let f = fold_byte(b);
            pm[f as usize] |= 1u64 << i;
            if !seen[f as usize] {
                seen[f as usize] = true;
                distinct.push(f);
            }
        }
        Self {
            span: NeedleMask::new(needle),
            len_mask: if m >= 64 { u64::MAX } else { (1u64 << m) - 1 },
            needle: needle.to_vec().into_boxed_slice(),
            typos,
            need: m.saturating_sub(typos),
            pm,
            distinct: distinct.into_boxed_slice(),
        }
    }

    /// Exact accept test: `LCS(needle, hay) >= need`.
    #[inline]
    fn decide(&self, hay: &[u8]) -> bool {
        if self.needle.is_empty() {
            return false;
        }
        if self.need == 0 {
            return true;
        }
        if hay.len() < self.need {
            return false;
        }
        if self.typos == 0 {
            return is_subseq_fold(&self.needle, hay);
        }
        if self.needle.len() > 64 {
            return skip_dp_accept_slow(&self.needle, hay, self.typos);
        }
        lcs_len_bitparallel(&self.pm, self.len_mask, hay) >= self.need
    }

    /// Gate one contiguous hay slice.
    pub fn hit(&self, hay: &[u8]) -> Option<GateHit> {
        if self.needle.is_empty() || hay.is_empty() {
            return None;
        }
        if self.need == 0 {
            return Some(GateHit {
                start: 0,
                end: hay.len().min(u16::MAX as usize) as u32,
                include_prefix: true,
            });
        }
        let (first, last) = needle_span(&self.span, hay, self.need)?;
        let hi = (last + 1).min(hay.len());
        if !self.decide(&hay[first..hi]) {
            return None;
        }
        Some(GateHit {
            start: first as u32,
            end: hi as u32,
            include_prefix: first == 0,
        })
    }

    /// Gate a `FileItem` directly: reads its chunk pointers from the arena and
    /// streams the LCS in one pass. No path copy, no per-chunk callback —
    /// the chunk walk is inlined here so the compiler can keep it in registers.
    pub fn hit_file(&self, file: &FileItem, arena: ArenaPtr) -> Option<GateHit> {
        if file.is_deleted() || self.needle.is_empty() {
            return None;
        }
        let byte_len = file.path.byte_len as usize;
        if byte_len == 0 {
            return None;
        }
        // `LCS(needle, hay) <= min(needle_len, hay_len)`, so a path shorter than
        // `need` cannot reach the threshold. Reject before touching a chunk:
        // `byte_len` is already in a register and prefix-heavy corpora (short
        // paths) hit this often.
        if byte_len < self.need {
            return None;
        }
        if self.needle.len() <= 64 {
            // Typos cover the whole needle: every path is a candidate and the
            // window is the full path (same short-circuit as `hit`).
            if self.need == 0 {
                return Some(GateHit {
                    start: 0,
                    end: byte_len as u32,
                    include_prefix: true,
                });
            }
            let distinct = &*self.distinct;
            // Span counting only: per-hit DP state is redundant here. LCS
            // over the full path equals LCS over the span slice (affixes
            // hold no needle bytes), and `decide` below runs that DP once.
            let mut first = usize::MAX;
            let mut last = 0usize;
            let mut n = 0usize;
            let path = &file.path;
            path.for_each_chunk(arena, |off, chunk| {
                let end = (off + chunk.len()).min(byte_len);
                let len = end - off;
                let view = &chunk[..len];
                // One vector op per 16 bytes; popcount replaces the per-hit
                // scalar walk (first/last need only the extreme set bits).
                let mut base = 0usize;
                while base < len {
                    let step = 16.min(len - base);
                    let mask = needle_byte_mask(&view[base..base + step], distinct, &self.span);
                    let c = mask.count_ones() as usize;
                    if c != 0 {
                        if n == 0 {
                            first = off + base + mask.trailing_zeros() as usize;
                        }
                        last = off + base + (16 - mask.leading_zeros() as usize - 1);
                        n += c;
                    }
                    base += step;
                }
            });
            // Fewer hits than the typo budget needs cannot match; the rest
            // are adjudicated on the staged span slice, exactly like the
            // staged `hit` path (same span, same decide).
            if n < self.need {
                return None;
            }
            let hi = (last + 1).min(byte_len);
            thread_local! {
                static SPAN: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            let mut span = SPAN.with(|c| {
                let mut b = c.borrow_mut();
                let len = hi - first;
                if b.len() < len {
                    b.resize(len, 0);
                }
                std::mem::take(&mut *b)
            });
            file.path
                .copy_window_to(arena, first as u32, hi as u32, &mut span[..hi - first]);
            let ok = self.decide(&span[..hi - first]);
            SPAN.with(|c| *c.borrow_mut() = span);
            if !ok {
                return None;
            }
            return Some(GateHit {
                start: first as u32,
                end: (last + 1).min(byte_len) as u32,
                include_prefix: first == 0,
            });
        }
        // Long needles need random access: stage the path, then skip-DP.
        thread_local! {
            static HAY: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let mut hay = HAY.with(|c| {
            let mut b = c.borrow_mut();
            if b.len() < byte_len {
                b.resize(byte_len, 0);
            } else {
                unsafe { b.set_len(byte_len) };
            }
            std::mem::take(&mut *b)
        });
        file.path.read_to_buf(arena, &mut hay[..byte_len]);
        let hit = self.hit(&hay[..byte_len]);
        HAY.with(|c| *c.borrow_mut() = hay);
        hit
    }
}

/// Single-haystack gate. `None` = not a match under the typo model.
pub fn gate_hit(needle: &[u8], hay: &[u8], max_typos: u16) -> Option<GateHit> {
    GateNeedle::new(needle, max_typos).hit(hay)
}

/// Parallel match-set over FileItems. Emits frizbee-compatible
/// `Match { score: start, end_col: end }` (skip_sw encoding).
/// `file_at` resolves an index to its FileItem, `file_index` maps a
/// position in `0..n` back to a candidate index.
pub fn gate_match_range<'a, F, G>(
    needle: &[u8],
    n: usize,
    max_typos: u16,
    threads: usize,
    file_index: F,
    file_at: G,
    arena: ArenaPtr,
) -> Vec<neo_frizbee::Match>
where
    F: Fn(u32) -> u32 + Sync,
    G: Fn(u32) -> &'a FileItem + Sync,
{
    // Needle state (span mask + bit-parallel position masks) is per query, so
    // Built once per query and shared by every path.
    let state = GateNeedle::new(needle, max_typos);
    // Presence pretest: paths missing more needle bytes than the typo
    // budget allows cannot match. Skipped below need 3.
    // `FFF_NO_GATE_PREFILTER=1` disables it for benchmarking.
    let presence = crate::presence::NeedlePresence::new(needle, max_typos);
    let check_presence = presence.should_check()
        && std::env::var_os("FFF_NO_GATE_PREFILTER").is_none();
    let scan = |i: u32| -> Option<neo_frizbee::Match> {
        // Warm the next candidate's path chunks: the gate is bound by arena
        // load latency, not by the LCS arithmetic.
        if i + 1 < n as u32 {
            file_at(i + 1).path.prefetch(arena);
        }
        let file = file_at(i);
        if check_presence {
            let mask = file.presence_mask();
            if !crate::presence::allows_or_unknown(&presence, mask, needle) {
                return None;
            }
        }
        let hit = state.hit_file(file, arena)?;
        Some(neo_frizbee::Match {
            score: hit.start as u16,
            index: file_index(i),
            exact: false,
            end_col: hit.end as u16,
        })
    };
    // The work estimate already decided serial-vs-parallel (see
    // `effective_match_threads`); the gate just honors it.
    if threads <= 1 {
        return (0..n as u32).filter_map(scan).collect();
    }
    use rayon::prelude::*;
    (0..n as u32).into_par_iter().filter_map(scan).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_rejects_too_short() {
        assert!(gate_hit(b"abc", b"ac", 0).is_none());
        assert!(gate_hit(b"abc", b"axc", 1).is_some());
    }

    #[test]
    fn typos_allow_missing_chars() {
        assert!(gate_hit(b"username", b"user_name", 1).is_some());
        assert!(gate_hit(b"username", b"userXname", 1).is_some());
        assert!(gate_hit(b"username", b"xxxx", 1).is_none());
    }

    #[test]
    fn case_insensitive() {
        assert!(gate_hit(b"UserName", b"user_name", 1).is_some());
    }

    #[test]
    fn window_covers_needle_bytes() {
        let h = gate_hit(b"score", b"src/scoring/mod.rs", 2).unwrap();
        assert!(h.end > h.start);
        assert_eq!(h.include_prefix, h.start == 0);
    }

    #[test]
    fn agrees_with_lcs_on_random() {
        let mut state = 0x12345678u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        for _ in 0..2000 {
            let nl = 2 + (next() % 8) as usize;
            let hl = 1 + (next() % 16) as usize;
            let needle: Vec<u8> = (0..nl).map(|_| b'a' + (next() % 5) as u8).collect();
            let hay: Vec<u8> = (0..hl).map(|_| b'a' + (next() % 5) as u8).collect();
            let t = (next() % 3) as u16;
            let need = nl as i32 - t as i32;
            // LCS via DP
            let mut dp = vec![0i32; hl + 1];
            for &nc in &needle {
                let mut prev = 0i32;
                for j in 1..=hl {
                    let tmp = dp[j];
                    dp[j] = if fold_byte(nc) == fold_byte(hay[j - 1]) {
                        prev + 1
                    } else {
                        dp[j].max(dp[j - 1])
                    };
                    prev = tmp;
                }
            }
            let lcs = dp[hl];
            let gate = gate_hit(&needle, &hay, t).is_some();
            let expect = lcs >= need.max(0);
            assert_eq!(gate, expect, "n={needle:?} h={hay:?} t={t} lcs={lcs}");
        }
    }
    #[test]
    fn differential_vs_frizbee_on_random() {
        let mut state = 0xC0FFEEu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        let mut gate_only = 0;
        let mut friz_only = 0;
        let mut both = 0;
        let mut neither = 0;
        let mut friz_wrong = 0;
        let mut gate_wrong = 0;
        for _ in 0..3000 {
            let nl = 3 + (next() % 8) as usize;
            let hl = 2 + (next() % 20) as usize;
            let needle: String = (0..nl)
                .map(|_| (b'a' + (next() % 6) as u8) as char)
                .collect();
            let hay: String = (0..hl)
                .map(|_| (b'a' + (next() % 6) as u8) as char)
                .collect();
            let t = (next() % 3) as u16;
            let need = nl as i32 - t as i32;
            // LCS
            let nbs = needle.as_bytes();
            let hbs = hay.as_bytes();
            let mut dp = vec![0i32; hl + 1];
            for &nc in nbs {
                let mut prev = 0i32;
                for j in 1..=hl {
                    let tmp = dp[j];
                    dp[j] = if fold_byte(nc) == fold_byte(hbs[j - 1]) {
                        prev + 1
                    } else {
                        dp[j].max(dp[j - 1])
                    };
                    prev = tmp;
                }
            }
            let lcs = dp[hl];
            let expect = lcs >= need.max(0);
            let g = gate_hit(nbs, hbs, t).is_some();
            let f = {
                let cfg = neo_frizbee::Config {
                    max_typos: Some(t),
                    sort: neo_frizbee::SortStrategy::Unsorted,
                    ..Default::default()
                };
                let mut m = neo_frizbee::Matcher::new(needle.as_str(), &cfg);
                !m.match_list(&[hay.as_str()]).is_empty()
            };
            if g && f {
                both += 1;
            } else if g {
                gate_only += 1;
            } else if f {
                friz_only += 1;
                if !expect {
                    friz_wrong += 1;
                }
            } else {
                neither += 1;
            }
            if g != expect {
                gate_wrong += 1;
            }
        }
        eprintln!(
            "DIFF both={both} gate_only={gate_only} friz_only={friz_only} neither={neither} gate_wrong={gate_wrong} friz_wrong={friz_wrong}"
        );
        assert_eq!(gate_wrong, 0, "gate must equal LCS");
    }

    #[test]
    fn hit_file_matches_staged_hit_across_chunk_boundaries() {
        // `hit_file` streams chunks from the arena; `hit` sees a flat slice.
        // They must agree for every length that straddles a 16-byte chunk.
        use crate::simd_path::build_chunked_path_store_from_strings;
        let mut state = 0x5EEDu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        for _ in 0..200 {
            let nl = 2 + (next() % 20) as usize;
            let hl = 1 + (next() % 80) as usize;
            let needle: Vec<u8> = (0..nl).map(|_| b'a' + (next() % 6) as u8).collect();
            let hay: Vec<u8> = (0..hl).map(|_| b'a' + (next() % 6) as u8).collect();
            let t = (next() % 3) as u16;

            let rel = String::from_utf8_lossy(&hay).to_string();
            let files = vec![FileItem::new_raw(0, 0, 0, None, false)];
            let (store, strings) = build_chunked_path_store_from_strings(&[rel], &files);
            let mut file = files.into_iter().next().unwrap();
            file.set_path(strings[0].clone());
            file.set_presence(crate::presence::mask_of(&hay));

            let arena = store.as_arena_ptr();
            let g = GateNeedle::new(&needle, t);
            let streamed = g.hit_file(&file, arena);
            let staged = g.hit(&hay);
            assert_eq!(
                streamed.is_some(),
                staged.is_some(),
                "hit disagreement n={needle:?} h={hay:?} t={t}"
            );
            if let (Some(a), Some(b)) = (streamed, staged) {
                assert_eq!(
                    (a.start, a.end, a.include_prefix),
                    (b.start, b.end, b.include_prefix),
                    "window disagreement n={needle:?} h={hay:?} t={t}"
                );
            }
        }
    }

    #[test]
    fn streamed_and_staged_agree_with_many_distinct_bytes() {
        // Needles with >8 distinct bytes take the scalar span path in
        // `hit_file` while `hit` stages flat; windows must still agree.
        use crate::simd_path::build_chunked_path_store_from_strings;
        let needles: &[&[u8]] = &[
            b"abcdefghijklmnop",
            b"qwertyuiopasdfgh",
            b"0123456789abcdef",
        ];
        let hays: &[&[u8]] = &[
            b"src/abcdefghijklmnop/mod.rs",
            b"xxxxxxxxxxxxxxxx",
            b"a",
            b"0123456789abcdef0123456789abcdef",
        ];
        for needle in needles {
            for hay in hays {
                for t in [0u16, 1, 2] {
                    let rel = String::from_utf8_lossy(hay).to_string();
                    let files = vec![FileItem::new_raw(0, 0, 0, None, false)];
                    let (store, strings) = build_chunked_path_store_from_strings(&[rel], &files);
                    let mut file = files.into_iter().next().unwrap();
                    file.set_path(strings[0].clone());
                    file.set_presence(crate::presence::mask_of(hay));
                    let arena = store.as_arena_ptr();
                    let g = GateNeedle::new(needle, t);
                    assert_eq!(
                        g.hit_file(&file, arena),
                        g.hit(hay),
                        "n={needle:?} h={hay:?} t={t}"
                    );
                }
            }
        }
    }

    #[test]
    fn differential_long_and_high_bytes() {
        let mut disagree = 0;
        for seed in 0..500u32 {
            let mut state = seed.wrapping_mul(0x9E3779B1);
            let mut next = move || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 16) as u16
            };
            let nl = 4 + (next() % 10) as usize;
            let hl = 100 + (next() % 400) as usize;
            let needle: Vec<u8> = (0..nl).map(|_| b'a' + (next() % 8) as u8).collect();
            let hay: Vec<u8> = (0..hl)
                .map(|_| {
                    let r = next() % 10;
                    if r < 8 {
                        b'a' + r as u8
                    } else if r == 8 {
                        0xC3
                    } else {
                        0xE9
                    }
                })
                .collect();
            let t = (next() % 4) as u16;
            let g = gate_hit(&needle, &hay, t).is_some();
            let nls = String::from_utf8_lossy(&needle).to_string();
            let hls = String::from_utf8_lossy(&hay).to_string();
            let f = {
                let cfg = neo_frizbee::Config {
                    max_typos: Some(t),
                    sort: neo_frizbee::SortStrategy::Unsorted,
                    ..Default::default()
                };
                let mut m = neo_frizbee::Matcher::new(nls.as_str(), &cfg);
                !m.match_list(&[hls.as_str()]).is_empty()
            };
            if g != f {
                disagree += 1;
            }
        }
        eprintln!("LONG disagree={disagree}");
    }

    #[test]
    fn presence_pretest_never_changes_gate_results() {
        // Masks must not change gate results, at any thread count.
        use crate::simd_path::build_chunked_path_store_from_strings;
        let rels: Vec<String> = (0..300)
            .map(|i| match i % 6 {
                0 => format!("src/mod{i}/handler_file.rs"),
                1 => format!("test/xyz_{i}.py"),
                2 => format!("docs/readme_{i}.md"),
                3 => format!("a/b/c/d/e/f/g_{i}.xml"),
                4 => format!("src/tttt_{i}.rs"),
                _ => format!("q_{i}.txt"),
            })
            .collect();
        let run = |masked: bool, needle: &[u8], typos: u16, threads: usize| {
            let raws: Vec<FileItem> = rels
                .iter()
                .map(|_| FileItem::new_raw(0, 0, 0, None, false))
                .collect();
            let (store, strings) = build_chunked_path_store_from_strings(&rels, &raws);
            let mut files = raws;
            for (i, file) in files.iter_mut().enumerate() {
                file.set_path(strings[i].clone());
                file.set_presence(if masked {
                    crate::presence::mask_of(rels[i].as_bytes())
                } else {
                    0
                });
            }
            let arena = store.as_arena_ptr();
            let out = gate_match_range(
                needle,
                files.len(),
                typos,
                threads,
                |i| i,
                |i| &files[i as usize],
                arena,
            );
            std::mem::forget(store);
            out.into_iter()
                .map(|m| (m.index, m.score, m.end_col, m.exact))
                .collect::<Vec<_>>()
        };
        for needle in [&b"handler"[..], b"test", b"src/mod", b"xyz", b"tt", b"qq"] {
            for typos in [0u16, 2] {
                for threads in [1usize, 4] {
                    assert_eq!(
                        run(true, needle, typos, threads),
                        run(false, needle, typos, threads),
                        "pretest changed results n={needle:?} t={typos}"
                    );
                }
            }
        }
    }
}
