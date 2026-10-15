//! Tolerant subsequence gate: span reject, skip-DP accept, span window.
//!
//! Span bounds needle bytes; skip-DP accepts iff `LCS >= m - T`.
//! Window is the needle-byte span. Sound: never rejects a true match;
//! skip-DP is exact, so false accepts are impossible.

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

// Span of needle-bytes in `hay`. None if fewer than `need` such bytes.
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

// Exact skip-DP over the whole hay: true iff
// `LCS(needle, hay) >= needle.len() - max_typos`.
// Fallback for needles longer than one 64-bit word.
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

// Hyyrö bit-parallel LCS length: one linear pass, no per-needle rescans.
// `pm[c]` must hold the folded-byte position masks and `len_mask` the
// low `needle.len()` bits. Exact for `needle.len() <= 64`.
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

// Exact subsequence (T=0) — faster than skip-DP.
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

// Bit i set when `chunk[i]` folds to a needle byte. Nibble lookup
// tables cover all 256 byte values in constant ops per 16 bytes,
// independent of distinct-needle count (SSSE3; scalar span test on
// tails, non-x86, or no-SSSE3). Same bits as the scalar path.
#[inline]
fn needle_byte_mask(chunk: &[u8], span: &NeedleMask, tables: &SpanTables) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        if chunk.len() == 16 && tables.simd {
            // SAFETY: SSSE3 verified in `GateNeedle::new`; 16 exact bytes.
            return unsafe { needle_byte_mask_shuf(chunk, &tables.lo, &tables.hi) };
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

// 16 parallel set-membership tests: `lo`/`hi` hold the span's high-nibble
// rows, `B0T`/`B1T` select the tested half without a blend (SSE4.1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn needle_byte_mask_shuf(chunk: &[u8], t0: &[u8; 16], t1: &[u8; 16]) -> u16 {
    use core::arch::x86_64::*;
    let v = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
    let lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
    let hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
    let a0 = _mm_shuffle_epi8(_mm_loadu_si128(t0.as_ptr() as *const __m128i), lo);
    let a1 = _mm_shuffle_epi8(_mm_loadu_si128(t1.as_ptr() as *const __m128i), lo);
    let b0 = _mm_shuffle_epi8(
        _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -128, 64, 32, 16, 8, 4, 2, 1),
        hi,
    );
    let b1 = _mm_shuffle_epi8(
        _mm_set_epi8(-128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        hi,
    );
    let r = _mm_or_si128(_mm_and_si128(a0, b0), _mm_and_si128(a1, b1));
    let nz = _mm_cmpeq_epi8(r, _mm_setzero_si128());
    ((!_mm_movemask_epi8(nz)) as u16) & 0xFFFF
}

// Per-query nibble tables for `needle_byte_mask_shuf`: `lo[l]`/`hi[l]`
// hold the span's high-nibble rows for low nibble `l`.
struct SpanTables {
    lo: [u8; 16],
    hi: [u8; 16],
    simd: bool,
}

impl SpanTables {
    fn new(span: &NeedleMask) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for l in 0..16usize {
            let mut row = 0u16;
            for h in 0..16u8 {
                if span.contains(h << 4 | l as u8) {
                    row |= 1 << h;
                }
            }
            lo[l] = row as u8;
            hi[l] = (row >> 8) as u8;
        }
        #[cfg(target_arch = "x86_64")]
        let simd = std::is_x86_feature_detected!("ssse3") && !force_scalar();
        #[cfg(not(target_arch = "x86_64"))]
        let simd = false;
        Self { lo, hi, simd }
    }
}

// `FFF_FORCE_SCALAR=1` disables all runtime SIMD dispatch so scalar
// fallbacks stay covered (same hatch as the SW kernels).
fn force_scalar() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_FORCE_SCALAR").is_some())
}

/// Per-query gate state: span mask + bit-parallel LCS position masks.
/// Built once per search so per-path work is a single linear pass.
pub struct GateNeedle {
    needle: Box<[u8]>,
    // Typo budget actually used (capped at `MAX_SKIPS`).
    typos: usize,
    // `needle.len() - typos`.
    need: usize,
    span: NeedleMask,
    // `pm[c]` = bitmask of needle positions holding folded byte `c`.
    pm: Box<[u64; 256]>,
    // Nibble membership tables for the vectorized span test.
    tables: SpanTables,
    len_mask: u64,
    // Right-tail cutoff (FFF_NO_SPAN_CUT escapes): stop the walk at the
    // first prefix whose LCS already reaches `need`. Sound by prefix
    // monotonicity (accept/reject never changes), but SW windows tighten
    // so scores may shift: adjudicate on match sets, not checksums.
    // Checked per 16B step, never inside the bit loop.
    cut: bool,
}

// Mutable state of one gate walk; splittable at any byte so a cached
// prefix walk can seed a file's suffix walk with identical results.
struct WalkState {
    s: u64,
    first: usize,
    last: usize,
    n: usize,
    end_cut: usize,
    allow_cut: bool,
}

impl WalkState {
    fn fresh(allow_cut: bool) -> Self {
        Self {
            s: 0,
            first: usize::MAX,
            last: 0,
            n: 0,
            end_cut: 0,
            allow_cut,
        }
    }
}

// Memoized prefix walk for sibling files: same dir means same prefix
// bytes, so the prefix Myers walk runs once per dir, not per file.
// Owned per sequential file loop (never shared across threads).
pub struct PrefixWalkCache {
    enabled: bool,
    dir: u32,
    split: usize,
    s: u64,
    first: Option<usize>,
    last: usize,
    n: usize,
    end_cut: usize,
    pub hits: u64,
    pub fills: u64,
    pub bytes_skipped: u64,
}

impl PrefixWalkCache {
    pub fn new() -> Self {
        Self {
            enabled: true,
            dir: u32::MAX,
            split: 0,
            s: 0,
            first: None,
            last: 0,
            n: 0,
            end_cut: 0,
            hits: 0,
            fills: 0,
            bytes_skipped: 0,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl GateNeedle {
    pub fn new(needle: &[u8], max_typos: u16) -> Self {
        let typos = (max_typos as usize).min(MAX_SKIPS);
        let m = needle.len();
        let mut pm = Box::new([0u64; 256]);
        for (i, &b) in needle.iter().enumerate().take(64) {
            pm[fold_byte(b) as usize] |= 1u64 << i;
        }
        let span = NeedleMask::new(needle);
        let tables = SpanTables::new(&span);
        Self {
            span,
            tables,
            len_mask: if m >= 64 { u64::MAX } else { (1u64 << m) - 1 },
            needle: needle.to_vec().into_boxed_slice(),
            typos,
            need: m.saturating_sub(typos),
            pm,
            cut: std::env::var_os("FFF_NO_SPAN_CUT").is_none(),
        }
    }

    #[cfg(test)]
    fn set_cut(&mut self, on: bool) {
        self.cut = on;
    }

    // One 16B step at absolute position `abs`: mask, extremes, Myers.
    // Identical for chunk and flat sources; only the byte fetch differs.
    #[inline]
    fn walk_step(&self, abs: usize, step: &[u8], st: &mut WalkState, want_cut: bool) {
        let tables = &self.tables;
        let pm = &*self.pm;
        let mask = needle_byte_mask(step, &self.span, tables);
        let c = mask.count_ones() as usize;
        if c == 0 {
            return;
        }
        if st.n == 0 {
            st.first = abs + mask.trailing_zeros() as usize;
        }
        st.last = abs + (16 - mask.leading_zeros() as usize - 1);
        // Myers bit-parallel LCS over hit bytes only; bytes outside the
        // needle alphabet are identity updates.
        let mut bits = mask;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let x = st.s | pm[fold_byte(step[bit]) as usize];
            let y = (st.s << 1) | 1;
            st.s = x & (x ^ x.wrapping_sub(y));
        }
        st.n += c;
        // Prefix already accepts: everything past this step is right-tail
        // slack the SW window does not need.
        if want_cut && st.end_cut == 0 && st.n >= self.need.max(1) {
            if (st.s & self.len_mask).count_ones() >= self.need as u32 {
                st.end_cut = abs + step.len();
            }
        }
    }

    // Walk path bytes in [lo, hi) via arena chunks, folding hits into
    // `st`. Interior 16B steps stay vectorized; only range edges go
    // scalar. With `cut`, the walk stops recording past the first
    // accepting prefix (later steps still iterate chunks but skip work).
    fn walk_range(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
    ) {
        let byte_len = file.path.byte_len as usize;
        let want_cut = self.cut && st.allow_cut;
        file.path.for_each_chunk(arena, |off, chunk| {
            if off >= hi {
                return;
            }
            let mut end = (off + chunk.len()).min(byte_len).min(hi);
            if st.end_cut != 0 {
                if off >= st.end_cut {
                    return;
                }
                end = end.min(st.end_cut);
            }
            let start = off.max(lo);
            if start >= end {
                return;
            }
            let view = &chunk[start - off..end - off];
            let len = end - start;
            let mut base = 0usize;
            while base < len {
                let step = 16.min(len - base);
                self.walk_step(start + base, &view[base..base + step], st, want_cut);
                base += step;
            }
        });
    }

    // Walk flat bytes in [lo, hi) where `flat[i]` holds path byte
    // `split + i`. Same step sequence as the chunk walk over the same
    // range, so state folds identically with no chunk chasing.
    fn walk_flat(
        &self,
        flat: &[u8],
        split: usize,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
    ) {
        let want_cut = self.cut && st.allow_cut;
        let mut pos = lo.max(split);
        let end = hi;
        while pos < end {
            if st.end_cut != 0 && pos >= st.end_cut {
                break;
            }
            let step = 16.min(end - pos);
            self.walk_step(pos, &flat[pos - split..pos - split + step], st, want_cut);
            pos += step;
        }
    }

    // Shared accept/window tail for the walk entry points.
    fn finish(&self, st: &WalkState, byte_len: usize) -> Option<GateHit> {
        if (st.s & self.len_mask).count_ones() < self.need as u32 {
            return None;
        }
        let mut end = (st.last + 1).min(byte_len);
        if st.end_cut != 0 {
            end = end.min(st.end_cut);
        }
        Some(GateHit {
            start: st.first as u32,
            end: end as u32,
            include_prefix: st.first == 0,
        })
    }

    // Exact accept test: `LCS(needle, hay) >= need`.
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
            // Single adjudicating walk: per-hit Myers state is exact (skipped
            // non-needle bytes are identity updates), so no second DP and no
            // staging. Extremes come from the mask bits, not a per-hit loop.
            let mut st = WalkState::fresh(true);
            self.walk_range(file, arena, 0, byte_len, &mut st);
            return self.finish(&st, byte_len);
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

    /// Gate with a memoized prefix walk: consecutive same-dir files share
    /// prefix bytes, so the prefix Myers walk runs once per dir. Accept and
    /// windows match `hit_file` (same ops, memoized; with FFF_SPAN_CUT the
    /// cut point may differ by step alignment, still sound). `cache` is
    /// owned per sequential file loop, never shared across threads.
    pub fn hit_file_cached(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        cache: &mut PrefixWalkCache,
        flat: Option<&[u8]>,
    ) -> Option<GateHit> {
        if file.is_deleted() || self.needle.is_empty() {
            return None;
        }
        let byte_len = file.path.byte_len as usize;
        if byte_len == 0 || byte_len < self.need {
            return None;
        }
        if self.needle.len() > 64 {
            return self.hit_file(file, arena);
        }
        if self.need == 0 {
            return Some(GateHit {
                start: 0,
                end: byte_len as u32,
                include_prefix: true,
            });
        }
        let split = (file.path.filename_offset as usize).min(byte_len);
        let dir = file.parent_dir_index;
        let mut st = WalkState::fresh(true);
        if cache.enabled && cache.dir == dir && cache.split == split {
            st.s = cache.s;
            st.first = cache.first.unwrap_or(usize::MAX);
            st.last = cache.last;
            st.n = cache.n;
            st.end_cut = cache.end_cut;
            cache.hits += 1;
            cache.bytes_skipped += split as u64;
        } else {
            self.walk_range(file, arena, 0, split, &mut st);
            if cache.enabled {
                cache.dir = dir;
                cache.split = split;
                cache.s = st.s;
                cache.first = (st.n > 0).then_some(st.first);
                cache.last = st.last;
                cache.n = st.n;
                cache.end_cut = st.end_cut;
                cache.fills += 1;
            }
        }
        // Prefix already accepted on its own: the window sits inside
        // it and the suffix walk is pure overhead.
        if st.end_cut == 0 || st.end_cut > split {
            match flat {
                Some(bytes) => self.walk_flat(bytes, split, split, byte_len, &mut st),
                None => self.walk_range(file, arena, split, byte_len, &mut st),
            }
        } else {
            cache.bytes_skipped += (byte_len - split) as u64;
        }
        self.finish(&st, byte_len)
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
    // `FFF_NO_GATE_PREFILTER=1` disables the pretest.
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
    // Work estimate selects serial vs parallel; the gate honors it.
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
            // Cut tightens streamed windows by design; what must hold is
            // accept-identity plus containment of the cut window.
            assert_eq!(
                streamed.is_some(),
                staged.is_some(),
                "hit disagreement n={needle:?} h={hay:?} t={t}"
            );
            if let (Some(a), Some(b)) = (staged, streamed) {
                assert!(
                    b.start >= a.start && b.end <= a.end,
                    "window escape n={needle:?} h={hay:?} t={t}"
                );
            }
        }
    }

    #[test]
    fn streamed_and_staged_agree_with_many_distinct_bytes() {
        // Membership is a nibble-table lookup either way; windows must
        // agree for many-distinct needles too.
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
                    let streamed = g.hit_file(&file, arena);
                    let staged = g.hit(hay);
                    assert_eq!(
                        streamed.is_some(),
                        staged.is_some(),
                        "n={needle:?} h={hay:?} t={t}"
                    );
                    if let (Some(a), Some(b)) = (staged, streamed) {
                        assert!(
                            b.start >= a.start && b.end <= a.end,
                            "window escape n={needle:?} h={hay:?} t={t}"
                        );
                    }
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

    #[test]
    fn range_walk_matches_full_walk_at_any_split() {
        // A walk split at any byte must fold identical state (cut off).
        use crate::simd_path::build_chunked_path_store_from_strings;
        let mut state = 0x5E1Fu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        for _ in 0..200 {
            let nl = 2 + (next() % 18) as usize;
            let hl = 1 + (next() % 120) as usize;
            let needle: Vec<u8> = (0..nl).map(|_| b'a' + (next() % 6) as u8).collect();
            let hay: Vec<u8> = (0..hl).map(|_| b'a' + (next() % 6) as u8).collect();
            let t = (next() % 4) as u16;
            let split = (next() as usize) % (hl + 1);
            let rel = String::from_utf8_lossy(&hay).to_string();
            let files = vec![FileItem::new_raw(0, 0, 0, None, false)];
            let (store, strings) = build_chunked_path_store_from_strings(&[rel], &files);
            let mut file = files.into_iter().next().unwrap();
            file.set_path(strings[0].clone());
            let arena = store.as_arena_ptr();
            let g = GateNeedle::new(&needle, t);
            let mut whole = WalkState::fresh(false);
            g.walk_range(&file, arena, 0, hl, &mut whole);
            let mut halves = WalkState::fresh(false);
            g.walk_range(&file, arena, 0, split, &mut halves);
            g.walk_range(&file, arena, split, hl, &mut halves);
            assert_eq!(
                (whole.s, whole.first, whole.last, whole.n),
                (halves.s, halves.first, halves.last, halves.n),
                "split={split} n={needle:?} h={hay:?} t={t}"
            );
        }
    }

    #[test]
    fn prefix_cache_matches_direct_walk() {
        // Sequential same-dir files via hit_file_cached must equal hit_file.
        use crate::simd_path::build_chunked_path_store_from_strings;
        let rels = [
            "src/alpha/one.rs",
            "src/alpha/two.rs",
            "src/beta/three.rs",
            "src/alpha/four.rs",
            "src/alpha/averylongfilename_that_crosses_chunks.rs",
            "other/root.rs",
        ];
        let dirs = [11u32, 11, 22, 11, 11, 33];
        let files = vec![FileItem::new_raw(0, 0, 0, None, false); 6];
        let owned: Vec<String> = rels.iter().map(|s| s.to_string()).collect();
        let (store, strings) = build_chunked_path_store_from_strings(&owned, &files);
        let mut files: Vec<FileItem> = files.into_iter().collect();
        for (((file, s), rel), &d) in files
            .iter_mut()
            .zip(strings.iter())
            .zip(owned.iter())
            .zip(dirs.iter())
        {
            file.set_path(s.clone());
            file.parent_dir_index = d;
            file.path.filename_offset =
                (rel.rfind('/').map(|i| i + 1).unwrap_or(0) as u16).min(file.path.byte_len);
        }
        let arena = store.as_arena_ptr();
        for needle in [
            &b"alpha"[..],
            b"handler",
            b"src",
            b"qq",
            b"averylongfilename",
        ] {
            for t in [0u16, 2, 4] {
                let mut g = GateNeedle::new(needle, t);
                // Cache-vs-direct must be window-identical: disable the
                // cutoff (its step alignment differs by construction).
                g.set_cut(false);
                let mut cache = PrefixWalkCache::new();
                for file in &files {
                    assert_eq!(
                        g.hit_file_cached(file, arena, &mut cache, None),
                        g.hit_file(file, arena),
                        "n={needle:?} t={t} f={:?}",
                        file.path.byte_len,
                    );
                }
                // need==0 short-circuits before the walk; otherwise every
                // first-per-dir file must fill and siblings must hit.
                if needle.len() > t as usize {
                    assert!(cache.fills > 0, "no dir fills recorded");
                    assert!(cache.hits > 0, "no sibling hits recorded");
                }
            }
        }
    }

    #[test]
    fn flat_walk_matches_chunk_walk() {
        // Same bytes in path order through the flat table must fold the
        // same windows as the chunk walk, with and without the prefix
        // cache and the cutoff.
        use crate::simd_path::build_chunked_path_store_from_strings;
        use crate::stable_vec::StableVec;
        use crate::suffix_proto::SuffixProto;
        let rels = [
            "src/alpha/one.rs",
            "src/alpha/two.rs",
            "src/beta/three.rs",
            "src/alpha/averylongfilename_that_crosses_chunks.rs",
            "other/root.rs",
        ];
        let dirs = [11u32, 11, 22, 11, 33];
        let files = vec![FileItem::new_raw(0, 0, 0, None, false); 5];
        let owned: Vec<String> = rels.iter().map(|s| s.to_string()).collect();
        let (store, strings) = build_chunked_path_store_from_strings(&owned, &files);
        let mut vec: Vec<FileItem> = files.into_iter().collect();
        for (((file, s), rel), &d) in vec
            .iter_mut()
            .zip(strings.iter())
            .zip(owned.iter())
            .zip(dirs.iter())
        {
            file.set_path(s.clone());
            file.parent_dir_index = d;
            file.path.filename_offset =
                (rel.rfind('/').map(|i| i + 1).unwrap_or(0) as u16).min(file.path.byte_len);
        }
        let stable = StableVec::from_vec_with_reserve(vec, 0);
        let arena = store.as_arena_ptr();
        let proto = SuffixProto::build(&stable, arena, stable.len(), 7);
        let files: &[FileItem] = &stable;
        for needle in [&b"alpha"[..], b"handler", b"src", b"qq", b"rs"] {
            for t in [0u16, 2, 4] {
                for cut in [false, true] {
                    let mut g = GateNeedle::new(needle, t);
                    g.set_cut(cut);
                    let mut cache = PrefixWalkCache::new();
                    for file in files.iter() {
                        let flat = proto.suffix_for(file);
                        assert!(flat.is_some(), "proto must cover base files");
                        let a =
                            g.hit_file_cached(file, arena, &mut cache, flat);
                        let b = g.hit_file(file, arena);
                        // Flat and chunk walks see the same bytes: strictly
                        // equal without the cutoff. With it, step alignment
                        // may cut at different points (still sound), so only
                        // accept-identity is pinned there.
                        if cut {
                            assert_eq!(
                                a.is_some(),
                                b.is_some(),
                                "accept changed n={needle:?} t={t}",
                            );
                        } else {
                            assert_eq!(a, b, "n={needle:?} t={t} cut={cut}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn suffix_proto_addressing_falls_back() {
        // Overflow, foreign and resized views miss the table by address.
        use crate::simd_path::build_chunked_path_store_from_strings;
        use crate::stable_vec::StableVec;
        use crate::suffix_proto::SuffixProto;
        let owned = vec!["src/a/one.rs".to_string(), "src/a/two.rs".to_string()];
        let files = vec![FileItem::new_raw(0, 0, 0, None, false); 2];
        let (store, strings) = build_chunked_path_store_from_strings(&owned, &files);
        let mut vec: Vec<FileItem> = files.into_iter().collect();
        for ((file, s), rel) in vec.iter_mut().zip(strings.iter()).zip(owned.iter()) {
            file.set_path(s.clone());
            file.path.filename_offset =
                (rel.rfind('/').map(|i| i + 1).unwrap_or(0) as u16).min(file.path.byte_len);
        }
        let stable = StableVec::from_vec_with_reserve(vec, 0);
        let arena = store.as_arena_ptr();
        let proto = SuffixProto::build(&stable, arena, 1, 7);
        let files: &[FileItem] = &stable;
        assert!(proto.suffix_for(&files[0]).is_some());
        assert!(proto.suffix_for(&files[1]).is_none());
        let foreign = FileItem::new_raw(0, 0, 0, None, false);
        assert!(proto.suffix_for(&foreign).is_none());
        assert_eq!(proto.generation(), 7);
    }

    #[test]
    fn span_cut_keeps_accept_and_tightens_window() {
        // Cut windows accept exactly when full windows do, and never
        // extend past them. Fuzzed against the contiguous oracle.
        use crate::simd_path::build_chunked_path_store_from_strings;
        let mut state = 0xC07u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        for _ in 0..300 {
            let nl = 2 + (next() % 14) as usize;
            let hl = 1 + (next() % 100) as usize;
            let needle: Vec<u8> = (0..nl).map(|_| b'a' + (next() % 6) as u8).collect();
            let hay: Vec<u8> = (0..hl).map(|_| b'a' + (next() % 6) as u8).collect();
            let t = (next() % 4) as u16;
            let rel = String::from_utf8_lossy(&hay).to_string();
            let files = vec![FileItem::new_raw(0, 0, 0, None, false)];
            let (store, strings) = build_chunked_path_store_from_strings(&[rel], &files);
            let mut file = files.into_iter().next().unwrap();
            file.set_path(strings[0].clone());
            let arena = store.as_arena_ptr();
            let mut g = GateNeedle::new(&needle, t);
            let plain = g.hit_file(&file, arena);
            g.set_cut(true);
            let cut = g.hit_file(&file, arena);
            assert_eq!(
                plain.is_some(),
                cut.is_some(),
                "accept changed n={needle:?} h={hay:?} t={t}"
            );
            if let (Some(a), Some(b)) = (plain, cut) {
                assert!(
                    b.start >= a.start && b.end <= a.end,
                    "cut window not inside plain n={needle:?} h={hay:?} t={t}"
                );
            }
            assert_eq!(
                cut.is_some(),
                g.hit(&hay).is_some(),
                "cut diverged from oracle n={needle:?} h={hay:?} t={t}"
            );
        }
    }
}
