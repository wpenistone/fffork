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

/// Case-folded needle-byte bitmask: 256 bits as 4×u64.
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
                // No early exit: still scanning for the last hit.
            }
        }
    }
    if n < need {
        return None;
    }
    Some((first.unwrap(), last))
}

// True iff `LCS(needle, hay) >= len - max_typos`; fallback past one 64-bit word.
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
            if s < t && cur < nxt[s + 1] {
                nxt[s + 1] = cur;
            }
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

// Bit-parallel LCS length; exact for `needle.len() <= 64`.
// `pm[c]` holds folded-byte masks, `len_mask` the low `len` bits.
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
    // Lowercase needle takes the memchr path.
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

// T=0 subsequence check; faster than skip-DP.
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

// Bit i set when `chunk[i]` folds to a needle byte, via nibble tables
// (SSSE3; scalar on tails, non-x86, or no-SSSE3).
#[inline]
fn needle_byte_mask(chunk: &[u8], span: &NeedleMask, tables: &SpanTables) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        if chunk.len() == 16 && tables.simd {
            // SAFETY: SSSE3 verified in `GateNeedle::new`; exactly 16 bytes.
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

// 16 parallel membership tests; halves selected without a blend.
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

// ASCII-needle half: high table is all zero, so hay bytes >= 128
// land on zero b0 entries. Same bits, 2 shuffles instead of 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn needle_byte_mask_ascii(chunk: &[u8], t0: &[u8; 16]) -> u16 {
    use core::arch::x86_64::*;
    let v = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
    let lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
    let hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
    let a0 = _mm_shuffle_epi8(_mm_loadu_si128(t0.as_ptr() as *const __m128i), lo);
    let b0 = _mm_shuffle_epi8(
        _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -128, 64, 32, 16, 8, 4, 2, 1),
        hi,
    );
    let r = _mm_and_si128(a0, b0);
    let nz = _mm_cmpeq_epi8(r, _mm_setzero_si128());
    ((!_mm_movemask_epi8(nz)) as u16) & 0xFFFF
}

// 32B AVX2 step (opt-in `FFF_GATE_AVX2`): one shuffle pass per two
// SSSE3 steps, u32 mask out. Same bits, coarser cut granularity.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn needle_byte_mask_avx2(chunk: &[u8; 32], t0: &[u8; 16], t1: &[u8; 16], ascii_only: bool) -> u32 {
    use core::arch::x86_64::*;
    let v = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
    let mask_0f = _mm256_set1_epi8(0x0F);
    let lo = _mm256_and_si256(v, mask_0f);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_0f);
    let lut0 = _mm256_broadcastsi128_si256(_mm_loadu_si128(t0.as_ptr() as *const __m128i));
    let bit_lo = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let a0 = _mm256_shuffle_epi8(lut0, lo);
    let b0 = _mm256_shuffle_epi8(bit_lo, hi);
    let r = if ascii_only {
        _mm256_and_si256(a0, b0)
    } else {
        let lut1 = _mm256_broadcastsi128_si256(_mm_loadu_si128(t1.as_ptr() as *const __m128i));
        let bit_hi = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let a1 = _mm256_shuffle_epi8(lut1, lo);
        let b1 = _mm256_shuffle_epi8(bit_hi, hi);
        _mm256_or_si256(_mm256_and_si256(a0, b0), _mm256_and_si256(a1, b1))
    };
    let nz = _mm256_cmpeq_epi8(r, _mm256_setzero_si256());
    !_mm256_movemask_epi8(nz) as u32
}

// Per-query nibble tables: `lo[l]`/`hi[l]` hold span rows for low nibble `l`.
struct SpanTables {
    lo: [u8; 16],
    hi: [u8; 16],
    simd: bool,
    // True when no needle byte is >= 128: the high-half tables are
    // all zero, so the a1/b1 shuffle pair is redundant.
    ascii_only: bool,
    // Opt-in 32B gate steps (`FFF_GATE_AVX2`); default off pending
    // adjudication (coarser cut granularity shifts windows).
    avx2: bool,
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
        let ascii_only = hi == [0u8; 16];
        Self { lo, hi, simd, ascii_only, avx2: false }
    }
}

// `FFF_FORCE_SCALAR=1` disables SIMD dispatch, covering scalar fallbacks.
fn force_scalar() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_FORCE_SCALAR").is_some())
}

// `FFF_GATE_AVX2=1` opts into 32B gate steps; cached, not per-query.
#[cfg(target_arch = "x86_64")]
fn gate_avx2_opt_in() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_GATE_AVX2").is_some())
}

/// Per-query gate state, built once per search; per-path work is one linear pass.
pub struct GateNeedle {
    needle: Box<[u8]>,
    typos: usize,
    need: usize,
    span: NeedleMask,
    // `pm[c]` = bitmask of needle positions holding folded byte `c`.
    pm: Box<[u64; 256]>,
    // Reversed-needle masks for the backward trim walk.
    pm_rev: Box<[u64; 256]>,
    tables: SpanTables,
    len_mask: u64,
    // Right-tail cutoff (`FFF_NO_SPAN_CUT` escapes): stops at the first accepting prefix.
    // Sound for accept/reject by prefix monotonicity; windows tighten so scores may shift.
    // Checked per 16B step, never inside the bit loop.
    cut: bool,
    // Left-trim to the latest accepting start (`FFF_TIGHT_SPAN`); default off.
    tight: bool,
}

// One gate walk's state; splittable at any byte with identical results.
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

// Memoized prefix walk: same-dir siblings share it, once per dir.
// Owned per sequential file loop, never shared across threads.
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
    pub tight_trim: u64,
    pub tight_files: u64,
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
            tight_trim: 0,
            tight_files: 0,
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
        // Pre-folded both cases: the walk indexes raw bytes, no
        // fold per hit. Uppercase needle bytes set both entries.
        for (i, &b) in needle.iter().enumerate().take(64) {
            let lo = fold_byte(b);
            pm[lo as usize] |= 1u64 << i;
            if lo.is_ascii_lowercase() {
                pm[(lo - 32) as usize] |= 1u64 << i;
            }
        }
        // LCS is reversal-invariant, so the backward walk accepts exactly when forward does.
        let mut pm_rev = Box::new([0u64; 256]);
        for (i, &b) in needle.iter().rev().enumerate().take(64) {
            let lo = fold_byte(b);
            pm_rev[lo as usize] |= 1u64 << i;
            if lo.is_ascii_lowercase() {
                pm_rev[(lo - 32) as usize] |= 1u64 << i;
            }
        }
        let span = NeedleMask::new(needle);
        let mut tables = SpanTables::new(&span);
        #[cfg(target_arch = "x86_64")]
        {
            tables.avx2 =
                gate_avx2_opt_in() && std::is_x86_feature_detected!("avx2") && !force_scalar();
        }
        Self {
            span,
            tables,
            len_mask: if m >= 64 { u64::MAX } else { (1u64 << m) - 1 },
            needle: needle.to_vec().into_boxed_slice(),
            typos,
            need: m.saturating_sub(typos),
            pm,
            cut: std::env::var_os("FFF_NO_SPAN_CUT").is_none(),
            pm_rev,
            tight: std::env::var_os("FFF_TIGHT_SPAN").is_some(),
        }
    }

    #[cfg(test)]
    fn set_tight(&mut self, on: bool) {
        self.tight = on;
    }

    #[cfg(test)]
    fn set_avx2_32(&mut self, on: bool) {
        self.tables.avx2 = on;
    }

    #[cfg(test)]
    fn set_cut(&mut self, on: bool) {
        self.cut = on;
    }

    // One 16B step at `abs`: fold precomputed mask into extremes
    // and Myers state. Callers pick the mask source (scalar table
    // test or inlined shuffle); folding is identical either way.
    #[inline]
    fn walk_step(&self, abs: usize, step: &[u8], mask: u16, st: &mut WalkState, want_cut: bool) {
        let pm = &*self.pm;
        let c = mask.count_ones() as usize;
        if c == 0 {
            return;
        }
        if st.n == 0 {
            st.first = abs + mask.trailing_zeros() as usize;
        }
        st.last = abs + (16 - mask.leading_zeros() as usize - 1);
        // Myers over hit bytes only; other bytes are identity updates.
        let mut bits = mask;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let x = st.s | pm[step[bit] as usize];
            let y = (st.s << 1) | 1;
            st.s = x & (x ^ x.wrapping_sub(y));
        }
        st.n += c;
        // Past an accepting prefix, the tail is slack the SW window does not need.
        if want_cut && st.end_cut == 0 && st.n >= self.need.max(1) {
            if (st.s & self.len_mask).count_ones() >= self.need as u32 {
                st.end_cut = abs + step.len();
            }
        }
    }

    // Range dispatch, once per walk: AVX2 32B when opted in, else
    // the SSSE3 range (inlined shuffle, hoisted tables), scalar last.
    fn walk_range(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
    ) {
        #[cfg(target_arch = "x86_64")]
        if self.tables.avx2 {
            return unsafe { self.walk_range_avx2(file, arena, lo, hi, st) };
        }
        #[cfg(target_arch = "x86_64")]
        if self.tables.simd {
            return unsafe { self.walk_range_simd(file, arena, lo, hi, st) };
        }
        self.walk_range_scalar(file, arena, lo, hi, st)
    }

    // 32B step: same fold as 16B over a u32 mask. Cut granularity
    // doubles, so windows may widen vs 16B (adjudicate on sets).
    #[inline]
    fn walk_step32(&self, abs: usize, step: &[u8], mask: u32, st: &mut WalkState, want_cut: bool) {
        let pm = &*self.pm;
        let c = mask.count_ones() as usize;
        if c == 0 {
            return;
        }
        if st.n == 0 {
            st.first = abs + mask.trailing_zeros() as usize;
        }
        st.last = abs + (32 - mask.leading_zeros() as usize - 1);
        let mut bits = mask;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let x = st.s | pm[step[bit] as usize];
            let y = (st.s << 1) | 1;
            st.s = x & (x ^ x.wrapping_sub(y));
        }
        st.n += c;
        if want_cut && st.end_cut == 0 && st.n >= self.need.max(1) {
            if (st.s & self.len_mask).count_ones() >= self.need as u32 {
                st.end_cut = abs + step.len();
            }
        }
    }

    // 32B range for the opt-in AVX2 path; partial tails go 16B.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn walk_range_avx2(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
    ) {
        let byte_len = file.path.byte_len as usize;
        let want_cut = self.cut && st.allow_cut;
        let ascii = self.tables.ascii_only;
        let (t0, t1) = (&self.tables.lo, &self.tables.hi);
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
                let step = 32.min(len - base);
                let seg = &view[base..base + step];
                if step == 32 {
                    let arr = &*(seg.as_ptr() as *const [u8; 32]);
                    let mask = needle_byte_mask_avx2(arr, t0, t1, ascii);
                    self.walk_step32(start + base, seg, mask, st, want_cut);
                } else {
                    // Partial tail: 16B sub-steps (scalar mask
                    // assumes <= 16B), same as the scalar range.
                    let mut b2 = 0usize;
                    while b2 < step {
                        let s2 = 16.min(step - b2);
                        let sg = &seg[b2..b2 + s2];
                        let m2 = needle_byte_mask(sg, &self.span, &self.tables);
                        self.walk_step(start + base + b2, sg, m2, st, want_cut);
                        b2 += s2;
                    }
                }
                base += step;
            }
        });
    }

    // Walks path bytes in [lo, hi) via arena chunks into `st`.
    // Interior steps stay vectorized; cut stops recording past the first accepting prefix.
    fn walk_range_scalar(
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
                let seg = &view[base..base + step];
                let mask = needle_byte_mask(seg, &self.span, &self.tables);
                self.walk_step(start + base, seg, mask, st, want_cut);
                base += step;
            }
        });
    }

    // Same walk with the shuffle inlined: tables stay in registers,
    // no per-step call. ASCII needles skip the high-half pair.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "ssse3")]
    unsafe fn walk_range_simd(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
    ) {
        let byte_len = file.path.byte_len as usize;
        let want_cut = self.cut && st.allow_cut;
        let ascii = self.tables.ascii_only;
        let (t0, t1) = (&self.tables.lo, &self.tables.hi);
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
                let seg = &view[base..base + step];
                let mask = if step == 16 {
                    if ascii {
                        needle_byte_mask_ascii(seg, t0)
                    } else {
                        needle_byte_mask_shuf(seg, t0, t1)
                    }
                } else {
                    needle_byte_mask(seg, &self.span, &self.tables)
                };
                self.walk_step(start + base, seg, mask, st, want_cut);
                base += step;
            }
        });
    }

    // Latest still-accepting window start, via backward Myers on reversed masks.
    // Step-granular accept, byte-exact refine within.
    fn latest_start(&self, window: &[u8]) -> usize {
        if self.need == 0 || window.is_empty() {
            return 0;
        }
        let pm = &*self.pm_rev;
        let mut s = 0u64;
        let mut pos = window.len();
        while pos > 0 {
            let step = 16.min(pos);
            let base = pos - step;
            let seg = &window[base..pos];
            let s_before = s;
            let mask = needle_byte_mask(seg, &self.span, &self.tables);
            let mut bits = mask;
            while bits != 0 {
                let bit = 15 - bits.leading_zeros() as usize;
                bits &= !(1u16 << bit);
                let x = s | pm[seg[bit] as usize];
                let y = (s << 1) | 1;
                s = x & (x ^ x.wrapping_sub(y));
            }
            if (s & self.len_mask).count_ones() >= self.need as u32 {
                let mut r = s_before;
                for (k, &b) in seg.iter().enumerate().rev() {
                    let x = r | pm[b as usize];
                    let y = (r << 1) | 1;
                    r = x & (x ^ x.wrapping_sub(y));
                    if (r & self.len_mask).count_ones() >= self.need as u32 {
                        return base + k;
                    }
                }
                return base;
            }
            pos = base;
        }
        0
    }

    // Left-trims an accepted window to its latest accepting start.
    // Accept never changes, scores may shift.
    fn maybe_tighten(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        hit: GateHit,
        trim_out: Option<&mut u64>,
    ) -> GateHit {
        if !self.tight || self.needle.len() > 64 || self.need == 0 {
            return hit;
        }
        let (start, end) = (hit.start as usize, hit.end as usize);
        if end - start <= self.need {
            return hit;
        }
        thread_local! {
            static TWIN: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let cut = TWIN.with(|c| {
            let mut b = c.borrow_mut();
            let len = end - start;
            if b.len() < len {
                b.resize(len, 0);
            }
            file.path
                .copy_window_to(arena, start as u32, end as u32, &mut b[..len]);
            let a = self.latest_start(&b[..len]);
            if let Some(out) = trim_out {
                *out += a as u64;
            }
            a
        });
        let new_start = start + cut;
        GateHit {
            start: new_start as u32,
            include_prefix: new_start == 0,
            ..hit
        }
    }

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

    /// Gates a `FileItem` from arena chunks in one LCS pass, without copying the path.
    pub fn hit_file(&self, file: &FileItem, arena: ArenaPtr) -> Option<GateHit> {
        if file.is_deleted() || self.needle.is_empty() {
            return None;
        }
        let byte_len = file.path.byte_len as usize;
        if byte_len == 0 {
            return None;
        }
        // `LCS <= min` lengths, so `byte_len < need` rejects before any chunk load.
        if byte_len < self.need {
            return None;
        }
        if self.needle.len() <= 64 {
            // `need == 0` accepts everything with a full-path window, as in `hit`.
            if self.need == 0 {
                return Some(GateHit {
                    start: 0,
                    end: byte_len as u32,
                    include_prefix: true,
                });
            }
            // Single adjudicating walk is exact (skips are identity updates);
            // extremes come from the mask bits.
            let mut st = WalkState::fresh(true);
            self.walk_range(file, arena, 0, byte_len, &mut st);
            let hit = self.finish(&st, byte_len)?;
            return Some(self.maybe_tighten(file, arena, hit, None));
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
                // SAFETY: shrink-only within initialized length.
                unsafe { b.set_len(byte_len) };
            }
            std::mem::take(&mut *b)
        });
        file.path.read_to_buf(arena, &mut hay[..byte_len]);
        let hit = self.hit(&hay[..byte_len]);
        HAY.with(|c| *c.borrow_mut() = hay);
        hit
    }

    /// Memoized prefix walk: same-dir files share it; accept matches `hit_file`.
    /// Cut alignment may differ by step, still sound; cache is per-loop, never shared.
    pub fn hit_file_cached(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        cache: &mut PrefixWalkCache,
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
        // An accepted prefix already contains the window; the suffix walk is overhead.
        if st.end_cut == 0 || st.end_cut > split {
            self.walk_range(file, arena, split, byte_len, &mut st);
        } else {
            cache.bytes_skipped += (byte_len - split) as u64;
        }
        let hit = self.finish(&st, byte_len)?;
        let tight_before = hit.start;
        let out = self.maybe_tighten(file, arena, hit, Some(&mut cache.tight_trim));
        cache.tight_files += (out.start != tight_before) as u64;
        Some(out)
    }
}

/// Single-haystack gate. `None` = not a match under the typo model.
pub fn gate_hit(needle: &[u8], hay: &[u8], max_typos: u16) -> Option<GateHit> {
    GateNeedle::new(needle, max_typos).hit(hay)
}

/// Parallel match-set over FileItems, emitting frizbee-compatible matches
/// with (start, end) in (score, end_col). `file_index` maps 0..n to candidates.
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
    let state = GateNeedle::new(needle, max_typos);
    // Sound pretest: paths missing more needle bytes than budget allows cannot match.
    let presence = crate::presence::NeedlePresence::new(needle, max_typos);
    let check_presence = presence.should_check()
        && std::env::var_os("FFF_NO_GATE_PREFILTER").is_none();
    let scan = |i: u32| -> Option<neo_frizbee::Match> {
        // Prefetch next path: the gate is bound by arena load latency.
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
        // Streamed chunks must agree with the flat slice across 16B boundaries.
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
            // Cut preserves accept and only tightens the window.
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
        // Many-distinct needles must agree too (nibble-table path).
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
                // Disable cut: its step alignment differs by construction.
                g.set_cut(false);
                let mut cache = PrefixWalkCache::new();
                for file in &files {
                    assert_eq!(
                        g.hit_file_cached(file, arena, &mut cache),
                        g.hit_file(file, arena),
                        "n={needle:?} t={t} f={:?}",
                        file.path.byte_len,
                    );
                }
                if needle.len() > t as usize {
                    assert!(cache.fills > 0, "no dir fills recorded");
                    assert!(cache.hits > 0, "no sibling hits recorded");
                }
            }
        }
    }

    #[test]
    fn span_cut_keeps_accept_and_tightens_window() {
        // Cut keeps accept and never widens the window.
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

    #[test]
    fn avx_mask_matches_scalar_masks() {
        // 32B AVX mask vs two scalar halves, over ASCII and high-byte
        // chunks (file-level fixtures cannot carry high bytes: the
        // store builder takes Strings, so lossy conversion would
        // rewrite them; masks take slices directly).
        let mut state = 0x9A7u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        for _ in 0..200 {
            let nl = 2 + (next() % 10) as usize;
            let needle: Vec<u8> = (0..nl)
                .map(|_| {
                    let v = b'a' + (next() % 6) as u8;
                    if next() % 4 == 0 { v.wrapping_add(128) } else { v }
                })
                .collect();
            let chunk: Vec<u8> = (0..32)
                .map(|_| {
                    let v = b'a' + (next() % 6) as u8;
                    if next() % 4 == 0 { v.wrapping_add(128) } else { v }
                })
                .collect();
            let g = GateNeedle::new(&needle, 2);
            let arr: &[u8; 32] = chunk.as_slice().try_into().unwrap();
            let a = unsafe {
                needle_byte_mask_avx2(arr, &g.tables.lo, &g.tables.hi, g.tables.ascii_only)
            };
            let s0 = needle_byte_mask(&chunk[..16], &g.span, &g.tables) as u32;
            let s1 = needle_byte_mask(&chunk[16..], &g.span, &g.tables) as u32;
            assert_eq!(a, s0 | (s1 << 16), "mask diverged n={needle:?}");
            // ASCII fast path must agree too when it applies.
            if g.tables.ascii_only {
                let h = unsafe { needle_byte_mask_ascii(&chunk[..16], &g.tables.lo) };
                assert_eq!(h, s0 as u16, "ascii mask diverged n={needle:?}");
            }
        }
    }

    #[test]
    fn avx32_steps_match_oracle_windows() {
        // 32B steps see the same bytes: accept-identical always. Cut
        // off: window-identical (same extremes). Cut on: accept-equal
        // with coarser cut firing later-or-equal. ASCII hays here
        // (lossy fixtures rewrite high bytes); high bytes are covered
        // at mask level in `avx_mask_matches_scalar_masks`.
        use crate::simd_path::build_chunked_path_store_from_strings;
        let mut state = 0xA32u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        for _ in 0..300 {
            let nl = 2 + (next() % 12) as usize;
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
            for cut in [false, true] {
                let mut g = GateNeedle::new(&needle, t);
                g.set_cut(cut);
                let plain = g.hit_file(&file, arena);
                g.set_avx2_32(true);
                let wide = g.hit_file(&file, arena);
                let oracle = g.hit(&hay);
                assert_eq!(
                    oracle.is_some(),
                    wide.is_some(),
                    "avx diverged from oracle cut={cut} n={needle:?} h={hay:?} t={t}"
                );
                assert_eq!(
                    plain.is_some(),
                    wide.is_some(),
                    "accept changed cut={cut} n={needle:?} h={hay:?} t={t}"
                );
                if let (Some(a), Some(b)) = (plain, wide) {
                    if cut {
                        // Coarser cut fires later-or-equal; same extremes.
                        assert!(
                            b.start == a.start && b.end >= a.end,
                            "avx window broke order cut={cut} n={needle:?} t={t}"
                        );
                    } else {
                        assert_eq!(a, b, "cut-off windows differ n={needle:?} t={t}");
                    }
                }
            }
        }
    }

    #[test]
    fn tight_span_keeps_accept_and_trims_left() {
        // Tight left-trims to the latest accepting start, keeping accept.
        use crate::simd_path::build_chunked_path_store_from_strings;
        let mut state = 0x716u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        for _ in 0..300 {
            let nl = 2 + (next() % 12) as usize;
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
            g.set_cut(true);
            let cut = g.hit_file(&file, arena);
            g.set_tight(true);
            let tight = g.hit_file(&file, arena);
            assert_eq!(
                cut.is_some(),
                tight.is_some(),
                "accept changed n={needle:?} h={hay:?} t={t}"
            );
            if let (Some(a), Some(b)) = (cut, tight) {
                assert!(
                    b.start >= a.start && b.end == a.end,
                    "tight not left-trim n={needle:?} h={hay:?} t={t}"
                );
                // Oracle: trimmed window accepts, one byte further rejects.
                if nl > t as usize {
                    let w = &hay[a.start as usize..a.end as usize];
                    let off = (b.start - a.start) as usize;
                    assert!(g.decide(&w[off..]), "trimmed rejects n={needle:?} t={t}");
                    if off + 1 < w.len() {
                        assert!(!g.decide(&w[off + 1..]), "not latest n={needle:?} t={t}");
                    }
                }
            }
        }
    }

    #[test]
    fn tight_cached_matches_direct_and_counts() {
        use crate::simd_path::build_chunked_path_store_from_strings;
        let rels = ["src/aaa/one.rs", "src/aaa/two.rs", "src/bbb/three.rs"];
        let owned: Vec<String> = rels.iter().map(|s| s.to_string()).collect();
        let files = vec![FileItem::new_raw(0, 0, 0, None, false); 3];
        let (store, strings) = build_chunked_path_store_from_strings(&owned, &files);
        let mut vec: Vec<FileItem> = files.into_iter().collect();
        let dirs = [11u32, 11, 22];
        for (((file, s), rel), &d) in vec
            .iter_mut()
            .zip(strings.iter())
            .zip(rels.iter())
            .zip(dirs.iter())
        {
            file.set_path(s.clone());
            file.parent_dir_index = d;
            file.path.filename_offset =
                (rel.rfind('/').map(|i| i + 1).unwrap_or(0) as u16).min(file.path.byte_len);
        }
        let arena = store.as_arena_ptr();
        let mut g = GateNeedle::new(b"aaors", 2);
        g.set_cut(true);
        g.set_tight(true);
        let mut cache = PrefixWalkCache::new();
        let mut expect_trim = 0u64;
        let mut expect_moved = 0u64;
        for file in &vec {
            let direct = g.hit_file(file, arena);
            assert_eq!(
                g.hit_file_cached(file, arena, &mut cache),
                direct,
                "cached diverged for {:?}",
                file.path.byte_len,
            );
            g.set_tight(false);
            let cut = g.hit_file(file, arena);
            g.set_tight(true);
            if let (Some(a), Some(b)) = (cut, direct) {
                expect_trim += (b.start - a.start) as u64;
                expect_moved += (b.start != a.start) as u64;
            }
        }
        assert_eq!(cache.tight_trim, expect_trim, "trim miscounted");
        assert_eq!(cache.tight_files, expect_moved, "file count off");
    }
}
