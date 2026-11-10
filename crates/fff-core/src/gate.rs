use crate::simd_path::{ArenaPtr, ChunkedString, SIMD_CHUNK_BYTES};
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
    if b.is_ascii_uppercase() {
        b + 32
    } else {
        b
    }
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

fn needle_span_chunks(
    mask: &NeedleMask,
    file: &FileItem,
    arena: ArenaPtr,
    byte_len: usize,
    need: usize,
) -> Option<(usize, usize)> {
    if byte_len < need {
        return None;
    }
    let mut first = None;
    let mut last = 0usize;
    let mut n = 0usize;
    file.path
        .for_each_chunk_span(arena, 0, byte_len, |base, chunk| {
            for (i, &b) in chunk.iter().enumerate() {
                if mask.contains(b) {
                    let abs = base + i;
                    if first.is_none() {
                        first = Some(abs);
                    }
                    last = abs;
                    n += 1;
                }
            }
            true
        });
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
            return unsafe {
                let regs = shuf_mask_regs(&tables.lo, &tables.hi);
                needle_byte_mask_shuf(chunk, &regs)
            };
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
struct ShufMaskRegs {
    nibble: core::arch::x86_64::__m128i,
    bits0: core::arch::x86_64::__m128i,
    bits1: core::arch::x86_64::__m128i,
    zero: core::arch::x86_64::__m128i,
    t0: core::arch::x86_64::__m128i,
    t1: core::arch::x86_64::__m128i,
}

#[cfg(target_arch = "x86_64")]
struct AsciiMaskRegs {
    nibble: core::arch::x86_64::__m128i,
    bits: core::arch::x86_64::__m128i,
    zero: core::arch::x86_64::__m128i,
    t0: core::arch::x86_64::__m128i,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn shuf_mask_regs(t0: &[u8; 16], t1: &[u8; 16]) -> ShufMaskRegs {
    use core::arch::x86_64::*;
    ShufMaskRegs {
        nibble: _mm_set1_epi8(0x0F),
        bits0: _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -128, 64, 32, 16, 8, 4, 2, 1),
        bits1: _mm_set_epi8(-128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        zero: _mm_setzero_si128(),
        t0: unsafe { _mm_loadu_si128(t0.as_ptr() as *const __m128i) },
        t1: unsafe { _mm_loadu_si128(t1.as_ptr() as *const __m128i) },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn ascii_mask_regs(t0: &[u8; 16]) -> AsciiMaskRegs {
    use core::arch::x86_64::*;
    AsciiMaskRegs {
        nibble: _mm_set1_epi8(0x0F),
        bits: _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -128, 64, 32, 16, 8, 4, 2, 1),
        zero: _mm_setzero_si128(),
        t0: unsafe { _mm_loadu_si128(t0.as_ptr() as *const __m128i) },
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn needle_byte_mask_shuf(chunk: &[u8], regs: &ShufMaskRegs) -> u16 {
    use core::arch::x86_64::*;
    let v = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const __m128i) };
    let lo = _mm_and_si128(v, regs.nibble);
    let hi = _mm_and_si128(_mm_srli_epi16(v, 4), regs.nibble);
    let a0 = _mm_shuffle_epi8(regs.t0, lo);
    let a1 = _mm_shuffle_epi8(regs.t1, lo);
    let b0 = _mm_shuffle_epi8(regs.bits0, hi);
    let b1 = _mm_shuffle_epi8(regs.bits1, hi);
    let r = _mm_or_si128(_mm_and_si128(a0, b0), _mm_and_si128(a1, b1));
    let nz = _mm_cmpeq_epi8(r, regs.zero);
    (!_mm_movemask_epi8(nz)) as u16
}

// ASCII-needle half: high table is all zero, so hay bytes >= 128
// land on zero b0 entries. Same bits, 2 shuffles instead of 4.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn needle_byte_mask_ascii(chunk: &[u8], regs: &AsciiMaskRegs) -> u16 {
    use core::arch::x86_64::*;
    let v = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const __m128i) };
    let lo = _mm_and_si128(v, regs.nibble);
    let hi = _mm_and_si128(_mm_srli_epi16(v, 4), regs.nibble);
    let a0 = _mm_shuffle_epi8(regs.t0, lo);
    let b0 = _mm_shuffle_epi8(regs.bits, hi);
    let r = _mm_and_si128(a0, b0);
    let nz = _mm_cmpeq_epi8(r, regs.zero);
    (!_mm_movemask_epi8(nz)) as u16
}

// Broadcast once per needle. A slot walk then only loads these registers.
#[cfg(target_arch = "x86_64")]
struct AsciiYmm {
    nibble: core::arch::x86_64::__m256i,
    bits: core::arch::x86_64::__m256i,
    t0: core::arch::x86_64::__m256i,
    zero: core::arch::x86_64::__m256i,
}

#[cfg(target_arch = "x86_64")]
struct ShufYmm {
    nibble: core::arch::x86_64::__m256i,
    bits0: core::arch::x86_64::__m256i,
    bits1: core::arch::x86_64::__m256i,
    t0: core::arch::x86_64::__m256i,
    t1: core::arch::x86_64::__m256i,
    zero: core::arch::x86_64::__m256i,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ascii_ymm(regs: &AsciiMaskRegs) -> AsciiYmm {
    use core::arch::x86_64::*;
    AsciiYmm {
        nibble: _mm256_broadcastsi128_si256(regs.nibble),
        bits: _mm256_broadcastsi128_si256(regs.bits),
        t0: _mm256_broadcastsi128_si256(regs.t0),
        zero: _mm256_broadcastsi128_si256(regs.zero),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn shuf_ymm(regs: &ShufMaskRegs) -> ShufYmm {
    use core::arch::x86_64::*;
    ShufYmm {
        nibble: _mm256_broadcastsi128_si256(regs.nibble),
        bits0: _mm256_broadcastsi128_si256(regs.bits0),
        bits1: _mm256_broadcastsi128_si256(regs.bits1),
        t0: _mm256_broadcastsi128_si256(regs.t0),
        t1: _mm256_broadcastsi128_si256(regs.t1),
        zero: _mm256_broadcastsi128_si256(regs.zero),
    }
}

// Two SSSE3 masks in one AVX2 pass. Low 16 bits are the first half.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mask_ascii_ymm(
    chunk: &[u8; 32],
    nibble: core::arch::x86_64::__m256i,
    bits: core::arch::x86_64::__m256i,
    t0: core::arch::x86_64::__m256i,
    zero: core::arch::x86_64::__m256i,
) -> u32 {
    use core::arch::x86_64::*;
    let v = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const __m256i) };
    let lo = _mm256_and_si256(v, nibble);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), nibble);
    let a0 = _mm256_shuffle_epi8(t0, lo);
    let b0 = _mm256_shuffle_epi8(bits, hi);
    let r = _mm256_and_si256(a0, b0);
    let nz = _mm256_cmpeq_epi8(r, zero);
    !_mm256_movemask_epi8(nz) as u32
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn needle_byte_mask_ascii_avx2(chunk: &[u8; 32], regs: &AsciiYmm) -> u32 {
    unsafe { mask_ascii_ymm(chunk, regs.nibble, regs.bits, regs.t0, regs.zero) }
}

// Same bits as two general SSSE3 masks. Low 16 bits are the first half.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mask_shuf_ymm(
    chunk: &[u8; 32],
    nibble: core::arch::x86_64::__m256i,
    bits0: core::arch::x86_64::__m256i,
    bits1: core::arch::x86_64::__m256i,
    t0: core::arch::x86_64::__m256i,
    t1: core::arch::x86_64::__m256i,
    zero: core::arch::x86_64::__m256i,
) -> u32 {
    use core::arch::x86_64::*;
    let v = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const __m256i) };
    let lo = _mm256_and_si256(v, nibble);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), nibble);
    let a0 = _mm256_shuffle_epi8(t0, lo);
    let a1 = _mm256_shuffle_epi8(t1, lo);
    let b0 = _mm256_shuffle_epi8(bits0, hi);
    let b1 = _mm256_shuffle_epi8(bits1, hi);
    let r = _mm256_or_si256(_mm256_and_si256(a0, b0), _mm256_and_si256(a1, b1));
    let nz = _mm256_cmpeq_epi8(r, zero);
    !_mm256_movemask_epi8(nz) as u32
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn needle_byte_mask_shuf_avx2(chunk: &[u8; 32], regs: &ShufYmm) -> u32 {
    unsafe {
        mask_shuf_ymm(
            chunk,
            regs.nibble,
            regs.bits0,
            regs.bits1,
            regs.t0,
            regs.t1,
            regs.zero,
        )
    }
}

// 32B AVX2 step (opt-in `FFF_GATE_AVX2`): one shuffle pass per two
// SSSE3 steps, u32 mask out. Same bits, coarser cut granularity.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn needle_byte_mask_avx2(
    chunk: &[u8; 32],
    t0: &[u8; 16],
    t1: &[u8; 16],
    ascii_only: bool,
) -> u32 {
    use core::arch::x86_64::*;
    let v = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const __m256i) };
    let mask_0f = _mm256_set1_epi8(0x0F);
    let lo = _mm256_and_si256(v, mask_0f);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_0f);
    let lut0 =
        _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(t0.as_ptr() as *const __m128i) });
    let bit_lo = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128, 0, 0,
        0, 0, 0, 0, 0, 0,
    );
    let a0 = _mm256_shuffle_epi8(lut0, lo);
    let b0 = _mm256_shuffle_epi8(bit_lo, hi);
    let r = if ascii_only {
        _mm256_and_si256(a0, b0)
    } else {
        let lut1 =
            _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(t1.as_ptr() as *const __m128i) });
        let bit_hi = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4,
            8, 16, 32, 64, -128,
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
    // AVX2 membership mask for a full slot. The walk still steps 16 bytes.
    mask_avx2: bool,
}

fn set_span_bit(lo: &mut [u8; 16], hi: &mut [u8; 16], v: u8) {
    let l = (v & 0x0f) as usize;
    let h = v >> 4;
    if h < 8 {
        lo[l] |= 1 << h;
    } else {
        hi[l] |= 1 << (h - 8);
    }
}

impl SpanTables {
    // One bit per folded needle byte and its uppercase form.
    fn new(needle: &[u8]) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for &b in needle {
            let folded = fold_byte(b);
            set_span_bit(&mut lo, &mut hi, folded);
            if folded.is_ascii_lowercase() {
                set_span_bit(&mut lo, &mut hi, folded - 32);
            }
        }
        let simd = gate_simd();
        let ascii_only = hi == [0u8; 16];
        Self {
            lo,
            hi,
            simd,
            ascii_only,
            avx2: false,
            mask_avx2: false,
        }
    }
}

// `FFF_FORCE_SCALAR=1` disables SIMD dispatch, covering scalar fallbacks.
fn force_scalar() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_FORCE_SCALAR").is_some())
}

fn gate_simd() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("ssse3") && !force_scalar()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

// `FFF_GATE_AVX2=1` opts into 32B gate steps; cached, not per-query.
#[cfg(target_arch = "x86_64")]
fn span_cut_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_NO_SPAN_CUT").is_none())
}

fn tight_span_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_TIGHT_SPAN").is_some())
}

fn gate_avx2_opt_in() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FFF_GATE_AVX2").is_some())
}

// CPU AVX2 for the 32-byte membership mask. Steps stay 16 bytes.
fn gate_mask_avx2() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2") && !force_scalar()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

#[cfg(target_arch = "x86_64")]
fn gate_avx2_hw() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        gate_avx2_opt_in() && std::is_x86_feature_detected!("avx2") && !force_scalar()
    })
}

/// Per-query gate state, built once per search; per-path work is one linear pass.
pub struct GateNeedle {
    needle: std::mem::ManuallyDrop<Vec<u8>>,
    typos: usize,
    need: usize,
    span: NeedleMask,
    // `pm[c]` = bitmask of needle positions holding folded byte `c`.
    pm: std::mem::ManuallyDrop<Box<[u64; 256]>>,
    // Reversed-needle masks for the backward trim walk.
    pm_rev: std::mem::ManuallyDrop<Box<[u64; 256]>>,
    tables: SpanTables,
    len_mask: u64,
    // Right-tail cutoff (`FFF_NO_SPAN_CUT` escapes): stops at the first accepting prefix.
    // Sound for accept/reject by prefix monotonicity; windows tighten so scores may shift.
    // Checked per 16B step, never inside the bit loop.
    cut: bool,
    // Left-trim to the latest accepting start (`FFF_TIGHT_SPAN`); default off.
    tight: bool,
    // `pm_rev` is filled only while `tight` is on.
    rev_ready: bool,
    // Record stepped bytes. Read once when the needle is built.
    count_walked: bool,
    // False only when the index has no tombstones, so the per-file load is skipped.
    check_deleted: bool,
    // SSSE3 shuffle constants for this needle. Built once; walks only borrow them.
    #[cfg(target_arch = "x86_64")]
    ascii_regs: Option<AsciiMaskRegs>,
    #[cfg(target_arch = "x86_64")]
    shuf_regs: Option<ShufMaskRegs>,
    // The same constants already broadcast. A slot does not broadcast again.
    #[cfg(target_arch = "x86_64")]
    ascii_ymm: Option<AsciiYmm>,
    #[cfg(target_arch = "x86_64")]
    shuf_ymm: Option<ShufYmm>,
}

// One gate walk's state; splittable at any byte with identical results.
struct WalkState {
    s: u64,
    first: usize,
    last: usize,
    n: usize,
    end_cut: usize,
    allow_cut: bool,
    walked: u64,
    // Prefix-snapshot slot. A first hit loads the byte before it from the path.
    prev: u8,
    lead_sep: bool,
    // Last LCS popcnt and the hit count it was taken at.
    lcs: u32,
    lcs_n: usize,
}

impl WalkState {
    #[inline]
    fn fresh(allow_cut: bool) -> Self {
        Self {
            s: 0,
            first: usize::MAX,
            last: 0,
            n: 0,
            end_cut: 0,
            allow_cut,
            walked: 0,
            prev: 0,
            lead_sep: false,
            lcs: 0,
            lcs_n: 0,
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
    first: usize,
    last: usize,
    n: usize,
    end_cut: usize,
    prev: u8,
    lead_sep: bool,
    lcs: u32,
    lcs_n: usize,
    // Prefix alone accepted. `prefix_start` is after the leading separator.
    prefix_accepts: bool,
    prefix_include: bool,
    prefix_start: u32,
    pub hits: u64,
    pub fills: u64,
    pub bytes_skipped: u64,
    pub tight_trim: u64,
    pub tight_files: u64,
    pub walked: u64,
}

impl PrefixWalkCache {
    pub fn new() -> Self {
        Self {
            enabled: true,
            dir: u32::MAX,
            split: 0,
            s: 0,
            first: usize::MAX,
            last: 0,
            n: 0,
            end_cut: 0,
            prev: 0,
            lead_sep: false,
            lcs: 0,
            lcs_n: 0,
            prefix_accepts: false,
            prefix_include: false,
            prefix_start: 0,
            hits: 0,
            fills: 0,
            bytes_skipped: 0,
            tight_trim: 0,
            tight_files: 0,
            walked: 0,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

fn take_gate_mask(slot: &std::cell::RefCell<Option<Box<[u64; 256]>>>) -> Box<[u64; 256]> {
    slot.borrow_mut()
        .take()
        .unwrap_or_else(|| Box::new([0u64; 256]))
}

fn stash_gate_mask(slot: &std::cell::RefCell<Option<Box<[u64; 256]>>>, mask: Box<[u64; 256]>) {
    *slot.borrow_mut() = Some(mask);
}

fn fill_rev_mask(pm: &mut [u64; 256], needle: &[u8]) {
    for (i, &b) in needle.iter().rev().enumerate().take(64) {
        let lo = fold_byte(b);
        pm[lo as usize] |= 1u64 << i;
        if lo.is_ascii_lowercase() {
            pm[(lo - 32) as usize] |= 1u64 << i;
        }
    }
}

fn clear_gate_mask(pm: &mut [u64; 256], bytes: impl Iterator<Item = u8>) {
    for b in bytes {
        let lo = fold_byte(b);
        pm[lo as usize] = 0;
        if lo.is_ascii_lowercase() {
            pm[(lo - 32) as usize] = 0;
        }
    }
}

// Mask tables stay on this thread and are cleared for the next needle.
thread_local! {
    static GATE_PM: std::cell::RefCell<Option<Box<[u64; 256]>>> = const { std::cell::RefCell::new(None) };
    static GATE_PM_REV: std::cell::RefCell<Option<Box<[u64; 256]>>> = const { std::cell::RefCell::new(None) };
    static GATE_NEEDLE: std::cell::RefCell<Option<Vec<u8>>> = const { std::cell::RefCell::new(None) };
}

fn take_needle_buf() -> Vec<u8> {
    GATE_NEEDLE.with(|slot| slot.borrow_mut().take().unwrap_or_default())
}

fn stash_needle_buf(mut buf: Vec<u8>) {
    buf.clear();
    if buf.capacity() > 256 {
        buf.shrink_to(256);
    }
    GATE_NEEDLE.with(|slot| *slot.borrow_mut() = Some(buf));
}

impl Drop for GateNeedle {
    fn drop(&mut self) {
        let mut pm = unsafe { std::mem::ManuallyDrop::take(&mut self.pm) };
        let mut pm_rev = unsafe { std::mem::ManuallyDrop::take(&mut self.pm_rev) };
        let needle = unsafe { std::mem::ManuallyDrop::take(&mut self.needle) };
        // Same bytes the builder ORs in. Entries this needle never used stay zero.
        clear_gate_mask(&mut pm, needle.iter().copied().take(64));
        if self.rev_ready {
            clear_gate_mask(&mut pm_rev, needle.iter().rev().copied().take(64));
        }
        GATE_PM.with(|slot| stash_gate_mask(slot, pm));
        GATE_PM_REV.with(|slot| stash_gate_mask(slot, pm_rev));
        stash_needle_buf(needle);
    }
}

impl GateNeedle {
    // New hits can raise the LCS by at most one each, so a short step skips the popcnt.
    #[inline(always)]
    fn cut_ready(&self, st: &mut WalkState) -> bool {
        if st.n < self.need {
            return false;
        }
        let grown = st.n - st.lcs_n;
        if (st.lcs as usize) + grown < self.need {
            return false;
        }
        let have = (st.s & self.len_mask).count_ones();
        st.lcs = have;
        st.lcs_n = st.n;
        have >= self.need as u32
    }

    pub fn new(needle: &[u8], max_typos: u16) -> Self {
        let typos = (max_typos as usize).min(MAX_SKIPS);
        let m = needle.len();
        let mut pm = GATE_PM.with(take_gate_mask);
        // Pre-folded both cases: the walk indexes raw bytes, no
        // fold per hit. Uppercase needle bytes set both entries.
        for (i, &b) in needle.iter().enumerate().take(64) {
            let lo = fold_byte(b);
            pm[lo as usize] |= 1u64 << i;
            if lo.is_ascii_lowercase() {
                pm[(lo - 32) as usize] |= 1u64 << i;
            }
        }
        // Reverse masks feed the tight-span trim. That trim is off unless requested.
        let tight = tight_span_enabled();
        let mut pm_rev = GATE_PM_REV.with(take_gate_mask);
        if tight {
            fill_rev_mask(&mut pm_rev, needle);
        }
        let span = NeedleMask::new(needle);
        let mut tables = SpanTables::new(needle);
        #[cfg(target_arch = "x86_64")]
        let (ascii_regs, shuf_regs) = if tables.simd {
            unsafe {
                if tables.ascii_only {
                    (Some(ascii_mask_regs(&tables.lo)), None)
                } else {
                    (None, Some(shuf_mask_regs(&tables.lo, &tables.hi)))
                }
            }
        } else {
            (None, None)
        };
        #[cfg(target_arch = "x86_64")]
        {
            tables.avx2 = gate_avx2_hw();
            tables.mask_avx2 = gate_mask_avx2();
        }
        #[cfg(target_arch = "x86_64")]
        let (ascii_ymm, shuf_ymm) = if tables.mask_avx2 {
            unsafe {
                if tables.ascii_only {
                    (ascii_regs.as_ref().map(|regs| ascii_ymm(regs)), None)
                } else {
                    (None, shuf_regs.as_ref().map(|regs| shuf_ymm(regs)))
                }
            }
        } else {
            (None, None)
        };
        Self {
            span,
            tables,
            len_mask: if m >= 64 { u64::MAX } else { (1u64 << m) - 1 },
            needle: std::mem::ManuallyDrop::new({
                let mut owned = take_needle_buf();
                owned.clear();
                owned.extend_from_slice(needle);
                owned
            }),
            typos,
            need: m.saturating_sub(typos),
            pm: std::mem::ManuallyDrop::new(pm),
            cut: span_cut_enabled(),
            pm_rev: std::mem::ManuallyDrop::new(pm_rev),
            tight,
            rev_ready: tight,
            count_walked: crate::score::phase_timing(),
            check_deleted: true,
            #[cfg(target_arch = "x86_64")]
            ascii_regs,
            #[cfg(target_arch = "x86_64")]
            shuf_regs,
            #[cfg(target_arch = "x86_64")]
            ascii_ymm,
            #[cfg(target_arch = "x86_64")]
            shuf_ymm,
        }
    }

    pub(crate) fn set_check_deleted(&mut self, on: bool) {
        self.check_deleted = on;
    }

    // Needle is 1..=64 and can miss. The scan skips those checks per file.
    pub(crate) fn live_scan(&self) -> bool {
        !self.needle.is_empty() && self.needle.len() <= 64 && self.need > 0
    }

    // Every accept is the whole path, and the stored end is that length.
    pub(crate) fn full_path_windows(&self) -> bool {
        self.need == 0
    }

    pub(crate) fn deletes_checked(&self) -> bool {
        self.check_deleted
    }

    #[cfg(test)]
    fn set_tight(&mut self, on: bool) {
        if on == self.tight {
            return;
        }
        if on {
            fill_rev_mask(&mut self.pm_rev, &self.needle);
            self.rev_ready = true;
        } else if self.rev_ready {
            clear_gate_mask(
                &mut self.pm_rev,
                self.needle.iter().rev().copied().take(64),
            );
            self.rev_ready = false;
        }
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

    // The preceding byte is read only when this file's first hit is the step's first byte.
    #[inline(always)]
    fn prev_if_first(
        st: &WalkState,
        mask: u32,
        chunk: &[u8; SIMD_CHUNK_BYTES],
        index: usize,
    ) -> Option<u8> {
        if st.n == 0 && mask & 1 != 0 {
            Self::held_prev(chunk, index)
        } else {
            None
        }
    }

    #[inline(always)]
    fn held_prev(chunk: &[u8; SIMD_CHUNK_BYTES], index: usize) -> Option<u8> {
        if index == 0 {
            None
        } else {
            Some(unsafe { *chunk.get_unchecked(index - 1) })
        }
    }

    // The next hit's position word is loaded while this hit updates `s`.
    // Mask bits are indices inside `step`. The scan order is LSB to MSB.
    #[inline(always)]
    fn myers_hits(pm: &[u64; 256], step: &[u8], mask: u32, s: &mut u64) -> (u32, usize) {
        let mut bits = mask;
        let mut bit = bits.trailing_zeros();
        let mut pos = pm[unsafe { *step.get_unchecked(bit as usize) } as usize];
        let mut n = 1usize;
        loop {
            bits &= bits - 1;
            if bits == 0 {
                let x = *s | pos;
                let y = (*s << 1) | 1;
                *s = x & (x ^ x.wrapping_sub(y));
                return (bit, n);
            }
            let nb = bits.trailing_zeros();
            let np = pm[unsafe { *step.get_unchecked(nb as usize) } as usize];
            let x = *s | pos;
            let y = (*s << 1) | 1;
            *s = x & (x ^ x.wrapping_sub(y));
            pos = np;
            bit = nb;
            n += 1;
        }
    }

    // One 16B step at `abs`. An empty mask returns. The byte before the
    // first hit is loaded only when that hit is the first byte of a step.
    #[inline(always)]
    fn walk_step(
        &self,
        pm: &[u64; 256],
        file: &FileItem,
        arena: ArenaPtr,
        abs: usize,
        step: &[u8],
        mask: u16,
        st: &mut WalkState,
        want_cut: bool,
        before: Option<u8>,
    ) {
        if mask == 0 {
            return;
        }
        // One set bit skips the walk. A multi-hit step counts the bits it visits.
        let single = mask & (mask - 1) == 0;
        let first_bit = if st.n == 0 || single {
            mask.trailing_zeros()
        } else {
            0
        };
        if st.n == 0 {
            // `before` is the preceding byte when this step starts inside the loaded slot.
            let prev_b = if first_bit == 0 {
                if let Some(b) = before {
                    b
                } else if abs == 0 {
                    0
                } else {
                    file.path.byte_at(arena, abs - 1)
                }
            } else {
                unsafe { *step.get_unchecked(first_bit as usize - 1) }
            };
            st.first = abs + first_bit as usize;
            st.lead_sep = prev_b == b'/' || prev_b == b'\\';
        }
        let c = if single {
            let bit = first_bit;
            let b = unsafe { *step.get_unchecked(bit as usize) };
            let x = st.s | pm[b as usize];
            let y = (st.s << 1) | 1;
            st.s = x & (x ^ x.wrapping_sub(y));
            st.last = abs + bit as usize;
            1
        } else {
            // Myers over hit bytes only; other bytes are identity updates.
            let (last_bit, n) = Self::myers_hits(pm, step, mask as u32, &mut st.s);
            st.last = abs + last_bit as usize;
            n
        };
        st.n += c;
        // Past an accepting prefix, the tail is slack the SW window does not need.
        if want_cut && st.end_cut == 0 && self.cut_ready(st) {
            st.end_cut = abs + step.len();
        }
    }

    // AVX2 32B only when opted in; SSSE3 otherwise; scalar last.
    #[inline(always)]
    fn walk_range(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
        slots: *const u32,
    ) {
        #[cfg(not(target_arch = "x86_64"))]
        let _ = slots;
        #[cfg(target_arch = "x86_64")]
        if self.tables.avx2 {
            // ASCII is fixed for the needle, so each step keeps one mask body.
            return unsafe {
                if self.tables.ascii_only {
                    self.walk_range_avx2::<true>(file, arena, lo, hi, st)
                } else {
                    self.walk_range_avx2::<false>(file, arena, lo, hi, st)
                }
            };
        }
        #[cfg(target_arch = "x86_64")]
        if self.tables.simd {
            // Timing is fixed for the process. The byte counter stays out of the step.
            return unsafe {
                if self.count_walked {
                    if self.tables.ascii_only {
                        self.walk_range_simd::<true, true>(file, arena, lo, hi, st, slots)
                    } else {
                        self.walk_range_simd::<false, true>(file, arena, lo, hi, st, slots)
                    }
                } else if self.tables.ascii_only {
                    self.walk_range_simd::<true, false>(file, arena, lo, hi, st, slots)
                } else {
                    self.walk_range_simd::<false, false>(file, arena, lo, hi, st, slots)
                }
            };
        }
        self.walk_range_scalar(file, arena, lo, hi, st)
    }

    // 32B step: same fold as 16B over a u32 mask. Cut granularity
    // doubles, so windows may widen vs 16B (adjudicate on sets).
    #[inline(always)]
    fn walk_step32(
        &self,
        pm: &[u64; 256],
        file: &FileItem,
        arena: ArenaPtr,
        abs: usize,
        step: &[u8],
        mask: u32,
        st: &mut WalkState,
        want_cut: bool,
        before: Option<u8>,
    ) {
        if mask == 0 {
            return;
        }
        let single = mask & (mask - 1) == 0;
        let first_bit = if st.n == 0 || single {
            mask.trailing_zeros()
        } else {
            0
        };
        if st.n == 0 {
            let prev_b = if first_bit == 0 {
                if let Some(b) = before {
                    b
                } else if abs == 0 {
                    0
                } else {
                    file.path.byte_at(arena, abs - 1)
                }
            } else {
                unsafe { *step.get_unchecked(first_bit as usize - 1) }
            };
            st.first = abs + first_bit as usize;
            st.lead_sep = prev_b == b'/' || prev_b == b'\\';
        }
        let c = if single {
            let bit = first_bit;
            let b = unsafe { *step.get_unchecked(bit as usize) };
            let x = st.s | pm[b as usize];
            let y = (st.s << 1) | 1;
            st.s = x & (x ^ x.wrapping_sub(y));
            st.last = abs + bit as usize;
            1
        } else {
            let (last_bit, n) = Self::myers_hits(pm, step, mask, &mut st.s);
            st.last = abs + last_bit as usize;
            n
        };
        st.n += c;
        if want_cut && st.end_cut == 0 && self.cut_ready(st) {
            st.end_cut = abs + step.len();
        }
    }

    // A prefetch does not fault.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn prefetch_slot(arena: ArenaPtr, idx: u32) {
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                arena.as_ptr().add(idx as usize * SIMD_CHUNK_BYTES) as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }

    // Indexed chunks stay immutable for the search.
    #[inline(always)]
    fn slot_chunk<'a>(arena: ArenaPtr, idx: u32) -> &'a [u8; SIMD_CHUNK_BYTES] {
        unsafe {
            &*(arena.as_ptr().add(idx as usize * SIMD_CHUNK_BYTES) as *const [u8; SIMD_CHUNK_BYTES])
        }
    }

    // 32B range for the opt-in AVX2 path; partial tails go 16B.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn walk_range_avx2<const ASCII: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
    ) {
        let want_cut = self.cut && st.allow_cut;
        let counted = self.count_walked;
        if ASCII {
            let regs = unsafe { self.ascii_regs.as_ref().unwrap_unchecked() };
            unsafe {
                self.walk_range_avx2_masked::<ASCII>(
                    file,
                    arena,
                    lo,
                    hi,
                    st,
                    want_cut,
                    counted,
                    |w| needle_byte_mask_ascii(w, regs),
                );
            }
        } else {
            let regs = unsafe { self.shuf_regs.as_ref().unwrap_unchecked() };
            unsafe {
                self.walk_range_avx2_masked::<ASCII>(
                    file,
                    arena,
                    lo,
                    hi,
                    st,
                    want_cut,
                    counted,
                    |w| needle_byte_mask_shuf(w, regs),
                );
            }
        }
    }

    // Partial 16-byte tails reuse `mask16`. The 32-byte steps stay on the AVX2 mask.
    #[cfg(target_arch = "x86_64")]
    unsafe fn walk_range_avx2_masked<const ASCII: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
        want_cut: bool,
        counted: bool,
        mask16: impl Fn(&[u8]) -> u16,
    ) {
        let pm = &*self.pm;
        // Callers pass `hi` inside the path, so the walk does not reload the length.
        debug_assert!(hi <= file.path.byte_len as usize);
        if lo >= hi {
            return;
        }
        let start_chunk = lo / SIMD_CHUNK_BYTES;
        let mut off = start_chunk * SIMD_CHUNK_BYTES;
        let mut index = file.path.slot_index(arena, start_chunk);
        // One index pointer, then each slot. Prefetch the next only while this span reads it.
        // `end_cut` applies before the slot, not between its two steps.
        while off < hi {
            let take = SIMD_CHUNK_BYTES.min(hi - off);
            let idx = if off + take < hi {
                let pair = unsafe { core::ptr::read_unaligned(index as *const u64) };
                let pair = u64::from_le(pair);
                Self::prefetch_slot(arena, (pair >> 32) as u32);
                if off + take + SIMD_CHUNK_BYTES < hi {
                    let ahead = unsafe { *index.add(2) };
                    Self::prefetch_slot(arena, ahead);
                }
                pair as u32
            } else {
                unsafe { *index }
            };
            let chunk = Self::slot_chunk(arena, idx);
            let mut end = (off + take).min(hi);
            if st.end_cut != 0 {
                if off >= st.end_cut {
                    break;
                }
                end = end.min(st.end_cut);
            }
            let start = off.max(lo);
            if start >= end {
                break;
            }
            let len = end - start;
            if counted {
                st.walked += len as u64;
            }
            let mut base = 0usize;
            while base < len {
                let step = 32.min(len - base);
                let p = start - off + base;
                if step == 32 {
                    let mask = unsafe {
                        needle_byte_mask_avx2(chunk, &self.tables.lo, &self.tables.hi, ASCII)
                    };
                    self.walk_step32(
                        pm,
                        file,
                        arena,
                        off + p,
                        &chunk[p..p + step],
                        mask,
                        st,
                        want_cut,
                        Self::prev_if_first(st, mask, chunk, p),
                    );
                } else {
                    // Partial tail: 16B sub-steps, same as the scalar range;
                    // sub-16B sub-steps take a 16B load inside the entry.
                    let mut b2 = 0usize;
                    while b2 < step {
                        let s2 = 16.min(step - b2);
                        let q = p + b2;
                        let sg = &chunk[q..q + s2];
                        let m2 = if s2 == 16 {
                            mask16(sg)
                        } else {
                            let src = q.min(16);
                            let m = mask16(&chunk[src..src + 16]);
                            (m >> (q - src)) & (u16::MAX >> (16 - s2))
                        };
                        self.walk_step(
                            pm,
                            file,
                            arena,
                            off + q,
                            sg,
                            m2,
                            st,
                            want_cut,
                            Self::prev_if_first(st, m2 as u32, chunk, q),
                        );
                        b2 += s2;
                    }
                }
                base += step;
            }
            off += take;
            index = unsafe { index.add(1) };
        }
    }

    // One index base, then each slot. Prefetch the next only while this span reads it.
    // `end_cut` applies before the slot, not between its two steps.
    fn walk_range_scalar(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
    ) {
        let pm = &*self.pm;
        let want_cut = self.cut && st.allow_cut;
        let counted = self.count_walked;
        // Callers pass `hi` inside the path, so the walk does not reload the length.
        debug_assert!(hi <= file.path.byte_len as usize);
        if lo >= hi {
            return;
        }
        let start_chunk = lo / SIMD_CHUNK_BYTES;
        let mut off = start_chunk * SIMD_CHUNK_BYTES;
        let mut index = file.path.slot_index(arena, start_chunk);
        while off < hi {
            let take = SIMD_CHUNK_BYTES.min(hi - off);
            let idx = if off + take < hi {
                let pair = unsafe { core::ptr::read_unaligned(index as *const u64) };
                let pair = u64::from_le(pair);
                #[cfg(target_arch = "x86_64")]
                {
                    Self::prefetch_slot(arena, (pair >> 32) as u32);
                    if off + take + SIMD_CHUNK_BYTES < hi {
                        let ahead = unsafe { *index.add(2) };
                        Self::prefetch_slot(arena, ahead);
                    }
                }
                pair as u32
            } else {
                unsafe { *index }
            };
            let chunk = Self::slot_chunk(arena, idx);
            let mut end = (off + take).min(hi);
            if st.end_cut != 0 {
                if off >= st.end_cut {
                    break;
                }
                end = end.min(st.end_cut);
            }
            let start = off.max(lo);
            if start >= end {
                break;
            }
            let view = &chunk[start - off..end - off];
            let len = end - start;
            if counted {
                st.walked += len as u64;
            }
            if len > 16 {
                let seg0 = &view[..16];
                let mask0 = needle_byte_mask(seg0, &self.span, &self.tables);
                if mask0 != 0 {
                    self.walk_step(
                        pm,
                        file,
                        arena,
                        start,
                        seg0,
                        mask0,
                        st,
                        want_cut,
                        Self::prev_if_first(st, mask0 as u32, chunk, start - off),
                    );
                }
                let seg1 = &view[16..];
                let mask1 = needle_byte_mask(seg1, &self.span, &self.tables);
                if mask1 != 0 {
                    self.walk_step(
                        pm,
                        file,
                        arena,
                        start + 16,
                        seg1,
                        mask1,
                        st,
                        want_cut,
                        Self::prev_if_first(st, mask1 as u32, chunk, start - off + 16),
                    );
                }
            } else if len > 0 {
                let mask = needle_byte_mask(view, &self.span, &self.tables);
                if mask != 0 {
                    self.walk_step(
                        pm,
                        file,
                        arena,
                        start,
                        view,
                        mask,
                        st,
                        want_cut,
                        Self::prev_if_first(st, mask as u32, chunk, start - off),
                    );
                }
            }
            off += take;
            index = unsafe { index.add(1) };
        }
    }

    // The low half is the first 16 bytes. A cut still lands between the halves.
    #[inline(always)]
    fn walk_two_halves(
        &self,
        pm: &[u64; 256],
        file: &FileItem,
        arena: ArenaPtr,
        chunk: &[u8; SIMD_CHUNK_BYTES],
        abs: usize,
        bits: u32,
        st: &mut WalkState,
        want_cut: bool,
    ) {
        let low = bits as u16;
        let high = (bits >> 16) as u16;
        if low != 0 {
            let seg0 = unsafe { core::slice::from_raw_parts(chunk.as_ptr(), 16) };
            self.walk_step(pm, file, arena, abs, seg0, low, st, want_cut, None);
        }
        if high != 0 {
            let seg1 = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(16), 16) };
            self.walk_step(
                pm,
                file,
                arena,
                abs + 16,
                seg1,
                high,
                st,
                want_cut,
                Self::prev_if_first(st, high as u32, chunk, 16),
            );
        }
    }

    // 16-byte steps inside one slot. The two masks of a long step overlap.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn walk_loaded_chunk<const ASCII: bool, const COUNTED: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        chunk: &[u8; SIMD_CHUNK_BYTES],
        at: usize,
        len: usize,
        abs: usize,
        st: &mut WalkState,
        want_cut: bool,
    ) {
        if len > 0 && at + len <= SIMD_CHUNK_BYTES && self.tables.mask_avx2 {
            // `ASCII` is the needle, so this monomorphization keeps one mask body.
            let bits = unsafe {
                if ASCII {
                    needle_byte_mask_ascii_avx2(
                        chunk,
                        self.ascii_ymm.as_ref().unwrap_unchecked(),
                    )
                } else {
                    needle_byte_mask_shuf_avx2(chunk, self.shuf_ymm.as_ref().unwrap_unchecked())
                }
            };
            let pm = &*self.pm;
            self.walk_from_slot_bits::<COUNTED>(
                pm, file, arena, chunk, at, len, abs, bits, st, want_cut,
            );
            return;
        }
        if ASCII {
            let regs = unsafe { self.ascii_regs.as_ref().unwrap_unchecked() };
            unsafe {
                self.walk_loaded_masked::<COUNTED>(
                    file,
                    arena,
                    chunk,
                    at,
                    len,
                    abs,
                    st,
                    want_cut,
                    |w| needle_byte_mask_ascii(w, regs),
                );
            }
        } else {
            let regs = unsafe { self.shuf_regs.as_ref().unwrap_unchecked() };
            unsafe {
                self.walk_loaded_masked::<COUNTED>(
                    file,
                    arena,
                    chunk,
                    at,
                    len,
                    abs,
                    st,
                    want_cut,
                    |w| needle_byte_mask_shuf(w, regs),
                );
            }
        }
    }

    // Bits of one slot, shifted so bit 0 is `chunk[offset]`. `n` is 0..=16.
    #[inline(always)]
    fn slot_step_mask(bits: u32, offset: usize, n: usize) -> u16 {
        let n = n.min(16) as u32;
        let shifted = (bits >> offset) as u32;
        (shifted & ((1u32 << n).wrapping_sub(1))) as u16
    }

    // Same 16-byte steps as the shuffle walk. `bits` is the whole slot.
    #[inline(always)]
    fn walk_from_slot_bits<const COUNTED: bool>(
        &self,
        pm: &[u64; 256],
        file: &FileItem,
        arena: ArenaPtr,
        chunk: &[u8; SIMD_CHUNK_BYTES],
        at: usize,
        len: usize,
        abs: usize,
        bits: u32,
        st: &mut WalkState,
        want_cut: bool,
    ) {
        if COUNTED {
            st.walked += len as u64;
        }
        // No needle byte in this slot. The steps would return immediately.
        if bits == 0 {
            return;
        }
        if len == 32 && at == 0 {
            self.walk_two_halves(pm, file, arena, chunk, abs, bits, st, want_cut);
            return;
        }
        if len > 16 {
            let mask0 = (bits >> at) as u16;
            let step1 = len - 16;
            let p1 = at + 16;
            // The second piece is shorter than 16, so the high bits of the shift are unused.
            let mask1 = ((bits >> p1) as u16) & (u16::MAX >> (16 - step1));
            if mask0 != 0 {
                let seg0 = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(at), 16) };
                self.walk_step(
                    pm,
                    file,
                    arena,
                    abs,
                    seg0,
                    mask0,
                    st,
                    want_cut,
                    Self::prev_if_first(st, mask0 as u32, chunk, at),
                );
            }
            if mask1 != 0 {
                let seg1 = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(p1), step1) };
                self.walk_step(
                    pm,
                    file,
                    arena,
                    abs + 16,
                    seg1,
                    mask1,
                    st,
                    want_cut,
                    Self::prev_if_first(st, mask1 as u32, chunk, p1),
                );
            }
            return;
        }
        let mask = Self::slot_step_mask(bits, at, len);
        // The slot may have hits outside this span. Those bytes are not walked.
        if mask == 0 {
            return;
        }
        let seg = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(at), len) };
        self.walk_step(
            pm,
            file,
            arena,
            abs,
            seg,
            mask,
            st,
            want_cut,
            Self::prev_if_first(st, mask as u32, chunk, at),
        );
    }

    // `mask16` reads one 16-byte window. The needle registers stay in the closure.
    unsafe fn walk_loaded_masked<const COUNTED: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        chunk: &[u8; SIMD_CHUNK_BYTES],
        at: usize,
        len: usize,
        abs: usize,
        st: &mut WalkState,
        want_cut: bool,
        mask16: impl Fn(&[u8]) -> u16,
    ) {
        let pm = &*self.pm;
        if COUNTED {
            st.walked += len as u64;
        }
        // A full slot is two aligned halves. The shift below would be zero.
        if len == 32 && at == 0 {
            let seg0 = unsafe { core::slice::from_raw_parts(chunk.as_ptr(), 16) };
            let seg1 = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(16), 16) };
            let mask0 = mask16(seg0);
            let mask1 = mask16(seg1);
            self.walk_step(pm, file, arena, abs, seg0, mask0, st, want_cut, None);
            self.walk_step(
                pm,
                file,
                arena,
                abs + 16,
                seg1,
                mask1,
                st,
                want_cut,
                Self::prev_if_first(st, mask1 as u32, chunk, 16),
            );
            return;
        }
        if len > 16 {
            let seg0 = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(at), 16) };
            let mask0 = mask16(seg0);
            let step1 = len - 16;
            let p1 = at + 16;
            let seg1 = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(p1), step1) };
            let src = p1.min(16);
            let window = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(src), 16) };
            let mask1 = (mask16(window) >> (p1 - src)) & (u16::MAX >> (16 - step1));
            self.walk_step(
                pm,
                file,
                arena,
                abs,
                seg0,
                mask0,
                st,
                want_cut,
                Self::prev_if_first(st, mask0 as u32, chunk, at),
            );
            self.walk_step(
                pm,
                file,
                arena,
                abs + 16,
                seg1,
                mask1,
                st,
                want_cut,
                Self::prev_if_first(st, mask1 as u32, chunk, p1),
            );
            return;
        }
        if len == 0 {
            return;
        }
        // One step. A slot longer than 16 bytes already returned.
        let step = len;
        let p = at;
        let seg = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(p), step) };
        let mask = if step == 16 {
            mask16(seg)
        } else {
            let src = p.min(16);
            let window = unsafe { core::slice::from_raw_parts(chunk.as_ptr().add(src), 16) };
            (mask16(window) >> (p - src)) & (u16::MAX >> (16 - step))
        };
        self.walk_step(
            pm,
            file,
            arena,
            abs,
            seg,
            mask,
            st,
            want_cut,
            Self::prev_if_first(st, mask as u32, chunk, p),
        );
    }

    // One to four slots. Cut is not rechecked between a slot's two steps.
    // A fifth slot, or a cut already set, stays on the general walk.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn walk_short_span<const ASCII: bool, const COUNTED: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
    ) -> bool {
        if st.end_cut != 0 || hi <= lo {
            return false;
        }
        let span = hi - lo;
        let within = lo % SIMD_CHUNK_BYTES;
        let want_cut = self.cut && st.allow_cut;
        if within + span <= SIMD_CHUNK_BYTES {
            let chunk = unsafe {
                ChunkedString::chunk_at(
                    arena,
                    *file.path.slot_index(arena, lo / SIMD_CHUNK_BYTES),
                )
            };
            unsafe {
                self.walk_loaded_chunk::<ASCII, COUNTED>(
                    file, arena, chunk, within, span, lo, st, want_cut,
                );
            }
            return true;
        }
        let first = SIMD_CHUNK_BYTES - within;
        let second = span - first;
        if second > SIMD_CHUNK_BYTES * 2 {
            if second > SIMD_CHUNK_BYTES * 3 {
                return false;
            }
            let fourth = second - SIMD_CHUNK_BYTES * 2;
            // The span stays inside the path, so these four slots exist.
            let index = file.path.slot_index(arena, lo / SIMD_CHUNK_BYTES);
            let (i0, i1) = crate::simd_path::load_two_slots(index);
            let (i2, i3) = crate::simd_path::load_two_slots(unsafe { index.add(2) });
            let (chunk0, chunk1, chunk2, chunk3) = (
                ChunkedString::chunk_at(arena, i0),
                ChunkedString::chunk_at(arena, i1),
                ChunkedString::chunk_at(arena, i2),
                ChunkedString::chunk_at(arena, i3),
            );
            let abs1 = lo + first;
            let abs2 = abs1 + SIMD_CHUNK_BYTES;
            let abs3 = abs2 + SIMD_CHUNK_BYTES;
            unsafe {
                #[cfg(target_arch = "x86_64")]
                {
                    std::arch::x86_64::_mm_prefetch(
                        chunk1.as_ptr() as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                    std::arch::x86_64::_mm_prefetch(
                        chunk2.as_ptr() as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                    std::arch::x86_64::_mm_prefetch(
                        chunk3.as_ptr() as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
                self.walk_loaded_chunk::<ASCII, COUNTED>(
                    file, arena, chunk0, within, first, lo, st, want_cut,
                );
                let n1 = if st.end_cut == 0 {
                    SIMD_CHUNK_BYTES
                } else if abs1 < st.end_cut {
                    SIMD_CHUNK_BYTES.min(st.end_cut - abs1)
                } else {
                    0
                };
                if n1 > 0 {
                    self.walk_loaded_chunk::<ASCII, COUNTED>(
                        file, arena, chunk1, 0, n1, abs1, st, want_cut,
                    );
                }
                let n2 = if st.end_cut == 0 {
                    SIMD_CHUNK_BYTES
                } else if abs2 < st.end_cut {
                    SIMD_CHUNK_BYTES.min(st.end_cut - abs2)
                } else {
                    0
                };
                if n2 > 0 {
                    self.walk_loaded_chunk::<ASCII, COUNTED>(
                        file, arena, chunk2, 0, n2, abs2, st, want_cut,
                    );
                }
                let n3 = if st.end_cut == 0 {
                    fourth
                } else if abs3 < st.end_cut {
                    fourth.min(st.end_cut - abs3)
                } else {
                    0
                };
                if n3 > 0 {
                    self.walk_loaded_chunk::<ASCII, COUNTED>(
                        file, arena, chunk3, 0, n3, abs3, st, want_cut,
                    );
                }
            }
            return true;
        }
        let third = second.saturating_sub(SIMD_CHUNK_BYTES);
        // The span stays inside the path, so the next slot exists.
        let index = file.path.slot_index(arena, lo / SIMD_CHUNK_BYTES);
        let (i0, i1) = crate::simd_path::load_two_slots(index);
        let (chunk0, chunk1, chunk2) = (
            ChunkedString::chunk_at(arena, i0),
            ChunkedString::chunk_at(arena, i1),
            if third > 0 {
                Some(ChunkedString::chunk_at(arena, unsafe { *index.add(2) }))
            } else {
                None
            },
        );
        // A third slot is the bytes past 64. The middle slot is a full 32.
        let mid = if third > 0 { SIMD_CHUNK_BYTES } else { second };
        let abs1 = lo + first;
        let abs2 = abs1 + mid;
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::x86_64::_mm_prefetch(
                    chunk1.as_ptr() as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
                if let Some(chunk2) = chunk2 {
                    std::arch::x86_64::_mm_prefetch(
                        chunk2.as_ptr() as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
            self.walk_loaded_chunk::<ASCII, COUNTED>(
                file, arena, chunk0, within, first, lo, st, want_cut,
            );
            let n1 = if st.end_cut == 0 {
                mid
            } else if abs1 < st.end_cut {
                mid.min(st.end_cut - abs1)
            } else {
                0
            };
            if n1 > 0 {
                self.walk_loaded_chunk::<ASCII, COUNTED>(
                    file, arena, chunk1, 0, n1, abs1, st, want_cut,
                );
            }
            if let Some(chunk2) = chunk2 {
                let n2 = if st.end_cut == 0 {
                    third
                } else if abs2 < st.end_cut {
                    third.min(st.end_cut - abs2)
                } else {
                    0
                };
                if n2 > 0 {
                    self.walk_loaded_chunk::<ASCII, COUNTED>(
                        file, arena, chunk2, 0, n2, abs2, st, want_cut,
                    );
                }
            }
        }
        true
    }

    // Same walk with the shuffle inlined: tables stay in registers,
    // no per-step call. ASCII needles skip the high-half pair.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "ssse3")]
    unsafe fn walk_range_simd<const ASCII: bool, const COUNTED: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
        slots: *const u32,
    ) {
        let want_cut = self.cut && st.allow_cut;
        // A span that crosses a slot keeps the broadcast registers. One slot stays on the short walk.
        let span = hi.saturating_sub(lo);
        if self.tables.mask_avx2 && st.end_cut == 0 && span > 0 {
            let within = lo % SIMD_CHUNK_BYTES;
            if within + span <= SIMD_CHUNK_BYTES {
                let chunk = unsafe {
                    ChunkedString::chunk_at(arena, *slots.add(lo / SIMD_CHUNK_BYTES))
                };
                unsafe {
                    self.walk_loaded_chunk::<ASCII, COUNTED>(
                        file, arena, chunk, within, span, lo, st, want_cut,
                    );
                }
                return;
            }
        }
        if self.tables.mask_avx2
            && (span > SIMD_CHUNK_BYTES
                || lo % SIMD_CHUNK_BYTES + span > SIMD_CHUNK_BYTES)
        {
            unsafe { self.walk_range_ymm::<ASCII, COUNTED>(file, arena, lo, hi, st, slots) };
            return;
        }
        if unsafe { self.walk_short_span::<ASCII, COUNTED>(file, arena, lo, hi, st) } {
            return;
        }
        if self.tables.mask_avx2 {
            unsafe { self.walk_range_ymm::<ASCII, COUNTED>(file, arena, lo, hi, st, slots) };
            return;
        }
        if ASCII {
            let regs = unsafe { self.ascii_regs.as_ref().unwrap_unchecked() };
            unsafe {
                self.walk_range_masked::<COUNTED>(
                    file,
                    arena,
                    lo,
                    hi,
                    st,
                    slots,
                    want_cut,
                    |w| needle_byte_mask_ascii(w, regs),
                    |_| None,
                );
            }
        } else {
            let regs = unsafe { self.shuf_regs.as_ref().unwrap_unchecked() };
            unsafe {
                self.walk_range_masked::<COUNTED>(
                    file,
                    arena,
                    lo,
                    hi,
                    st,
                    slots,
                    want_cut,
                    |w| needle_byte_mask_shuf(w, regs),
                    |_| None,
                );
            }
        }
    }

    // The long walk holds the broadcast registers. A full slot does not reload them.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn walk_range_ymm<const ASCII: bool, const COUNTED: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
        slots: *const u32,
    ) {
        let want_cut = self.cut && st.allow_cut;
        if ASCII {
            let y = unsafe { self.ascii_ymm.as_ref().unwrap_unchecked() };
            let (nibble, bits, t0, zero) = (y.nibble, y.bits, y.t0, y.zero);
            let regs = unsafe { self.ascii_regs.as_ref().unwrap_unchecked() };
            unsafe {
                self.walk_range_masked::<COUNTED>(
                    file,
                    arena,
                    lo,
                    hi,
                    st,
                    slots,
                    want_cut,
                    |w| needle_byte_mask_ascii(w, regs),
                    |chunk| Some(mask_ascii_ymm(chunk, nibble, bits, t0, zero)),
                );
            }
        } else {
            let y = unsafe { self.shuf_ymm.as_ref().unwrap_unchecked() };
            let (nibble, bits0, bits1, t0, t1, zero) =
                (y.nibble, y.bits0, y.bits1, y.t0, y.t1, y.zero);
            let regs = unsafe { self.shuf_regs.as_ref().unwrap_unchecked() };
            unsafe {
                self.walk_range_masked::<COUNTED>(
                    file,
                    arena,
                    lo,
                    hi,
                    st,
                    slots,
                    want_cut,
                    |w| needle_byte_mask_shuf(w, regs),
                    |chunk| Some(mask_shuf_ymm(chunk, nibble, bits0, bits1, t0, t1, zero)),
                );
            }
        }
    }

    // `mask16` reads one 16-byte window. Needle registers stay in the closure.
    #[inline(always)]
    unsafe fn walk_range_masked<const COUNTED: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        st: &mut WalkState,
        slots: *const u32,
        want_cut: bool,
        mask16: impl Fn(&[u8]) -> u16,
        mask32: impl Fn(&[u8; SIMD_CHUNK_BYTES]) -> Option<u32>,
    ) {
        let pm = &*self.pm;
        // Callers pass `hi` inside the path, so the walk does not reload the length.
        debug_assert!(hi <= file.path.byte_len as usize);
        if lo >= hi {
            return;
        }
        let start_chunk = lo / SIMD_CHUNK_BYTES;
        let mut off = start_chunk * SIMD_CHUNK_BYTES;
        let mut index = unsafe { slots.add(start_chunk) };
        // One index pointer, then each slot. Prefetch the next only while this span reads it.
        // `end_cut` applies before the slot, not between its two steps.
        while off < hi {
            let take = SIMD_CHUNK_BYTES.min(hi - off);
            let idx = if off + take < hi {
                let pair = unsafe { core::ptr::read_unaligned(index as *const u64) };
                let pair = u64::from_le(pair);
                Self::prefetch_slot(arena, (pair >> 32) as u32);
                if off + take + SIMD_CHUNK_BYTES < hi {
                    let ahead = unsafe { *index.add(2) };
                    Self::prefetch_slot(arena, ahead);
                }
                pair as u32
            } else {
                unsafe { *index }
            };
            let chunk = Self::slot_chunk(arena, idx);
            let mut end = (off + take).min(hi);
            if st.end_cut != 0 {
                if off >= st.end_cut {
                    break;
                }
                end = end.min(st.end_cut);
            }
            let start = off.max(lo);
            if start >= end {
                break;
            }
            let len = end - start;
            if COUNTED {
                st.walked += len as u64;
            }
            // The 32-byte mask covers a partial slot too. The count is already added.
            if let Some(bits) = mask32(chunk) {
                self.walk_from_slot_bits::<false>(
                    pm,
                    file,
                    arena,
                    chunk,
                    start - off,
                    len,
                    start,
                    bits,
                    st,
                    want_cut,
                );
                off += take;
                index = unsafe { index.add(1) };
                continue;
            }
            // A full slot is two aligned halves. The shift below would be zero.
            if len == 32 && start == off {
                let seg0 = &chunk[..16];
                let seg1 = &chunk[16..32];
                let mask0 = mask16(seg0);
                let mask1 = mask16(seg1);
                if mask0 != 0 {
                    self.walk_step(pm, file, arena, off, seg0, mask0, st, want_cut, None);
                }
                if mask1 != 0 {
                    self.walk_step(
                        pm,
                        file,
                        arena,
                        off + 16,
                        seg1,
                        mask1,
                        st,
                        want_cut,
                        Self::prev_if_first(st, mask1 as u32, chunk, 16),
                    );
                }
            } else if len > 16 {
                // Step boundaries match the slicing walk. Sub-16B tails take a
                // 16B load at min(p, 16) (inside the entry) and keep their bits.
                let p0 = start - off;
                let seg0 = &chunk[p0..p0 + 16];
                let mask0 = mask16(seg0);
                let step1 = len - 16;
                let p1 = p0 + 16;
                let seg1 = &chunk[p1..p1 + step1];
                let src = p1.min(16);
                let m1 = mask16(&chunk[src..src + 16]);
                let mask1 = (m1 >> (p1 - src)) & (u16::MAX >> (16 - step1));
                if mask0 != 0 {
                    self.walk_step(
                        pm,
                        file,
                        arena,
                        off + p0,
                        seg0,
                        mask0,
                        st,
                        want_cut,
                        Self::prev_if_first(st, mask0 as u32, chunk, p0),
                    );
                }
                if mask1 != 0 {
                    self.walk_step(
                        pm,
                        file,
                        arena,
                        off + p1,
                        seg1,
                        mask1,
                        st,
                        want_cut,
                        Self::prev_if_first(st, mask1 as u32, chunk, p1),
                    );
                }
            } else if len > 0 {
                // One step. A longer piece took the branch above.
                let step = len;
                let p = start - off;
                let seg = &chunk[p..p + step];
                let mask = if step == 16 {
                    mask16(seg)
                } else {
                    let src = p.min(16);
                    let m = mask16(&chunk[src..src + 16]);
                    (m >> (p - src)) & (u16::MAX >> (16 - step))
                };
                if mask != 0 {
                    self.walk_step(
                        pm,
                        file,
                        arena,
                        off + p,
                        seg,
                        mask,
                        st,
                        want_cut,
                        Self::prev_if_first(st, mask as u32, chunk, p),
                    );
                }
            }
            off += take;
            index = unsafe { index.add(1) };
        }
    }

    // Directory missed. Same 16B steps and cut as `walk_range` on `[split, byte_len)`.
    // Prefix LCS plus one per suffix byte still short of `need`: chunk stays unread.
    #[inline(always)]
    fn walk_unaccepted_suffix(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        split: usize,
        byte_len: usize,
        st: &mut WalkState,
        slots: *const u32,
    ) {
        #[cfg(not(target_arch = "x86_64"))]
        let _ = slots;
        let slen = byte_len - split;
        if slen == 0 {
            return;
        }
        // Alphabet hits are an upper bound on LCS. The popcnt is only for a tighter bound.
        if st.n.saturating_add(slen) < self.need {
            return;
        }
        if slen < self.need {
            let have = (st.s & self.len_mask).count_ones() as usize;
            if have.saturating_add(slen) < self.need {
                return;
            }
        }
        if self.tables.avx2 {
            self.walk_range(file, arena, split, byte_len, st, slots);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if self.tables.simd {
            unsafe {
                if self.count_walked {
                    if self.tables.ascii_only {
                        self.walk_suffix_simd::<true, true>(file, arena, split, byte_len, st, slots)
                    } else {
                        self.walk_suffix_simd::<false, true>(file, arena, split, byte_len, st, slots)
                    }
                } else if self.tables.ascii_only {
                    self.walk_suffix_simd::<true, false>(file, arena, split, byte_len, st, slots)
                } else {
                    self.walk_suffix_simd::<false, false>(file, arena, split, byte_len, st, slots)
                }
            };
            return;
        }
        self.walk_range(file, arena, split, byte_len, st, slots);
    }

    // The filename is the same slot walk as the directory. One index pass, not one load per slot.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn walk_suffix_simd<const ASCII: bool, const COUNTED: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        split: usize,
        byte_len: usize,
        st: &mut WalkState,
        slots: *const u32,
    ) {
        unsafe {
            // A name inside one slot skips the crossing-range setup.
            if self.tables.mask_avx2 && st.end_cut == 0 {
                let span = byte_len - split;
                let within = split % SIMD_CHUNK_BYTES;
                if span > 0 && within + span <= SIMD_CHUNK_BYTES {
                    let chunk = ChunkedString::chunk_at(
                        arena,
                        *slots.add(split / SIMD_CHUNK_BYTES),
                    );
                    self.walk_loaded_chunk::<ASCII, COUNTED>(
                        file,
                        arena,
                        chunk,
                        within,
                        span,
                        split,
                        st,
                        self.cut && st.allow_cut,
                    );
                    return;
                }
            }
            self.walk_range_simd::<ASCII, COUNTED>(file, arena, split, byte_len, st, slots);
        }
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

    #[inline]
    fn finish(&self, st: &WalkState, byte_len: usize) -> Option<GateHit> {
        // A cut is stored only after the LCS length already reached `need`.
        if st.end_cut == 0 {
            if self.need > 0 && st.n < self.need {
                return None;
            }
            // A popcnt already stored for this hit count is still the LCS length.
            let have = if st.lcs_n == st.n {
                st.lcs
            } else {
                (st.s & self.len_mask).count_ones()
            };
            if have < self.need as u32 {
                return None;
            }
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
        if (self.check_deleted && file.is_deleted()) || self.needle.is_empty() {
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
            let slots = file.path.slot_index(arena, 0);
            self.walk_range(file, arena, 0, byte_len, &mut st, slots);
            let mut hit = self.finish(&st, byte_len)?;
            apply_lead_sep(&mut hit, st.lead_sep);
            let split = (file.path.filename_offset as usize).min(byte_len);
            widen_to_filename(&mut hit, byte_len, split, st.end_cut != 0);
            if !self.tight {
                return Some(hit);
            }
            return Some(self.maybe_tighten(file, arena, hit, None));
        }
        // Span the chunks first. Rejects never copy. Accepts copy only [first, last].
        if self.need == 0 {
            return Some(GateHit {
                start: 0,
                end: byte_len as u32,
                include_prefix: true,
            });
        }
        let Some((first, last)) =
            needle_span_chunks(&self.span, file, arena, byte_len, self.need)
        else {
            return None;
        };
        let hi = (last + 1).min(byte_len);
        let len = hi - first;
        thread_local! {
            static HAY: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let mut hay = HAY.with(|c| {
            let mut b = c.borrow_mut();
            if b.len() < len {
                b.resize(len, 0);
            }
            std::mem::take(&mut *b)
        });
        file.path
            .copy_window_to(arena, first as u32, hi as u32, &mut hay[..len]);
        let hit = if self.decide(&hay[..len]) {
            let mut hit = GateHit {
                start: first as u32,
                end: hi as u32,
                include_prefix: first == 0,
            };
            include_leading_separator(file, arena, &mut hit);
            let split = (file.path.filename_offset as usize).min(byte_len);
            widen_to_filename(&mut hit, byte_len, split, false);
            Some(hit)
        } else {
            None
        };
        HAY.with(|c| *c.borrow_mut() = hay);
        hit
    }

    /// Memoized prefix walk: same-dir files share it; accept matches `hit_file`.
    /// Cut alignment may differ by step, still sound; cache is per-loop, never shared.
    #[inline(always)]
    pub fn hit_file_cached(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        cache: &mut PrefixWalkCache,
    ) -> Option<GateHit> {
        if self.needle.is_empty() {
            return None;
        }
        if self.needle.len() > 64 {
            return self.hit_file(file, arena);
        }
        if self.need == 0 {
            let byte_len = file.path.byte_len as usize;
            if byte_len == 0 {
                return None;
            }
            return Some(GateHit {
                start: 0,
                end: byte_len as u32,
                include_prefix: true,
            });
        }
        if self.check_deleted {
            self.hit_file_live::<true>(file, arena, cache)
        } else {
            self.hit_file_live::<false>(file, arena, cache)
        }
    }

    // `need > 0` and the needle fits the bit walk. `CHECK_DELETED` is fixed for the scan.
    #[inline(always)]
    pub(crate) fn hit_file_live<const CHECK_DELETED: bool>(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        cache: &mut PrefixWalkCache,
    ) -> Option<GateHit> {
        if CHECK_DELETED && file.is_deleted() {
            return None;
        }
        let (index_offset, byte_len_u, name_off) = file.path.header_parts();
        let byte_len = byte_len_u as usize;
        if byte_len == 0 || byte_len < self.need {
            return None;
        }
        // The header already held the index. The filename walk reuses that base.
        let slots = ChunkedString::slot_base(arena, index_offset);
        let split = (name_off as usize).min(byte_len);
        let dir = file.parent_dir_index;
        if cache.enabled && cache.dir == dir && cache.split == split {
            self.hit_after_prefix(file, arena, cache, split, byte_len, slots)
        } else {
            self.fill_prefix(file, arena, cache, dir, split, byte_len, slots)
        }
    }

    // Sibling of a cached directory. A sealed prefix reuses one window.
    #[inline(always)]
    fn hit_after_prefix(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        cache: &mut PrefixWalkCache,
        split: usize,
        byte_len: usize,
        slots: *const u32,
    ) -> Option<GateHit> {
        if cache.end_cut != 0 && cache.end_cut <= split {
            cache.hits += 1;
            cache.bytes_skipped += byte_len as u64;
            if !cache.prefix_accepts {
                return None;
            }
            return Some(self.note_tight(
                file,
                arena,
                cache,
                GateHit {
                    start: cache.prefix_start,
                    end: byte_len as u32,
                    include_prefix: cache.prefix_include,
                },
            ));
        }
        let mut st = WalkState {
            s: cache.s,
            first: cache.first,
            last: cache.last,
            n: cache.n,
            end_cut: cache.end_cut,
            allow_cut: true,
            walked: 0,
            prev: cache.prev,
            lead_sep: cache.lead_sep,
            lcs: cache.lcs,
            lcs_n: cache.lcs_n,
        };
        cache.hits += 1;
        cache.bytes_skipped += split as u64;
        self.walk_unaccepted_suffix(file, arena, split, byte_len, &mut st, slots);
        cache.walked += st.walked;
        self.commit_hit(file, arena, cache, &st, byte_len, split, false)
    }

    #[inline(never)]
    fn fill_prefix(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        cache: &mut PrefixWalkCache,
        dir: u32,
        split: usize,
        byte_len: usize,
        slots: *const u32,
    ) -> Option<GateHit> {
        let mut st = WalkState::fresh(true);
        self.walk_range(file, arena, 0, split, &mut st, slots);
        if cache.enabled {
            cache.dir = dir;
            cache.split = split;
            cache.s = st.s;
            cache.first = if st.n > 0 { st.first } else { usize::MAX };
            cache.last = st.last;
            cache.n = st.n;
            cache.end_cut = st.end_cut;
            cache.prev = st.prev;
            cache.lead_sep = st.lead_sep;
            cache.lcs = st.lcs;
            cache.lcs_n = st.lcs_n;
            cache.prefix_accepts = false;
            cache.fills += 1;
        }
        let skip = st.end_cut != 0 && st.end_cut <= split;
        if !skip {
            self.walk_unaccepted_suffix(file, arena, split, byte_len, &mut st, slots);
        } else {
            cache.bytes_skipped += (byte_len - split) as u64;
        }
        cache.walked += st.walked;
        self.commit_hit(file, arena, cache, &st, byte_len, split, skip && cache.enabled)
    }

    // `seal` stores the post-separator start when the prefix already accepts.
    #[inline]
    fn commit_hit(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        cache: &mut PrefixWalkCache,
        st: &WalkState,
        byte_len: usize,
        split: usize,
        seal: bool,
    ) -> Option<GateHit> {
        let mut hit = self.finish(st, byte_len)?;
        apply_lead_sep(&mut hit, st.lead_sep);
        widen_to_filename(&mut hit, byte_len, split, st.end_cut != 0);
        if seal {
            cache.prefix_start = hit.start;
            cache.prefix_include = hit.include_prefix;
            cache.prefix_accepts = true;
        }
        Some(self.note_tight(file, arena, cache, hit))
    }

    #[inline]
    fn note_tight(
        &self,
        file: &FileItem,
        arena: ArenaPtr,
        cache: &mut PrefixWalkCache,
        hit: GateHit,
    ) -> GateHit {
        if !self.tight {
            return hit;
        }
        let tight_before = hit.start;
        let out = self.maybe_tighten(file, arena, hit, Some(&mut cache.tight_trim));
        cache.tight_files += (out.start != tight_before) as u64;
        out
    }
}

/// Single-haystack gate. `None` = not a match under the typo model.
pub fn gate_hit(needle: &[u8], hay: &[u8], max_typos: u16) -> Option<GateHit> {
    GateNeedle::new(needle, max_typos).hit(hay)
}

/// Parallel match-set over FileItems, emitting frizbee-compatible matches
/// with (start, end) in (score, end_col). `file_index` maps 0..n to candidates.
#[allow(dead_code)]
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
    let check_presence =
        presence.should_check() && std::env::var_os("FFF_NO_GATE_PREFILTER").is_none();
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
            exact: hit.end as usize == file.path.byte_len as usize,
            end_col: hit.end as u16,
        })
    };
    if threads <= 1 {
        return (0..n as u32).filter_map(scan).collect();
    }
    use rayon::prelude::*;
    (0..n as u32).into_par_iter().filter_map(scan).collect()
}

// SW reads the byte before the match for its delimiter bonus. `st.first` is
// the first needle byte, so pull a preceding separator in; `include_prefix` stays false.
#[inline]
fn apply_lead_sep(hit: &mut GateHit, lead_sep: bool) {
    if hit.start != 0 && !hit.include_prefix && lead_sep {
        hit.start -= 1;
    }
}

// Needles longer than 64 copy a window and do not walk step-by-step.
fn include_leading_separator(file: &FileItem, arena: ArenaPtr, hit: &mut GateHit) {
    let start = hit.start as usize;
    if start == 0 || hit.include_prefix {
        return;
    }
    let b = file.path.byte_at(arena, start - 1);
    apply_lead_sep(hit, b == b'/' || b == b'\\');
}

// Early cut can stop before the filename; widen so the filename is scored.
// `early_cut` is a real stop, not a match that already ended before the tail.
#[inline]
fn widen_to_filename(hit: &mut GateHit, byte_len: usize, split: usize, early_cut: bool) {
    let end = hit.end as usize;
    if end < byte_len && (early_cut || end <= split) {
        hit.end = byte_len.min(u32::MAX as usize) as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_includes_separator_before_filename_match() {
        use crate::simd_path::build_chunked_path_store_from_strings;
        let rel = "lib/score.rs";
        let mut file = FileItem::new_raw(0, 0, 0, None, false);
        let (store, mut strings) =
            build_chunked_path_store_from_strings(&[rel.to_string()], &[file.clone()]);
        file.set_path(strings.remove(0));
        file.path.filename_offset = (rel.rfind('/').unwrap() + 1) as u16;
        let arena = store.as_arena_ptr();
        let g = GateNeedle::new(b"score", 0);
        let h = g.hit_file(&file, arena).expect("score should match");
        assert_eq!(h.start, 3, "window must include the '/' delimiter");
        assert!(!h.include_prefix);
    }

    #[test]
    fn early_cut_inside_filename_widens_to_full_path() {
        use crate::simd_path::build_chunked_path_store_from_strings;
        let rel = "aa/b_long_filename.rs";
        let mut file = FileItem::new_raw(0, 0, 0, None, false);
        let (store, mut strings) =
            build_chunked_path_store_from_strings(&[rel.to_string()], &[file.clone()]);
        file.set_path(strings.remove(0));
        file.path.filename_offset = (rel.rfind('/').unwrap() + 1) as u16;
        let arena = store.as_arena_ptr();
        let g = GateNeedle::new(b"aab", 0);
        let h = g.hit_file(&file, arena).expect("aab should match");
        assert!(h.end as usize > file.path.filename_offset as usize);
        assert_eq!(h.end as usize, file.path.byte_len as usize);
    }

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
            // Cut preserves accept; separator/filename widening may cover
            // bytes outside the staged span.
            assert_eq!(
                streamed.is_some(),
                staged.is_some(),
                "hit disagreement n={needle:?} h={hay:?} t={t}"
            );
            if let (Some(a), Some(b)) = (staged, streamed) {
                assert!(
                    b.start <= a.start && b.end >= a.end,
                    "window lost a staged byte n={needle:?} h={hay:?} t={t}"
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
                            b.start <= a.start && b.end >= a.end,
                            "window lost a staged byte n={needle:?} h={hay:?} t={t}"
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
            let slots = file.path.slot_index(arena, 0);
            g.walk_range(&file, arena, 0, hl, &mut whole, slots);
            let mut halves = WalkState::fresh(false);
            g.walk_range(&file, arena, 0, split, &mut halves, slots);
            g.walk_range(&file, arena, split, hl, &mut halves, slots);
            assert_eq!(
                (whole.s, whole.first, whole.last, whole.n),
                (halves.s, halves.first, halves.last, halves.n),
                "split={split} n={needle:?} h={hay:?} t={t}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_range_matches_scalar_range_with_cut() {
        // Vector tails must fold the byte loop's state: same masks, step
        // boundaries, and cut byte.
        use crate::simd_path::build_chunked_path_store_from_strings;
        let mut state = 0x51EDu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        for _ in 0..300 {
            let nl = 2 + (next() % 18) as usize;
            let hl = 1 + (next() % 120) as usize;
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
            if !g.tables.simd {
                return;
            }
            g.set_cut(true);
            let mut a = WalkState::fresh(true);
            let slots = file.path.slot_index(arena, 0);
            unsafe {
                if g.tables.ascii_only {
                    g.walk_range_simd::<true, false>(&file, arena, 0, hl, &mut a, slots)
                } else {
                    g.walk_range_simd::<false, false>(&file, arena, 0, hl, &mut a, slots)
                }
            };
            let mut b = WalkState::fresh(true);
            g.walk_range_scalar(&file, arena, 0, hl, &mut b);
            assert_eq!(
                (a.s, a.first, a.last, a.n, a.end_cut, g.finish(&a, hl)),
                (b.s, b.first, b.last, b.n, b.end_cut, g.finish(&b, hl)),
                "n={needle:?} h={hay:?} t={t}"
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
                    if next() % 4 == 0 {
                        v.wrapping_add(128)
                    } else {
                        v
                    }
                })
                .collect();
            let chunk: Vec<u8> = (0..32)
                .map(|_| {
                    let v = b'a' + (next() % 6) as u8;
                    if next() % 4 == 0 {
                        v.wrapping_add(128)
                    } else {
                        v
                    }
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
                let h = unsafe {
                    let regs = ascii_mask_regs(&g.tables.lo);
                    needle_byte_mask_ascii(&chunk[..16], &regs)
                };
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
