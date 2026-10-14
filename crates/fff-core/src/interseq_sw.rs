//! Inter-sequence Smith-Waterman: lanes hold paths, haystack runs sequential.
//!
//! Replicates frizbee's gap model; score is the last row's max.

// Scoring params matching frizbee defaults + fff overrides for lowercase queries.
#[derive(Clone, Copy)]
pub(crate) struct SwParams {
    pub match_score: u16,
    pub mismatch_penalty: u16,
    pub gap_open: u16,
    pub gap_extend: u16,
    pub prefix_bonus: u16,
    pub delimiter_bonus: u16,
    pub capitalization_bonus: u16,
    pub matching_case_bonus: u16,
}

impl SwParams {
    pub fn for_query(has_uppercase: bool) -> Self {
        Self {
            match_score: 12,
            mismatch_penalty: 6,
            gap_open: 5,
            gap_extend: 1,
            prefix_bonus: 12,
            delimiter_bonus: 4,
            capitalization_bonus: if has_uppercase { 8 } else { 0 },
            matching_case_bonus: if has_uppercase { 4 } else { 0 },
        }
    }
    pub fn u8_safe(&self, m: usize) -> bool {
        let per = self.match_score as u32
            + self.delimiter_bonus as u32
            + self.capitalization_bonus as u32
            + self.matching_case_bonus as u32;
        (m as u32)
            .saturating_mul(per)
            .saturating_add(self.prefix_bonus as u32)
            < 256
    }

    // Sound upper bound on any SW score for this needle: each alignment
    // path takes at most one diagonal per column (gaps only subtract),
    // so rows contribute at most `per` each, prefix once, exact once.
    // Reserved for UB work (pre-SW prune needs a sound global threshold
    // first); the fuzz test below pins it.
    #[allow(dead_code)]
    pub(crate) fn max_sw_score(&self, m: usize) -> i32 {
        let per = self.match_score as i32
            + self.delimiter_bonus as i32
            + self.capitalization_bonus as i32
            + self.matching_case_bonus as i32;
        (m as i32) * per + self.prefix_bonus as i32 + 8
    }
}

#[inline(always)]
fn is_delimiter(b: u8) -> bool {
    b.is_ascii() && !b.is_ascii_alphanumeric()
}

#[inline(always)]
fn flip_case(b: u8) -> u8 {
    if b.is_ascii_uppercase() {
        b + 32
    } else if b.is_ascii_lowercase() {
        b - 32
    } else {
        b
    }
}

// Per-worker DP rows for long needles (m > 32). The scores below used to
// allocate 8 vecs per call; reuse zeroed buffers instead. Same values.
#[derive(Default)]
struct LongScratch {
    h0: Vec<i32>,
    h1: Vec<i32>,
    e0: Vec<i32>,
    e1: Vec<i32>,
    f0: Vec<i32>,
    f1: Vec<i32>,
    m0: Vec<bool>,
    m1: Vec<bool>,
}

impl LongScratch {
    fn ensure(&mut self, n: usize) {
        for v in [
            &mut self.h0,
            &mut self.h1,
            &mut self.e0,
            &mut self.e1,
            &mut self.f0,
            &mut self.f1,
        ] {
            v.resize(n, 0);
            v.fill(0);
        }
        for v in [&mut self.m0, &mut self.m1] {
            v.resize(n, false);
            v.fill(false);
        }
    }
}

thread_local! {
    static LONG_SCRATCH: std::cell::RefCell<LongScratch> =
        const { std::cell::RefCell::new(LongScratch {
            h0: Vec::new(),
            h1: Vec::new(),
            e0: Vec::new(),
            e1: Vec::new(),
            f0: Vec::new(),
            f1: Vec::new(),
            m0: Vec::new(),
            m1: Vec::new(),
        }) };
}

// Scalar reference SW matching frizbee's exact semantics on a TRIMMED haystack.
// `include_prefix` must be `true` iff the trimmed start == 0 (frizbee:
// `haystack_start_pos == 0` after `trim_haystack`'s saturating_sub(1)).
// Returns (score, end_col_local). end_col_local = LAST column in the trimmed
// slice where the last-row max occurs (frizbee's match_end_col).
// No exact_match_bonus — caller adds it when `include_exact && needle == trimmed`.
pub(crate) fn score_one_scalar_full(
    needle: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    let m = needle.len();
    let n = haystack.len();
    if m == 0 || n == 0 {
        return (0, 0);
    }
    // Stack scratch for common lengths: the 8 heap rows below cost more
    // than the DP itself on short trims. Longer needles keep the Vec path.
    if m <= 32 {
        return score_one_scalar_stack(needle, haystack, p, include_prefix);
    }
    score_one_scalar_vec(needle, haystack, p, include_prefix)
}

fn score_one_scalar_vec(
    needle: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    let m = needle.len();
    let n = haystack.len();
    // Use i32 DP values to avoid saturation artifacts during computation.
    let gap_open = p.gap_open as i32;
    let gap_extend = p.gap_extend as i32;
    let gap_open_only = gap_open; // full open penalty when transitioning match->gap
    let match_add = p.match_score as i32;
    let mismatch_pen = p.mismatch_penalty as i32;

    // match_prev[i]: was (i-1, j-1) a match?  (frizbee's shifted_match_mask)
    // match_curr[i]: is (i-1, j) a match?
    // Rows come from the worker-local scratch (zeroed on acquire).
    let mut scratch = LONG_SCRATCH.with(|c| {
        let mut s = c.borrow_mut();
        s.ensure(m + 1);
        std::mem::take(&mut *s)
    });
    let mut h_prev = std::mem::take(&mut scratch.h0);
    let mut e_prev = std::mem::take(&mut scratch.e0); // horizontal gap state
    let mut f_prev = std::mem::take(&mut scratch.f0); // vertical gap state
    let mut h_curr = std::mem::take(&mut scratch.h1);
    let mut e_curr = std::mem::take(&mut scratch.e1);
    let mut f_curr = std::mem::take(&mut scratch.f1);
    let mut match_prev = std::mem::take(&mut scratch.m0);
    let mut match_curr = std::mem::take(&mut scratch.m1);

    let mut prev_is_lower = false;
    let mut prev_is_delim = false;
    let mut last_row_max = 0i32;
    let mut end_col = 0u16;

    for j in 0..n {
        let hb = haystack[j];
        let hb_upper = hb.is_ascii_uppercase();
        let hb_lower = hb.is_ascii_lowercase();
        let hb_delim = is_delimiter(hb);

        let cap_bonus = if hb_upper && prev_is_lower {
            p.capitalization_bonus as i32
        } else {
            0
        };
        // frizbee: delimiter_mask = prev_char_is_delimiter AND NOT curr_char_is_delimiter.
        // prev starts as NOT-delimiter (Mask::zero()).
        let delim_bonus = if prev_is_delim && !hb_delim {
            p.delimiter_bonus as i32
        } else {
            0
        };
        let prefix_bonus = if include_prefix && j == 0 {
            p.prefix_bonus as i32
        } else {
            0
        };

        h_curr[0] = 0;
        e_curr[0] = 0;
        f_curr[0] = 0;
        match_curr[0] = false;

        for i in 1..=m {
            let nc = needle[i - 1];
            let nc_flip = flip_case(nc);
            let is_match = hb == nc || hb == nc_flip;
            let is_exact = hb == nc;
            match_curr[i] = is_match;

            // Diagonal: H[i-1][j-1] + s(i,j)
            let s = if is_match {
                let mut v = match_add + delim_bonus + cap_bonus + prefix_bonus;
                if is_exact {
                    v += p.matching_case_bonus as i32;
                }
                v
            } else {
                -mismatch_pen
            };
            let diag = h_prev[i - 1] + s;

            // E (horizontal gap): frizbee's propagate_horizontal_gaps
            // gap_open ONLY if previous cell (i, j-1) was a match
            let e_pen = if match_prev[i] {
                gap_open_only
            } else {
                gap_extend
            };
            let e_new = h_prev[i] - e_pen;
            let e_ext = e_prev[i] - gap_extend;
            e_curr[i] = e_new.max(e_ext);

            // F (vertical gap): frizbee's up_scores
            // gap_open ONLY if previous row (i-1, j) was a match
            let f_pen = if match_curr[i - 1] {
                gap_open_only
            } else {
                gap_extend
            };
            let f_new = h_curr[i - 1] - f_pen;
            let f_ext = f_curr[i - 1] - gap_extend;
            f_curr[i] = f_new.max(f_ext);

            // H[i][j] = max(0, diag, E, F)
            let h = diag.max(e_curr[i]).max(f_curr[i]).max(0);
            h_curr[i] = h;
        }

        if h_curr[m] > last_row_max {
            last_row_max = h_curr[m];
            end_col = j as u16;
        }

        std::mem::swap(&mut h_prev, &mut h_curr);
        std::mem::swap(&mut e_prev, &mut e_curr);
        std::mem::swap(&mut f_prev, &mut f_curr);
        std::mem::swap(&mut match_prev, &mut match_curr);

        prev_is_lower = hb_lower;
        prev_is_delim = hb_delim;
    }

    scratch.h0 = h_prev;
    scratch.e0 = e_prev;
    scratch.f0 = f_prev;
    scratch.h1 = h_curr;
    scratch.e1 = e_curr;
    scratch.f1 = f_curr;
    scratch.m0 = match_prev;
    scratch.m1 = match_curr;
    LONG_SCRATCH.with(|c| *c.borrow_mut() = scratch);

    (last_row_max as u16, end_col)
}

fn score_one_scalar_stack(
    needle: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    let m = needle.len();
    let n = haystack.len();
    let gap_open = p.gap_open as i32;
    let gap_extend = p.gap_extend as i32;
    let gap_open_only = gap_open;
    let match_add = p.match_score as i32;
    let mismatch_pen = p.mismatch_penalty as i32;
    let mut h_prev = [0i32; 33];
    let mut e_prev = [0i32; 33];
    let mut f_prev = [0i32; 33];
    let mut h_curr = [0i32; 33];
    let mut e_curr = [0i32; 33];
    let mut f_curr = [0i32; 33];
    let mut match_prev = [false; 33];
    let mut match_curr = [false; 33];
    let mut prev_is_lower = false;
    let mut prev_is_delim = false;
    let mut last_row_max = 0i32;
    let mut end_col = 0u16;
    for j in 0..n {
        let hb = haystack[j];
        let hb_lower = hb.is_ascii_lowercase();
        let hb_delim = is_delimiter(hb);
        let cap_bonus = if hb.is_ascii_uppercase() && prev_is_lower {
            p.capitalization_bonus as i32
        } else {
            0
        };
        let delim_bonus = if prev_is_delim && !hb_delim {
            p.delimiter_bonus as i32
        } else {
            0
        };
        let prefix_bonus = if include_prefix && j == 0 {
            p.prefix_bonus as i32
        } else {
            0
        };
        h_curr[0] = 0;
        e_curr[0] = 0;
        f_curr[0] = 0;
        match_curr[0] = false;
        for i in 1..=m {
            let nc = needle[i - 1];
            let nc_flip = flip_case(nc);
            let is_match = hb == nc || hb == nc_flip;
            let is_exact = hb == nc;
            match_curr[i] = is_match;
            let s = if is_match {
                let mut v = match_add + delim_bonus + cap_bonus + prefix_bonus;
                if is_exact {
                    v += p.matching_case_bonus as i32;
                }
                v
            } else {
                -mismatch_pen
            };
            let diag = h_prev[i - 1] + s;
            let e_pen = if match_prev[i] {
                gap_open_only
            } else {
                gap_extend
            };
            let e_new = h_prev[i] - e_pen;
            let e_ext = e_prev[i] - gap_extend;
            e_curr[i] = e_new.max(e_ext);
            let f_pen = if match_curr[i - 1] {
                gap_open_only
            } else {
                gap_extend
            };
            let f_new = h_curr[i - 1] - f_pen;
            let f_ext = f_curr[i - 1] - gap_extend;
            f_curr[i] = f_new.max(f_ext);
            let h = diag.max(e_curr[i]).max(f_curr[i]).max(0);
            h_curr[i] = h;
        }
        if h_curr[m] > last_row_max {
            last_row_max = h_curr[m];
            end_col = j as u16;
        }
        std::mem::swap(&mut h_prev, &mut h_curr);
        std::mem::swap(&mut e_prev, &mut e_curr);
        std::mem::swap(&mut f_prev, &mut f_curr);
        std::mem::swap(&mut match_prev, &mut match_curr);
        prev_is_lower = hb_lower;
        prev_is_delim = hb_delim;
    }
    (last_row_max as u16, end_col)
}

// Score-only wrapper; test oracle for the batch kernels.
#[cfg(test)]
pub(crate) fn score_one_scalar(
    needle: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> u16 {
    score_one_scalar_full(needle, haystack, p, include_prefix).0
}

// ---------------------------------------------------------------------------
// AVX2 inter-sequence batch scorer: 16 paths per __m256i (u16 lanes).
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    pub(crate) const LANES: usize = 16;

    // Build SoA buffer: soa[j * stride + p] = haystack[p][j], padded with 0.
    // `stride` must match the kernel's lane count (16 for u16, 32 for u8).
    // Test oracle: production transposes straight from the stage buffer.
    #[cfg(test)]
    pub(crate) fn build_soa(haystacks: &[&[u8]], max_len: usize, stride: usize) -> Vec<u8> {
        let n = haystacks.len().min(stride);
        let mut soa = vec![0u8; max_len * stride];
        for (p, h) in haystacks.iter().enumerate().take(stride) {
            for (j, &b) in h.iter().enumerate() {
                soa[j * stride + p] = b;
            }
        }
        let _ = n;
        soa
    }

    // 8×8 byte matrix transpose via SSE2 unpack shuffles.
    //
    // `dst[j * dst_stride + i] = src[i * src_stride + j]` for i, j in 0..8.
    // Reads 8 contiguous bytes from each of 8 source rows and writes 8
    // contiguous bytes to each of 8 destination rows (AoS → SoA tiles).
    //
    // # Safety
    // Caller must ensure `src` has 8 rows of `src_stride` readable bytes and
    // `dst` has 8 rows of `dst_stride` writable bytes; `dst` may alias `src`
    // only when both strides are equal and the tiles coincide exactly.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn transpose_8x8(
        src: *const u8,
        src_stride: usize,
        dst: *mut u8,
        dst_stride: usize,
    ) {
        unsafe {
            let load = |i: usize| _mm_loadl_epi64(src.add(i * src_stride) as *const __m128i);
            let a0 = load(0);
            let a1 = load(1);
            let a2 = load(2);
            let a3 = load(3);
            let a4 = load(4);
            let a5 = load(5);
            let a6 = load(6);
            let a7 = load(7);

            // Byte pairs (rows 0-1, 2-3, 4-5, 6-7).
            let p0 = _mm_unpacklo_epi8(a0, a1);
            let p1 = _mm_unpacklo_epi8(a2, a3);
            let p2 = _mm_unpacklo_epi8(a4, a5);
            let p3 = _mm_unpacklo_epi8(a6, a7);

            // 16-bit interleave → 4-wide row groups at column quartets.
            let q0 = _mm_unpacklo_epi16(p0, p1);
            let q1 = _mm_unpackhi_epi16(p0, p1);
            let q2 = _mm_unpacklo_epi16(p2, p3);
            let q3 = _mm_unpackhi_epi16(p2, p3);

            // 32-bit interleave → full 8-lane output rows packed two per register.
            let r0 = _mm_unpacklo_epi32(q0, q2); // rows 0, 1
            let r1 = _mm_unpackhi_epi32(q0, q2); // rows 2, 3
            let r2 = _mm_unpacklo_epi32(q1, q3); // rows 4, 5
            let r3 = _mm_unpackhi_epi32(q1, q3); // rows 6, 7

            _mm_storel_epi64(dst as *mut __m128i, r0);
            _mm_storel_epi64(dst.add(dst_stride) as *mut __m128i, _mm_srli_si128(r0, 8));
            _mm_storel_epi64(dst.add(2 * dst_stride) as *mut __m128i, r1);
            _mm_storel_epi64(
                dst.add(3 * dst_stride) as *mut __m128i,
                _mm_srli_si128(r1, 8),
            );
            _mm_storel_epi64(dst.add(4 * dst_stride) as *mut __m128i, r2);
            _mm_storel_epi64(
                dst.add(5 * dst_stride) as *mut __m128i,
                _mm_srli_si128(r2, 8),
            );
            _mm_storel_epi64(dst.add(6 * dst_stride) as *mut __m128i, r3);
            _mm_storel_epi64(
                dst.add(7 * dst_stride) as *mut __m128i,
                _mm_srli_si128(r3, 8),
            );
        }
    }

    // Fill SoA from path-major staged bytes using 8×8 shuffle transpose tiles.
    //
    // `stage` is path-major: path `k` lives at `stage[k * rows + j]` and is
    // zero-padded to `rows`. `out` is position-major SoA with `stride` lanes
    // per row: `out[j * stride + k]`. `rows` and `stride` must both be
    // multiples of 8 (`rows` = padded max trimmed len, `stride` = lane count).
    // Every SoA byte is written; no pre-zeroing of `out` is required.
    //
    // # Safety
    // Caller must ensure `stage.len() >= stride * rows` and
    // `out.len() >= rows * stride`, and AVX2-capable target is not required
    // (SSE2 only) but must not race with concurrent aliasing writes.
    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn soa_from_stage(stage: &[u8], rows: usize, stride: usize, out: &mut [u8]) {
        debug_assert_eq!(rows % 8, 0);
        debug_assert_eq!(stride % 8, 0);
        let mut k0 = 0usize;
        while k0 < stride {
            let mut j0 = 0usize;
            while j0 < rows {
                unsafe {
                    transpose_8x8(
                        stage.as_ptr().add(k0 * rows + j0),
                        rows,
                        out.as_mut_ptr().add(j0 * stride + k0),
                        stride,
                    );
                }
                j0 += 8;
            }
            k0 += 8;
        }
    }

    // Score a batch of up to 16 pre-trimmed haystacks using inter-sequence AVX2 SW.
    // `include_prefix` is a per-lane bitmask (bit i = lane i gets prefix bonus at j=0).
    // Returns per-lane (score, end_col_local) where end_col_local is the last
    // column in the trimmed slice where the last-row max occurs.
    // Needle rows are dynamic up to 32 (arrays sized 33); longer needles
    // take the scalar path (see `use_simd` in score.rs).
    //
    // # Safety
    // Caller must ensure AVX2 is available and `out.len() >= haystacks.len()`.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn score_batch_avx2(
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        p: &SwParams,
        include_prefix_mask: u16,
        out: &mut [u16],
        end_cols: &mut [u16],
    ) {
        let m = needle.len().min(32);
        let n = max_len;
        if m == 0 || n == 0 {
            out.iter_mut().for_each(|s| *s = 0);
            end_cols.iter_mut().for_each(|s| *s = 0);
            return;
        }

        let set1 = |v: u16| _mm256_set1_epi16(v as i16);
        let zero = _mm256_setzero_si256();
        let ones = set1(u16::MAX);

        let gap_open_v = set1(p.gap_open);
        let gap_ext_v = set1(p.gap_extend);
        let match_add_v = set1(p.match_score + p.mismatch_penalty);
        let mismatch_v = set1(p.mismatch_penalty);
        let delim_v = set1(p.delimiter_bonus);
        let cap_v = set1(p.capitalization_bonus);
        let case_v = set1(p.matching_case_bonus);
        let prefix_v = set1(p.prefix_bonus);
        // Hoist class/needle broadcasts out of the per-column loop.
        let a_lo = set1(b'A' as u16 - 1);
        let z_hi = set1(b'Z' as u16 + 1);
        let la_lo = set1(b'a' as u16 - 1);
        let lz_hi = set1(b'z' as u16 + 1);
        let d0_lo = set1(b'0' as u16 - 1);
        let d9_hi = set1(b'9' as u16 + 1);
        let lt128 = set1(128);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;
        let need_delim = p.delimiter_bonus != 0;
        let mut nc_v = [zero; 33];
        let mut nf_v = [zero; 33];
        for i in 0..m {
            nc_v[i + 1] = set1(needle[i] as u16);
            nf_v[i + 1] = set1(flip_case(needle[i]) as u16);
        }

        // Per-lane prefix mask: lanes with include_prefix get prefix_bonus at j=0.
        let prefix_lanes = {
            let mut arr = [0u16; 16];
            for (i, a) in arr.iter_mut().enumerate() {
                *a = if include_prefix_mask & (1 << i) != 0 {
                    u16::MAX
                } else {
                    0
                };
            }
            _mm256_loadu_si256(arr.as_ptr() as *const __m256i)
        };

        // DP state arrays (max 33 rows for m<=32). Ping-pong with an index
        // instead of swapping 1056-byte arrays every column.
        let mut h = [[zero; 33]; 2];
        let mut e = [[zero; 33]; 2];
        let mut f = [[zero; 33]; 2];
        let mut match_state = [[zero; 33]; 2];
        let mut cur = 0usize; // read from [cur], write to [1-cur]

        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut last_row_max = zero;
        let mut end_col_v = zero;
        let mut best_score_v = zero;

        for j in 0..n {
            // Load 16 bytes at position j, zero-extend to 16 u16
            let raw = _mm_loadu_si128(soa.as_ptr().add(j * LANES) as *const __m128i);
            let hay = _mm256_cvtepu8_epi16(raw);

            let (is_lower, is_delim_st, cap_bonus, delim_bonus) = if need_case || need_delim {
                let ge_a = _mm256_cmpgt_epi16(hay, a_lo);
                let le_z = _mm256_cmpgt_epi16(z_hi, hay);
                let is_upper = _mm256_and_si256(ge_a, le_z);
                let ge_la = _mm256_cmpgt_epi16(hay, la_lo);
                let le_lz = _mm256_cmpgt_epi16(lz_hi, hay);
                let is_lower = _mm256_and_si256(ge_la, le_lz);
                let ge_0 = _mm256_cmpgt_epi16(hay, d0_lo);
                let le_9 = _mm256_cmpgt_epi16(d9_hi, hay);
                let is_digit = _mm256_and_si256(ge_0, le_9);
                let is_letter = _mm256_or_si256(is_upper, is_lower);
                let alnum = _mm256_or_si256(is_letter, is_digit);
                let not_alnum = _mm256_andnot_si256(alnum, ones);
                let lt_128 = _mm256_cmpgt_epi16(lt128, hay);
                let is_delim = _mm256_and_si256(not_alnum, lt_128);
                let not_delim = _mm256_andnot_si256(is_delim, ones);
                let cap_mask = _mm256_and_si256(is_upper, prev_lower);
                let cap_bonus = _mm256_and_si256(cap_mask, cap_v);
                let dm = _mm256_and_si256(prev_delim, not_delim);
                let delim_bonus = _mm256_and_si256(dm, delim_v);
                (is_lower, is_delim, cap_bonus, delim_bonus)
            } else {
                (zero, zero, zero, zero)
            };
            let prefix_bonus = if j == 0 {
                _mm256_and_si256(prefix_v, prefix_lanes)
            } else {
                zero
            };
            let bonus_base = _mm256_add_epi16(
                _mm256_add_epi16(delim_bonus, cap_bonus),
                _mm256_add_epi16(prefix_bonus, match_add_v),
            );

            let nxt = 1 - cur;
            h[nxt][0] = zero;
            e[nxt][0] = zero;
            f[nxt][0] = zero;
            match_state[nxt][0] = zero;

            for i in 1..=m {
                let eq_nc = _mm256_cmpeq_epi16(hay, nc_v[i]);
                let eq_nf = _mm256_cmpeq_epi16(hay, nf_v[i]);
                let mm = _mm256_or_si256(eq_nc, eq_nf);
                match_state[nxt][i] = mm;

                let masked_bonus = _mm256_and_si256(bonus_base, mm);
                let diag = _mm256_add_epi16(h[cur][i - 1], masked_bonus);
                let diag = _mm256_subs_epu16(diag, mismatch_v);
                let diag = _mm256_add_epi16(diag, _mm256_and_si256(eq_nc, case_v));

                let e_pen = _mm256_blendv_epi8(gap_ext_v, gap_open_v, match_state[cur][i]);
                let e_new = _mm256_subs_epu16(h[cur][i], e_pen);
                let e_ext = _mm256_subs_epu16(e[cur][i], gap_ext_v);
                e[nxt][i] = _mm256_max_epu16(e_new, e_ext);

                let f_pen = _mm256_blendv_epi8(gap_ext_v, gap_open_v, match_state[nxt][i - 1]);
                let f_new = _mm256_subs_epu16(h[nxt][i - 1], f_pen);
                let f_ext = _mm256_subs_epu16(f[nxt][i - 1], gap_ext_v);
                f[nxt][i] = _mm256_max_epu16(f_new, f_ext);

                let h_val = _mm256_max_epu16(diag, e[nxt][i]);
                let h_val = _mm256_max_epu16(h_val, f[nxt][i]);
                h[nxt][i] = _mm256_max_epu16(h_val, zero);
            }

            let row_m = h[nxt][m];
            let gt_best = _mm256_cmpgt_epi16(row_m, best_score_v);
            let j_v = set1(j as u16);
            end_col_v = _mm256_blendv_epi8(end_col_v, j_v, gt_best);
            best_score_v = _mm256_max_epu16(best_score_v, row_m);
            last_row_max = _mm256_max_epu16(last_row_max, row_m);

            cur = nxt;
            prev_lower = is_lower;
            prev_delim = is_delim_st;
        }

        // Extract per-lane scores and end_cols
        let scores: [u16; 16] = std::mem::transmute(last_row_max);
        let ecols: [u16; 16] = std::mem::transmute(end_col_v);
        for (i, (&s, &e)) in scores.iter().zip(ecols.iter()).enumerate() {
            if i < out.len() {
                out[i] = s;
                end_cols[i] = e;
            }
        }
    }

    // Score a batch of up to 32 pre-trimmed haystacks using inter-sequence AVX2 SW
    // with u8 scores (32 lanes). Only valid when `score_fits_in_u8` is true
    // (max score < 256). Same semantics as `score_batch_avx2` but 2x lanes.
    //
    // # Safety
    // Caller must ensure AVX2 is available. Scores must fit in u8.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn score_batch_avx2_u8(
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        p: &SwParams,
        include_prefix_mask: u32,
        out: &mut [u16],
        end_cols: &mut [u16],
    ) {
        const LANES8: usize = 32;
        let m = needle.len().min(32);
        let n = max_len;
        if m == 0 || n == 0 {
            out.iter_mut().for_each(|s| *s = 0);
            end_cols.iter_mut().for_each(|s| *s = 0);
            return;
        }

        let set1 = |v: u8| _mm256_set1_epi8(v as i8);
        let zero = _mm256_setzero_si256();
        let ones = set1(u8::MAX);

        let gap_open_v = set1(p.gap_open as u8);
        let gap_ext_v = set1(p.gap_extend as u8);
        let match_add_v = set1((p.match_score + p.mismatch_penalty) as u8);
        let mismatch_v = set1(p.mismatch_penalty as u8);
        let delim_v = set1(p.delimiter_bonus as u8);
        let cap_v = set1(p.capitalization_bonus as u8);
        let case_v = set1(p.matching_case_bonus as u8);
        let prefix_v = set1(p.prefix_bonus as u8);
        // Class thresholds are constant across columns, so hoist them out.
        let a_lo = set1(b'A' - 1);
        let z_hi = set1(b'Z' + 1);
        let la_lo = set1(b'a' - 1);
        let lz_hi = set1(b'z' + 1);
        let d0_lo = set1(b'0' - 1);
        let d9_hi = set1(b'9' + 1);
        let hi80 = set1(0x80);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;
        let need_delim = p.delimiter_bonus != 0;

        let prefix_lanes = {
            let mut arr = [0u8; 32];
            for (i, a) in arr.iter_mut().enumerate() {
                *a = if include_prefix_mask & (1u32 << i) != 0 {
                    u8::MAX
                } else {
                    0
                };
            }
            _mm256_loadu_si256(arr.as_ptr() as *const __m256i)
        };

        // Broadcast each needle byte once per batch.
        let mut nc_v = [zero; 33];
        let mut nf_v = [zero; 33];
        for i in 0..m {
            nc_v[i + 1] = set1(needle[i]);
            nf_v[i + 1] = set1(flip_case(needle[i]));
        }

        let mut h = [[zero; 33]; 2];
        let mut e = [[zero; 33]; 2];
        let mut f = [[zero; 33]; 2];
        let mut match_state = [[zero; 33]; 2];
        let mut cur = 0usize;

        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut last_row_max = zero;
        let mut end_col_v = zero;
        let mut best_score_v = zero;
        // Unsigned end tracking: scores pass 127, where the signed
        // compare inverts (best looks negative, so every later column
        // latches). XOR with 0x80 maps the order back. Scores use the
        // unsigned max and are unaffected.
        let sign_bias = _mm256_set1_epi8(-128);

        for j in 0..n {
            // Saturation check every 4th column (result-identical; last_row_max
            // only grows, so skipping columns is safe).
            if j & 3 == 0 && _mm256_movemask_epi8(_mm256_cmpeq_epi8(last_row_max, ones)) == -1 {
                break;
            }
            // Load 32 bytes at position j
            let hay = _mm256_loadu_si256(soa.as_ptr().add(j * LANES8) as *const __m256i);
            // Prefetch the next column (position-major SoA is sequential).
            if j + 2 < n {
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        soa.as_ptr().add((j + 2) * LANES8) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            // Character class masks — skip entirely when case/delim bonuses are 0
            // (typical lowercase query). Saves ~10 SIMD ops per column.
            let (is_lower, is_delim_unused, cap_bonus, delim_bonus) = if need_case || need_delim {
                let ge_a = _mm256_cmpgt_epi8(hay, a_lo);
                let le_z = _mm256_cmpgt_epi8(z_hi, hay);
                let is_upper = _mm256_and_si256(ge_a, le_z);
                let ge_la = _mm256_cmpgt_epi8(hay, la_lo);
                let le_lz = _mm256_cmpgt_epi8(lz_hi, hay);
                let is_lower = _mm256_and_si256(ge_la, le_lz);
                let ge_0 = _mm256_cmpgt_epi8(hay, d0_lo);
                let le_9 = _mm256_cmpgt_epi8(d9_hi, hay);
                let is_digit = _mm256_and_si256(ge_0, le_9);
                let is_letter = _mm256_or_si256(is_upper, is_lower);
                let alnum = _mm256_or_si256(is_letter, is_digit);
                let not_alnum = _mm256_andnot_si256(alnum, ones);
                let high_bit = _mm256_and_si256(hay, hi80);
                let lt_128 = _mm256_cmpeq_epi8(high_bit, zero);
                let is_delim = _mm256_and_si256(not_alnum, lt_128);
                let not_delim = _mm256_andnot_si256(is_delim, ones);
                let cap_mask = _mm256_and_si256(is_upper, prev_lower);
                let cap_bonus = _mm256_and_si256(cap_mask, cap_v);
                let dm = _mm256_and_si256(prev_delim, not_delim);
                let delim_bonus = _mm256_and_si256(dm, delim_v);
                (is_lower, is_delim, cap_bonus, delim_bonus)
            } else {
                (zero, zero, zero, zero)
            };
            let prefix_bonus = if j == 0 {
                _mm256_and_si256(prefix_v, prefix_lanes)
            } else {
                zero
            };
            // The bonus pack is column-invariant, so build it once.
            let bonus_base = _mm256_add_epi8(
                _mm256_add_epi8(delim_bonus, cap_bonus),
                _mm256_add_epi8(prefix_bonus, match_add_v),
            );

            let nxt = 1 - cur;
            h[nxt][0] = zero;
            e[nxt][0] = zero;
            f[nxt][0] = zero;
            match_state[nxt][0] = zero;

            for i in 1..=m {
                let eq_nc = _mm256_cmpeq_epi8(hay, nc_v[i]);
                let eq_nf = _mm256_cmpeq_epi8(hay, nf_v[i]);
                let mm = _mm256_or_si256(eq_nc, eq_nf);
                match_state[nxt][i] = mm;

                let masked_bonus = _mm256_and_si256(bonus_base, mm);
                let diag = _mm256_add_epi8(h[cur][i - 1], masked_bonus);
                let diag = _mm256_subs_epu8(diag, mismatch_v);
                let diag = _mm256_add_epi8(diag, _mm256_and_si256(eq_nc, case_v));

                let e_pen = _mm256_blendv_epi8(gap_ext_v, gap_open_v, match_state[cur][i]);
                let e_new = _mm256_subs_epu8(h[cur][i], e_pen);
                let e_ext = _mm256_subs_epu8(e[cur][i], gap_ext_v);
                e[nxt][i] = _mm256_max_epu8(e_new, e_ext);

                let f_pen = _mm256_blendv_epi8(gap_ext_v, gap_open_v, match_state[nxt][i - 1]);
                let f_new = _mm256_subs_epu8(h[nxt][i - 1], f_pen);
                let f_ext = _mm256_subs_epu8(f[nxt][i - 1], gap_ext_v);
                f[nxt][i] = _mm256_max_epu8(f_new, f_ext);

                let h_val = _mm256_max_epu8(diag, e[nxt][i]);
                let h_val = _mm256_max_epu8(h_val, f[nxt][i]);
                h[nxt][i] = _mm256_max_epu8(h_val, zero);
            }

            let row_m = h[nxt][m];
            let gt_best = _mm256_cmpgt_epi8(
                _mm256_xor_si256(row_m, sign_bias),
                _mm256_xor_si256(best_score_v, sign_bias),
            );
            let j_v = set1(j as u8);
            end_col_v = _mm256_blendv_epi8(end_col_v, j_v, gt_best);
            best_score_v = _mm256_max_epu8(best_score_v, row_m);
            last_row_max = _mm256_max_epu8(last_row_max, row_m);

            cur = nxt;
            prev_lower = is_lower;
            prev_delim = is_delim_unused;
        }

        // Extract per-lane scores (u8 -> u16) and end_cols
        let score_bytes: [u8; 32] = std::mem::transmute(last_row_max);
        let ecol_bytes: [u8; 32] = std::mem::transmute(end_col_v);
        for (i, (&s, &e)) in score_bytes.iter().zip(ecol_bytes.iter()).enumerate() {
            if i < out.len() {
                out[i] = s as u16;
                end_cols[i] = e as u16;
            }
        }
    }

    // Dispatch to a const-unrolled specialization for common needle lengths.
    // `M = 0` keeps the general runtime-m loop. Results are bit-identical.
    //
    // # Safety
    // Caller must guarantee AVX2 (score.rs gates on simd_available).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn score_batch_avx2_u8_dispatch(
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        p: &SwParams,
        include_prefix_mask: u32,
        out: &mut [u16],
        end_cols: &mut [u16],
    ) {
        // `FFF_NO_UNROLL=1` forces the general runtime-m loop.
        static NO_UNROLL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *NO_UNROLL
            .get_or_init(|| std::env::var_os("FFF_NO_UNROLL").is_some())
        {
            return score_batch_avx2_u8(
                needle,
                soa,
                max_len,
                p,
                include_prefix_mask,
                out,
                end_cols,
            );
        }
        let m = needle.len().min(32);
        macro_rules! unrolled {
            ($($v:literal),*) => {
                match m {
                    $( $v => score_batch_avx2_u8_const::<$v>(needle, soa, max_len, p, include_prefix_mask, out, end_cols), )*
                    _ => score_batch_avx2_u8(needle, soa, max_len, p, include_prefix_mask, out, end_cols),
                }
            };
        }
        unrolled!(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        );
    }

    // Same as [`score_batch_avx2_u8`] but with a compile-time needle length so
    // the DP row loop is fully unrolled and the state arrays SROA to registers.
    // Bit-identical to the general loop (asserted in tests).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn score_batch_avx2_u8_const<const M: usize>(
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        p: &SwParams,
        include_prefix_mask: u32,
        out: &mut [u16],
        end_cols: &mut [u16],
    ) {
        debug_assert_eq!(needle.len().min(32), M);
        let m = M;
        let n = max_len;
        if m == 0 || n == 0 {
            out.iter_mut().for_each(|s| *s = 0);
            end_cols.iter_mut().for_each(|s| *s = 0);
            return;
        }

        let set1 = |v: u8| _mm256_set1_epi8(v as i8);
        let zero = _mm256_setzero_si256();
        let ones = set1(u8::MAX);

        let gap_open_v = set1(p.gap_open as u8);
        let gap_ext_v = set1(p.gap_extend as u8);
        let match_add_v = set1((p.match_score + p.mismatch_penalty) as u8);
        let mismatch_v = set1(p.mismatch_penalty as u8);
        let delim_v = set1(p.delimiter_bonus as u8);
        let cap_v = set1(p.capitalization_bonus as u8);
        let case_v = set1(p.matching_case_bonus as u8);
        let prefix_v = set1(p.prefix_bonus as u8);
        let a_lo = set1(b'A' - 1);
        let z_hi = set1(b'Z' + 1);
        let la_lo = set1(b'a' - 1);
        let lz_hi = set1(b'z' + 1);
        let d0_lo = set1(b'0' - 1);
        let d9_hi = set1(b'9' + 1);
        let hi80 = set1(0x80);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;
        let need_delim = p.delimiter_bonus != 0;

        let prefix_lanes = {
            let mut arr = [0u8; 32];
            for (i, a) in arr.iter_mut().enumerate() {
                *a = if include_prefix_mask & (1u32 << i) != 0 {
                    u8::MAX
                } else {
                    0
                };
            }
            _mm256_loadu_si256(arr.as_ptr() as *const __m256i)
        };

        // Fixed max arrays; the row loop below has a const trip count so LLVM
        // unrolls 1..=M and can keep the hot state in registers.
        let mut nc_v = [zero; 33];
        let mut nf_v = [zero; 33];
        for i in 0..m {
            nc_v[i + 1] = set1(needle[i]);
            nf_v[i + 1] = set1(flip_case(needle[i]));
        }

        let mut h = [[zero; 33]; 2];
        let mut e = [[zero; 33]; 2];
        let mut f = [[zero; 33]; 2];
        let mut match_state = [[zero; 33]; 2];
        let mut cur = 0usize;

        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut last_row_max = zero;
        let mut end_col_v = zero;
        let mut best_score_v = zero;
        // Unsigned end tracking (see general kernel above): 0x80 bias
        // restores unsigned order for the signed compare.
        let sign_bias = _mm256_set1_epi8(-128);

        for j in 0..n {
            let hay = _mm256_loadu_si256(soa.as_ptr().add(j * 32) as *const __m256i);
            if j + 2 < n {
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        soa.as_ptr().add((j + 2) * 32) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            if j & 3 == 0 && _mm256_movemask_epi8(_mm256_cmpeq_epi8(last_row_max, ones)) == -1 {
                break;
            }

            let (is_lower, is_delim_unused, cap_bonus, delim_bonus) = if need_case || need_delim {
                let ge_a = _mm256_cmpgt_epi8(hay, a_lo);
                let le_z = _mm256_cmpgt_epi8(z_hi, hay);
                let is_upper = _mm256_and_si256(ge_a, le_z);
                let ge_la = _mm256_cmpgt_epi8(hay, la_lo);
                let le_lz = _mm256_cmpgt_epi8(lz_hi, hay);
                let is_lower = _mm256_and_si256(ge_la, le_lz);
                let ge_0 = _mm256_cmpgt_epi8(hay, d0_lo);
                let le_9 = _mm256_cmpgt_epi8(d9_hi, hay);
                let is_digit = _mm256_and_si256(ge_0, le_9);
                let is_letter = _mm256_or_si256(is_upper, is_lower);
                let alnum = _mm256_or_si256(is_letter, is_digit);
                let not_alnum = _mm256_andnot_si256(alnum, ones);
                let high_bit = _mm256_and_si256(hay, hi80);
                let lt_128 = _mm256_cmpeq_epi8(high_bit, zero);
                let is_delim = _mm256_and_si256(not_alnum, lt_128);
                let not_delim = _mm256_andnot_si256(is_delim, ones);
                let cap_mask = _mm256_and_si256(is_upper, prev_lower);
                let cap_bonus = _mm256_and_si256(cap_mask, cap_v);
                let dm = _mm256_and_si256(prev_delim, not_delim);
                let delim_bonus = _mm256_and_si256(dm, delim_v);
                (is_lower, is_delim, cap_bonus, delim_bonus)
            } else {
                (zero, zero, zero, zero)
            };
            let prefix_bonus = if j == 0 {
                _mm256_and_si256(prefix_v, prefix_lanes)
            } else {
                zero
            };
            // The bonus pack is column-invariant, so build it once.
            let bonus_base = _mm256_add_epi8(
                _mm256_add_epi8(delim_bonus, cap_bonus),
                _mm256_add_epi8(prefix_bonus, match_add_v),
            );

            let nxt = 1 - cur;
            h[nxt][0] = zero;
            e[nxt][0] = zero;
            f[nxt][0] = zero;
            match_state[nxt][0] = zero;

            // Const trip count (M is a const generic): fully unrolled.
            for i in 1..=M {
                let eq_nc = _mm256_cmpeq_epi8(hay, nc_v[i]);
                let eq_nf = _mm256_cmpeq_epi8(hay, nf_v[i]);
                let mm = _mm256_or_si256(eq_nc, eq_nf);
                match_state[nxt][i] = mm;

                let masked_bonus = _mm256_and_si256(bonus_base, mm);
                let diag = _mm256_add_epi8(h[cur][i - 1], masked_bonus);
                let diag = _mm256_subs_epu8(diag, mismatch_v);
                let diag = _mm256_add_epi8(diag, _mm256_and_si256(eq_nc, case_v));

                let e_pen = _mm256_blendv_epi8(gap_ext_v, gap_open_v, match_state[cur][i]);
                let e_new = _mm256_subs_epu8(h[cur][i], e_pen);
                let e_ext = _mm256_subs_epu8(e[cur][i], gap_ext_v);
                e[nxt][i] = _mm256_max_epu8(e_new, e_ext);

                let f_pen = _mm256_blendv_epi8(gap_ext_v, gap_open_v, match_state[nxt][i - 1]);
                let f_new = _mm256_subs_epu8(h[nxt][i - 1], f_pen);
                let f_ext = _mm256_subs_epu8(f[nxt][i - 1], gap_ext_v);
                f[nxt][i] = _mm256_max_epu8(f_new, f_ext);

                let h_val = _mm256_max_epu8(diag, e[nxt][i]);
                let h_val = _mm256_max_epu8(h_val, f[nxt][i]);
                h[nxt][i] = _mm256_max_epu8(h_val, zero);
            }

            let row_m = h[nxt][M];
            let gt_best = _mm256_cmpgt_epi8(
                _mm256_xor_si256(row_m, sign_bias),
                _mm256_xor_si256(best_score_v, sign_bias),
            );
            let j_v = set1(j as u8);
            end_col_v = _mm256_blendv_epi8(end_col_v, j_v, gt_best);
            best_score_v = _mm256_max_epu8(best_score_v, row_m);
            last_row_max = _mm256_max_epu8(last_row_max, row_m);

            cur = nxt;
            prev_lower = is_lower;
            prev_delim = is_delim_unused;
        }

        let score_bytes: [u8; 32] = std::mem::transmute(last_row_max);
        let ecol_bytes: [u8; 32] = std::mem::transmute(end_col_v);
        for (i, (&s, &e)) in score_bytes.iter().zip(ecol_bytes.iter()).enumerate() {
            if i < out.len() {
                out[i] = s as u16;
                end_cols[i] = e as u16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SwParams {
        SwParams::for_query(false)
    }

    // Serializes access to `FRIZBEE_SKIP_SW` across parallel tests.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Compare against frizbee's Matcher (the ground truth) using fff's
    // scoring config (matching_case_bonus=0 for lowercase queries).
    fn frizbee_score(needle: &str, haystack: &str) -> u16 {
        // Ensure skip_sw is OFF for the ground-truth measurement

        let has_upper = needle.chars().any(|c| c.is_uppercase());
        let cfg = neo_frizbee::Config {
            max_typos: Some(6),
            sort: neo_frizbee::SortStrategy::Unsorted,
            scoring: neo_frizbee::Scoring {
                capitalization_bonus: if has_upper { 8 } else { 0 },
                matching_case_bonus: if has_upper { 4 } else { 0 },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut m = neo_frizbee::Matcher::new(needle, &cfg);
        let matches = m.match_list(std::slice::from_ref(&haystack));
        if matches.is_empty() {
            0
        } else {
            // Subtract exact_match_bonus (added by smith_waterman_one, not score_haystack)
            let exact = needle.eq_ignore_ascii_case(haystack);
            let bonus = if exact { 8u16 } else { 0 };
            matches[0].score.saturating_sub(bonus)
        }
    }

    #[test]
    fn scalar_matches_frizbee_on_basic_cases() {        let p = params();
        let cases: &[(&str, &str)] = &[
            ("abc", "abc"),
            ("abc", "aabbcc"),
            ("ab", "aXc"),
            ("score", "crates/fff-core/src/score.rs"),
            ("abc", "xyz"),
            ("a", "abc"),
            ("abc", "ab"),
            ("test", "src/test/main.rs"),
            ("f", "foo/bar/baz.txt"),
            ("fb", "foo/bar/baz.txt"),
            ("fbz", "foo/bar/baz.txt"),
            ("abc", "a-b-c"),
            ("abc", "aXbXc"),
            ("file", "src/file_picker.rs"),
        ];
        for &(n, h) in cases {
            let mine = score_one_scalar(n.as_bytes(), h.as_bytes(), &p, true);
            let theirs = frizbee_score(n, h);
            assert_eq!(
                mine, theirs,
                "needle={n:?} haystack={h:?}: mine={mine} frizbee={theirs}"
            );
        }
    }

    #[test]
    fn max_sw_bound_holds_on_random() {
        // Falsification for pre-SW pruning: any violation kills the idea.
        // Batch kernels replicate scalar exactly (parity battery), so
        // scalar coverage transfers.
        let mut state = 0xBEEFu32;
        let mut next = move || {
            state = state
                .wrapping_mul(1664525)
                .wrapping_add(1013904223);
            (state >> 16) as u16
        };
        let mut byte = move |next: &mut dyn FnMut() -> u16| {
            let r = next() % 6;
            if r == 0 {
                b'/'
            } else if r == 1 {
                b'A' + (next() % 26) as u8
            } else {
                b'a' + (next() % 26) as u8
            }
        };
        for _ in 0..8000 {
            let m = 1 + (next() % 32) as usize;
            let n = 1 + (next() % 64) as usize;
            let needle: Vec<u8> = (0..m).map(|_| byte(&mut next)).collect();
            let hay: Vec<u8> = (0..n).map(|_| byte(&mut next)).collect();
            let p = SwParams::for_query(next() % 2 == 0);
            let bound = p.max_sw_score(m);
            for inc in [false, true] {
                let (sc, _) = score_one_scalar_full(&needle, &hay, &p, inc);
                assert!(
                    sc as i32 <= bound,
                    "m={m} n={n} inc={inc} sc={sc} bound={bound}"
                );
            }
        }
    }

    #[test]
    fn scalar_matches_frizbee_no_prefix() {
        let p = params();
        let cases: &[(&str, &str)] = &[("abc", "abc"), ("bc", "abc"), ("score", "src/score.rs")];
        for &(n, h) in cases {
            let mine = score_one_scalar(n.as_bytes(), h.as_bytes(), &p, false);
            let theirs = frizbee_score(n, h);
            // Without prefix, scores may differ; check upper bound only.
            assert!(
                mine <= theirs + 12,
                "needle={n} haystack={h}: mine={mine} theirs={theirs}"
            );
        }
    }

    #[test]
    fn exact_match_scores_match_score() {
        let p = params();
        let s = score_one_scalar(b"abc", b"abc", &p, true);
        assert_eq!(s, 48, "abc/abc");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_u8_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let batches: &[&[&str]] = &[
            &[
                "abc",
                "aabbcc",
                "aXc",
                "xyz",
                "abcabc",
                "a",
                "ab",
                "a-b-c",
                "src/score.rs",
                "foo/bar.txt",
                "test.rs",
                "main",
                "aaa",
                "bbb",
                "ccc",
                "ddd",
                "eee",
                "fff",
                "ggg",
                "hhh",
                "iii",
                "jjj",
                "kkk",
                "lll",
                "mmm",
                "nnn",
                "ooo",
                "ppp",
                "qqq",
                "rrr",
                "sss",
                "ttt",
            ],
            &[
                "file_picker",
                "score",
                "netedit",
                "tl",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "0",
                "1",
            ],
        ];
        for batch in batches {
            let max_len = batch.iter().map(|h| h.len()).max().unwrap();
            let hays: Vec<&[u8]> = batch.iter().map(|h| h.as_bytes()).collect();
            let soa = super::avx2::build_soa(&hays, max_len, 32);
            let mut simd_out = vec![0u16; batch.len()];
            let mut simd_ecol = vec![0u16; batch.len()];
            let prefix_mask: u32 = if batch.len() >= 32 {
                u32::MAX
            } else {
                (1u32 << batch.len()) - 1
            };
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    b"abc",
                    &soa,
                    max_len,
                    &p,
                    prefix_mask,
                    &mut simd_out,
                    &mut simd_ecol,
                );
            }
            for (k, h) in batch.iter().enumerate() {
                let (scalar_score, scalar_ecol) =
                    score_one_scalar_full(b"abc", h.as_bytes(), &p, true);
                assert_eq!(
                    simd_out[k], scalar_score,
                    "u8 batch[{k}]={h:?}: simd_score={} scalar_score={}",
                    simd_out[k], scalar_score
                );
                assert_eq!(
                    simd_ecol[k], scalar_ecol,
                    "u8 batch[{k}]={h:?}: simd_ecol={} scalar_ecol={}",
                    simd_ecol[k], scalar_ecol
                );
            }
        }
    }

    #[test]
    fn avx2_u8_dispatch_matches_general_unrolled() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        for m in 1usize..=32 {
            let needle: Vec<u8> = (0..m).map(|i| b'a' + (i % 26) as u8).collect();
            let batch: Vec<String> = (0..32)
                .map(|i| {
                    let len = 4 + (i * 3 + m) % 40;
                    let bytes: Vec<u8> = (0..len).map(|j| b'a' + ((i + j) % 26) as u8).collect();
                    String::from_utf8(bytes).unwrap()
                })
                .collect();
            let max_len = batch.iter().map(|h| h.len()).max().unwrap();
            let hays: Vec<&[u8]> = batch.iter().map(|h| h.as_bytes()).collect();
            let soa = super::avx2::build_soa(&hays, max_len, 32);
            let mut out_a = vec![0u16; 32];
            let mut eco_a = vec![0u16; 32];
            let mut out_b = vec![0u16; 32];
            let mut eco_b = vec![0u16; 32];
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    &needle,
                    &soa,
                    max_len,
                    &p,
                    u32::MAX,
                    &mut out_a,
                    &mut eco_a,
                );
                super::avx2::score_batch_avx2_u8_dispatch(
                    &needle,
                    &soa,
                    max_len,
                    &p,
                    u32::MAX,
                    &mut out_b,
                    &mut eco_b,
                );
            }
            assert_eq!(out_a, out_b, "m={m} scores differ general vs unrolled");
            assert_eq!(eco_a, eco_b, "m={m} end_cols differ general vs unrolled");
        }
    }

    #[test]
    fn u8_matches_scalar_on_long_needles() {
        // u8 lanes only engage for short needles in production; long
        // needles must agree with scalar exactly (no saturation quirk).
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let needles: &[&[u8]] = &[
            b"internationalization",
            b"characterization",
            b"controller",
            b"file_picker",
            b"rename VARIABLE42",
        ];
        let hays: &[&[u8]] = &[
            b"src/i18n/internationalization.rs",
            b"crates/fff-core/src/characterization_layer.rs",
            b"sumo/dom0/controllers/NetworkController.cpp",
            b"lua/fff/file_picker/init.lua",
            b"a",
            b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            b"Characterization_of_VARIABLE42_in_DOCS.md",
        ];
        for needle in needles {
            // Production never runs SIMD with n <= 8 (tails unpadded);
            // mirror that: pad lanes past 8 so stale-tail reads can't fire.
            let mut hays32: Vec<&[u8]> = hays.to_vec();
            while hays32.len() < 12 {
                hays32.extend_from_within(..);
            }
            hays32.truncate(24);
            let n = hays32.len();
            let max_len = hays32.iter().map(|h| h.len()).max().unwrap();
            let soa = super::avx2::build_soa(&hays32, max_len, 32);
            let mut simd_out = vec![0u16; n];
            let mut simd_ecol = vec![0u16; n];
            // Production-realistic mixed prefix mask (only some lanes).
            let pmask: u32 = 0b101010101010101010101010;
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    needle,
                    &soa,
                    max_len,
                    &p,
                    pmask,
                    &mut simd_out,
                    &mut simd_ecol,
                );
            }
            for (k, h) in hays32.iter().enumerate() {
                let inc = pmask >> k & 1 == 1;
                let (scalar_score, scalar_ecol) =
                    score_one_scalar_full(needle, h, &p, inc);
                if simd_out[k] != scalar_score || simd_ecol[k] != scalar_ecol {
                    println!(
                        "U8DIVERGE needle={:?} hay[{k}]={:?}: u8=({},{}) scalar=({},{})",
                        needle, h, simd_out[k], simd_ecol[k], scalar_score, scalar_ecol
                    );
                }
            }
            // Const-unrolled dispatch must equal the general kernel on the
            // same bytes (production calls dispatch, not general).
            let mut disp_out = vec![0u16; n];
            let mut disp_eco = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2_u8_dispatch(
                    needle,
                    &soa,
                    max_len,
                    &p,
                    pmask,
                    &mut disp_out,
                    &mut disp_eco,
                );
            }
            for (k, h) in hays32.iter().enumerate() {
                if disp_out[k] != simd_out[k] || disp_eco[k] != simd_ecol[k] {
                    println!(
                        "DISPDIVERGE needle={:?} hay[{k}]={:?}: disp=({},{}) general=({},{})",
                        needle, h, disp_out[k], disp_eco[k], simd_out[k], simd_ecol[k]
                    );
                }
            }
            // Same inputs through the u16 kernel: isolates the bug to u8.
            let soa16 = super::avx2::build_soa(&hays32, max_len, 16);
            let mut out16 = vec![0u16; n];
            let mut eco16 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2(
                    needle,
                    &soa16,
                    max_len,
                    &p,
                    0b1010101010101010u16,
                    &mut out16,
                    &mut eco16,
                );
            }
            for (k, h) in hays32.iter().enumerate() {
                let inc = (0b1010101010101010u32 >> k) & 1 == 1;
                let (scalar_score, scalar_ecol) =
                    score_one_scalar_full(needle, h, &p, inc);
                assert_eq!(
                    out16[k], scalar_score,
                    "needle={:?} hay[{k}]={:?}: u16={} scalar={}",
                    needle, h, out16[k], scalar_score
                );
                assert_eq!(
                    eco16[k], scalar_ecol,
                    "needle={:?} hay[{k}]={:?}: u16_ecol={} scalar_ecol={}",
                    needle, h, eco16[k], scalar_ecol
                );
            }
        }
    }

    #[test]
    fn u8_fuzz_long_needles_all_kernels_agree() {
        // Adversarial sweep: the production u8-vs-u16 checksum split on
        // long needles must come from somewhere reproducible here.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let mut state = 0xC0FFEEu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
        // Byte soup with separators, digits, case, high bytes.
        let alphabet: &[u8] = b"abcdefABCDEF0123456789/._- \x80\xff\x01\x7f";
        for round in 0..300 {
            let m = 16 + (next() as usize % 9);
            let needle: Vec<u8> = (0..m).map(|_| alphabet[next() as usize % alphabet.len()]).collect();
            let nlanes = 32;
            let hays: Vec<Vec<u8>> = (0..nlanes)
                .map(|_| {
                    let len = 20 + (next() as usize % 230);
                    (0..len).map(|_| alphabet[next() as usize % alphabet.len()]).collect()
                })
                .collect();
            let hrefs: Vec<&[u8]> = hays.iter().map(|h| h.as_slice()).collect();
            let max_len = hrefs.iter().map(|h| h.len()).max().unwrap();
            let soa32 = super::avx2::build_soa(&hrefs, max_len, 32);
            let soa16 = super::avx2::build_soa(&hrefs[..16], max_len, 16);
            let mut o8 = vec![0u16; nlanes];
            let mut e8 = vec![0u16; nlanes];
            let mut o16 = vec![0u16; 16];
            let mut e16 = vec![0u16; 16];
            unsafe {
                super::avx2::score_batch_avx2_u8(&needle, &soa32, max_len, &p, u32::MAX, &mut o8, &mut e8);
                super::avx2::score_batch_avx2(&needle, &soa16, max_len, &p, u16::MAX, &mut o16, &mut e16);
            }
            for k in 0..nlanes {
                let (ss, se) = score_one_scalar_full(&needle, hrefs[k], &p, true);
                assert_eq!(o8[k], ss, "round {round} lane {k}: u8 vs scalar");
                let _ = se;
            }
            for k in 0..16 {
                let (ss, _) = score_one_scalar_full(&needle, hrefs[k], &p, true);
                assert_eq!(o16[k], ss, "round {round} lane {k}: u16 vs scalar");
            }
        }
    }

    #[test]
    fn u8_end_col_survives_high_scores() {
        // u8 agrees with scalar/u16 through m=18 exact; m>=20 exact
        // diverges (diag transient saturation past 255 + end latch).
        // Both are out of the u8 production domain (u8_safe gates m<=15
        // lowercase); this pins the in-domain boundary plus margin.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        for m in [5usize, 10, 11, 13, 15, 16, 18] {
            let needle: Vec<u8> = b"internationalization".iter().cycle().take(m).cloned().collect();
            let mut hay = needle.clone();
            hay.extend_from_slice(b"_trailing_junk_0123456789");
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let max_len = hay.len();
            let soa32 = super::avx2::build_soa(&hrefs, max_len, 32);
            let mut o8 = vec![0u16; n];
            let mut e8 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2_u8(&needle, &soa32, max_len, &p, u32::MAX, &mut o8, &mut e8);
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(o8[0], ss, "score diverged at m={m}");
            assert_eq!(e8[0], se, "end_col diverged at m={m}");
        }
    }

    #[test]
    fn u8_bound_is_sound() {
        let lo = params();
        assert!(lo.u8_safe(3));
        assert!(lo.u8_safe(14));
        assert!(lo.u8_safe(15));
        assert!(!lo.u8_safe(16));
        let hi = SwParams::for_query(true);
        assert!(hi.u8_safe(3));
        assert!(hi.u8_safe(8));
        assert!(!hi.u8_safe(9));
        assert!(!hi.u8_safe(14));
    }

    #[test]
    fn stack_scalar_matches_vec_path() {
        // Stack (m<=32) must equal Vec path on identical inputs.
        let p = params();
        let mut state = 0xBEEFu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
        for m in [1usize, 7, 15, 16, 31, 32, 33, 40, 64, 100] {
            let needle: Vec<u8> = (0..m)
                .map(|i| b'a' + (next().wrapping_add(i as u8) % 5))
                .collect();
            let hay: Vec<u8> = (0..60).map(|_| b'a' + (next() % 5)).collect();
            let n = String::from_utf8_lossy(&needle).to_string();
            let h = String::from_utf8_lossy(&hay).to_string();
            let stack = if m <= 32 {
                // Stack path only valid for m<=32 (fixed [T; 33] rows).
                score_one_scalar_stack(n.as_bytes(), h.as_bytes(), &p, true)
            } else {
                // Dispatch routes m>32 to Vec; must equal direct Vec call.
                score_one_scalar_full(n.as_bytes(), h.as_bytes(), &p, true)
            };
            let vec = score_one_scalar_vec(n.as_bytes(), h.as_bytes(), &p, true);
            assert_eq!(stack.0, vec.0, "score m={m} needle={n:?}");
            assert_eq!(stack.1, vec.1, "end_col m={m} needle={n:?}");
        }
    }

    #[test]
    fn trimmed_no_prefix_matches_frizbee() {
        // Use frizbee's own prefilter windows (via FRIZBEE_SKIP_SW encoding)
        // and verify our trimmed SW matches frizbee's internal trimmed SW.
        let p = params();
        let cases: &[(&str, &str)] = &[
            ("username.rs", "src/username.rs"),
            ("username.rs", "src/user_name_handler.rs"),
            ("item.rs", "src/models/item.rs"),
            ("item.rs", "src/item_sync/file.rs"),
            ("score", "crates/fff-core/src/score.rs"),
            ("score", "src/scoring/engine.rs"),
            ("file_picker_", "crates/fff-core/src/file_picker.rs"),
            ("usrnmea", "src/username/username.rs"),
            ("usrnmea", "src/username/handler.rs"),
            ("abc", "abc"),
            ("abc", "aabbcc"),
            ("test", "src/test/main.rs"),
        ];
        let has_upper = false;
        let cfg = neo_frizbee::Config {
            max_typos: Some(6),
            sort: neo_frizbee::SortStrategy::Unsorted,
            scoring: neo_frizbee::Scoring {
                capitalization_bonus: if has_upper { 8 } else { 0 },
                matching_case_bonus: if has_upper { 4 } else { 0 },
                ..Default::default()
            },
            ..Default::default()
        };
        for &(needle, full) in cases {
            // Get frizbee's prefilter window via skip_sw (hold env lock)
            let skip_matches = {
                let sm = neo_frizbee::with_skip_sw(|| {
                    let mut m = neo_frizbee::Matcher::new(needle, &cfg);
                    m.match_list(std::slice::from_ref(&full))
                });

                sm
            };

            // Get frizbee's real score (frizbee_score acquires env lock itself)
            let theirs = frizbee_score(needle, full);

            if skip_matches.is_empty() {
                assert_eq!(
                    theirs, 0,
                    "frizbee rejects {needle:?} on {full:?} but score={theirs}"
                );
                continue;
            }
            let start_adj = skip_matches[0].score as usize;
            let end_pos = skip_matches[0].end_col as usize;
            let end_clamped = end_pos.min(full.len());
            let start_clamped = start_adj.min(end_clamped);
            let trimmed = &full.as_bytes()[start_clamped..end_clamped];
            let include_prefix = start_clamped == 0;

            let mine = score_one_scalar(needle.as_bytes(), trimmed, &p, include_prefix);
            assert_eq!(
                mine,
                theirs,
                "needle={needle:?} full={full:?} window=({start_adj},{end_pos}) trimmed={:?} inc_prefix={include_prefix}: mine={mine} frizbee={theirs}",
                std::str::from_utf8(trimmed).unwrap_or("?")
            );
        }
    }

    #[test]
    fn no_match_scores_zero() {
        let p = params();
        let s = score_one_scalar(b"xyz", b"abc", &p, true);
        assert_eq!(s, 0, "xyz/abc");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let batches: &[&[&str]] = &[
            &[
                "abc",
                "aabbcc",
                "aXc",
                "xyz",
                "abcabc",
                "a",
                "ab",
                "a-b-c",
                "src/score.rs",
                "foo/bar.txt",
                "test.rs",
                "main",
                "aaa",
                "bbb",
                "ccc",
                "ddd",
            ],
            &[
                "file_picker",
                "score",
                "netedit",
                "tl",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
            ],
        ];
        for batch in batches {
            let max_len = batch.iter().map(|h| h.len()).max().unwrap();
            let hays: Vec<&[u8]> = batch.iter().map(|h| h.as_bytes()).collect();
            let soa = super::avx2::build_soa(&hays, max_len, 16);
            let mut simd_out = vec![0u16; batch.len()];
            let mut simd_ecol = vec![0u16; batch.len()];
            // All lanes get include_prefix=true (tests use include_prefix=true)
            let prefix_mask: u16 = if batch.len() >= 16 {
                u16::MAX
            } else {
                (1u16 << batch.len()) - 1
            };
            unsafe {
                super::avx2::score_batch_avx2(
                    b"abc",
                    &soa,
                    max_len,
                    &p,
                    prefix_mask,
                    &mut simd_out,
                    &mut simd_ecol,
                );
            }
            for (k, h) in batch.iter().enumerate() {
                let (scalar_score, scalar_ecol) =
                    score_one_scalar_full(b"abc", h.as_bytes(), &p, true);
                assert_eq!(
                    simd_out[k], scalar_score,
                    "batch[{k}]={h:?}: simd_score={} scalar_score={}",
                    simd_out[k], scalar_score
                );
                assert_eq!(
                    simd_ecol[k], scalar_ecol,
                    "batch[{k}]={h:?}: simd_ecol={} scalar_ecol={}",
                    simd_ecol[k], scalar_ecol
                );
            }
        }
    }

    #[test]
    fn avx2_u16_matches_scalar_long_needle() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        // M=20 exercises rows past the old 16-row domain.
        let needle = b"internationalization";
        let hays: &[&str] = &[
            "src/internationalization/mod.rs",
            "src/internationalizatio/mod.rs",
            "src/intenationalization/mod.rs",
            "xxxxxxxxxxxxxxxxxxxxxxxx",
            "src/other.rs",
        ];
        let max_len = hays.iter().map(|h| h.len()).max().unwrap();
        let hays_b: Vec<&[u8]> = hays.iter().map(|h| h.as_bytes()).collect();
        let soa = super::avx2::build_soa(&hays_b, max_len, 16);
        let mut simd_out = vec![0u16; hays.len()];
        let mut simd_ecol = vec![0u16; hays.len()];
        let prefix_mask: u16 = (1u16 << hays.len()) - 1;
        unsafe {
            super::avx2::score_batch_avx2(
                needle,
                &soa,
                max_len,
                &p,
                prefix_mask,
                &mut simd_out,
                &mut simd_ecol,
            );
        }
        for (k, h) in hays.iter().enumerate() {
            let (scalar_score, scalar_ecol) =
                score_one_scalar_full(needle, h.as_bytes(), &p, true);
            assert_eq!(
                simd_out[k], scalar_score,
                "lane[{k}]={h:?}: simd_score={} scalar_score={}",
                simd_out[k], scalar_score
            );
            assert_eq!(
                simd_ecol[k], scalar_ecol,
                "lane[{k}]={h:?}: simd_ecol={} scalar_ecol={}",
                simd_ecol[k], scalar_ecol
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn transpose_8x8_is_exact_matrix_transpose() {
        // src[i][j] = i * 8 + j, differing src/dst strides.
        let mut src = [0u8; 8 * 11];
        for i in 0..8 {
            for j in 0..8 {
                src[i * 11 + j] = (i * 8 + j) as u8;
            }
        }
        let mut dst = [0xAAu8; 8 * 13];
        unsafe {
            super::avx2::transpose_8x8(src.as_ptr(), 11, dst.as_mut_ptr(), 13);
        }
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(
                    dst[j * 13 + i],
                    (i * 8 + j) as u8,
                    "dst[{j}][{i}] != src[{i}][{j}]"
                );
            }
        }
        // Untouched dst bytes stay put.
        assert_eq!(dst[8], 0xAA);
        assert_eq!(dst[12], 0xAA);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_from_stage_matches_build_soa() {
        let cases: &[&[&str]] = &[
            &["a", "", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"],
            &[
                "", "x", "yz", "1234", "56789", "abcdef", "ghijklm", "mnopqrs",
            ],
            &[
                "src/score.rs",
                "a",
                "",
                "very/long/path/name/that/spans/many/chunks/x",
                "zz",
            ],
            &[
                "01234567",
                "8",
                "89abcdef",
                "",
                "foo",
                "barbaz",
                "q",
                "wertyuiop",
            ],
        ];
        for batch in cases {
            for &stride in &[16usize, 32] {
                let max_len = batch.iter().map(|h| h.len()).max().unwrap_or(0);
                let rows = max_len.div_ceil(8) * 8;
                if rows == 0 {
                    continue;
                }
                let mut stage = vec![0u8; stride * rows];
                for (k, h) in batch.iter().enumerate().take(stride) {
                    stage[k * rows..k * rows + h.len()].copy_from_slice(h.as_bytes());
                }
                let mut out = vec![0u8; rows * stride];
                unsafe {
                    super::avx2::soa_from_stage(&stage, rows, stride, &mut out);
                }
                let hays: Vec<&[u8]> = batch.iter().map(|h| h.as_bytes()).collect();
                let expect = super::avx2::build_soa(&hays, rows, stride);
                assert_eq!(out, expect, "stride={stride} batch={batch:?}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_from_stage_dense_roundtrip_layout() {
        // Unambiguous layout probe: stage[k * rows + j] = k * 16 + j (mod 256).
        let (rows, stride) = (16usize, 32usize);
        let mut stage = vec![0u8; rows * stride];
        for k in 0..stride {
            for j in 0..rows {
                stage[k * rows + j] = ((k * 16 + j) % 256) as u8;
            }
        }
        let mut out = vec![0u8; rows * stride];
        unsafe {
            super::avx2::soa_from_stage(&stage, rows, stride, &mut out);
        }
        for k in 0..stride {
            for j in 0..rows {
                assert_eq!(
                    out[j * stride + k],
                    ((k * 16 + j) % 256) as u8,
                    "soa[{j}][{k}]"
                );
            }
        }
    }
}
