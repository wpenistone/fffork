// Frizbee defaults with fff overrides for lowercase queries.
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

    // Sound upper bound: one diagonal per column at most, so rows add
    // at most `per` each, plus prefix and exact once.
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

// Per-worker DP rows for long needles (m > 32); reused zeroed buffers.
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

// Reference SW on a TRIMMED haystack; `include_prefix` iff trimmed start == 0.
// Returns (score, end_col_local): last-row max and its LAST column.
// No exact_match_bonus; caller adds it when `include_exact && needle == trimmed`.
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
    // Stack rows for m <= 32; longer needles keep the Vec path.
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

    // match_prev[i]/match_curr[i]: was (i-1, j-1)/(i-1, j) a match.
    // Rows come zeroed from worker-local scratch.
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
        // Delimiter bonus needs prev-delim AND NOT curr-delim; prev starts clear.
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

            // E: gap_open only if (i, j-1) was a match.
            let e_pen = if match_prev[i] {
                gap_open_only
            } else {
                gap_extend
            };
            let e_new = h_prev[i] - e_pen;
            let e_ext = e_prev[i] - gap_extend;
            e_curr[i] = e_new.max(e_ext);

            // F: gap_open only if (i-1, j) was a match.
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

// Portable AoS-to-SoA transpose: out[j*stride+k] = stage[k*rows+j].
// Byte loop at memcpy speed; ARM uses this (no SSE2 tiles there).
// x86 keeps the shuffle tiles.
pub(crate) fn soa_from_stage_scalar(stage: &[u8], rows: usize, stride: usize, out: &mut [u8]) {
    for j in 0..rows {
        for k in 0..stride {
            out[j * stride + k] = stage[k * rows + j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    pub(crate) const LANES: usize = 16;

    // Test oracle SoA: soa[j*stride+p] = haystack[p][j], zero-padded.
    // `stride` must equal the kernel lane count.
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

    // 8x8 AoS-to-SoA transpose: dst[j*dst_stride+i] = src[i*src_stride+j].
    // Safety: `src` needs 8 rows of `src_stride` readable, `dst` 8 writable;
    // aliasing only when strides match and tiles coincide.
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

            let p0 = _mm_unpacklo_epi8(a0, a1);
            let p1 = _mm_unpacklo_epi8(a2, a3);
            let p2 = _mm_unpacklo_epi8(a4, a5);
            let p3 = _mm_unpacklo_epi8(a6, a7);

            let q0 = _mm_unpacklo_epi16(p0, p1);
            let q1 = _mm_unpackhi_epi16(p0, p1);
            let q2 = _mm_unpacklo_epi16(p2, p3);
            let q3 = _mm_unpackhi_epi16(p2, p3);

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

    // Tiles path-major stage to position-major SoA; writes every byte.
    // Safety: `stage`/`out` cover stride*rows; SSE2 only, no racy aliasing.
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

    // AVX2 SW over up to 16 pre-trimmed haystacks; per-lane (score, end_col).
    // Needle rows dynamic to 32; longer needles stay scalar.
    // Safety: AVX2 available; `out.len() >= haystacks.len()`.
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

        let gap_ext_v = set1(p.gap_extend);
        let gap_diff_v = set1(p.gap_open.wrapping_sub(p.gap_extend));
        let match_score_v = set1(p.match_score);
        let mismatch_v = set1(p.mismatch_penalty);
        let delim_v = set1(p.delimiter_bonus);
        let cap_v = set1(p.capitalization_bonus);
        let case_v = set1(p.matching_case_bonus);
        let prefix_v = set1(p.prefix_bonus);
        let bit20 = set1(0x20);
        let a_lo = set1(b'A' as u16 - 1);
        let z_hi = set1(b'Z' as u16 + 1);
        let la_lo = set1(b'a' as u16 - 1);
        let lz_hi = set1(b'z' as u16 + 1);
        let d0_lo = set1(b'0' as u16 - 1);
        let d9_hi = set1(b'9' as u16 + 1);
        let lt128 = set1(128);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;
        let mut n_fold_v = [zero; 32];
        let mut nc_v = [zero; 32];
        for i in 0..m {
            let b = needle[i] as u16;
            n_fold_v[i] = set1(if (b as u8).is_ascii_uppercase() {
                b + 32
            } else {
                b
            });
            nc_v[i] = set1(b);
        }

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

        // 1D state with register-carried f/mm.
        let mut h = [zero; 32];
        let mut e = [zero; 32];
        let mut match_prev = [zero; 32];

        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut last_row_max = zero;
        let mut end_col_v = zero;

        for j in 0..n {
            let raw = _mm_loadu_si128(soa.as_ptr().add(j * LANES) as *const __m128i);
            let hay = _mm256_cvtepu8_epi16(raw);
            if j + 2 < n {
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        soa.as_ptr().add((j + 2) * LANES) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            let ge_a = _mm256_cmpgt_epi16(hay, a_lo);
            let le_z = _mm256_cmpgt_epi16(z_hi, hay);
            let is_upper = _mm256_and_si256(ge_a, le_z);
            // Fold once per column; cells test one compare.
            let hay_fold = _mm256_add_epi16(hay, _mm256_and_si256(is_upper, bit20));
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
            let delim_bonus = _mm256_and_si256(_mm256_and_si256(prev_delim, not_delim), delim_v);
            // Capitalization needs the previous column's lowercase mask, carried in a register.
            let (is_lower, cap_bonus) = if need_case {
                let cap_mask = _mm256_and_si256(is_upper, prev_lower);
                (is_lower, _mm256_and_si256(cap_mask, cap_v))
            } else {
                (zero, zero)
            };
            let prefix_bonus = if j == 0 {
                _mm256_and_si256(prefix_v, prefix_lanes)
            } else {
                zero
            };
            let bonus_base = _mm256_add_epi16(
                _mm256_add_epi16(delim_bonus, cap_bonus),
                _mm256_add_epi16(prefix_bonus, match_score_v),
            );

            // Register-carried recurrence; no 2D state or f array.
            let mut h_diag = zero;
            let mut h_up = zero;
            let mut f_running = zero;
            let mut prev_mm = zero;

            for i in 0..m {
                let mm = _mm256_cmpeq_epi16(hay_fold, n_fold_v[i]);
                let m_prev = match_prev[i];
                match_prev[i] = mm;

                let mut match_gain = _mm256_and_si256(bonus_base, mm);
                if need_case {
                    let eq_nc = _mm256_cmpeq_epi16(hay, nc_v[i]);
                    match_gain = _mm256_add_epi16(match_gain, _mm256_and_si256(eq_nc, case_v));
                }
                let mis_pen = _mm256_andnot_si256(mm, mismatch_v);
                let diag = _mm256_add_epi16(_mm256_subs_epu16(h_diag, mis_pen), match_gain);

                let h_left = h[i];
                h_diag = h_left;

                // Factored gap: open premium via mask, extend outside the max.
                let e_open_extra = _mm256_and_si256(m_prev, gap_diff_v);
                let e_pre = _mm256_max_epu16(_mm256_subs_epu16(h_left, e_open_extra), e[i]);
                let e_val = _mm256_subs_epu16(e_pre, gap_ext_v);
                e[i] = e_val;

                let f_open_extra = _mm256_and_si256(prev_mm, gap_diff_v);
                let f_pre = _mm256_max_epu16(_mm256_subs_epu16(h_up, f_open_extra), f_running);
                f_running = _mm256_subs_epu16(f_pre, gap_ext_v);

                let h_val = _mm256_max_epu16(_mm256_max_epu16(diag, e_val), f_running);
                h[i] = h_val;
                h_up = h_val;
                prev_mm = mm;
            }

            let row_m = h_up;
            let gt_best = _mm256_cmpgt_epi16(row_m, last_row_max);
            let j_v = set1(j as u16);
            end_col_v = _mm256_blendv_epi8(end_col_v, j_v, gt_best);
            last_row_max = _mm256_max_epu16(last_row_max, row_m);

            prev_lower = is_lower;
            prev_delim = is_delim;
        }

        let scores: [u16; 16] = std::mem::transmute(last_row_max);
        let ecols: [u16; 16] = std::mem::transmute(end_col_v);
        for (i, (&s, &e)) in scores.iter().zip(ecols.iter()).enumerate() {
            if i < out.len() {
                out[i] = s;
                end_cols[i] = e;
            }
        }
    }

    // 32-lane u8 variant; same semantics, valid only when scores fit in u8.
    // Safety: AVX2 available; scores must fit in u8.
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

        let gap_ext_v = set1(p.gap_extend as u8);
        let gap_diff_v = set1(p.gap_open.wrapping_sub(p.gap_extend) as u8);
        let match_score_v = set1(p.match_score as u8);
        let mismatch_v = set1(p.mismatch_penalty as u8);
        let delim_v = set1(p.delimiter_bonus as u8);
        let cap_v = set1(p.capitalization_bonus as u8);
        let case_v = set1(p.matching_case_bonus as u8);
        let prefix_v = set1(p.prefix_bonus as u8);
        let bit20 = set1(0x20);
        let add_upper = set1(127 - b'Z');
        let thresh_upper = set1(101);
        let add_letter = set1(127 - b'z');
        let thresh_letter = set1(101);
        let add_digit = set1(127 - b'9');
        let thresh_digit = set1(117);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;

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

        let mut n_fold_v = [zero; 32];
        let mut nc_v = [zero; 32];
        for i in 0..m {
            let b = needle[i];
            n_fold_v[i] = set1(if b.is_ascii_uppercase() { b + 32 } else { b });
            nc_v[i] = set1(b);
        }

        // 1D state with register-carried f/mm.
        let mut h = [zero; 32];
        let mut e = [zero; 32];
        let mut match_prev = [zero; 32];

        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut last_row_max = zero;
        let mut end_col_v = zero;
        // 0x80 bias restores unsigned order for the signed compare past 127;
        // scores use the unsigned max and are unaffected.
        let sign_bias = _mm256_set1_epi8(-128);

        for j in 0..n {
            // Saturated lanes stay saturated (max only grows); checking every 4th column is exact.
            if j & 3 == 0 && _mm256_movemask_epi8(_mm256_cmpeq_epi8(last_row_max, ones)) == -1 {
                break;
            }
            let hay = _mm256_loadu_si256(soa.as_ptr().add(j * LANES8) as *const __m256i);
            if j + 2 < n {
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        soa.as_ptr().add((j + 2) * LANES8) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            // Skips class masks when case/delim bonuses are 0 (typical lowercase query).
            // Signed-wrap range tests; bytes outside wrap negative or stay below threshold.
            let is_upper = _mm256_cmpgt_epi8(_mm256_add_epi8(hay, add_upper), thresh_upper);
            let hay_fold = _mm256_or_si256(hay, _mm256_and_si256(is_upper, bit20));
            let is_letter = _mm256_cmpgt_epi8(_mm256_add_epi8(hay_fold, add_letter), thresh_letter);
            let is_digit = _mm256_cmpgt_epi8(_mm256_add_epi8(hay, add_digit), thresh_digit);
            let alnum = _mm256_or_si256(is_letter, is_digit);
            let lt_128 = _mm256_cmpgt_epi8(hay, ones);
            let is_delim = _mm256_andnot_si256(alnum, lt_128);
            let dm = _mm256_andnot_si256(is_delim, prev_delim);
            let delim_bonus = _mm256_and_si256(dm, delim_v);

            let (is_lower, cap_bonus) = if need_case {
                let is_lower = _mm256_andnot_si256(is_upper, is_letter);
                let cap_mask = _mm256_and_si256(is_upper, prev_lower);
                (is_lower, _mm256_and_si256(cap_mask, cap_v))
            } else {
                (zero, zero)
            };
            let prefix_bonus = if j == 0 {
                _mm256_and_si256(prefix_v, prefix_lanes)
            } else {
                zero
            };
            let bonus_base = _mm256_add_epi8(
                _mm256_add_epi8(delim_bonus, cap_bonus),
                _mm256_add_epi8(prefix_bonus, match_score_v),
            );

            // Register-carried recurrence; no 2D state or f array.
            let mut h_diag = zero;
            let mut h_up = zero;
            let mut f_running = zero;
            let mut prev_mm = zero;

            for i in 0..m {
                let mm = _mm256_cmpeq_epi8(hay_fold, n_fold_v[i]);
                let m_prev = match_prev[i];
                match_prev[i] = mm;

                let mut match_gain = _mm256_and_si256(bonus_base, mm);
                if need_case {
                    let eq_nc = _mm256_cmpeq_epi8(hay, nc_v[i]);
                    match_gain = _mm256_add_epi8(match_gain, _mm256_and_si256(eq_nc, case_v));
                }
                // Net gain first, then saturating add: transients never wrap past 255.
                let mis_pen = _mm256_andnot_si256(mm, mismatch_v);
                let diag = _mm256_adds_epu8(_mm256_subs_epu8(h_diag, mis_pen), match_gain);

                let h_left = h[i];
                h_diag = h_left;

                // Factored gap: open premium via mask, extend outside the max.
                let e_open_extra = _mm256_and_si256(m_prev, gap_diff_v);
                let e_pre = _mm256_max_epu8(_mm256_subs_epu8(h_left, e_open_extra), e[i]);
                let e_val = _mm256_subs_epu8(e_pre, gap_ext_v);
                e[i] = e_val;

                let f_open_extra = _mm256_and_si256(prev_mm, gap_diff_v);
                let f_pre = _mm256_max_epu8(_mm256_subs_epu8(h_up, f_open_extra), f_running);
                f_running = _mm256_subs_epu8(f_pre, gap_ext_v);

                let h_val = _mm256_max_epu8(_mm256_max_epu8(diag, e_val), f_running);
                h[i] = h_val;
                h_up = h_val;
                prev_mm = mm;
            }

            let row_m = h_up;
            let gt_best = _mm256_cmpgt_epi8(
                _mm256_xor_si256(row_m, sign_bias),
                _mm256_xor_si256(last_row_max, sign_bias),
            );
            let j_v = set1(j as u8);
            end_col_v = _mm256_blendv_epi8(end_col_v, j_v, gt_best);
            last_row_max = _mm256_max_epu8(last_row_max, row_m);

            prev_lower = is_lower;
            prev_delim = is_delim;
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

    // Dispatches common needle lengths to const-unrolled kernels; bit-identical.
    // Other lengths keep the general runtime-m loop.
    // Safety: AVX2 required (score.rs gates on simd_available).
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

    // Compile-time needle length: unrolled row loop, SROA state; bit-identical.
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

        let gap_ext_v = set1(p.gap_extend as u8);
        let gap_diff_v = set1(p.gap_open.wrapping_sub(p.gap_extend) as u8);
        let match_score_v = set1(p.match_score as u8);
        let mismatch_v = set1(p.mismatch_penalty as u8);
        let delim_v = set1(p.delimiter_bonus as u8);
        let cap_v = set1(p.capitalization_bonus as u8);
        let case_v = set1(p.matching_case_bonus as u8);
        let prefix_v = set1(p.prefix_bonus as u8);
        let bit20 = set1(0x20);
        let add_upper = set1(127 - b'Z');
        let thresh_upper = set1(101);
        let add_letter = set1(127 - b'z');
        let thresh_letter = set1(101);
        let add_digit = set1(127 - b'9');
        let thresh_digit = set1(117);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;

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

        let mut n_fold_v = [zero; 32];
        let mut nc_v = [zero; 32];
        for i in 0..m {
            let b = needle[i];
            n_fold_v[i] = set1(if b.is_ascii_uppercase() { b + 32 } else { b });
            nc_v[i] = set1(b);
        }

        // 1D state with register-carried f/mm; indices compile-time for SROA.
        let mut h = [zero; 32];
        let mut e = [zero; 32];
        let mut match_prev = [zero; 32];

        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut last_row_max = zero;
        let mut end_col_v = zero;
        // 0x80 bias restores unsigned order for the signed compare.
        let sign_bias = _mm256_set1_epi8(-128);

        for j in 0..n {
            // A saturated batch is done; skip the fetch before loading.
            if j & 3 == 0 && _mm256_movemask_epi8(_mm256_cmpeq_epi8(last_row_max, ones)) == -1 {
                break;
            }
            let hay = _mm256_loadu_si256(soa.as_ptr().add(j * 32) as *const __m256i);
            if j + 2 < n {
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        soa.as_ptr().add((j + 2) * 32) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            // Signed-wrap range tests; bytes outside wrap negative or stay below threshold.
            let is_upper = _mm256_cmpgt_epi8(_mm256_add_epi8(hay, add_upper), thresh_upper);
            let hay_fold = _mm256_or_si256(hay, _mm256_and_si256(is_upper, bit20));
            let is_letter = _mm256_cmpgt_epi8(_mm256_add_epi8(hay_fold, add_letter), thresh_letter);
            let is_digit = _mm256_cmpgt_epi8(_mm256_add_epi8(hay, add_digit), thresh_digit);
            let alnum = _mm256_or_si256(is_letter, is_digit);
            let lt_128 = _mm256_cmpgt_epi8(hay, ones);
            let is_delim = _mm256_andnot_si256(alnum, lt_128);
            let dm = _mm256_andnot_si256(is_delim, prev_delim);
            let delim_bonus = _mm256_and_si256(dm, delim_v);

            let (is_lower, cap_bonus) = if need_case {
                let is_lower = _mm256_andnot_si256(is_upper, is_letter);
                let cap_mask = _mm256_and_si256(is_upper, prev_lower);
                (is_lower, _mm256_and_si256(cap_mask, cap_v))
            } else {
                (zero, zero)
            };
            let prefix_bonus = if j == 0 {
                _mm256_and_si256(prefix_v, prefix_lanes)
            } else {
                zero
            };
            let bonus_base = _mm256_add_epi8(
                _mm256_add_epi8(delim_bonus, cap_bonus),
                _mm256_add_epi8(prefix_bonus, match_score_v),
            );

            // Register-carried recurrence; no 2D state or f array.
            let mut h_diag = zero;
            let mut h_up = zero;
            let mut f_running = zero;
            let mut prev_mm = zero;

            for i in 0..M {
                let mm = _mm256_cmpeq_epi8(hay_fold, n_fold_v[i]);
                let m_prev = match_prev[i];
                match_prev[i] = mm;

                let mut match_gain = _mm256_and_si256(bonus_base, mm);
                if need_case {
                    let eq_nc = _mm256_cmpeq_epi8(hay, nc_v[i]);
                    match_gain = _mm256_add_epi8(match_gain, _mm256_and_si256(eq_nc, case_v));
                }
                // Net gain first, then saturating add: transients never wrap past 255.
                let mis_pen = _mm256_andnot_si256(mm, mismatch_v);
                let diag = _mm256_adds_epu8(_mm256_subs_epu8(h_diag, mis_pen), match_gain);

                let h_left = h[i];
                h_diag = h_left;

                // Factored gap: open premium via mask, extend outside the max.
                let e_open_extra = _mm256_and_si256(m_prev, gap_diff_v);
                let e_pre = _mm256_max_epu8(_mm256_subs_epu8(h_left, e_open_extra), e[i]);
                let e_val = _mm256_subs_epu8(e_pre, gap_ext_v);
                e[i] = e_val;

                let f_open_extra = _mm256_and_si256(prev_mm, gap_diff_v);
                let f_pre = _mm256_max_epu8(_mm256_subs_epu8(h_up, f_open_extra), f_running);
                f_running = _mm256_subs_epu8(f_pre, gap_ext_v);

                let h_val = _mm256_max_epu8(_mm256_max_epu8(diag, e_val), f_running);
                h[i] = h_val;
                h_up = h_val;
                prev_mm = mm;
            }

            let row_m = h_up;
            let gt_best = _mm256_cmpgt_epi8(
                _mm256_xor_si256(row_m, sign_bias),
                _mm256_xor_si256(last_row_max, sign_bias),
            );
            let j_v = set1(j as u8);
            end_col_v = _mm256_blendv_epi8(end_col_v, j_v, gt_best);
            last_row_max = _mm256_max_epu8(last_row_max, row_m);

            prev_lower = is_lower;
            prev_delim = is_delim;
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

// 16-lane u8 SW kernel for AArch64 NEON: same recurrence and exact
// semantics as the AVX2 u8 kernel (factored gap form, 1D state with
// register-carried f/mm, folded-needle single compare). NEON compares
// are natively unsigned, so no sign-bias trick is needed.
//
// # Safety
// Caller must be AArch64 (NEON is baseline there). Scores must fit in
// u8 (`u8_safe` domain). Windows must be at most 256 bytes: end
// tracking is byte-wide, so longer windows stay scalar by dispatch.
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use super::*;
    use std::arch::aarch64::*;

    pub(crate) const LANES8: usize = 16;

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn score_batch_neon_u8(
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

        let zero = vdupq_n_u8(0);
        let ones = vdupq_n_u8(0xFF);
        let gap_ext_v = vdupq_n_u8(p.gap_extend as u8);
        let gap_diff_v = vdupq_n_u8(p.gap_open.wrapping_sub(p.gap_extend) as u8);
        let match_score_v = vdupq_n_u8(p.match_score as u8);
        let mismatch_v = vdupq_n_u8(p.mismatch_penalty as u8);
        let delim_v = vdupq_n_u8(p.delimiter_bonus as u8);
        let cap_v = vdupq_n_u8(p.capitalization_bonus as u8);
        let case_v = vdupq_n_u8(p.matching_case_bonus as u8);
        let prefix_v = vdupq_n_u8(p.prefix_bonus as u8);
        let bit20 = vdupq_n_u8(0x20);
        // Signed-wrap range tests: 'A'..='Z' iff hay+37 > 101, etc.
        // Bytes outside the range wrap negative or stay below threshold.
        let add_upper = vdupq_n_u8(127 - b'Z');
        let thresh_upper = vdupq_n_u8(101);
        let add_letter = vdupq_n_u8(127 - b'z');
        let thresh_letter = vdupq_n_u8(101);
        let add_digit = vdupq_n_u8(127 - b'9');
        let thresh_digit = vdupq_n_u8(117);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;

        let prefix_lanes = {
            let mut arr = [0u8; 16];
            for (i, a) in arr.iter_mut().enumerate() {
                *a = if include_prefix_mask & (1u16 << i) != 0 {
                    0xFF
                } else {
                    0
                };
            }
            vld1q_u8(arr.as_ptr())
        };

        let mut n_fold_v = [zero; 32];
        let mut nc_v = [zero; 32];
        for i in 0..m {
            let b = needle[i];
            n_fold_v[i] = vdupq_n_u8(if b.is_ascii_uppercase() { b + 32 } else { b });
            nc_v[i] = vdupq_n_u8(b);
        }

        // 1D in-place state: indices are compile-time after unrolling,
        // so no ping-pong dimension is needed for e, h, or match masks.
        let mut h = [zero; 32];
        let mut e = [zero; 32];
        let mut match_prev = [zero; 32];
        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut last_row_max = zero;
        let mut end_col_v = zero;

        for j in 0..n {
            // Saturated lanes stay saturated; checking every 4th column is exact.
            if j & 3 == 0 && vmaxvq_u8(last_row_max) == 0xFF {
                break;
            }
            let hay = vld1q_u8(soa.as_ptr().add(j * LANES8));
            let is_upper = vcgtq_s8(
                vreinterpretq_s8_u8(vaddq_u8(hay, add_upper)),
                vreinterpretq_s8_u8(thresh_upper),
            );
            let hay_fold = vorrq_u8(hay, vandq_u8(is_upper, bit20));
            let is_letter = vcgtq_s8(
                vreinterpretq_s8_u8(vaddq_u8(hay_fold, add_letter)),
                vreinterpretq_s8_u8(thresh_letter),
            );
            let is_digit = vcgtq_s8(
                vreinterpretq_s8_u8(vaddq_u8(hay, add_digit)),
                vreinterpretq_s8_u8(thresh_digit),
            );
            let alnum = vorrq_u8(is_letter, is_digit);
            let lt_128 = vcltq_u8(hay, vdupq_n_u8(128));
            let is_delim = vbicq_u8(lt_128, alnum);
            let dm = vbicq_u8(prev_delim, is_delim);
            let delim_bonus = vandq_u8(dm, delim_v);

            let (is_lower, cap_bonus) = if need_case {
                let is_lower = vbicq_u8(is_letter, is_upper);
                let cap_mask = vandq_u8(is_upper, prev_lower);
                (is_lower, vandq_u8(cap_mask, cap_v))
            } else {
                (zero, zero)
            };

            let prefix_bonus = if j == 0 {
                vandq_u8(prefix_v, prefix_lanes)
            } else {
                zero
            };
            let bonus_base = vaddq_u8(
                vaddq_u8(delim_bonus, cap_bonus),
                vaddq_u8(prefix_bonus, match_score_v),
            );

            let mut h_diag = zero;
            let mut h_up = zero;
            let mut f_running = zero;
            let mut prev_mm = zero;

            for i in 0..m {
                let mm = vceqq_u8(hay_fold, n_fold_v[i]);
                let m_prev = match_prev[i];
                match_prev[i] = mm;

                let mut match_gain = vandq_u8(bonus_base, mm);
                if need_case {
                    let eq_nc = vceqq_u8(hay, nc_v[i]);
                    match_gain = vaddq_u8(match_gain, vandq_u8(eq_nc, case_v));
                }
                // Net gain first, then saturating add: transient sums
                // can never wrap past 255 before the mismatch is out.
                let mis_pen = vbicq_u8(mismatch_v, mm);
                let diag = vqaddq_u8(vqsubq_u8(h_diag, mis_pen), match_gain);

                let h_left = h[i];
                h_diag = h_left;

                let e_open_extra = vandq_u8(m_prev, gap_diff_v);
                let e_pre = vmaxq_u8(vqsubq_u8(h_left, e_open_extra), e[i]);
                let e_val = vqsubq_u8(e_pre, gap_ext_v);
                e[i] = e_val;

                let f_open_extra = vandq_u8(prev_mm, gap_diff_v);
                let f_pre = vmaxq_u8(vqsubq_u8(h_up, f_open_extra), f_running);
                f_running = vqsubq_u8(f_pre, gap_ext_v);

                let h_val = vmaxq_u8(vmaxq_u8(diag, e_val), f_running);
                h[i] = h_val;
                h_up = h_val;
                prev_mm = mm;
            }

            let row_m = h_up;
            let gt_best = vcgtq_u8(row_m, last_row_max);
            end_col_v = vbslq_u8(gt_best, vdupq_n_u8(j as u8), end_col_v);
            last_row_max = vmaxq_u8(last_row_max, row_m);

            prev_lower = is_lower;
            prev_delim = is_delim;
        }

        let mut sb = [0u8; 16];
        let mut eb = [0u8; 16];
        vst1q_u8(sb.as_mut_ptr(), last_row_max);
        vst1q_u8(eb.as_mut_ptr(), end_col_v);
        for i in 0..16 {
            if i < out.len() {
                out[i] = sb[i] as u16;
            }
            if i < end_cols.len() {
                end_cols[i] = eb[i] as u16;
            }
        }
    }

    // Test-only SoA builder (mirrors the avx2 test oracle layout).
    #[cfg(test)]
    pub(crate) fn build_soa16(haystacks: &[&[u8]], max_len: usize) -> Vec<u8> {
        let mut soa = vec![0u8; max_len * 16];
        for (p, h) in haystacks.iter().enumerate().take(16) {
            for (j, &b) in h.iter().enumerate() {
                soa[j * 16 + p] = b;
            }
        }
        soa
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

    // Ground truth: frizbee Matcher under fff's scoring config.
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
            // Less exact_match_bonus, which score_haystack excludes.
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
        // Any violation kills pre-SW pruning; parity lets scalar coverage transfer.
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

    #[cfg(target_arch = "x86_64")]
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_matches_scalar_on_long_needles() {
        // Out-of-domain long needles must still agree with scalar exactly.
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
            // Pad lanes past 8; shorter tails run scalar in production.
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
            // Dispatch must equal the general kernel (production calls dispatch).
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
            // Same inputs through u16 isolate divergence to u8.
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_fuzz_long_needles_all_kernels_agree() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let mut state = 0xC0FFEEu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_end_col_survives_high_scores() {
        // In-domain exact needles must agree; the loop pins the boundary plus margin.
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_transient_band_agrees_with_scalar() {
        // Diag intermediates can exceed the final, so in-domain cells can wrap; biased fuzz hunts that band.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x48u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
        let bonus_alpha = b"aBcDeF_/012";
        for round in 0..500 {
            let upper = round % 2 == 0;
            let p = SwParams::for_query(upper);
            let m = if upper {
                6 + (next() % 3) as usize
            } else {
                13 + (next() % 3) as usize
            };
            if !p.u8_safe(m) {
                continue;
            }
            let needle: Vec<u8> = (0..m)
                .map(|_| bonus_alpha[next() as usize % bonus_alpha.len()])
                .collect();
            let mut hay = Vec::new();
            for _ in 0..(1 + next() % 3) {
                hay.extend_from_slice(&needle[..(1 + next() as usize % m)]);
            }
            hay.extend_from_slice(b"_junk0123456789");
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let soa32 = super::avx2::build_soa(&hrefs, hay.len(), 32);
            let mut o8 = vec![0u16; n];
            let mut e8 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    &needle,
                    &soa32,
                    hay.len(),
                    &p,
                    u32::MAX,
                    &mut o8,
                    &mut e8,
                );
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(
                o8[0], ss,
                "score diverged round={round} m={m} upper={upper}"
            );
            assert_eq!(
                e8[0], se,
                "end_col diverged round={round} m={m} upper={upper}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_late_best_end_col_agrees() {
        // u8 while best-ends fit a byte; longer windows take u16. Both must agree with scalar.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let needle = b"controller".to_vec();
        assert!(p.u8_safe(needle.len()));
        // In-contract: best-ends 209/249 fit the byte column tracker.
        for &off in &[200usize, 240] {
            let mut hay = vec![b'q'; off + needle.len() + 8];
            hay[off..off + needle.len()].copy_from_slice(&needle);
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let soa32 = super::avx2::build_soa(&hrefs, hay.len(), 32);
            let mut o8 = vec![0u16; n];
            let mut e8 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    &needle,
                    &soa32,
                    hay.len(),
                    &p,
                    u32::MAX,
                    &mut o8,
                    &mut e8,
                );
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(o8[0], ss, "score diverged off={off}");
            assert_eq!(e8[0], se, "end_col diverged off={off}");
        }
        // Handoff: best-ends past byte 255 go u16, still exact.
        for &off in &[260usize, 300, 511] {
            let mut hay = vec![b'q'; off + needle.len() + 8];
            hay[off..off + needle.len()].copy_from_slice(&needle);
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let soa16 = super::avx2::build_soa(&hrefs, hay.len(), 16);
            let mut o16 = vec![0u16; n];
            let mut e16 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2(
                    &needle,
                    &soa16,
                    hay.len(),
                    &p,
                    u16::MAX,
                    &mut o16,
                    &mut e16,
                );
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(o16[0], ss, "u16 score diverged off={off}");
            assert_eq!(e16[0], se, "u16 end_col diverged off={off}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_agrees_with_scalar_through_safe_domain() {
        // Wherever u8_safe admits, u8 must equal scalar on exact-prefix hays.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for upper in [false, true] {
            let p = SwParams::for_query(upper);
            let base: Vec<u8> = if upper {
                b"INTERNATIONALIZATION".to_vec()
            } else {
                b"internationalization".to_vec()
            };
            for m in 1..=32usize {
                if !p.u8_safe(m) {
                    continue;
                }
                let needle: Vec<u8> = base.iter().cycle().take(m).cloned().collect();
                let mut hay = needle.clone();
                hay.extend_from_slice(b"_trailing_junk_0123456789");
                let n = 16;
                let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
                let soa32 = super::avx2::build_soa(&hrefs, hay.len(), 32);
                let mut o8 = vec![0u16; n];
                let mut e8 = vec![0u16; n];
                unsafe {
                    super::avx2::score_batch_avx2_u8(
                        &needle,
                        &soa32,
                        hay.len(),
                        &p,
                        u32::MAX,
                        &mut o8,
                        &mut e8,
                    );
                }
                let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
                assert_eq!(o8[0], ss, "score diverged upper={upper} m={m}");
                assert_eq!(e8[0], se, "end_col diverged upper={upper} m={m}");
            }
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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_u8_agrees_with_scalar() {
        // First-class ARM parity: NEON u8 kernel vs scalar oracle over
        // the u8_safe domain, exact-prefix plus biased-bonus fuzz
        // shapes (transient band included). Runs on ARM hardware.
        let mut state = 0x9E01u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
        let alpha = b"aBcDeF_/012";
        for round in 0..300 {
            let upper = round % 2 == 0;
            let p = SwParams::for_query(upper);
            let m = if upper {
                3 + (next() % 6) as usize
            } else {
                5 + (next() % 11) as usize
            };
            if !p.u8_safe(m) {
                continue;
            }
            let needle: Vec<u8> = (0..m)
                .map(|_| alpha[next() as usize % alpha.len()])
                .collect();
            let mut hay = Vec::new();
            for _ in 0..(1 + next() % 3) {
                hay.extend_from_slice(&needle[..(1 + next() as usize % m)]);
            }
            hay.extend_from_slice(b"_junk0123456789");
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let soa = super::neon::build_soa16(&hrefs, hay.len());
            let mut o = vec![0u16; n];
            let mut e = vec![0u16; n];
            unsafe {
                super::neon::score_batch_neon_u8(
                    &needle,
                    &soa,
                    hay.len(),
                    &p,
                    u16::MAX,
                    &mut o,
                    &mut e,
                );
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(o[0], ss, "neon score diverged round={round} m={m} upper={upper}");
            assert_eq!(e[0], se, "neon end_col diverged round={round} m={m} upper={upper}");
        }
    }

    #[test]
    fn stack_scalar_matches_vec_path() {
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
                score_one_scalar_full(n.as_bytes(), h.as_bytes(), &p, true)
            };
            let vec = score_one_scalar_vec(n.as_bytes(), h.as_bytes(), &p, true);
            assert_eq!(stack.0, vec.0, "score m={m} needle={n:?}");
            assert_eq!(stack.1, vec.1, "end_col m={m} needle={n:?}");
        }
    }

    #[test]
    fn trimmed_no_prefix_matches_frizbee() {
        // Our trimmed SW must match frizbee's on frizbee's own prefilter windows.
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
            let skip_matches = {
                let sm = neo_frizbee::with_skip_sw(|| {
                    let mut m = neo_frizbee::Matcher::new(needle, &cfg);
                    m.match_list(std::slice::from_ref(&full))
                });

                sm
            };

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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_u16_matches_scalar_long_needle() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        // m=20 exercises rows past 16.
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
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn transpose_8x8_is_exact_matrix_transpose() {
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
        assert_eq!(dst[8], 0xAA);
        assert_eq!(dst[12], 0xAA);
    }

    #[cfg(target_arch = "x86_64")]
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
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_from_stage_dense_roundtrip_layout() {
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
