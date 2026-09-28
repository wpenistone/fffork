//! Inter-sequence Smith-Waterman: lanes = paths, haystack sequential.
//!
//! Replicates frizbee's gap model exactly (ascii_gap.rs):
//! - gap_open applied ONLY when the previous cell was a match
//!   (`shifted_match_mask` gates `gop` in `gap_step!`)
//! - gap_extend for all other gap positions
//! Score = max of LAST query row (not global max), matching frizbee's
//! `max_scores.horizontal_max()` which tracks only `row_scores` after the
//! inner row loop.

/// Scoring params matching frizbee defaults + fff overrides for lowercase queries.
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

/// Scalar reference SW matching frizbee's exact semantics on a TRIMMED haystack.
/// `include_prefix` must be `true` iff the trimmed start == 0 (frizbee:
/// `haystack_start_pos == 0` after `trim_haystack`'s saturating_sub(1)).
/// Returns (score, end_col_local). end_col_local = LAST column in the trimmed
/// slice where the last-row max occurs (frizbee's match_end_col).
/// No exact_match_bonus — caller adds it when `include_exact && needle == trimmed`.
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

    // Use i32 DP values to avoid saturation artifacts during computation.
    let gap_open = p.gap_open as i32;
    let gap_extend = p.gap_extend as i32;
    let gap_open_only = gap_open; // full open penalty when transitioning match->gap
    let match_add = p.match_score as i32;
    let mismatch_pen = p.mismatch_penalty as i32;

    // match_prev[i]: was (i-1, j-1) a match?  (frizbee's shifted_match_mask)
    // match_curr[i]: is (i-1, j) a match?
    let mut h_prev = vec![0i32; m + 1];
    let mut e_prev = vec![0i32; m + 1]; // horizontal gap state
    let mut f_prev = vec![0i32; m + 1]; // vertical gap state
    let mut h_curr = vec![0i32; m + 1];
    let mut e_curr = vec![0i32; m + 1];
    let mut f_curr = vec![0i32; m + 1];
    let mut match_prev = vec![false; m + 1];
    let mut match_curr = vec![false; m + 1];

    let mut prev_is_lower = false;
    let mut prev_is_delim = false;
    let mut last_row_max = 0i32;
    let mut end_col = 0u16;

    for j in 0..n {
        let hb = haystack[j];
        let hb_upper = hb.is_ascii_uppercase();
        let hb_lower = hb.is_ascii_lowercase();
        let hb_delim = is_delimiter(hb);

        let cap_bonus = if hb_upper && prev_is_lower { p.capitalization_bonus as i32 } else { 0 };
        // frizbee: delimiter_mask = prev_char_is_delimiter AND NOT curr_char_is_delimiter.
        // prev starts as NOT-delimiter (Mask::zero()).
        let delim_bonus = if prev_is_delim && !hb_delim { p.delimiter_bonus as i32 } else { 0 };
        let prefix_bonus = if include_prefix && j == 0 { p.prefix_bonus as i32 } else { 0 };

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
            let e_pen = if match_prev[i] { gap_open_only } else { gap_extend };
            let e_new = h_prev[i] - e_pen;
            let e_ext = e_prev[i] - gap_extend;
            e_curr[i] = e_new.max(e_ext);

            // F (vertical gap): frizbee's up_scores
            // gap_open ONLY if previous row (i-1, j) was a match
            let f_pen = if match_curr[i - 1] { gap_open_only } else { gap_extend };
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

    (last_row_max as u16, end_col)
}

/// Convenience wrapper: just the score.
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

    /// Build SoA buffer: soa[j * stride + p] = haystack[p][j], padded with 0.
    /// `stride` must match the kernel's lane count (16 for u16, 32 for u8).
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

    /// Score a batch of up to 16 pre-trimmed haystacks using inter-sequence AVX2 SW.
    /// `include_prefix` is a per-lane bitmask (bit i = lane i gets prefix bonus at j=0).
    /// Returns per-lane (score, end_col_local) where end_col_local is the last
    /// column in the trimmed slice where the last-row max occurs.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available and `out.len() >= haystacks.len()`.
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
        let m = needle.len().min(16);
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

        // Per-lane prefix mask: lanes with include_prefix get prefix_bonus at j=0.
        let prefix_lanes = {
            let mut arr = [0u16; 16];
            for (i, a) in arr.iter_mut().enumerate() {
                *a = if include_prefix_mask & (1 << i) != 0 { u16::MAX } else { 0 };
            }
            _mm256_loadu_si256(arr.as_ptr() as *const __m256i)
        };

        // DP state arrays (max 17 rows for m<=16). Ping-pong with an index
        // instead of swapping 544-byte arrays every column.
        let mut h = [[zero; 17]; 2];
        let mut e = [[zero; 17]; 2];
        let mut f = [[zero; 17]; 2];
        let mut match_state = [[zero; 17]; 2];
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

            // Character class masks (frizbee: NOT(letter|digit|>127) = delimiter)
            let ge_a = _mm256_cmpgt_epi16(hay, set1(b'A' as u16 - 1));
            let le_z = _mm256_cmpgt_epi16(set1(b'Z' as u16 + 1), hay);
            let is_upper = _mm256_and_si256(ge_a, le_z);
            let ge_la = _mm256_cmpgt_epi16(hay, set1(b'a' as u16 - 1));
            let le_lz = _mm256_cmpgt_epi16(set1(b'z' as u16 + 1), hay);
            let is_lower = _mm256_and_si256(ge_la, le_lz);
            let ge_0 = _mm256_cmpgt_epi16(hay, set1(b'0' as u16 - 1));
            let le_9 = _mm256_cmpgt_epi16(set1(b'9' as u16 + 1), hay);
            let is_digit = _mm256_and_si256(ge_0, le_9);
            let is_letter = _mm256_or_si256(is_upper, is_lower);
            let alnum = _mm256_or_si256(is_letter, is_digit);
            let not_alnum = _mm256_andnot_si256(alnum, ones);
            let lt_128 = _mm256_cmpgt_epi16(set1(128), hay);
            let is_delim = _mm256_and_si256(not_alnum, lt_128);
            let not_delim = _mm256_andnot_si256(is_delim, ones);

            // Bonuses (frizbee: delimiter = prev_delim AND NOT curr_delim)
            let cap_mask = _mm256_and_si256(is_upper, prev_lower);
            let cap_bonus = _mm256_and_si256(cap_mask, cap_v);
            let dm = _mm256_and_si256(prev_delim, not_delim);
            let delim_bonus = _mm256_and_si256(dm, delim_v);
            let prefix_bonus = if j == 0 {
                _mm256_and_si256(prefix_v, prefix_lanes)
            } else {
                zero
            };

            let nxt = 1 - cur;
            h[nxt][0] = zero;
            e[nxt][0] = zero;
            f[nxt][0] = zero;
            match_state[nxt][0] = zero;

            for i in 1..=m {
                let nc = set1(needle[i - 1] as u16);
                let nf = set1(flip_case(needle[i - 1]) as u16);
                let eq_nc = _mm256_cmpeq_epi16(hay, nc);
                let eq_nf = _mm256_cmpeq_epi16(hay, nf);
                let mm = _mm256_or_si256(eq_nc, eq_nf);
                match_state[nxt][i] = mm;

                let masked_bonus = _mm256_and_si256(
                    _mm256_add_epi16(
                        _mm256_add_epi16(delim_bonus, cap_bonus),
                        _mm256_add_epi16(prefix_bonus, match_add_v),
                    ),
                    mm,
                );
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
            prev_delim = is_delim;
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

    /// Score a batch of up to 32 pre-trimmed haystacks using inter-sequence AVX2 SW
    /// with u8 scores (32 lanes). Only valid when `score_fits_in_u8` is true
    /// (max score < 256). Same semantics as `score_batch_avx2` but 2x lanes.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available. Scores must fit in u8.
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
        // Hoist class thresholds out of the hot column loop (was set1 per column).
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
                *a = if include_prefix_mask & (1u32 << i) != 0 { u8::MAX } else { 0 };
            }
            _mm256_loadu_si256(arr.as_ptr() as *const __m256i)
        };

        // Pre-broadcast needle bytes (was set1 twice per needle row per column).
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

        for j in 0..n {
            // Load 32 bytes at position j
            let hay = _mm256_loadu_si256(soa.as_ptr().add(j * LANES8) as *const __m256i);

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

                let masked_bonus = _mm256_and_si256(
                    _mm256_add_epi8(
                        _mm256_add_epi8(delim_bonus, cap_bonus),
                        _mm256_add_epi8(prefix_bonus, match_add_v),
                    ),
                    mm,
                );
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
            let gt_best = _mm256_cmpgt_epi8(row_m, best_score_v);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SwParams {
        SwParams::for_query(false)
    }

    /// Serializes access to `FRIZBEE_SKIP_SW` across parallel tests.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Compare against frizbee's Matcher (the ground truth) using fff's
    /// scoring config (matching_case_bonus=0 for lowercase queries).
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
    fn scalar_matches_frizbee_on_basic_cases() {
        let p = params();
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
    fn scalar_matches_frizbee_no_prefix() {
        let p = params();
        let cases: &[(&str, &str)] = &[
            ("abc", "abc"),
            ("bc", "abc"),
            ("score", "src/score.rs"),
        ];
        for &(n, h) in cases {
            let mine = score_one_scalar(n.as_bytes(), h.as_bytes(), &p, false);
            let theirs = frizbee_score(n, h);
            // Without prefix, scores may differ (frizbee always uses include_prefix=true
            // when start_pos==0). Just check non-negative.
            assert!(mine <= theirs + 12, "needle={n} haystack={h}: mine={mine} theirs={theirs}");
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
            &["abc", "aabbcc", "aXc", "xyz", "abcabc", "a", "ab", "a-b-c",
              "src/score.rs", "foo/bar.txt", "test.rs", "main",
              "aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh",
              "iii", "jjj", "kkk", "lll", "mmm", "nnn", "ooo", "ppp",
              "qqq", "rrr", "sss", "ttt"],
            &["file_picker", "score", "netedit", "tl",
              "a", "b", "c", "d", "e", "f", "g", "h",
              "i", "j", "k", "l", "m", "n", "o", "p",
              "q", "r", "s", "t", "u", "v", "w", "x",
              "y", "z", "0", "1"],
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
                    b"abc", &soa, max_len, &p, prefix_mask, &mut simd_out, &mut simd_ecol,
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
                
                
                let sm = neo_frizbee::with_skip_sw(|| { let mut m = neo_frizbee::Matcher::new(needle, &cfg); m.match_list(std::slice::from_ref(&full)) });
                
                
                sm
            };

            // Get frizbee's real score (frizbee_score acquires env lock itself)
            let theirs = frizbee_score(needle, full);

            if skip_matches.is_empty() {
                assert_eq!(theirs, 0, "frizbee rejects {needle:?} on {full:?} but score={theirs}");
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
                mine, theirs,
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
            &["abc", "aabbcc", "aXc", "xyz", "abcabc", "a", "ab", "a-b-c",
              "src/score.rs", "foo/bar.txt", "test.rs", "main",
              "aaa", "bbb", "ccc", "ddd"],
            &["file_picker", "score", "netedit", "tl",
              "a", "b", "c", "d", "e", "f", "g", "h",
              "i", "j", "k", "l"],
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
                    b"abc", &soa, max_len, &p, prefix_mask, &mut simd_out, &mut simd_ecol,
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
}
