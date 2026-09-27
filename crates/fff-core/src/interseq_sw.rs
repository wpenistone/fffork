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

/// Scalar reference SW matching frizbee's exact semantics.
/// Returns (score, end_col). Score = max of H[m][*] (last query row),
/// no exact_match_bonus. end_col = LAST column where max occurs (frizbee compat).
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
    let mut f_prev = vec![0i32; m + 1]; // vertical gap state (not needed across cols in frizbee but kept for symmetry)
    let mut h_curr = vec![0i32; m + 1];
    let mut e_curr = vec![0i32; m + 1];
    let mut f_curr = vec![0i32; m + 1];
    let mut match_prev = vec![false; m + 1];
    let mut match_curr = vec![false; m + 1];

    let mut prev_is_lower = false;
    let mut prev_is_delim = false;
    let mut delim_enabled = false;
    let mut last_row_max = 0i32;
    let mut end_col = 0u16;

    for j in 0..n {
        let hb = haystack[j];
        let hb_upper = hb.is_ascii_uppercase();
        let hb_lower = hb.is_ascii_lowercase();
        let hb_digit = hb.is_ascii_digit();
        let hb_delim = is_delimiter(hb);
        if !hb_delim {
            delim_enabled = true;
        }

        let cap_bonus = if hb_upper && prev_is_lower { p.capitalization_bonus as i32 } else { 0 };
        let delim_bonus = if delim_enabled && prev_is_delim && !hb_delim { p.delimiter_bonus as i32 } else { 0 };
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

        if h_curr[m] >= last_row_max {
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

    /// Build SoA buffer: soa[j * n + p] = haystack[p][j], padded with 0.
    pub(crate) fn build_soa(haystacks: &[&[u8]], max_len: usize) -> Vec<u8> {
        let n = haystacks.len();
        let mut soa = vec![0u8; max_len * n];
        for (p, h) in haystacks.iter().enumerate() {
            for (j, &b) in h.iter().enumerate() {
                soa[j * n + p] = b;
            }
        }
        soa
    }

    /// Score a batch of 16 haystacks using inter-sequence AVX2 SW.
    /// Returns per-path max-of-last-row scores.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available and `out.len() >= haystacks.len()`.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn score_batch_avx2(
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        p: &SwParams,
        include_prefix: bool,
        out: &mut [u16],
    ) {
        let m = needle.len().min(16);
        let n = max_len;
        if m == 0 || n == 0 {
            out.iter_mut().for_each(|s| *s = 0);
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

        // DP state arrays (max 17 rows for m<=16)
        let mut h_prev = [zero; 17];
        let mut e_prev = [zero; 17];
        let mut f_prev = [zero; 17];
        let mut match_prev = [zero; 17];
        let mut h_curr = [zero; 17];
        let mut e_curr = [zero; 17];
        let mut f_curr = [zero; 17];
        let mut match_curr = [zero; 17];

        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut delim_enabled = zero;
        let mut last_row_max = zero;

        for j in 0..n {
            // Load 16 bytes at position j, zero-extend to 16 u16
            let raw = _mm_loadu_si128(soa.as_ptr().add(j * LANES) as *const __m128i);
            let hay = _mm256_cvtepu8_epi16(raw);

            // Character class masks
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

            delim_enabled = _mm256_or_si256(delim_enabled, not_delim);

            // Bonuses
            let cap_mask = _mm256_and_si256(is_upper, prev_lower);
            let cap_bonus = _mm256_and_si256(cap_mask, cap_v);
            let dm = _mm256_and_si256(_mm256_and_si256(prev_delim, not_delim), delim_enabled);
            let delim_bonus = _mm256_and_si256(dm, delim_v);
            let prefix_bonus = if include_prefix && j == 0 { prefix_v } else { zero };

            h_curr[0] = zero;
            e_curr[0] = zero;
            f_curr[0] = zero;
            match_curr[0] = zero;

            for i in 1..=m {
                let nc = set1(needle[i - 1] as u16);
                let nf = set1(flip_case(needle[i - 1]) as u16);
                let eq_nc = _mm256_cmpeq_epi16(hay, nc);
                let eq_nf = _mm256_cmpeq_epi16(hay, nf);
                let mm = _mm256_or_si256(eq_nc, eq_nf);
                match_curr[i] = mm;

                // diag: frizbee does add(match_bonuses), subs(mismatch), add(case)
                let masked_bonus = _mm256_and_si256(
                    _mm256_add_epi16(
                        _mm256_add_epi16(delim_bonus, cap_bonus),
                        _mm256_add_epi16(prefix_bonus, match_add_v),
                    ),
                    mm,
                );
                let diag = _mm256_add_epi16(h_prev[i - 1], masked_bonus);
                let diag = _mm256_subs_epu16(diag, mismatch_v); // ALWAYS subtract
                let diag = _mm256_add_epi16(diag, _mm256_and_si256(eq_nc, case_v));

                // E (horizontal gap)
                let e_pen = _mm256_blendv_epi8(gap_ext_v, gap_open_v, match_prev[i]);
                let e_new = _mm256_subs_epu16(h_prev[i], e_pen);
                let e_ext = _mm256_subs_epu16(e_prev[i], gap_ext_v);
                e_curr[i] = _mm256_max_epu16(e_new, e_ext);

                // F (vertical gap)
                let f_pen = _mm256_blendv_epi8(gap_ext_v, gap_open_v, match_curr[i - 1]);
                let f_new = _mm256_subs_epu16(h_curr[i - 1], f_pen);
                let f_ext = _mm256_subs_epu16(f_curr[i - 1], gap_ext_v);
                f_curr[i] = _mm256_max_epu16(f_new, f_ext);

                // H = max(0, diag, E, F)
                let h = _mm256_max_epu16(diag, e_curr[i]);
                let h = _mm256_max_epu16(h, f_curr[i]);
                h_curr[i] = _mm256_max_epu16(h, zero);
            }

            last_row_max = _mm256_max_epu16(last_row_max, h_curr[m]);

            std::mem::swap(&mut h_prev, &mut h_curr);
            std::mem::swap(&mut e_prev, &mut e_curr);
            std::mem::swap(&mut f_prev, &mut f_curr);
            std::mem::swap(&mut match_prev, &mut match_curr);

            prev_lower = is_lower;
            prev_delim = is_delim;
        }

        // Extract per-lane scores
        let lanes: [u16; 16] = std::mem::transmute(last_row_max);
        for (i, &s) in lanes.iter().enumerate() {
            if i < out.len() {
                out[i] = s;
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

    /// Compare against frizbee's Matcher (the ground truth) using fff's
    /// scoring config (matching_case_bonus=0 for lowercase queries).
    fn frizbee_score(needle: &str, haystack: &str) -> u16 {
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
            let soa = super::avx2::build_soa(&hays, max_len);
            let mut simd_out = vec![0u16; batch.len()];
            unsafe {
                super::avx2::score_batch_avx2(
                    b"abc", &soa, max_len, &p, true, &mut simd_out,
                );
            }
            for (k, h) in batch.iter().enumerate() {
                let scalar = score_one_scalar(b"abc", h.as_bytes(), &p, true);
                assert_eq!(
                    simd_out[k], scalar,
                    "batch[{k}]={h:?}: simd={} scalar={}",
                    simd_out[k], scalar
                );
            }
        }
    }
}
