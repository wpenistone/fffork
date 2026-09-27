//! Bit-parallel short-needle matching: Myers edit distance + LCS gate.
//!
//! FFF's typo model is *missing needle characters*: a hit with
//! `max_typos = T` satisfies `LCS(needle, path) >= |needle| - T`. That is the
//! exact accept condition of frizbee's ASCII typo prefilter (an ordered
//! alignment after deleting at most `T` needle bytes), so rejecting on
//! `LCS < |needle|-T` never drops a true match. Scoring stays on
//! Smith-Waterman — these kernels only gate.
//!
//! Why this beats affine-gap SW for short needles (2–32 bytes):
//! - SW: `O(|needle| * |haystack|)` cell updates, each carrying affine-gap
//!   state (diag/up/left + gap-open mask) through a SIMD matrix.
//! - LCS bit-vector: one word op per haystack byte for `|needle| <= 64`
//!   (`O(|haystack| * ceil(|needle|/64))`), no matrix, no gap state.
//! - Myers edit distance (Myers 1999) is the same shape and is kept for
//!   differential tests / callers that want a true Levenshtein bound.
//!
//! ## Algorithm
//!
//! LCS column `D[i] = LCS(needle[0..i], hay[0..j])` has `D[i]-D[i-1] ∈ {0,1}`,
//! so it is fully described by the *successor* bitmask `s` (bit `i` set iff
//! `D[i+1] == D[i] + 1`) and `LCS = popcount(s)`. Processing one haystack
//! byte `c` is the column recurrence
//! `D'[i] = max(D[i], D'[i-1], D[i-1] + [needle[i]==c])`
//! compiled to:
//! ```text
//! x = Peq[c] | s
//! y = ((s << 1) | 1) & mask
//! s = x & !(x - y)
//! ```
//! Pinned by `proptest_lcs_bit_matches_scalar` (2k random cases vs scalar DP).

/// Maximum needle length handled by the u64 bit-vector kernels.
pub const MAX_NEEDLE_BITS: usize = 64;

/// Case-fold an ASCII byte (uppercase → lowercase).
#[inline(always)]
pub fn fold(b: u8) -> u8 {
    if b.is_ascii_uppercase() {
        b + 32
    } else {
        b
    }
}

/// Per-byte needle mask: bit `i` set iff `fold(needle[i]) == c`.
/// Bits at or above `needle.len()` are always 0.
pub fn peq_table(needle: &[u8]) -> [u64; 256] {
    let mut peq = [0u64; 256];
    for (i, &b) in needle.iter().enumerate().take(MAX_NEEDLE_BITS) {
        peq[fold(b) as usize] |= 1u64 << i;
    }
    peq
}

#[inline]
fn bit_mask(m: usize) -> u64 {
    debug_assert!(m > 0 && m <= MAX_NEEDLE_BITS);
    if m == 64 {
        u64::MAX
    } else {
        (1u64 << m) - 1
    }
}

/// Exact LCS length (case-folded). Reference and long-needle fallback.
pub fn lcs_len_scalar(needle: &[u8], hay: &[u8]) -> usize {
    let m = needle.len();
    if m == 0 {
        return 0;
    }
    // Stack rows for the short-needle case (the common one) so the hot
    // prefilter never touches the allocator.
    if m <= 64 {
        let mut prev = [0u8; 65];
        let mut row = [0u8; 65];
        for &h in hay {
            let hf = fold(h);
            row[0] = 0;
            for j in 1..=m {
                if hf == fold(needle[j - 1]) {
                    row[j] = prev[j - 1] + 1;
                } else if row[j - 1] >= prev[j] {
                    row[j] = row[j - 1];
                } else {
                    row[j] = prev[j];
                }
            }
            std::mem::swap(&mut prev, &mut row);
        }
        return prev[m] as usize;
    }
    let mut prev = vec![0u16; m + 1];
    let mut row = vec![0u16; m + 1];
    for &h in hay {
        let hf = fold(h);
        row[0] = 0;
        for j in 1..=m {
            if hf == fold(needle[j - 1]) {
                row[j] = prev[j - 1] + 1;
            } else if row[j - 1] >= prev[j] {
                row[j] = row[j - 1];
            } else {
                row[j] = prev[j];
            }
        }
        std::mem::swap(&mut prev, &mut row);
    }
    prev[m] as usize
}

/// Exact LCS length for `|needle| <= 64` via the successor bit-vector.
/// Falls back to the scalar DP for longer needles.
pub fn lcs_len_bit(needle: &[u8], hay: &[u8]) -> usize {
    let m = needle.len();
    if m == 0 {
        return 0;
    }
    if m > MAX_NEEDLE_BITS {
        return lcs_len_scalar(needle, hay);
    }
    let mask = bit_mask(m);
    let peq = peq_table(needle);
    let mut s = 0u64;
    for &h in hay {
        let x = peq[fold(h) as usize] | s;
        let y = ((s << 1) | 1) & mask;
        s = x & !x.wrapping_sub(y);
    }
    s.count_ones() as usize
}

/// True if `LCS(needle, hay) >= need`. Early-exits once `need` successor
/// bits are set, or the remaining hay cannot close the gap.
#[inline]
pub fn lcs_ge_bit(needle: &[u8], hay: &[u8], need: usize) -> bool {
    let peq = peq_table(needle);
    lcs_ge_bit_peq(&peq, needle.len(), hay, need)
}

/// Same as [`lcs_ge_bit`] but reuses a precomputed `peq` table (2 KiB).
/// Hot path: Prefilter caches one table per needle instead of rebuilding
/// it for every haystack.
#[inline]
pub fn lcs_ge_bit_peq(peq: &[u64; 256], m: usize, hay: &[u8], need: usize) -> bool {
    if need == 0 {
        return true;
    }
    if m > MAX_NEEDLE_BITS {
        return false; // caller must not use peq path for long needles
    }
    if m == 0 || hay.len() < need {
        return false;
    }
    let mask = bit_mask(m);
    let mut s = 0u64;
    let mut hits = 0usize;
    let mut left = hay.len();
    for &h in hay {
        let x = peq[fold(h) as usize] | s;
        let y = ((s << 1) | 1) & mask;
        let ns = x & !x.wrapping_sub(y);
        if ns != s {
            s = ns;
            hits = s.count_ones() as usize;
            if hits >= need {
                return true;
            }
        }
        left -= 1;
        if hits + left < need {
            return false;
        }
    }
    hits >= need
}

/// Myers (1999) bit-vector Levenshtein distance, unit costs, `|needle| <= 64`.
pub fn edit_distance_bit(needle: &[u8], hay: &[u8]) -> u32 {
    let m = needle.len();
    if m == 0 {
        return hay.len() as u32;
    }
    if m > MAX_NEEDLE_BITS {
        return edit_distance_scalar(needle, hay);
    }
    let peq = peq_table(needle);
    let high = 1u64 << (m - 1);
    let mask = bit_mask(m);

    let mut pv = mask; // all ones
    let mut mv = 0u64;
    let mut score = m as i32;

    for &h in hay {
        let eq = peq[fold(h) as usize];
        let xv = eq | mv;
        let xh = (((eq & pv).wrapping_add(pv)) ^ pv) | eq;
        let ph = mv | !(xh | pv);
        let mh = pv & xh;

        if ph & high != 0 {
            score += 1;
        }
        if mh & high != 0 {
            score -= 1;
        }
        let ph = (ph << 1) | 1;
        let mh = mh << 1;
        pv = (mh | !(xv | ph)) & mask;
        mv = (ph & xv) & mask;
    }
    score as u32
}

fn edit_distance_scalar(needle: &[u8], hay: &[u8]) -> u32 {
    let n = needle.len();
    let mut prev: Vec<u32> = (0..=n as u32).collect();
    let mut row = vec![0u32; n + 1];
    for (j, &h) in hay.iter().enumerate() {
        row[0] = j as u32 + 1;
        let hf = fold(h);
        for i in 1..=n {
            let sub = prev[i - 1] + u32::from(fold(needle[i - 1]) != hf);
            row[i] = sub.min(prev[i] + 1).min(row[i - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut row);
    }
    prev[n]
}

/// Batched LCS gate over 8 haystacks sharing one needle (≤ 32 bytes).
/// Returns a bitmask: bit `k` set iff `LCS(needle, hays[k]) >= need`.
///
/// On x86-64 with AVX2 the eight successor vectors live in one `__m256i`
/// (8 × u32); only the 256-entry `Peq` gather stays scalar. That is the
/// "pack 8 short paths" shape from the perf mandate.
pub fn lcs_ge_batch8(needle: &[u8], hays: [&[u8]; 8], need: usize) -> u8 {
    let m = needle.len();
    if need == 0 {
        return 0xFF;
    }
    if m == 0 {
        return 0;
    }
    if m > 32 {
        let mut out = 0u8;
        for (k, hay) in hays.iter().enumerate() {
            if lcs_ge_bit(needle, hay, need) {
                out |= 1 << k;
            }
        }
        return out;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { lcs_ge_batch8_avx2(needle, hays, need) };
        }
    }
    lcs_ge_batch8_scalar(needle, hays, need)
}

fn lcs_ge_batch8_scalar(needle: &[u8], hays: [&[u8]; 8], need: usize) -> u8 {
    let m = needle.len();
    let peq = peq_table(needle);
    let mask = if m == 32 { u32::MAX } else { (1u32 << m) - 1 };

    let mut s = [0u32; 8];
    let mut hits = [0u32; 8];
    let mut left = [0usize; 8];
    for k in 0..8 {
        left[k] = hays[k].len();
    }
    let mut alive = 0xFFu8;
    let max_len = hays.iter().map(|h| h.len()).max().unwrap_or(0);

    for i in 0..max_len {
        for k in 0..8 {
            if alive & (1 << k) == 0 {
                continue;
            }
            let Some(&hb) = hays[k].get(i) else {
                alive &= !(1 << k);
                continue;
            };
            let msk = peq[fold(hb) as usize] as u32 & mask;
            let x = msk | s[k];
            let y = ((s[k] << 1) | 1) & mask;
            let ns = x & !x.wrapping_sub(y);
            if ns != s[k] {
                s[k] = ns;
                hits[k] = s[k].count_ones();
                if hits[k] as usize >= need {
                    alive &= !(1 << k);
                    continue;
                }
            }
            left[k] = left[k].saturating_sub(1);
            if hits[k] as usize + left[k] < need {
                alive &= !(1 << k);
            }
        }
        if alive == 0 {
            break;
        }
    }

    let mut out = 0u8;
    for k in 0..8 {
        if hits[k] as usize >= need {
            out |= 1 << k;
        }
    }
    out
}

/// Safety: caller must have confirmed `avx2` and `needle.len() <= 32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lcs_ge_batch8_avx2(needle: &[u8], hays: [&[u8]; 8], need: usize) -> u8 {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_andnot_si256, _mm256_or_si256, _mm256_set1_epi32,
        _mm256_set_epi32, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_storeu_si256,
        _mm256_sub_epi32,
    };

    let m = needle.len();
    let peq = peq_table(needle);
    let mask = if m == 32 { u32::MAX } else { (1u32 << m) - 1 };
    let mask_v = _mm256_set1_epi32(mask as i32);
    let one_v = _mm256_set1_epi32(1);

    let mut s_v = _mm256_setzero_si256();
    let mut hits = [0u32; 8];
    let mut left = [0usize; 8];
    for k in 0..8 {
        left[k] = hays[k].len();
    }
    let mut alive = 0xFFu8;
    let max_len = hays.iter().map(|h| h.len()).max().unwrap_or(0);

    for i in 0..max_len {
        // Gather Peq[c] for the 8 lanes (scalar table lookup is cheaper
        // than AVX2 gather on a 256-entry random table).
        let mut msk = [0i32; 8];
        let mut lane_alive = [false; 8];
        for k in 0..8 {
            if alive & (1 << k) == 0 {
                continue;
            }
            match hays[k].get(i) {
                Some(&hb) => {
                    msk[k] = peq[fold(hb) as usize] as i32 & mask as i32;
                    lane_alive[k] = true;
                }
                None => alive &= !(1 << k),
            }
        }
        let msk_v = _mm256_set_epi32(
            msk[7], msk[6], msk[5], msk[4], msk[3], msk[2], msk[1], msk[0],
        );

        // x = msk | s ; y = ((s << 1) | 1) & mask ; s' = x & !(x - y)
        let x = _mm256_or_si256(msk_v, s_v);
        let s_shift = _mm256_slli_epi32(s_v, 1);
        let y = _mm256_and_si256(_mm256_or_si256(s_shift, one_v), mask_v);
        let x_minus_y = _mm256_sub_epi32(x, y);
        let ns = _mm256_and_si256(x, _mm256_andnot_si256(x_minus_y, _mm256_set1_epi32(-1)));
        s_v = ns;

        let mut s_arr = [0i32; 8];
        _mm256_storeu_si256(s_arr.as_mut_ptr() as *mut __m256i, s_v);

        for k in 0..8 {
            if !lane_alive[k] {
                continue;
            }
            let bits = s_arr[k] as u32;
            if bits != hits[k] {
                // recount only when the vector changed
                hits[k] = bits.count_ones();
                if hits[k] as usize >= need {
                    alive &= !(1 << k);
                    continue;
                }
            }
            left[k] = left[k].saturating_sub(1);
            if hits[k] as usize + left[k] < need {
                alive &= !(1 << k);
            }
        }
        if alive == 0 {
            break;
        }
    }

    let mut out = 0u8;
    for k in 0..8 {
        if hits[k] as usize >= need {
            out |= 1 << k;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcs_ref(needle: &str, hay: &str) -> usize {
        lcs_len_scalar(needle.as_bytes(), hay.as_bytes())
    }

    #[test]
    fn lcs_bit_matches_scalar_basic() {
        let cases = [
            ("abc", "abc", 3),
            ("abc", "acb", 2),
            ("abc", "cba", 1),
            ("abc", "axc", 2),
            ("abc", "ba", 1),
            ("abc", "zzz", 0),
            ("abc", "", 0),
            ("", "abc", 0),
            // path has "file_picker." — trailing '_' is absent, LCS=11
            ("file_picker_", "crates/fff-core/src/file_picker.rs", 11),
            ("score", "crates/fff-core/src/score.rs", 5),
            ("aaa", "aa", 2),
            ("aaa", "a", 1),
            ("abcd", "acbd", 3),
            ("kitten", "sitting", 4),
        ];
        for (n, h, expect) in cases {
            assert_eq!(lcs_len_bit(n.as_bytes(), h.as_bytes()), expect, "{n} {h}");
            assert_eq!(lcs_ref(n, h), expect, "ref {n} {h}");
        }
    }

    #[test]
    fn lcs_ge_bit_agrees() {
        for (n, h) in [
            ("file_picker_", "src/file_picker.rs"),
            ("file_picker_", "nope"),
            ("abc", "aabbcc"),
            ("abc", "ab"),
            ("abc", "ba"),
        ] {
            for need in 0..=n.len() {
                assert_eq!(
                    lcs_ge_bit(n.as_bytes(), h.as_bytes(), need),
                    lcs_ref(n, h) >= need,
                    "{n} {h} need={need}"
                );
            }
        }
    }

    #[test]
    fn edit_distance_bit_matches_scalar() {
        let cases = [
            ("abc", "abc", 0),
            ("abc", "abd", 1),
            ("abc", "axc", 1),
            // two substitutions (a↔c) beat three indels
            ("abc", "cba", 2),
            ("abc", "", 3),
            ("", "abc", 3),
            ("kitten", "sitting", 3),
        ];
        for (n, h, expect) in cases {
            assert_eq!(
                edit_distance_bit(n.as_bytes(), h.as_bytes()),
                expect,
                "{n} {h}"
            );
            assert_eq!(
                edit_distance_scalar(n.as_bytes(), h.as_bytes()),
                expect,
                "ref {n} {h}"
            );
        }
    }

    #[test]
    fn batch8_matches_scalar() {
        let needle = "file_picker_";
        let need = needle.len() - 1;
        let hays = [
            &b"crates/fff-core/src/file_picker.rs"[..],
            &b"nope_at_all_here"[..],
            &b"src/file_picker.rs"[..],
            &b"ffffffffffffffffffffffff"[..],
            &b"a/b/c/d/e/f/g/h"[..],
            &b"file_picker_"[..],
            &b"picker_file_"[..],
            &b""[..],
        ];
        let bits = lcs_ge_batch8(needle.as_bytes(), hays, need);
        for k in 0..8 {
            let expect = lcs_ge_bit(needle.as_bytes(), hays[k], need);
            assert_eq!(bits & (1 << k) != 0, expect, "lane {k}");
        }
    }

    #[test]
    fn proptest_lcs_bit_matches_scalar() {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = |n: u32| -> u32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) % n as u64) as u32
        };
        for _ in 0..2000 {
            let nl = 1 + next(12) as usize;
            let hl = next(40) as usize;
            let needle: Vec<u8> = (0..nl).map(|_| b'a' + next(3) as u8).collect();
            let hay: Vec<u8> = (0..hl).map(|_| b'a' + next(3) as u8).collect();
            let a = lcs_len_scalar(&needle, &hay);
            let b = lcs_len_bit(&needle, &hay);
            assert_eq!(a, b, "needle={needle:?} hay={hay:?}");
            let need = a.saturating_sub(1);
            assert_eq!(lcs_ge_bit(&needle, &hay, need), a >= need);
        }
    }

    #[test]
    fn proptest_batch8_matches_scalar() {
        let mut seed = 0xDEADBEEFCAFEF00Du64;
        let mut next = |n: u32| -> u32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) % n as u64) as u32
        };
        for _ in 0..200 {
            let nl = 1 + next(16) as usize;
            let needle: Vec<u8> = (0..nl).map(|_| b'a' + next(3) as u8).collect();
            let mut hays: [&[u8]; 8] = [&[]; 8];
            let mut backing: Vec<Vec<u8>> = Vec::new();
            for _ in 0..8 {
                let hl = next(30) as usize;
                backing.push((0..hl).map(|_| b'a' + next(3) as u8).collect());
            }
            for k in 0..8 {
                hays[k] = &backing[k];
            }
            let need = needle.len().saturating_sub(1).max(1);
            let bits = lcs_ge_batch8(&needle, hays, need);
            for k in 0..8 {
                let expect = lcs_ge_bit(&needle, hays[k], need);
                assert_eq!(bits & (1 << k) != 0, expect, "lane {k} needle={needle:?}");
            }
        }
    }
}
