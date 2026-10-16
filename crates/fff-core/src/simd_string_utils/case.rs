#[inline]
pub fn ascii_swap_case(b: u8) -> u8 {
    b ^ 0x20
}

#[inline]
fn eq_lowered_scalar(h: *const u8, needle_lower: &[u8]) -> bool {
    for (i, &n) in needle_lower.iter().enumerate() {
        if unsafe { *h.add(i) }.to_ascii_lowercase() != n {
            return false;
        }
    }
    true
}

/// AVX2 only has a **signed** byte compare (`cmpgt`), but we need an
/// **unsigned** range check (`'A' <= byte <= 'Z'`). XOR-ing every byte with
/// `0x80` maps the unsigned range `[0, 255]` into the signed range
/// `[-128, 127]` preserving order, so signed `cmpgt` becomes correct
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn eq_lowered_avx2(h: *const u8, needle_lower: &[u8]) -> bool {
    use core::arch::x86_64::*;

    let len = needle_lower.len();
    let mut i = 0usize;
    // Signed wrap maps 'A'..='Z' above 101; all else wraps below or stays short.
    let add_upper = _mm256_set1_epi8((127 - b'Z') as i8);
    let thresh = _mm256_set1_epi8(101);
    let bit20 = _mm256_set1_epi8(0x20);

    while i + 32 <= len {
        let hv = unsafe { _mm256_loadu_si256(h.add(i) as *const __m256i) };
        let nv = unsafe { _mm256_loadu_si256(needle_lower.as_ptr().add(i) as *const __m256i) };

        let upper = _mm256_cmpgt_epi8(_mm256_add_epi8(hv, add_upper), thresh);
        let folded = _mm256_or_si256(hv, _mm256_and_si256(upper, bit20));

        let eq = _mm256_cmpeq_epi8(folded, nv);
        if _mm256_movemask_epi8(eq) != -1i32 {
            return false;
        }
        i += 32;
    }
    // Overlapped tail: one more full vector covers any remainder.
    if len >= 32 && i < len {
        let tail = len - 32;
        let hv = unsafe { _mm256_loadu_si256(h.add(tail) as *const __m256i) };
        let nv = unsafe { _mm256_loadu_si256(needle_lower.as_ptr().add(tail) as *const __m256i) };
        let upper = _mm256_cmpgt_epi8(_mm256_add_epi8(hv, add_upper), thresh);
        let folded = _mm256_or_si256(hv, _mm256_and_si256(upper, bit20));
        return _mm256_movemask_epi8(_mm256_cmpeq_epi8(folded, nv)) == -1i32;
    }
    true
}

/// Unsigned range checks (`vcge`/`vcle`) detect uppercase ASCII, bit 5 folds
/// to lowercase, then one horizontal max over the xor tells equal or not.
/// Baseline NEON only: no dotprod gate, runs on every aarch64 target.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn eq_lowered_neon(h: *const u8, needle_lower: &[u8]) -> bool {
    use core::arch::aarch64::*;

    let len = needle_lower.len();
    let mut i = 0usize;
    let a_val = vdupq_n_u8(b'A');
    let z_val = vdupq_n_u8(b'Z');
    let bit20 = vdupq_n_u8(0x20);

    while i + 16 <= len {
        let hv = unsafe { vld1q_u8(h.add(i)) };
        let nv = unsafe { vld1q_u8(needle_lower.as_ptr().add(i)) };
        let upper = vandq_u8(vcgeq_u8(hv, a_val), vcleq_u8(hv, z_val));
        let folded = vorrq_u8(hv, vandq_u8(upper, bit20));
        let xored = veorq_u8(folded, nv);
        if vmaxvq_u32(vreinterpretq_u32_u8(xored)) != 0 {
            return false;
        }
        i += 16;
    }
    if len >= 16 && i < len {
        let tail = len - 16;
        let hv = unsafe { vld1q_u8(h.add(tail)) };
        let nv = unsafe { vld1q_u8(needle_lower.as_ptr().add(tail)) };
        let upper = vandq_u8(vcgeq_u8(hv, a_val), vcleq_u8(hv, z_val));
        let folded = vorrq_u8(hv, vandq_u8(upper, bit20));
        let xored = veorq_u8(folded, nv);
        return vmaxvq_u32(vreinterpretq_u32_u8(xored)) == 0;
    }
    true
}

/// 8-byte SWAR fold-and-compare: uppercase lanes fold via bit tricks,
/// then one u64 compare covers 8 bytes for short needles.
#[inline(always)]
unsafe fn eq_lowered_swar_u64(h: *const u8, n: *const u8) -> bool {
    let hv = unsafe { core::ptr::read_unaligned(h as *const u64) };
    let nv = unsafe { core::ptr::read_unaligned(n as *const u64) };
    // A byte in 0..=127 is 'A'..='Z' iff (b + 0x3F) keeps bit 7
    // while (b + 0x25) drops it; masked adds cannot carry lanes.
    let is_ascii = !hv & 0x8080_8080_8080_8080u64;
    let ge_a = (hv & 0x7F7F_7F7F_7F7F_7F7Fu64).wrapping_add(0x3F3F_3F3F_3F3F_3F3Fu64);
    let gt_z = (hv & 0x7F7F_7F7F_7F7F_7F7Fu64).wrapping_add(0x2525_2525_2525_2525u64);
    let upper_mask = ((ge_a & !gt_z & is_ascii) >> 2) & 0x2020_2020_2020_2020u64;
    (hv | upper_mask) == nv
}

/// Case-insensitive equality of `needle_lower` against the haystack bytes
/// starting at `h`. `needle_lower` must be pre-lowercased (ASCII).
///
/// # Safety
/// `h` must be valid for reads of `needle_lower.len()` bytes.
#[inline]
pub(crate) unsafe fn eq_lowered_case(haystack: *const u8, needle_lower: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if needle_lower.len() >= 32 && super::memmem::has_avx2() {
            return unsafe { eq_lowered_avx2(haystack, needle_lower) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if needle_lower.len() >= 16 {
            return unsafe { eq_lowered_neon(haystack, needle_lower) };
        }
    }
    // Short needles: one SWAR word per 8 bytes beats the byte loop.
    if needle_lower.len() >= 8 {
        let len = needle_lower.len();
        let mut i = 0usize;
        while i + 8 <= len {
            if !unsafe { eq_lowered_swar_u64(haystack.add(i), needle_lower.as_ptr().add(i)) } {
                return false;
            }
            i += 8;
        }
        if i < len {
            let tail = len - 8;
            return unsafe {
                eq_lowered_swar_u64(haystack.add(tail), needle_lower.as_ptr().add(tail))
            };
        }
        return true;
    }
    eq_lowered_scalar(haystack, needle_lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eq_lowered(haystack: &[u8], needle_lower: &[u8]) -> bool {
        assert!(haystack.len() >= needle_lower.len());
        unsafe { eq_lowered_case(haystack.as_ptr(), needle_lower) }
    }

    #[test]
    fn swap_case_toggles_letters() {
        assert_eq!(ascii_swap_case(b'n'), b'N');
        assert_eq!(ascii_swap_case(b'N'), b'n');
        assert_eq!(ascii_swap_case(b'z'), b'Z');
    }

    #[test]
    fn eq_matches_std_semantics() {
        assert!(eq_lowered(b"Hello", b"hello"));
        assert!(eq_lowered(b"HELLO WORLD", b"hello"));
        assert!(!eq_lowered(b"Hellp", b"hello"));
        // Non-letters must not fold: '[' (0x5B) vs '{' (0x7B) differ only in bit 5.
        assert!(!eq_lowered(b"A[", b"a{"));
        assert!(eq_lowered(b"A{", b"a{"));
        // Long inputs exercise the SIMD kernels.
        let hay = b"INT STRUCT MUTEX *LOCK(STRUCT MUTEX *LOCK) { RETURN 0; }";
        let needle: Vec<u8> = hay.iter().map(|b| b.to_ascii_lowercase()).collect();
        assert!(eq_lowered(hay, &needle));
        let mut bad = needle.clone();
        *bad.last_mut().unwrap() = b'!';
        assert!(!eq_lowered(hay, &bad));
        // Mid lengths exercise the SWAR tier (8..32 bytes).
        let hay12 = b"AbC_dEf-GhI";
        assert!(eq_lowered(hay12, b"abc_def-ghi"));
        assert!(!eq_lowered(hay12, b"abc_def-ghj"));
        let hay31 = b"AbCdEfGhIjKlMnOpQrStUvWxYz01239";
        assert_eq!(hay31.len(), 31);
        let ndl31: Vec<u8> = hay31.iter().map(|b| b.to_ascii_lowercase()).collect();
        assert!(eq_lowered(hay31, &ndl31));
        let mut bad31 = ndl31.clone();
        bad31[15] = b'!';
        assert!(!eq_lowered(hay31, &bad31));
    }
}
