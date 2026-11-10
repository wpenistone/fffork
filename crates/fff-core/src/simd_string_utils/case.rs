#[inline]
pub fn ascii_swap_case(b: u8) -> u8 {
    b ^ 0x20
}

#[inline(always)]
fn eq_lowered_scalar(h: *const u8, needle_lower: &[u8]) -> bool {
    for (i, &n) in needle_lower.iter().enumerate() {
        if unsafe { *h.add(i) }.to_ascii_lowercase() != n {
            return false;
        }
    }
    true
}

// `rem` is 1, 2, or 3. Compares only `[at, at + rem)`.
#[inline(always)]
unsafe fn eq_rem(h: *const u8, n: *const u8, at: usize, rem: usize) -> bool {
    unsafe {
        debug_assert!(rem >= 1 && rem < 4);
        if (*h.add(at)).to_ascii_lowercase() != *n.add(at) {
            return false;
        }
        if rem == 1 {
            return true;
        }
        if (*h.add(at + 1)).to_ascii_lowercase() != *n.add(at + 1) {
            return false;
        }
        rem == 2 || (*h.add(at + 2)).to_ascii_lowercase() == *n.add(at + 2)
    }
}

// Signed `cmpgt` sees 'A'..='Z' only: `b.wrapping_add(37)` lands in 102..=127 there.
// Caller requires `needle_lower.len() >= 32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn eq_lowered_avx2(h: *const u8, needle_lower: &[u8]) -> bool {
    use core::arch::x86_64::*;

    let len = needle_lower.len();
    debug_assert!(len >= 32);
    let mut i = 0usize;
    let add_upper = _mm256_set1_epi8((127 - b'Z') as i8);
    let thresh = _mm256_set1_epi8(101);
    let bit20 = _mm256_set1_epi8(0x20);

    while i + 64 <= len {
        let hv0 = unsafe { _mm256_loadu_si256(h.add(i) as *const __m256i) };
        let hv1 = unsafe { _mm256_loadu_si256(h.add(i + 32) as *const __m256i) };
        let nv0 = unsafe { _mm256_loadu_si256(needle_lower.as_ptr().add(i) as *const __m256i) };
        let nv1 =
            unsafe { _mm256_loadu_si256(needle_lower.as_ptr().add(i + 32) as *const __m256i) };
        let upper0 = _mm256_cmpgt_epi8(_mm256_add_epi8(hv0, add_upper), thresh);
        let upper1 = _mm256_cmpgt_epi8(_mm256_add_epi8(hv1, add_upper), thresh);
        let folded0 = _mm256_or_si256(hv0, _mm256_and_si256(upper0, bit20));
        let folded1 = _mm256_or_si256(hv1, _mm256_and_si256(upper1, bit20));
        let eq0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(folded0, nv0));
        let eq1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(folded1, nv1));
        if eq0 != -1i32 || eq1 != -1i32 {
            return false;
        }
        i += 64;
    }
    while i + 32 <= len {
        let hv = unsafe { _mm256_loadu_si256(h.add(i) as *const __m256i) };
        let nv = unsafe { _mm256_loadu_si256(needle_lower.as_ptr().add(i) as *const __m256i) };
        let upper = _mm256_cmpgt_epi8(_mm256_add_epi8(hv, add_upper), thresh);
        let folded = _mm256_or_si256(hv, _mm256_and_si256(upper, bit20));
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(folded, nv)) != -1i32 {
            return false;
        }
        i += 32;
    }
    if i == len {
        return true;
    }
    // The last 32 bytes end at `len`, so the unread tail is inside this load.
    let tail = len - 32;
    let hv = unsafe { _mm256_loadu_si256(h.add(tail) as *const __m256i) };
    let nv = unsafe { _mm256_loadu_si256(needle_lower.as_ptr().add(tail) as *const __m256i) };
    let upper = _mm256_cmpgt_epi8(_mm256_add_epi8(hv, add_upper), thresh);
    let folded = _mm256_or_si256(hv, _mm256_and_si256(upper, bit20));
    _mm256_movemask_epi8(_mm256_cmpeq_epi8(folded, nv)) == -1i32
}

// Signed `cmpgt` sees 'A'..='Z' only. Caller requires `16 <= needle_lower.len() < 32`.
// A shorter tail is the last 16 bytes of the same slice.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn eq_lowered_avx2_16(h: *const u8, needle_lower: &[u8]) -> bool {
    use core::arch::x86_64::*;

    let len = needle_lower.len();
    debug_assert!(len >= 16 && len < 32);
    let n = needle_lower.as_ptr();
    let add_upper = _mm_set1_epi8((127 - b'Z') as i8);
    let thresh = _mm_set1_epi8(101);
    let bit20 = _mm_set1_epi8(0x20);

    let hv0 = unsafe { _mm_loadu_si128(h as *const __m128i) };
    let nv0 = unsafe { _mm_loadu_si128(n as *const __m128i) };
    if len == 16 {
        let upper = _mm_cmpgt_epi8(_mm_add_epi8(hv0, add_upper), thresh);
        let folded = _mm_or_si128(hv0, _mm_and_si128(upper, bit20));
        return _mm_movemask_epi8(_mm_cmpeq_epi8(folded, nv0)) == 0xFFFF;
    }
    // The last 16 bytes end at `len`. Both folds are in range and do not depend on each other.
    let tail = len - 16;
    let hv1 = unsafe { _mm_loadu_si128(h.add(tail) as *const __m128i) };
    let nv1 = unsafe { _mm_loadu_si128(n.add(tail) as *const __m128i) };
    let upper0 = _mm_cmpgt_epi8(_mm_add_epi8(hv0, add_upper), thresh);
    let upper1 = _mm_cmpgt_epi8(_mm_add_epi8(hv1, add_upper), thresh);
    let folded0 = _mm_or_si128(hv0, _mm_and_si128(upper0, bit20));
    let folded1 = _mm_or_si128(hv1, _mm_and_si128(upper1, bit20));
    let eq0 = _mm_movemask_epi8(_mm_cmpeq_epi8(folded0, nv0));
    let eq1 = _mm_movemask_epi8(_mm_cmpeq_epi8(folded1, nv1));
    eq0 == 0xFFFF && eq1 == 0xFFFF
}

// Low 8 bytes. Caller requires `8 <= needle_lower.len() < 16`.
// The unread tail is an overlapping 8-byte load that ends at `len`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn eq_lowered_sse8(h: *const u8, needle_lower: &[u8]) -> bool {
    use core::arch::x86_64::*;

    let len = needle_lower.len();
    debug_assert!(len >= 8 && len < 16);
    let n = needle_lower.as_ptr();
    let add_upper = _mm_set1_epi8((127 - b'Z') as i8);
    let thresh = _mm_set1_epi8(101);
    let bit20 = _mm_set1_epi8(0x20);
    let hv0 = unsafe { _mm_loadl_epi64(h as *const __m128i) };
    let nv0 = unsafe { _mm_loadl_epi64(n as *const __m128i) };
    if len == 8 {
        let upper = _mm_cmpgt_epi8(_mm_add_epi8(hv0, add_upper), thresh);
        let folded = _mm_or_si128(hv0, _mm_and_si128(upper, bit20));
        return (_mm_movemask_epi8(_mm_cmpeq_epi8(folded, nv0)) & 0xFF) == 0xFF;
    }
    // The last 8 bytes end at `len`. Both folds are in range and do not depend on each other.
    let tail = len - 8;
    let hv1 = unsafe { _mm_loadl_epi64(h.add(tail) as *const __m128i) };
    let nv1 = unsafe { _mm_loadl_epi64(n.add(tail) as *const __m128i) };
    let upper0 = _mm_cmpgt_epi8(_mm_add_epi8(hv0, add_upper), thresh);
    let upper1 = _mm_cmpgt_epi8(_mm_add_epi8(hv1, add_upper), thresh);
    let folded0 = _mm_or_si128(hv0, _mm_and_si128(upper0, bit20));
    let folded1 = _mm_or_si128(hv1, _mm_and_si128(upper1, bit20));
    let eq0 = _mm_movemask_epi8(_mm_cmpeq_epi8(folded0, nv0)) & 0xFF;
    let eq1 = _mm_movemask_epi8(_mm_cmpeq_epi8(folded1, nv1)) & 0xFF;
    eq0 == 0xFF && eq1 == 0xFF
}

// Uppercase lanes fold with bit 5; any nonzero xor byte fails the match.
// Caller requires `needle_lower.len() >= 16`. Baseline NEON, no dotprod.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn eq_lowered_neon(h: *const u8, needle_lower: &[u8]) -> bool {
    use core::arch::aarch64::*;

    let len = needle_lower.len();
    debug_assert!(len >= 16);
    let mut i = 0usize;
    let a_val = vdupq_n_u8(b'A');
    let z_val = vdupq_n_u8(b'Z');
    let bit20 = vdupq_n_u8(0x20);

    while i + 16 <= len {
        let hv = unsafe { vld1q_u8(h.add(i)) };
        let nv = unsafe { vld1q_u8(needle_lower.as_ptr().add(i)) };
        let upper = vandq_u8(vcgeq_u8(hv, a_val), vcleq_u8(hv, z_val));
        let folded = vorrq_u8(hv, vandq_u8(upper, bit20));
        if vmaxvq_u32(vreinterpretq_u32_u8(veorq_u8(folded, nv))) != 0 {
            return false;
        }
        i += 16;
    }
    if i == len {
        return true;
    }
    // The last 16 bytes end at `len`, so the unread tail is inside this load.
    let tail = len - 16;
    let hv = unsafe { vld1q_u8(h.add(tail)) };
    let nv = unsafe { vld1q_u8(needle_lower.as_ptr().add(tail)) };
    let upper = vandq_u8(vcgeq_u8(hv, a_val), vcleq_u8(hv, z_val));
    let folded = vorrq_u8(hv, vandq_u8(upper, bit20));
    vmaxvq_u32(vreinterpretq_u32_u8(veorq_u8(folded, nv))) == 0
}

// 'A'..='Z' keeps bit 7 of (b + 0x3F) and clears it for (b + 0x25), only when b < 128.
// Those adds are masked so they cannot carry; the surviving bit shifts onto 0x20.
#[inline(always)]
unsafe fn eq_swar64(h: *const u8, n: *const u8) -> bool {
    let hv = unsafe { core::ptr::read_unaligned(h as *const u64) };
    let nv = unsafe { core::ptr::read_unaligned(n as *const u64) };
    let is_ascii = !hv & 0x8080_8080_8080_8080u64;
    let masked = hv & 0x7F7F_7F7F_7F7F_7F7Fu64;
    let ge_a = masked.wrapping_add(0x3F3F_3F3F_3F3F_3F3Fu64);
    let gt_z = masked.wrapping_add(0x2525_2525_2525_2525u64);
    let upper = ((ge_a & !gt_z & is_ascii) >> 2) & 0x2020_2020_2020_2020u64;
    (hv | upper) == nv
}

#[inline(always)]
unsafe fn eq_swar32(h: *const u8, n: *const u8) -> bool {
    let hv = unsafe { core::ptr::read_unaligned(h as *const u32) };
    let nv = unsafe { core::ptr::read_unaligned(n as *const u32) };
    let is_ascii = !hv & 0x8080_8080u32;
    let masked = hv & 0x7F7F_7F7Fu32;
    let ge_a = masked.wrapping_add(0x3F3F_3F3F);
    let gt_z = masked.wrapping_add(0x2525_2525);
    let upper = ((ge_a & !gt_z & is_ascii) >> 2) & 0x2020_2020u32;
    (hv | upper) == nv
}

#[inline(always)]
unsafe fn eq_lowered_swar(h: *const u8, needle_lower: &[u8]) -> bool {
    let len = needle_lower.len();
    debug_assert!(len >= 4);
    let n = needle_lower.as_ptr();
    if len >= 8 {
        let mut i = 0usize;
        while i + 8 <= len {
            if unsafe { !eq_swar64(h.add(i), n.add(i)) } {
                return false;
            }
            i += 8;
        }
        if i == len {
            return true;
        }
        let rem = len - i;
        // 1..=4 new bytes are compared without rereading the previous word.
        if rem < 4 {
            return unsafe { eq_rem(h, n, i, rem) };
        }
        if rem == 4 {
            return unsafe { eq_swar32(h.add(i), n.add(i)) };
        }
        return unsafe { eq_swar64(h.add(len - 8), n.add(len - 8)) };
    }
    if unsafe { !eq_swar32(h, n) } {
        return false;
    }
    if len == 4 {
        return true;
    }
    // Remaining 1..=3 bytes sit past the first word.
    unsafe { eq_rem(h, n, 4, len - 4) }
}

// Caller is already on AVX2. `needle_lower` is pre-lowercased ASCII.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn confirm_avx2(h: *const u8, needle_lower: &[u8]) -> bool {
    let len = needle_lower.len();
    if len >= 32 {
        return unsafe { eq_lowered_avx2(h, needle_lower) };
    }
    if len >= 16 {
        return unsafe { eq_lowered_avx2_16(h, needle_lower) };
    }
    if len >= 8 {
        return unsafe { eq_lowered_sse8(h, needle_lower) };
    }
    if len >= 4 {
        return unsafe { eq_lowered_swar(h, needle_lower) };
    }
    eq_lowered_scalar(h, needle_lower)
}

// `needle_lower` must be pre-lowercased ASCII.
// `haystack` must be readable for `needle_lower.len()` bytes.
#[inline]
pub(crate) unsafe fn eq_lowered_case(haystack: *const u8, needle_lower: &[u8]) -> bool {
    let len = needle_lower.len();
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 16 && super::memmem::has_avx2() {
            if len >= 32 {
                return unsafe { eq_lowered_avx2(haystack, needle_lower) };
            }
            return unsafe { eq_lowered_avx2_16(haystack, needle_lower) };
        }
        if len >= 8 && len < 16 {
            return unsafe { eq_lowered_sse8(haystack, needle_lower) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if len >= 16 {
            return unsafe { eq_lowered_neon(haystack, needle_lower) };
        }
    }
    if len >= 4 {
        return unsafe { eq_lowered_swar(haystack, needle_lower) };
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
        for len in 4..=96 {
            let hay: Vec<u8> = (0..len).map(|i| b'A' + (i % 26) as u8).collect();
            let needle: Vec<u8> = hay.iter().map(|b| b.to_ascii_lowercase()).collect();
            assert!(eq_lowered(&hay, &needle));
            for at in 0..len {
                let mut bad_h = hay.clone();
                bad_h[at] = b'0';
                assert!(!eq_lowered(&bad_h, &needle));
            }
        }
    }
}
