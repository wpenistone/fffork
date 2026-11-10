use super::case::{ascii_swap_case, eq_lowered_case};
#[cfg(target_arch = "x86_64")]
use super::case::confirm_avx2;
use std::sync::OnceLock;

// Cache the CPUID probe: `is_x86_feature_detected!` costs more than a short
// memmem itself, and grep calls `find` once per file.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

// Byte frequency table stolen from memchr
const BYTE_FREQUENCIES: [u8; 256] = [
    55, 52, 51, 50, 49, 48, 47, 46, 45, 103, 242, 66, 67, 229, 44, 43, // 0x00
    42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 56, 32, 31, 30, 29, 28, // 0x10
    255, 148, 164, 149, 136, 160, 155, 173, 221, 222, 134, 122, 232, 202, 215, 224, // 0x20
    208, 220, 204, 187, 183, 179, 177, 168, 178, 200, 226, 195, 154, 184, 174, 126, // 0x30
    120, 191, 157, 194, 170, 189, 162, 161, 150, 193, 142, 137, 171, 176, 185,
    167, // 0x40 A-O
    186, 112, 175, 192, 188, 156, 140, 143, 123, 133, 128, 147, 138, 146, 114,
    223, // 0x50 P-_
    151, 249, 216, 238, 236, 253, 227, 218, 230, 247, 135, 180, 241, 233, 246,
    244, // 0x60 a-o
    231, 139, 245, 243, 251, 235, 201, 196, 240, 214, 152, 182, 205, 181, 127,
    27, // 0x70 p-DEL
    212, 211, 210, 213, 228, 197, 169, 159, 131, 172, 105, 80, 98, 96, 97, 81, // 0x80
    207, 145, 116, 115, 144, 130, 153, 121, 107, 132, 109, 110, 124, 111, 82, 108, // 0x90
    118, 141, 113, 129, 119, 125, 165, 117, 92, 106, 83, 72, 99, 93, 65, 79, // 0xa0
    166, 237, 163, 199, 190, 225, 209, 203, 198, 217, 219, 206, 234, 248, 158, 239, // 0xb0
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, // 0xc0
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, // 0xd0
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, // 0xe0
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, // 0xf0
];

#[inline]
fn rank(lower: u8) -> u8 {
    if lower.is_ascii_lowercase() {
        BYTE_FREQUENCIES[lower as usize].max(BYTE_FREQUENCIES[ascii_swap_case(lower) as usize])
    } else {
        BYTE_FREQUENCIES[lower as usize]
    }
}

/// Pick two distinct needle positions with the rarest bytes (case-insensitive).
/// Rank 255 is a real frequency, so a sentinel of 255 must not drop the second index.
fn select_rare_pair(needle_lower: &[u8]) -> (usize, usize) {
    debug_assert!(needle_lower.len() >= 2);

    let mut best1 = (rank(needle_lower[0]), 0usize);
    let mut best2 = (rank(needle_lower[1]), 1usize);
    if best2.0 < best1.0 {
        std::mem::swap(&mut best1, &mut best2);
    }

    for (i, &b) in needle_lower.iter().enumerate().skip(2) {
        let r = rank(b);
        if r < best1.0 {
            best2 = best1;
            best1 = (r, i);
        } else if r < best2.0 {
            best2 = (r, i);
        }
    }

    let i1 = best1.1.min(best2.1);
    let i2 = best1.1.max(best2.1);
    debug_assert!(i1 != i2);
    (i1, i2)
}

#[inline]
fn alt_byte(b: u8) -> u8 {
    if b.is_ascii_lowercase() {
        ascii_swap_case(b)
    } else {
        b
    }
}

// Built once per needle. `find` repeats it per haystack; grep must not.
#[derive(Clone, Copy)]
pub(crate) struct PairPlan {
    i1: usize,
    i2: usize,
    rare_pos: usize,
    b1: u8,
    b1_alt: u8,
    b2: u8,
    b2_alt: u8,
    #[cfg(target_arch = "aarch64")]
    first_rank: u8,
    #[cfg(target_arch = "x86_64")]
    avx2: bool,
}

impl PairPlan {
    // Rare byte, then the other pair byte. `alt` is the other ASCII case.
    #[inline]
    pub(crate) fn pair_ends(self) -> (usize, u8, u8, usize, u8, u8) {
        if self.rare_pos == self.i1 {
            (self.rare_pos, self.b1, self.b1_alt, self.i2, self.b2, self.b2_alt)
        } else {
            (self.rare_pos, self.b2, self.b2_alt, self.i1, self.b1, self.b1_alt)
        }
    }
}

pub(crate) fn plan_for(needle_lower: &[u8]) -> PairPlan {
    if needle_lower.len() < 2 {
        return PairPlan {
            i1: 0,
            i2: 0,
            rare_pos: 0,
            b1: 0,
            b1_alt: 0,
            b2: 0,
            b2_alt: 0,
            #[cfg(target_arch = "aarch64")]
            first_rank: 0,
            #[cfg(target_arch = "x86_64")]
            avx2: false,
        };
    }
    let (i1, i2) = select_rare_pair(needle_lower);
    let b1 = needle_lower[i1];
    let b2 = needle_lower[i2];
    PairPlan {
        i1,
        i2,
        rare_pos: if rank(b1) <= rank(b2) { i1 } else { i2 },
        b1,
        b1_alt: alt_byte(b1),
        b2,
        b2_alt: alt_byte(b2),
        #[cfg(target_arch = "aarch64")]
        first_rank: rank(needle_lower[0]),
        #[cfg(target_arch = "x86_64")]
        avx2: has_avx2(),
    }
}

/// Nibble *i* is 0xF iff byte *i* matched. One narrowing shift, no horizontal add.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_nibble_mask(v: core::arch::aarch64::uint8x16_t) -> u64 {
    use core::arch::aarch64::*;
    let narrowed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(v));
    vget_lane_u64::<0>(vreinterpret_u64_u8(narrowed))
}

// `other_pos` is the pair byte memchr did not test. Length 2 needs only that byte.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn tail_hit(
    ptr: *const u8,
    needle_lower: &[u8],
    candidate: usize,
    rare_pos: usize,
    other_pos: usize,
    other: u8,
    other_alt: u8,
) -> bool {
    unsafe {
        let b = *ptr.add(candidate + other_pos);
        if b != other && b != other_alt {
            return false;
        }
        let n = needle_lower.len();
        if n == 2 {
            return true;
        }
        if n == 3 {
            let m = 3 - rare_pos - other_pos;
            return (*ptr.add(candidate + m)).to_ascii_lowercase() == needle_lower[m];
        }
        eq_lowered_case(ptr.add(candidate), needle_lower)
    }
}

// Match starts at `offset + pos`, where `pos` is the index in the tail slice.
#[cfg(target_arch = "aarch64")]
fn scan_rare_tail(
    haystack: &[u8],
    needle_lower: &[u8],
    offset: usize,
    last_start: usize,
    rare_pos: usize,
    other_pos: usize,
) -> Option<usize> {
    if offset > last_start {
        return None;
    }
    let rare_byte = needle_lower[rare_pos];
    let tail_start = offset + rare_pos;
    let tail_end = last_start + rare_pos + 1;
    if tail_start < tail_end {
        let ptr = haystack.as_ptr();
        let tail_space = &haystack[tail_start..tail_end];
        let other = needle_lower[other_pos];
        let other_alt = alt_byte(other);
        if rare_byte.is_ascii_lowercase() {
            for pos in memchr::memchr2_iter(rare_byte, ascii_swap_case(rare_byte), tail_space) {
                let candidate = offset + pos;
                if unsafe {
                    tail_hit(
                        ptr,
                        needle_lower,
                        candidate,
                        rare_pos,
                        other_pos,
                        other,
                        other_alt,
                    )
                } {
                    return Some(candidate);
                }
            }
        } else {
            for pos in memchr::memchr_iter(rare_byte, tail_space) {
                let candidate = offset + pos;
                if unsafe {
                    tail_hit(
                        ptr,
                        needle_lower,
                        candidate,
                        rare_pos,
                        other_pos,
                        other,
                        other_alt,
                    )
                } {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

// Bits are visited low to high, so the first confirmed candidate is leftmost.
// Caller passes a nonzero mask. Length 2: both pair lanes are the whole needle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn first_confirmed(
    ptr: *const u8,
    needle_lower: &[u8],
    offset: usize,
    mut mask: u32,
    i1: usize,
    i2: usize,
) -> Option<usize> {
    debug_assert!(mask != 0);
    let n = needle_lower.len();
    // Both lanes are the whole needle only when the pair is exactly the two bytes.
    if n == 2 && i1 + i2 == 1 {
        return Some(offset + mask.trailing_zeros() as usize);
    }
    if n == 3 {
        let m = 3 - i1 - i2;
        let expect = needle_lower[m];
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            if unsafe { *ptr.add(offset + bit + m) }.to_ascii_lowercase() == expect {
                return Some(offset + bit);
            }
            mask &= mask - 1;
        }
        return None;
    }
    while mask != 0 {
        let bit = mask.trailing_zeros() as usize;
        if unsafe { confirm_avx2(ptr.add(offset + bit), needle_lower) } {
            return Some(offset + bit);
        }
        mask &= mask - 1;
    }
    None
}

/// AVX2 packed pair: two rare bytes, case-insensitive, leftmost hit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_packed_pair_avx2(
    haystack: &[u8],
    needle_lower: &[u8],
    plan: PairPlan,
) -> Option<usize> {
    use core::arch::x86_64::*;

    let n = needle_lower.len();
    let hlen = haystack.len();
    let ptr = haystack.as_ptr();
    let last_start = hlen - n; // last valid match-start position
    let i1 = plan.i1;
    let i2 = plan.i2;

    let v1_lo = _mm256_set1_epi8(plan.b1 as i8);
    let v1_hi = _mm256_set1_epi8(plan.b1_alt as i8);
    let v2_lo = _mm256_set1_epi8(plan.b2 as i8);
    let v2_hi = _mm256_set1_epi8(plan.b2_alt as i8);

    // Case-insensitive match: OR both case variants, then AND the two positions.
    macro_rules! pair_mask {
        ($off:expr) => {{
            let chunk1 = unsafe { _mm256_loadu_si256(ptr.add($off + i1) as *const __m256i) };
            let chunk2 = unsafe { _mm256_loadu_si256(ptr.add($off + i2) as *const __m256i) };
            let eq1 = _mm256_or_si256(
                _mm256_cmpeq_epi8(chunk1, v1_lo),
                _mm256_cmpeq_epi8(chunk1, v1_hi),
            );
            let eq2 = _mm256_or_si256(
                _mm256_cmpeq_epi8(chunk2, v2_lo),
                _mm256_cmpeq_epi8(chunk2, v2_hi),
            );
            _mm256_movemask_epi8(_mm256_and_si256(eq1, eq2)) as u32
        }};
    }

    // Full windows only: every lane starts at or before `last_start`.
    let max_idx = i1.max(i2);
    let max_offset = hlen.saturating_sub(max_idx + 32);
    let mut offset = 0usize;

    // Two windows per step. The low window is confirmed before the high one.
    if last_start >= 31 {
        let vector_limit = max_offset.min(last_start - 31);
        while offset + 32 <= vector_limit {
            let m0 = pair_mask!(offset);
            let m1 = pair_mask!(offset + 32);
            if m0 != 0 {
                if let Some(hit) =
                    unsafe { first_confirmed(ptr, needle_lower, offset, m0, i1, i2) }
                {
                    return Some(hit);
                }
            }
            if m1 != 0 {
                if let Some(hit) =
                    unsafe { first_confirmed(ptr, needle_lower, offset + 32, m1, i1, i2) }
                {
                    return Some(hit);
                }
            }
            offset += 64;
        }
        if offset <= vector_limit {
            let m0 = pair_mask!(offset);
            if m0 != 0 {
                if let Some(hit) =
                    unsafe { first_confirmed(ptr, needle_lower, offset, m0, i1, i2) }
                {
                    return Some(hit);
                }
            }
            offset += 32;
        }
    }

    if offset <= last_start {
        // One in-bounds window holds every start in `offset..=last_start`.
        // Drop bits already scanned and bits past `last_start`.
        let start = if offset <= max_offset { offset } else { max_offset };
        debug_assert!(start <= last_start);
        let mut valid = pair_mask!(start);
        if start < offset {
            let low = offset - start;
            debug_assert!(low < 32);
            valid &= !((1u32 << low) - 1);
        }
        let max_bit = last_start - start;
        if max_bit < 31 {
            valid &= (1u32 << (max_bit + 1)) - 1;
        }
        if valid != 0 {
            if let Some(hit) = unsafe { first_confirmed(ptr, needle_lower, start, valid, i1, i2) }
            {
                return Some(hit);
            }
        }
    }
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_packed_pair_neon(
    haystack: &[u8],
    needle_lower: &[u8],
    plan: PairPlan,
) -> Option<usize> {
    use core::arch::aarch64::*;

    let n = needle_lower.len();
    let hlen = haystack.len();
    let ptr = haystack.as_ptr();
    let last_start = hlen - n;
    let i1 = plan.i1;
    let i2 = plan.i2;

    let v1_lo = vdupq_n_u8(plan.b1);
    let v1_hi = vdupq_n_u8(plan.b1_alt);
    let v2_lo = vdupq_n_u8(plan.b2);
    let v2_hi = vdupq_n_u8(plan.b2_alt);

    // Full windows only: every lane starts at or before `last_start`.
    let max_idx = i1.max(i2);
    let max_offset = hlen.saturating_sub(max_idx + 16);
    let mut offset = 0usize;

    if last_start >= 15 {
        let vector_limit = max_offset.min(last_start - 15);
        while offset <= vector_limit {
            let chunk1 = unsafe { vld1q_u8(ptr.add(offset + i1)) };
            let chunk2 = unsafe { vld1q_u8(ptr.add(offset + i2)) };

            // Case-insensitive match: OR both case variants, then AND the two positions.
            let eq1 = vorrq_u8(vceqq_u8(chunk1, v1_lo), vceqq_u8(chunk1, v1_hi));
            let eq2 = vorrq_u8(vceqq_u8(chunk2, v2_lo), vceqq_u8(chunk2, v2_hi));

            let mut mask = unsafe { neon_nibble_mask(vandq_u8(eq1, eq2)) };
            // Nibbles increase with position, so the first confirmed hit is leftmost.
            while mask != 0 {
                let bit = (mask.trailing_zeros() >> 2) as usize;
                let candidate = offset + bit;
                let h = ptr.add(candidate);
                let ok = if n == 2 {
                    true
                } else if n == 3 {
                    let m = 3 - i1 - i2;
                    unsafe { *h.add(m) }.to_ascii_lowercase() == needle_lower[m]
                } else {
                    unsafe { eq_lowered_case(h, needle_lower) }
                };
                if ok {
                    return Some(candidate);
                }
                mask &= !(0xFu64 << (bit * 4));
            }

            offset += 16;
        }
    }

    let other_pos = if plan.rare_pos == i1 { i2 } else { i1 };
    scan_rare_tail(
        haystack,
        needle_lower,
        offset,
        last_start,
        plan.rare_pos,
        other_pos,
    )
}

fn find_first_byte_with_memchr(haystack: &[u8], needle_lower: &[u8]) -> Option<usize> {
    let n = needle_lower.len();
    debug_assert!(n >= 1 && n <= haystack.len());

    let search_space = &haystack[..=haystack.len() - n];
    let first = needle_lower[0];

    if first.is_ascii_lowercase() {
        let alt = ascii_swap_case(first);
        for pos in memchr::memchr2_iter(first, alt, search_space) {
            if unsafe { eq_lowered_case(haystack.as_ptr().add(pos), needle_lower) } {
                return Some(pos);
            }
        }
    } else {
        for pos in memchr::memchr_iter(first, search_space) {
            if unsafe { eq_lowered_case(haystack.as_ptr().add(pos), needle_lower) } {
                return Some(pos);
            }
        }
    }
    None
}

/// ASCII case-insensitive substring search returning the leftmost match
/// position. `needle_lower` must be pre-lowercased (ASCII).
// Visible for external consumers; hidden from docs.
#[doc(hidden)] // Public for external use; hidden from docs.
pub fn find(haystack: &[u8], needle_lower: &[u8]) -> Option<usize> {
    let n = needle_lower.len();
    if n == 0 {
        return Some(0);
    }
    if n > haystack.len() {
        return None;
    }
    if n == 1 {
        let first = needle_lower[0];
        return if first.is_ascii_lowercase() {
            memchr::memchr2(first, ascii_swap_case(first), haystack)
        } else {
            memchr::memchr(first, haystack)
        };
    }
    find_planned(haystack, needle_lower, plan_for(needle_lower))
}

// `plan` is `plan_for(needle_lower)`. Needles shorter than 2 ignore it.
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(unused_variables)
)]
#[inline]
pub(crate) fn find_planned(haystack: &[u8], needle_lower: &[u8], plan: PairPlan) -> Option<usize> {
    let n = needle_lower.len();
    if n == 0 {
        return Some(0);
    }
    if n > haystack.len() {
        return None;
    }

    if n == 1 {
        let first = needle_lower[0];
        return if first.is_ascii_lowercase() {
            memchr::memchr2(first, ascii_swap_case(first), haystack)
        } else {
            memchr::memchr(first, haystack)
        };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if plan.avx2 {
            // One vector load needs `max(i1, i2) + 32` haystack bytes.
            let max_idx = plan.i1.max(plan.i2);
            if haystack.len() >= max_idx + 32 {
                return unsafe { find_packed_pair_avx2(haystack, needle_lower, plan) };
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Packed pair only when the first byte is common (rank >= 200).
        // A rarer first byte stays on memchr2.
        let max_idx = plan.i1.max(plan.i2);
        if plan.first_rank >= 200 && haystack.len() >= max_idx + 16 {
            return unsafe { find_packed_pair_neon(haystack, needle_lower, plan) };
        }
    }

    // Falls back to the memchr implementation, which covers more SIMD backends.
    find_first_byte_with_memchr(haystack, needle_lower)
}

/// Leftmost ASCII case-insensitive match. Haystack length is under 1024.
/// `find` gets this needle when no byte is in A..=Z; otherwise the folded copy.
pub fn find_case_insensitive_short(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    debug_assert!(haystack.len() < 1024);
    if !crate::simd_path::needle_has_ascii_upper(needle) {
        return find(haystack, needle);
    }
    thread_local! {
        static LOWER_BUF: std::cell::RefCell<Vec<u8>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    LOWER_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.resize(needle.len(), 0);
        crate::simd_path::fold_ascii_needle(needle, &mut buf);
        find(haystack, &buf)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_find(haystack: &[u8], needle_lower: &[u8]) -> Option<usize> {
        if needle_lower.is_empty() {
            return Some(0);
        }
        if needle_lower.len() > haystack.len() {
            return None;
        }
        haystack
            .windows(needle_lower.len())
            .position(|w| w.eq_ignore_ascii_case(needle_lower))
    }

    #[test]
    fn basic_case_insensitive() {
        assert_eq!(find(b"Hello World", b"hello"), Some(0));
        assert_eq!(find(b"Hello World", b"world"), Some(6));
        assert_eq!(find(b"NOMORE bugs", b"nomore"), Some(0));
        assert_eq!(find(b"Hello World", b"xyz"), None);
        assert!(find(b"Hello World", b"o w").is_some());
    }

    #[test]
    fn edge_cases() {
        assert_eq!(find(b"ab", b"ab"), Some(0));
        assert_eq!(find(b"AB", b"ab"), Some(0));
        assert_eq!(find(b"a", b"ab"), None);
        assert_eq!(find(b"anything", b""), Some(0));
        assert_eq!(find(b"", b"x"), None);
        assert_eq!(find(b"xxA", b"a"), Some(2));
        assert_eq!(find(b"xx:", b":"), Some(2));
    }

    #[test]
    fn returns_leftmost_match() {
        assert_eq!(find(b"foo FOO foo", b"foo"), Some(0));
        let mut big = vec![b'.'; 300];
        big[100..103].copy_from_slice(b"FoO");
        big[200..203].copy_from_slice(b"foo");
        assert_eq!(find(&big, b"foo"), Some(100));
    }

    #[test]
    fn non_letter_bytes_do_not_case_fold() {
        // '[' (0x5B) and '{' (0x7B) differ only in bit 5 but are not letters.
        // A fold implemented as a bare `| 0x20` would falsely match these.
        assert_eq!(find(b"A[", b"a{"), None);
        assert_eq!(find(b"x@y", b"x`y"), None);
        assert_eq!(find(b"a]b", b"a}b"), None);
        assert_eq!(find(b"A{", b"a{"), Some(0));
    }

    #[test]
    fn matches_reference_on_random_inputs() {
        // Deterministic xorshift PRNG — no external deps.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // Alphabet with letters, both-case pairs, and 0x20-differing symbols.
        let alphabet = b"aAbBzZ [{@`]}^~_0.\n";
        for _ in 0..2000 {
            let hlen = (next() % 200) as usize;
            let nlen = (next() % 8) as usize;
            let haystack: Vec<u8> = (0..hlen)
                .map(|_| alphabet[(next() % alphabet.len() as u64) as usize])
                .collect();
            let needle: Vec<u8> = (0..nlen)
                .map(|_| alphabet[(next() % alphabet.len() as u64) as usize].to_ascii_lowercase())
                .collect();

            assert_eq!(
                find(&haystack, &needle),
                reference_find(&haystack, &needle),
                "mismatch for haystack={:?} needle={:?}",
                haystack,
                needle,
            );
        }
    }

    #[test]
    fn long_haystack_simd_paths() {
        let haystack =
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaTHIS_IS_A_LONG_NEEDLE_TESTbbbbbbbbbbbbbbbbbb";
        assert_eq!(find(haystack, b"this_is_a_long_needle_test"), Some(32));
        assert_eq!(find(haystack, b"this_is_a_long_needle_testz"), None);

        // Needle >= 16 bytes exercises SIMD verify.
        let haystack2 = b"int STRUCT MUTEX *LOCK(struct mutex *lock) { return 0; }";
        assert_eq!(find(haystack2, b"struct mutex *lock"), Some(4));

        let upper_hay = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        assert_eq!(find(upper_hay, b"qrstuvwxyz0123456789a"), Some(16));
        assert_eq!(find(upper_hay, b"qrstuvwxyz01234567899"), None);

        // Needle at either end.
        let end_hay = b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxfind_me";
        assert_eq!(find(end_hay, b"find_me"), Some(end_hay.len() - 7));
        assert_eq!(find(end_hay, b"xx"), Some(0));

        // 1KB haystack with needle near the end.
        let mut big = vec![b'z'; 1024];
        big[1000..1010].copy_from_slice(b"hElLo_WoRl");
        assert_eq!(find(&big, b"hello_wo"), Some(1000));
        assert_eq!(find(&big, b"hello_world"), None);
    }

    #[test]
    fn rare_pair_selection() {
        let (i1, i2) = select_rare_pair(b"nomore");
        let ranks: Vec<u8> = b"nomore".iter().map(|&b| rank(b)).collect();
        let (r1, r2) = (ranks[i1], ranks[i2]);
        for (i, &r) in ranks.iter().enumerate() {
            if i != i1 && i != i2 {
                assert!(r1 <= r || r2 <= r, "pair ({i1},{i2}) not optimal");
            }
        }
    }
}
