//! Scan-time per-path presence index: 64-bit mask per path (8 bytes).
//!
//! Sound pretest for `LCS >= |needle| - max_typos`; lossy fold may pass
//! non-matches, but never rejects a true match.

// Fold table: byte -> bit in `0..64`, uppercase pre-folded.
// Single load; identical mapping to the match below.
const FOLD_BIT_LUT: [u8; 256] = {
    let mut lut = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        let raw = i as u8;
        let b = if raw >= b'A' && raw <= b'Z' {
            raw + 32
        } else {
            raw
        };
        lut[i] = match b {
            b'a'..=b'z' => b - b'a',
            b'0'..=b'9' => 26 + (b - b'0'),
            b'.' => 36,
            b'_' => 37,
            b'-' => 38,
            b'/' => 39,
            b'\\' => 40,
            b' ' => 41,
            b',' => 42,
            b'@' => 43,
            b'#' => 44,
            b'(' => 45,
            b')' => 46,
            b'[' => 47,
            _ => 48 + (b & 15),
        };
        i += 1;
    }
    lut
};

// Map a byte to a bit in `0..64`; uppercase folds to lowercase so the
// pretest stays sound. Overflow bits only cost selectivity.
#[inline(always)]
pub(crate) fn fold_bit(b: u8) -> u32 {
    FOLD_BIT_LUT[b as usize] as u32
}

// Build the presence mask for a path (relative path bytes, any case).
#[inline]
pub(crate) fn mask_of(path: &[u8]) -> u64 {
    let mut m = 0u64;
    for &b in path {
        m |= 1u64 << fold_bit(b);
    }
    m
}

// Precomputed per-query presence check. Needle multiplicities split
// into four bit planes; hits are four popcounts, no per-byte loop.
#[derive(Clone, Copy)]
pub(crate) struct NeedlePresence {
    need: usize,
    min_len: usize,
    mask_1: u64,
    mask_2: u64,
    mask_3: u64,
    mask_4: u64,
    dup: bool,
    has_gt4: bool,
}

impl NeedlePresence {
    #[inline]
    pub(crate) fn new(needle: &[u8], max_typos: u16) -> Self {
        let typos = max_typos as usize;
        let need = needle.len().saturating_sub(typos);
        let min_len = need; // LCS ≤ path_len
        let mut mask_1 = 0u64;
        let mut mask_2 = 0u64;
        let mut mask_3 = 0u64;
        let mut mask_4 = 0u64;
        let mut has_gt4 = false;
        for &b in needle {
            let bit = 1u64 << fold_bit(b);
            if mask_4 & bit != 0 {
                has_gt4 = true;
            } else if mask_3 & bit != 0 {
                mask_4 |= bit;
            } else if mask_2 & bit != 0 {
                mask_3 |= bit;
            } else if mask_1 & bit != 0 {
                mask_2 |= bit;
            } else {
                mask_1 |= bit;
            }
        }
        let dup = mask_2 != 0;
        Self {
            need,
            min_len,
            mask_1,
            mask_2,
            mask_3,
            mask_4,
            dup,
            has_gt4,
        }
    }

    // Shortest path that can still match (LCS ≤ path_len).
    #[inline(always)]
    pub(crate) fn min_path_len(&self) -> usize {
        self.min_len
    }

    // False when the presence check is not worth running (need < 3:
    // short queries match almost every path, so the check is overhead).
    #[inline(always)]
    pub(crate) fn should_check(&self) -> bool {
        self.need >= 3
    }

    #[inline(always)]
    pub(crate) fn allows(&self, path_mask: u64, needle: &[u8]) -> bool {
        if self.need == 0 {
            return true;
        }
        if !self.dup {
            return (path_mask & self.mask_1).count_ones() as usize >= self.need;
        }
        // Multiplicity as popcount planes: bit b in mask_k iff the
        // needle holds b at least k times, so summed popcounts equal
        // the exact per-byte hit total. Branchless, no early exit needed.
        if !self.has_gt4 {
            let hits = (path_mask & self.mask_1).count_ones()
                + (path_mask & self.mask_2).count_ones()
                + (path_mask & self.mask_3).count_ones()
                + (path_mask & self.mask_4).count_ones();
            return hits as usize >= self.need;
        }
        // Past 4 repetitions (rare): exact per-byte loop.
        let mut hits = 0usize;
        for &b in needle {
            if path_mask & (1u64 << fold_bit(b)) != 0 {
                hits += 1;
                if hits >= self.need {
                    return true;
                }
            }
        }
        false
    }
}

// Sound pretest for one file: unknown masks (0) always scan.
#[inline]
pub(crate) fn allows_or_unknown(np: &NeedlePresence, path_mask: u64, needle: &[u8]) -> bool {
    path_mask == 0 || np.allows(path_mask, needle)
}

// Sound pretest for `LCS >= |needle| - max_typos`; counts with multiplicity
// (`need == 0` accepts everything).
#[inline]
pub(crate) fn allows(path_mask: u64, needle: &[u8], max_typos: u16) -> bool {
    let np = NeedlePresence::new(needle, max_typos);
    np.allows(path_mask, needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exhaustive soundness: for random path/needle pairs with a cheap LCS
    // oracle, `allows` never rejects when LCS >= need.
    fn lcs_len(a: &[u8], b: &[u8]) -> usize {
        // Case-insensitive to match frizbee's matcher semantics.
        let fold = |x: u8| if x.is_ascii_uppercase() { x + 32 } else { x };
        let mut prev = vec![0usize; b.len() + 1];
        let mut cur = vec![0usize; b.len() + 1];
        for &ca in a {
            for (j, &cb) in b.iter().enumerate() {
                cur[j + 1] = if fold(ca) == fold(cb) {
                    prev[j] + 1
                } else {
                    cur[j].max(prev[j + 1])
                };
            }
            std::mem::swap(&mut prev, &mut cur);
            cur.fill(0);
        }
        prev[b.len()]
    }

    #[test]
    fn soundness_vs_lcs() {
        // Deterministic LCG so the test is reproducible without a rand dep.
        let mut state = 0x2545F491_4F6CDD1Du64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u32
        };
        for _case in 0..20_000 {
            let path_len = (next() % 24) as usize + 1;
            let needle_len = (next() % 12) as usize + 1;
            let alphabet = b"abcXYZ01_/.";
            let path: Vec<u8> = (0..path_len)
                .map(|_| alphabet[(next() as usize) % alphabet.len()])
                .collect();
            let needle: Vec<u8> = (0..needle_len)
                .map(|_| alphabet[(next() as usize) % alphabet.len()])
                .collect();
            let max_typos = (next() % 5) as u16;
            let need = needle.len().saturating_sub(max_typos as usize);
            let lcs = lcs_len(&needle, &path);
            if lcs >= need {
                assert!(
                    allows(mask_of(&path), &needle, max_typos),
                    "FALSE REJECT path={:?} needle={:?} typos={} lcs={lcs} need={need}",
                    String::from_utf8_lossy(&path),
                    String::from_utf8_lossy(&needle),
                    max_typos,
                );
            }
        }
    }

    #[test]
    fn rejects_missing_needle_bytes() {
        let mask = mask_of(b"hello/world.rs");
        // 'z' is absent; with typos=0 need=1 and the z alone fails.
        assert!(!allows(mask, b"z", 0));
        // 'z' absent but typo budget covers it.
        assert!(allows(mask, b"z", 1));
        // All bytes present.
        assert!(allows(mask, b"hero", 0));
    }

    #[test]
    fn case_folds() {
        let mask = mask_of(b"FilePicker");
        assert!(allows(mask, b"filepicker", 0));
        assert!(allows(mask, b"FILEPICKER", 0));
    }

    #[test]
    fn multiplicity_is_counted() {
        // "aab" vs path "aab": LCS=3, need=3 -> must accept.
        let mask = mask_of(b"aab");
        assert!(allows(mask, b"aab", 0));
        // "aab" vs path "ab": LCS=2 < need=3. The mask check counts needle
        // bytes with multiplicity against binary presence, so this is a
        // known false positive (the matcher rejects later). Presence masks
        // only guarantee no false *rejections*.
        let mask = mask_of(b"ab");
        assert!(allows(mask, b"aab", 0));
        // Truly absent needle bytes still reject.
        let mask = mask_of(b"ab");
        assert!(!allows(mask, b"xyz", 0));
    }

    #[test]
    fn fold_bit_is_stable_for_case() {
        assert_eq!(fold_bit(b'A'), fold_bit(b'a'));
        assert_eq!(fold_bit(b'Z'), fold_bit(b'z'));
        assert_ne!(fold_bit(b'a'), fold_bit(b'b'));
    }

    #[test]
    fn fold_lut_covers_all_bytes() {
        // Every byte maps in range; case pairs agree; classes stay distinct.
        for b in 0..=255u8 {
            assert!(fold_bit(b) < 64, "bit out of range for {b}");
        }
        for b in b'A'..=b'Z' {
            assert_eq!(fold_bit(b), fold_bit(b + 32));
        }
        let distinct = [b'a', b'0', b'.', b'_', b'-', b'/', b' ', b',', b'@', b'#'];
        for (i, &x) in distinct.iter().enumerate() {
            for &y in &distinct[i + 1..] {
                assert_ne!(fold_bit(x), fold_bit(y), "{x} vs {y}");
            }
        }
        // Overflow folding stays a nibble function of the raw byte.
        for b in [b':', b';', 0x80u8, 0xFFu8, 0x01u8] {
            let up = if b.is_ascii_uppercase() { b + 32 } else { b };
            assert_eq!(fold_bit(b), 48 + (up & 15) as u32);
        }
    }

    #[test]
    fn planes_agree_with_naive_counts() {
        // Naive oracle: per-byte hits with multiplicity, no planes.
        let naive = |mask: u64, needle: &[u8], need: usize| {
            let mut hits = 0usize;
            for &b in needle {
                hits += ((mask >> fold_bit(b)) & 1) as usize;
            }
            hits >= need
        };
        let mut state = 0x1234_5678_9ABCu64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u32
        };
        // Alphabet biased to repetitions, forcing dup and gt4 paths.
        let alphabet = b"aaaabbbccde";
        for _ in 0..5000 {
            let nlen = (next() % 16) as usize + 1;
            let needle: Vec<u8> = (0..nlen)
                .map(|_| alphabet[(next() as usize) % alphabet.len()])
                .collect();
            let typos = (next() % 4) as u16;
            let np = NeedlePresence::new(&needle, typos);
            let mask = ((next() as u64) << 32) | next() as u64;
            assert_eq!(
                np.allows(mask, &needle),
                naive(mask, &needle, np.need),
                "needle={:?} typos={typos} mask={mask:016x}",
                String::from_utf8_lossy(&needle),
            );
        }
        // Pinned gt4 shapes: 5+ repetitions take the fallback loop.
        for needle in [&b"aaaaa"[..], b"aaaaaaaa", b"aabaaaabaaa"] {
            let np = NeedlePresence::new(needle, 0);
            assert!(np.has_gt4, "{needle:?} should flag gt4");
            for mask in [0u64, 1u64, u64::MAX, mask_of(b"a"), mask_of(b"b")] {
                assert_eq!(np.allows(mask, needle), naive(mask, needle, np.need));
            }
        }
    }
}
