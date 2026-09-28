//! Scan-time per-path character-presence index.
//!
//! Each path gets a 64-bit mask: bit `fold_bit(b)` is set iff byte `b`
//! occurs anywhere in the path (ASCII case-folded). The mask is a *sound
//! necessary condition* for `LCS(needle, path) >= |needle| - max_typos`:
//! any true match must have at least `|needle| - max_typos` needle bytes
//! present in the path, so if fewer needle bytes have their bit set the
//! path cannot match and can be skipped without touching its bytes.
//!
//! The fold is lossy (`b & 15` overflow bucket), so the check can pass
//! paths that still cannot match (false positives) — the matcher remains
//! the authority. It never rejects a true match (no false negatives).
//!
//! 64 bits keep the index at 8 bytes/path (contiguous, sequential scan)
//! versus 32 bytes for a 256-bit mask.

/// Map a byte to a bit index in `0..64`.
///
/// ASCII uppercase folds onto its lowercase bit so case-insensitive
/// matching stays sound. The first 48 bits are injective on the path
/// alphabet (a-z, 0-9, and twelve common punctuation bytes); the rest of
/// the byte space shares 16 overflow bits, which only costs selectivity.
#[inline(always)]
pub(crate) fn fold_bit(b: u8) -> u32 {
    let b = if b.is_ascii_uppercase() { b + 32 } else { b };
    match b {
        b'a'..=b'z' => (b - b'a') as u32,
        b'0'..=b'9' => 26 + (b - b'0') as u32,
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
        _ => 48 + (b & 15) as u32,
    }
}

/// Build the presence mask for a path (relative path bytes, any case).
#[inline]
pub(crate) fn mask_of(path: &[u8]) -> u64 {
    let mut m = 0u64;
    for &b in path {
        m |= 1u64 << fold_bit(b);
    }
    m
}

/// Sound necessary condition for `LCS(needle, path) >= |needle| - max_typos`.
///
/// Counts needle bytes (with multiplicity) whose folded bit is set in the
/// path mask. Multiplicity is required: counting distinct needle bits would
/// reject `needle="aab"` vs `path="aab"` at `need=3`.
///
/// `need == 0` (typo budget covers the whole needle) accepts everything.
#[inline]
pub(crate) fn allows(path_mask: u64, needle: &[u8], max_typos: u16) -> bool {
    let need = needle.len().saturating_sub(max_typos as usize);
    if need == 0 {
        return true;
    }
    let mut hits = 0usize;
    for &b in needle {
        if path_mask & (1u64 << fold_bit(b)) != 0 {
            hits += 1;
            if hits >= need {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exhaustive soundness: for random path/needle pairs with a cheap LCS
    /// oracle, `allows` never rejects when LCS >= need.
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
}
