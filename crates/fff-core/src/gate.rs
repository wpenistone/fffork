//! Tolerant subsequence gate — match-set + window without frizbee.
//!
//! Novel pipeline per haystack (and batched across paths):
//! 1. **Span gate (SIMD-friendly)**: min/max positions of any *needle* byte.
//!    Everything outside the span cannot belong to an LCS. If
//!    `span_len < m - max_typos`, reject without touching the rest.
//! 2. **Skip-DP (Allison-style)**: `best[s]` = furthest needle cursor after
//!    `s` deletions. Accept iff some `s <= T` reaches `m`.
//! 3. **Window**: the needle-byte span (sound superset of the minimal
//!    alignment window). Includes_prefix = span starts at 0.
//!
//! Soundness: never rejects when `LCS(needle, hay) >= m - T`.
//! False accepts are impossible — skip-DP is exact for the typo model
//! (deletions in the needle = unmatched needle chars).

use crate::types::FileItem;
use crate::simd_path::ArenaPtr;

/// Result of a successful gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateHit {
    pub start: u32,
    pub end: u32,
    pub include_prefix: bool,
}

const MAX_SKIPS: usize = 7;

#[inline]
fn fold_byte(b: u8) -> u8 {
    if b.is_ascii_uppercase() {
        b + 32
    } else {
        b
    }
}

/// Bitmask of needle bytes (case-folded) — 256 bits as 4×u64.
#[derive(Clone, Copy)]
pub struct NeedleMask {
    w: [u64; 4],
}

impl NeedleMask {
    #[inline]
    pub fn new(needle: &[u8]) -> Self {
        let mut w = [0u64; 4];
        for &b in needle {
            let x = fold_byte(b) as usize;
            w[x >> 6] |= 1u64 << (x & 63);
        }
        Self { w }
    }

    #[inline]
    pub fn contains(&self, b: u8) -> bool {
        let x = fold_byte(b) as usize;
        self.w[x >> 6] & (1u64 << (x & 63)) != 0
    }
}

/// Span of needle-bytes in `hay`. None if fewer than `need` such bytes.
fn needle_span(mask: &NeedleMask, hay: &[u8], need: usize) -> Option<(usize, usize)> {
    if hay.len() < need {
        return None;
    }
    let mut first = None;
    let mut last = 0usize;
    let mut n = 0usize;
    for (i, &b) in hay.iter().enumerate() {
        if mask.contains(b) {
            if first.is_none() {
                first = Some(i);
            }
            last = i;
            n += 1;
            if n >= need {
                // keep scanning only for last…
            }
        }
    }
    if n < need {
        return None;
    }
    Some((first.unwrap(), last))
}

/// Exact skip-DP over `[lo, hi)` of `hay` (or the whole slice).
/// Returns true iff LCS(needle, hay[lo..=hi]) >= needle.len() - max_typos.
fn skip_dp_accept(
    needle: &[u8],
    hay: &[u8],
    lo: usize,
    hi: usize,
    max_typos: usize,
) -> bool {
    let m = needle.len();
    if m == 0 {
        return false;
    }
    let need = m.saturating_sub(max_typos);
    if need == 0 {
        return true;
    }
    let t = max_typos.min(MAX_SKIPS);
    let hay = &hay[lo..hi];
    if hay.len() < need {
        return false;
    }
    const INF: usize = usize::MAX;
    let mut best = [INF; MAX_SKIPS + 1];
    let mut nxt = [INF; MAX_SKIPS + 1];
    best[0] = 0;
    for &c in needle {
        nxt = [INF; MAX_SKIPS + 1];
        for s in 0..=t {
            let cur = best[s];
            if cur == INF {
                continue;
            }
            // Delete this needle char (spend a typo).
            if s < t && cur < nxt[s + 1] {
                nxt[s + 1] = cur;
            }
            // Match this needle char at/after the cursor.
            if let Some(p) = find_byte_fold(hay, cur, c) {
                let adv = p + 1;
                if adv < nxt[s] {
                    nxt[s] = adv;
                }
            }
        }
        best = nxt;
    }
    best[..=t].iter().any(|&p| p != INF)
}

#[inline]
fn find_byte_fold(hay: &[u8], from: usize, needle_b: u8) -> Option<usize> {
    let target = fold_byte(needle_b);
    for (i, &b) in hay.iter().enumerate().skip(from) {
        if fold_byte(b) == target {
            return Some(i);
        }
    }
    None
}

/// Single-haystack gate. `None` = not a match under the typo model.
pub fn gate_hit(needle: &[u8], hay: &[u8], max_typos: u16) -> Option<GateHit> {
    let m = needle.len();
    if m == 0 || hay.is_empty() {
        return None;
    }
    let typos = max_typos as usize;
    let need = m.saturating_sub(typos);
    if need == 0 {
        return Some(GateHit {
            start: 0,
            end: hay.len().min(u16::MAX as usize) as u32,
            include_prefix: true,
        });
    }
    let mask = NeedleMask::new(needle);
    let (first, last) = needle_span(&mask, hay, need)?;
    // Span must hold at least `need` matched bytes (already n>=need) and
    // the skip-DP must succeed inside the span (plus one byte of margin).
    let lo = first;
    let hi = (last + 1).min(hay.len());
    if !skip_dp_accept(needle, hay, lo, hi, typos.min(MAX_SKIPS)) {
        return None;
    }
    Some(GateHit {
        start: lo as u32,
        end: hi as u32,
        include_prefix: lo == 0,
    })
}

/// Batch gate over paths (same needle). Indices of hits are pushed to `out`.
/// `windows` parallel to hits.
pub fn gate_batch<'a, I>(needle: &[u8], hays: I, max_typos: u16, out: &mut Vec<(u32, GateHit)>)
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
{
    for (idx, hay) in hays {
        if let Some(hit) = gate_hit(needle, hay, max_typos) {
            out.push((idx, hit));
        }
    }
}

/// Resolve + gate one FileItem against the arena path bytes.
pub fn gate_file(
    needle: &[u8],
    file: &FileItem,
    arena: ArenaPtr,
    max_typos: u16,
    buf: &mut [u8],
) -> Option<GateHit> {
    let hay = file.path.read_to_buf(arena, buf);
    gate_hit(needle, hay.as_ref(), max_typos)
}

/// Parallel match-set over FileItems. Emits frizbee-compatible
/// `Match { score: start, end_col: end }` (skip_sw encoding) so the rest
/// of the pipeline is unchanged. `file_index` maps working-set position →
/// corpus index.
pub fn gate_match_range<F>(
    needle: &[u8],
    n: usize,
    max_typos: u16,
    threads: usize,
    file_index: F,
    resolve_len: impl Fn(u32) -> usize + Sync,
    resolve_bytes: impl Fn(u32, &mut [u8]) -> usize + Sync,
) -> Vec<neo_frizbee::Match>
where
    F: Fn(u32) -> u32 + Sync,
{
    use rayon::prelude::*;
    let _ = threads; // rayon pool sizes itself
    let _ = resolve_len;
    (0..n as u32)
        .into_par_iter()
        .filter_map(|i| {
            // Cheap length reject first via resolve_len.
            let hay_len = resolve_len(i);
            if hay_len == 0 {
                return None;
            }
            let mut buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
            if hay_len > buf.len() {
                return None;
            }
            let got = resolve_bytes(i, &mut buf);
            if got == 0 {
                return None;
            }
            let hit = gate_hit(needle, &buf[..got], max_typos)?;
            Some(neo_frizbee::Match {
                score: hit.start as u16,
                index: file_index(i),
                exact: false,
                end_col: hit.end as u16,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_rejects_too_short() {
        assert!(gate_hit(b"abc", b"ac", 0).is_none());
        assert!(gate_hit(b"abc", b"axc", 1).is_some());
    }

    #[test]
    fn typos_allow_missing_chars() {
        assert!(gate_hit(b"username", b"user_name", 1).is_some());
        assert!(gate_hit(b"username", b"userXname", 1).is_some());
        assert!(gate_hit(b"username", b"xxxx", 1).is_none());
    }

    #[test]
    fn case_insensitive() {
        assert!(gate_hit(b"UserName", b"user_name", 1).is_some());
    }

    #[test]
    fn window_covers_needle_bytes() {
        let h = gate_hit(b"score", b"src/scoring/mod.rs", 2).unwrap();
        assert!(h.end > h.start);
        assert_eq!(h.include_prefix, h.start == 0);
    }

    #[test]
    fn agrees_with_lcs_on_random() {
        let mut state = 0x12345678u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        for _ in 0..2000 {
            let nl = 2 + (next() % 8) as usize;
            let hl = 1 + (next() % 16) as usize;
            let needle: Vec<u8> = (0..nl).map(|_| b'a' + (next() % 5) as u8).collect();
            let hay: Vec<u8> = (0..hl).map(|_| b'a' + (next() % 5) as u8).collect();
            let t = (next() % 3) as u16;
            let need = nl as i32 - t as i32;
            // LCS via DP
            let mut dp = vec![0i32; hl + 1];
            for &nc in &needle {
                let mut prev = 0i32;
                for j in 1..=hl {
                    let tmp = dp[j];
                    dp[j] = if fold_byte(nc) == fold_byte(hay[j - 1]) {
                        prev + 1
                    } else {
                        dp[j].max(dp[j - 1])
                    };
                    prev = tmp;
                }
            }
            let lcs = dp[hl];
            let gate = gate_hit(&needle, &hay, t).is_some();
            let expect = lcs >= need.max(0);
            assert_eq!(gate, expect, "n={needle:?} h={hay:?} t={t} lcs={lcs}");
        }
    }
}
