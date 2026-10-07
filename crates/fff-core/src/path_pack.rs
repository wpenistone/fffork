//! Packed relative-path corpus + pigeonhole prefilter for fuzzy matching.
//!
//! Paths live in one contiguous byte buffer so we can scan them with tight
//! SIMD-friendly loops. The prefilter is the pigeonhole principle for
//! Levenshtein distance: split the needle into `max_typos + 1` contiguous
//! pieces; a true match must contain **at least one** piece as an exact
//! substring (pieces stay in order across indels, but existence is enough
//! for a reject). OR of memmem hits is a necessary condition.

/// Flat table of packed relative paths (lowercase only).
pub(crate) struct PackedPaths {
    /// Concatenated lowercased path bytes.
    lower: Vec<u8>,
    /// Byte offset of path `i` in `lower` (`i` in `0..count`).
    starts: Vec<u32>,
    /// Length of path `i`.
    lens: Vec<u16>,
    /// FileItem index for path `i`.
    file_idx: Vec<u32>,
    count: usize,
}

impl PackedPaths {
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn file_index(&self, i: usize) -> u32 {
        self.file_idx[i]
    }

    #[inline]
    fn slice_lower(&self, i: usize) -> &[u8] {
        let s = self.starts[i] as usize;
        let n = self.lens[i] as usize;
        &self.lower[s..s + n]
    }
}

fn pack_lower(raw: &[u8], lower: &mut Vec<u8>) {
    lower.extend(
        raw.iter()
            .map(|&b| if b.is_ascii_uppercase() { b + 32 } else { b }),
    );
}

/// Pack paths from a (FileItem, index) iterator.
pub(crate) fn pack_from_iter<'a>(
    items: impl Iterator<Item = (usize, &'a crate::types::FileItem)>,
    arena: crate::simd_path::ArenaPtr,
) -> PackedPaths {
    let mut lower = Vec::new();
    let mut starts = Vec::new();
    let mut lens = Vec::new();
    let mut file_idx = Vec::new();
    let mut buf = [0u8; crate::simd_path::PATH_BUF_SIZE];

    for (fi, file) in items {
        if file.is_deleted() {
            continue;
        }
        let path = file.path.read_to_buf(arena, &mut buf);
        let raw = path.as_bytes();
        if raw.is_empty() {
            continue;
        }
        starts.push(lower.len() as u32);
        lens.push(raw.len() as u16);
        file_idx.push(fi as u32);
        pack_lower(raw, &mut lower);
    }
    starts.push(lower.len() as u32);
    lens.push(0);
    file_idx.push(u32::MAX);
    let count = lens.len().saturating_sub(1);
    PackedPaths {
        lower,
        starts,
        lens,
        file_idx,
        count,
    }
}

/// Build the packed table from arena-backed file paths.
pub(crate) fn pack_paths(
    files: &[crate::types::FileItem],
    arena: crate::simd_path::ArenaPtr,
    limit: usize,
) -> PackedPaths {
    pack_from_iter(
        files.iter().enumerate().take(limit).map(|(fi, f)| (fi, f)),
        arena,
    )
}

/// Split `needle` into `k+1` contiguous lowercase pieces (pigeonhole).
fn pigeon_pieces(needle: &[u8], max_typos: u16) -> Vec<Vec<u8>> {
    let lower: Vec<u8> = needle
        .iter()
        .map(|&b| if b.is_ascii_uppercase() { b + 32 } else { b })
        .collect();
    let k = max_typos as usize;
    let n = lower.len();
    if n == 0 {
        return Vec::new();
    }
    let parts = k + 1;
    let mut pieces = Vec::with_capacity(parts);
    let mut off = 0usize;
    for p in 0..parts {
        let take = (n - off) / (parts - p);
        let take = take.max(if p < n { 1 } else { 0 });
        if off >= n {
            break;
        }
        let end = (off + take).min(n);
        pieces.push(lower[off..end].to_vec());
        off = end;
    }
    if pieces.is_empty() {
        pieces.push(lower);
    }
    pieces
}

/// True if `hay` (already lowercase) contains `needle` as a substring.
#[inline]
fn contains_lower(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > hay.len() {
        return false;
    }
    if needle.len() == 1 {
        return memchr::memchr(needle[0], hay).is_some();
    }
    memchr::memmem::find(hay, needle).is_some()
}

/// Candidate file indices: packed slots whose path contains at least one
/// pigeonhole piece. Returns `None` when the filter cannot reject anything
/// (empty needle or single empty piece) — caller should match everything.
///
/// **Rejected for FFF:** matching is subsequence-with-gaps (Smith-Waterman),
/// not substring embedding — a true hit need not contain a contiguous piece.
pub(crate) fn pigeonhole_candidates(
    packed: &PackedPaths,
    needle: &[u8],
    max_typos: u16,
) -> Option<Vec<u32>> {
    let pieces = pigeon_pieces(needle, max_typos);
    if pieces.is_empty() {
        return None;
    }
    if pieces.iter().all(|p| p.is_empty()) {
        return None;
    }

    let mut out = Vec::new();
    for i in 0..packed.len() {
        let hay = packed.slice_lower(i);
        for piece in &pieces {
            if contains_lower(hay, piece) {
                out.push(packed.file_index(i));
                break;
            }
        }
    }
    Some(out)
}

#[inline]
fn fold(b: u8) -> u8 {
    if b.is_ascii_uppercase() { b + 32 } else { b }
}

/// Fold a needle into `out`, returning its length. Caps at 64 (bit-parallel
/// width); longer needles fall back to the scalar path with the full length.
fn fold_needle(needle: &[u8], out: &mut [u8; 64]) -> usize {
    let n = needle.len().min(64);
    for i in 0..n {
        out[i] = fold(needle[i]);
    }
    n
}

/// T=0 fast path: is `needle` a subsequence of `hay`?
#[inline]
fn is_subseq(needle: &[u8], hay: &[u8]) -> bool {
    let m = needle.len();
    if m == 0 {
        return true;
    }
    if hay.len() < m {
        return false;
    }
    let mut j = 0usize;
    for &b in hay {
        if j < m && needle[j] == b {
            j += 1;
            if j == m {
                return true;
            }
        }
    }
    j == m
}

/// First index of `c` in `hay[from..]`, or `None`.
#[inline]
fn find_byte(hay: &[u8], from: usize, c: u8) -> Option<usize> {
    if from >= hay.len() {
        return None;
    }
    // std memchr is a tight SIMD loop; for short paths the call is cheap.
    hay[from..].iter().position(|&b| b == c).map(|p| from + p)
}

const MAX_SKIPS: usize = 7;

/// Thresholded LCS: true iff `LCS(needle, hay) >= need`.
///
/// Sound prefilter for frizbee with `max_typos = T`: frizbee's typo model is
/// *unmatched / deleted needle characters* (`Config::max_typos` = "characters
/// missing from the needle"; the SW test `typo_threshold_distinguishes_mismatch_
/// deletion_and_haystack_gap` charges 1 typo for both `abc→axc` and `abc→ac`,
/// and 0 for `abc→abbc`). So a true hit has at most `T` unmatched needle
/// chars, i.e. `LCS(needle, path) ≥ |needle| − T`.
///
/// Algorithm (needle-forward DP, "≤T needle-deletions subsequence"):
/// `best[s]` = earliest hay cursor after consuming the current needle prefix
/// with exactly `s` deletions. Per needle char, either delete it (`s+1`, same
/// cursor) or match it at/after the cursor (`find_byte`). Alive state after all
/// needle chars ⇒ matched `m−s ≥ m−T` chars.
///
/// This is Myers-style bit-parallel in spirit: for `T=0` it collapses to a
/// one-pointer greedy scan; for `T>0` the `find_byte` scans are sequential
/// over the packed lowercase lane (no DP table, no allocation).
fn lcs_ge(needle: &[u8], hay: &[u8], need: usize) -> bool {
    let m = needle.len();
    if need == 0 {
        return true;
    }
    if m == 0 {
        return false;
    }
    if hay.len() < need {
        return false;
    }
    let t = m - need;
    if t == 0 {
        return is_subseq(needle, hay);
    }
    if t > MAX_SKIPS {
        // Fall back to the row DP for exotic typo budgets.
        return lcs_len_dp(needle, hay) >= need;
    }

    // best[s] = hay cursor (next index to search) after matching some prefix
    // of the needle with exactly s deletions. `INF` = unreachable.
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
            // Delete this needle char (spend one deletion).
            if s < t {
                if cur < nxt[s + 1] {
                    nxt[s + 1] = cur;
                }
            }
            // Match this needle char at/after the cursor.
            if let Some(p) = find_byte(hay, cur, c) {
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

/// Scalar one-row LCS DP — reference implementation and fallback for
/// `m > 64` or typo budgets above [`MAX_SKIPS`]. Stack-allocated, early-exit.
fn lcs_len_dp(needle: &[u8], hay: &[u8]) -> usize {
    let m = needle.len();
    if m == 0 {
        return 0;
    }
    // m ≤ 256 covers every FFF query; longer needles use a heap buffer.
    if m <= 256 {
        let mut prev = [0u16; 257];
        let mut row = [0u16; 257];
        for &h in hay {
            row[0] = 0;
            for j in 1..=m {
                row[j] = if needle[j - 1] == h {
                    prev[j - 1] + 1
                } else if row[j - 1] >= prev[j] {
                    row[j - 1]
                } else {
                    prev[j]
                };
            }
            std::mem::swap(&mut prev, &mut row);
        }
        return prev[m] as usize;
    }
    let mut prev = vec![0u16; m + 1];
    let mut row = vec![0u16; m + 1];
    for &h in hay {
        row[0] = 0;
        for j in 1..=m {
            row[j] = if needle[j - 1] == h {
                prev[j - 1] + 1
            } else if row[j - 1] >= prev[j] {
                row[j - 1]
            } else {
                prev[j]
            };
        }
        std::mem::swap(&mut prev, &mut row);
    }
    prev[m] as usize
}

/// LCS length (case-folded) of `needle` vs `hay` — public to the crate for
/// tests and callers that want the exact count, not just the threshold.
pub(crate) fn lcs_len(needle: &[u8], hay: &[u8]) -> usize {
    let mut folded = [0u8; 64];
    if needle.len() <= 64 {
        let n = fold_needle(needle, &mut folded);
        return lcs_len_dp(&folded[..n], hay);
    }
    let owned: Vec<u8> = needle.iter().map(|&b| fold(b)).collect();
    lcs_len_dp(&owned, hay)
}

/// Sound prefilter for fuzzy match with `max_typos = T`:
/// frizbee's typo model is *missing needle characters*, so
/// `LCS(needle, path) ≥ |needle| − T`. A true hit always satisfies this.
/// Returns candidate **working_files indices**.
///
/// Packed-lane driver: the needle is folded once; paths are gated 8-wide
/// through [`neo_frizbee::myers::lcs_ge_batch8`] (AVX2 8×u32 successor
/// bit-vectors). Remainder lanes use the scalar bit-vector. No per-path
/// allocation; O(path_len) word ops per path.
pub(crate) fn lcs_candidates(packed: &PackedPaths, needle: &[u8], max_typos: u16) -> Vec<u32> {
    let m = needle.len();
    let need = m.saturating_sub(max_typos as usize);
    let mut out = Vec::with_capacity(packed.len() / 4);
    if need == 0 {
        out.extend((0..packed.len()).map(|i| packed.file_index(i)));
        return out;
    }

    let mut folded = [0u8; 64];
    let nfold = if m <= 64 {
        fold_needle(needle, &mut folded)
    } else {
        0
    };
    let needle_folded: &[u8] = if m <= 64 { &folded[..nfold] } else { needle };

    let n = packed.len();
    let mut i = 0usize;
    while i + 8 <= n {
        let hays = [
            packed.slice_lower(i),
            packed.slice_lower(i + 1),
            packed.slice_lower(i + 2),
            packed.slice_lower(i + 3),
            packed.slice_lower(i + 4),
            packed.slice_lower(i + 5),
            packed.slice_lower(i + 6),
            packed.slice_lower(i + 7),
        ];
        let bits = neo_frizbee::myers::lcs_ge_batch8(needle_folded, hays, need);
        for k in 0..8 {
            if bits & (1 << k) != 0 {
                out.push(packed.file_index(i + k));
            }
        }
        i += 8;
    }
    while i < n {
        if neo_frizbee::myers::lcs_ge_bit(needle_folded, packed.slice_lower(i), need) {
            out.push(packed.file_index(i));
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: LCS ≥ need via the scalar DP.
    fn lcs_ge_ref(needle: &[u8], hay: &[u8], need: usize) -> bool {
        lcs_len_dp(needle, hay) >= need
    }

    #[test]
    fn lcs_ge_agrees_with_dp_on_hand_cases() {
        // (needle, hay, need, expect)
        let cases: &[(&[u8], &[u8], usize, bool)] = &[
            (b"abc", b"abc", 3, true),
            (b"abc", b"abc", 2, true),
            (b"abc", b"axc", 2, true), // 1 mismatch/deletion
            (b"abc", b"axc", 3, false),
            (b"abc", b"ac", 2, true), // 1 deletion in needle
            (b"abc", b"ac", 3, false),
            (b"abc", b"abbc", 3, true), // insertion in hay is free
            (b"abc", b"cba", 1, true),  // LCS=1
            (b"abc", b"cba", 2, false),
            (b"abcde", b"ace", 3, true), // LCS=3
            (b"abcde", b"ace", 4, false),
            (b"file_picker_", b"lua/fff/file_picker.lua", 9, true),
            (b"score", b"crates/fff-core/src/score.rs", 3, true),
            (b"score", b"crates/fff-core/src/score.rs", 5, true),
            // LCS("score","nothing_here") = "ore" = 3
            (b"score", b"nothing_here", 3, true),
            (b"score", b"nothing_here", 4, false),
            (b"", b"abc", 0, true),
            (b"a", b"a", 1, true),
            (b"a", b"b", 1, false),
        ];
        for &(n, h, need, expect) in cases {
            let got = lcs_ge(n, h, need);
            assert_eq!(
                got, expect,
                "lcs_ge({n:?}, {h:?}, {need}) = {got}, want {expect}"
            );
            assert_eq!(
                got,
                lcs_ge_ref(n, h, need),
                "ref mismatch for {n:?} {h:?} {need}"
            );
        }
    }

    #[test]
    fn lcs_ge_agrees_with_dp_on_random_pairs() {
        // Deterministic LCG — no external rng dependency.
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for _ in 0..2000 {
            let nlen = next() % 17; // 0..=16
            let hlen = next() % 40;
            let mut n = vec![0u8; nlen];
            let mut h = vec![0u8; hlen];
            for b in &mut n {
                *b = b'a' + (next() % 6) as u8;
            }
            for b in &mut h {
                *b = b'a' + (next() % 6) as u8;
            }
            let lcs = lcs_len_dp(&n, &h);
            for need in 0..=nlen {
                let got = lcs_ge(&n, &h, need);
                let want = lcs >= need;
                assert_eq!(
                    got, want,
                    "lcs_ge({n:?}, {h:?}, {need}) = {got}, want {want} (lcs={lcs})"
                );
            }
        }
    }

    #[test]
    fn lcs_ge_case_folds_needle() {
        // `lcs_ge` is the inner kernel: both sides are expected lowercase.
        // `lcs_candidates` folds the needle before calling. Fold here to
        // exercise the same path the driver uses.
        let fold_all = |s: &[u8]| -> Vec<u8> { s.iter().map(|&b| fold(b)).collect() };
        assert!(lcs_ge(&fold_all(b"ABC"), b"abc", 3));
        assert!(lcs_ge(&fold_all(b"AbC"), b"abc", 3));
        assert!(!lcs_ge(&fold_all(b"ABC"), b"abd", 3));
        // And the public wrapper folds for us:
        assert_eq!(lcs_len(b"ABC", b"abc"), 3);
    }

    #[test]
    fn lcs_len_matches_known_values() {
        assert_eq!(lcs_len(b"abc", b"abc"), 3);
        assert_eq!(lcs_len(b"abc", b"axc"), 2);
        assert_eq!(lcs_len(b"abc", b"ac"), 2);
        assert_eq!(lcs_len(b"abc", b"abbc"), 3);
        assert_eq!(lcs_len(b"abcde", b"ace"), 3);
        assert_eq!(lcs_len(b"abc", b"xyz"), 0);
    }

    #[test]
    fn lcs_candidates_returns_file_indices() {
        // Minimal packed table: two paths "aa" and "zz".
        let packed = PackedPaths {
            lower: b"aazz".to_vec(),
            starts: vec![0, 2, 4],
            lens: vec![2, 2, 0],
            file_idx: vec![10, 20, u32::MAX],
            count: 2,
        };
        let hits = lcs_candidates(&packed, b"aa", 0);
        assert_eq!(hits, vec![10]);
        let hits = lcs_candidates(&packed, b"zz", 0);
        assert_eq!(hits, vec![20]);
        let hits = lcs_candidates(&packed, b"az", 1);
        assert_eq!(hits, vec![10, 20]); // T=1 accepts both
    }
}
