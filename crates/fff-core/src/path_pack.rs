//! Packed relative-path corpus + pigeonhole prefilter for fuzzy matching.
//!
//! Paths live in one contiguous byte buffer so we can scan them with tight
//! SIMD-friendly loops. The prefilter is the pigeonhole principle for
//! Levenshtein distance: split the needle into `max_typos + 1` contiguous
//! pieces; a true match must contain **at least one** piece as an exact
//! substring (pieces stay in order across indels, but existence is enough
//! for a reject). OR of memmem hits is a necessary condition.

/// Flat table of packed relative paths (lowercase ASCII mirror optional).
pub(crate) struct PackedPaths {
    /// Concatenated path bytes (original case).
    bytes: Vec<u8>,
    /// Lowercased copy of `bytes` for case-insensitive needle search.
    lower: Vec<u8>,
    /// Byte offset of path `i` in `bytes`/`lower` (`i` in `0..count`).
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

/// Pack paths from a (FileItem, index) iterator.
pub(crate) fn pack_from_iter<'a>(
    items: impl Iterator<Item = (usize, &'a crate::types::FileItem)>,
    arena: crate::simd_path::ArenaPtr,
) -> PackedPaths {
    let mut bytes = Vec::new();
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
        starts.push(bytes.len() as u32);
        lens.push(raw.len() as u16);
        file_idx.push(fi as u32);
        bytes.extend_from_slice(raw);
        for &b in raw {
            lower.push(if b.is_ascii_uppercase() { b + 32 } else { b });
        }
    }
    starts.push(bytes.len() as u32);
    lens.push(0);
    file_idx.push(u32::MAX);
    let count = lens.len().saturating_sub(1);
    PackedPaths {
        bytes,
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
    let mut bytes = Vec::new();
    let mut lower = Vec::new();
    let mut starts = Vec::new();
    let mut lens = Vec::new();
    let mut file_idx = Vec::new();
    let mut buf = [0u8; crate::simd_path::PATH_BUF_SIZE];

    for (fi, file) in files.iter().enumerate().take(limit) {
        if file.is_deleted() {
            continue;
        }
        let path = file.path.read_to_buf(arena, &mut buf);
        let raw = path.as_bytes();
        if raw.is_empty() {
            continue;
        }
        starts.push(bytes.len() as u32);
        lens.push(raw.len() as u16);
        file_idx.push(fi as u32);
        bytes.extend_from_slice(raw);
        for &b in raw {
            lower.push(if b.is_ascii_uppercase() { b + 32 } else { b });
        }
    }
    // pad so slice end is always in range
    starts.push(bytes.len() as u32);
    lens.push(0);
    file_idx.push(u32::MAX);
    let count = lens.len().saturating_sub(1);

    PackedPaths {
        bytes,
        lower,
        starts,
        lens,
        file_idx,
        count,
    }
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
        return hay.contains(&needle[0]);
    }
    // Two-byte memchr-style scan, then confirm.
    let first = needle[0];
    let second = needle[1];
    let nlen = needle.len();
    let mut i = 0usize;
    let end = hay.len() - nlen + 1;
    while i < end {
        if hay[i] == first && hay[i + 1] == second && hay[i..i + nlen] == *needle {
            return true;
        }
        i += 1;
    }
    false
}

/// Candidate file indices: packed slots whose path contains at least one
/// pigeonhole piece. Returns `None` when the filter cannot reject anything
/// (empty needle or single empty piece) — caller should match everything.
pub(crate) fn pigeonhole_candidates(
    packed: &PackedPaths,
    needle: &[u8],
    max_typos: u16,
) -> Option<Vec<u32>> {
    let pieces = pigeon_pieces(needle, max_typos);
    if pieces.is_empty() {
        return None;
    }
    // If every piece is empty or the OR is trivially true for all, skip.
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
