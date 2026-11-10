use smallvec::SmallVec;
use std::path::{Path, PathBuf};

#[cfg(windows)]
pub fn canonicalize(path: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    dunce::canonicalize(path)
}

#[cfg(not(windows))]
pub fn canonicalize(path: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    std::fs::canonicalize(path)
}

// The index stores relative paths with `/` on every platform. These helpers
// convert between that canonical form and the OS-native separator, and are
// no-ops on non-Windows where `/` is already native.

/// Fold a relative path to the canonical `/` form (no-op off Windows).
#[cfg(windows)]
pub fn to_canonical_slashes(rel: &str) -> std::borrow::Cow<'_, str> {
    if rel.contains('\\') {
        std::borrow::Cow::Owned(rel.replace('\\', "/"))
    } else {
        std::borrow::Cow::Borrowed(rel)
    }
}

#[cfg(not(windows))]
#[inline]
pub fn to_canonical_slashes(rel: &str) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Borrowed(rel)
}

/// Rewrite canonical `/` bytes to the OS-native separator in place (no-op off
/// Windows). Used at OS/state boundaries (absolute-path reconstruction).
#[cfg(windows)]
#[inline]
pub fn nativize_slashes_in_place(bytes: &mut [u8]) {
    for b in bytes {
        if *b == b'/' {
            *b = b'\\';
        }
    }
}

#[cfg(not(windows))]
#[inline]
pub fn nativize_slashes_in_place(_bytes: &mut [u8]) {}

/// Git requires a normalized forward-slashed paths on windows
#[cfg(windows)]
pub fn normalize(path: PathBuf) -> PathBuf {
    let as_str = path.to_string_lossy();
    let with_backslashes: String = as_str.replace('/', "\\");
    let buf = PathBuf::from(with_backslashes);
    dunce::canonicalize(&buf).unwrap_or(buf)
}

#[cfg(not(windows))]
pub fn normalize(path: PathBuf) -> PathBuf {
    path
}

#[cfg(windows)]
pub fn expand_tilde(path: &str) -> PathBuf {
    PathBuf::from(path)
}

#[cfg(not(windows))]
pub fn expand_tilde(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/")
        && let Some(home_dir) = dirs::home_dir()
    {
        return home_dir.join(stripped);
    }

    PathBuf::from(path)
}

/// Calculate distance penalty based on directory proximity.
/// Returns a negative penalty score based on how far the candidate is from the current file.
///
/// `candidate_dir` is the directory portion of the candidate path (e.g. `"src/components/"`).
/// It may have a trailing `/` which is stripped internally.
///
/// Zero-allocation: byte walks both directory part iterators in lockstep.
// High bit set in each byte that is neither `/` nor `\`.
#[inline(always)]
fn non_sep_mask(v: u64) -> u64 {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let slash = {
        let x = v ^ 0x2f2f_2f2f_2f2f_2f2f;
        x.wrapping_sub(LO) & !x & HI
    };
    let back = {
        let x = v ^ 0x5c5c_5c5c_5c5c_5c5c;
        x.wrapping_sub(LO) & !x & HI
    };
    !(slash | back) & HI
}

pub fn calculate_distance_penalty(current_file: Option<&str>, candidate_dir: &str) -> i32 {
    CurrentDir::new(current_file).penalty_for(candidate_dir)
}

/// Pre-split current file directory, built once per query and reused
/// across all candidates. Same scoring as `calculate_distance_penalty`.
pub struct CurrentDir<'a> {
    parts: SmallVec<[&'a [u8]; 8]>,
}

impl<'a> CurrentDir<'a> {
    pub fn new(current_file: Option<&'a str>) -> Self {
        let Some(path) = current_file else {
            return Self {
                parts: SmallVec::new(),
            };
        };
        let bytes = path.as_bytes();
        let dir = match bytes.iter().rposition(|&b| b == b'/' || b == b'\\') {
            Some(pos) => &bytes[..pos],
            None => b"".as_slice(),
        };
        Self {
            parts: dir
                .split(|&b| b == b'/' || b == b'\\')
                .filter(|s| !s.is_empty())
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub fn penalty_for(&self, candidate_dir: &str) -> i32 {
        if self.parts.is_empty() || candidate_dir.is_empty() {
            return 0;
        }
        let cand_bytes = candidate_dir.as_bytes();
        let cand_trimmed = match cand_bytes.iter().rposition(|&b| b != b'/' && b != b'\\') {
            Some(pos) => &cand_bytes[..=pos],
            None => b"".as_slice(),
        };
        let mut cand_parts = cand_trimmed
            .split(|&b| b == b'/' || b == b'\\')
            .filter(|s| !s.is_empty());
        let mut common_len = 0usize;
        let mut current_total = 0usize;
        let mut ci = 0usize;
        loop {
            let cur = self.parts.get(ci).copied();
            let cand = cand_parts.next();
            ci += cur.is_some() as usize;
            match (cur, cand) {
                (Some(a), Some(b)) => {
                    current_total += 1;
                    if a == b {
                        common_len += 1;
                    } else {
                        current_total += self.parts.len() - ci;
                        break;
                    }
                }
                (Some(_), None) => {
                    current_total += 1 + (self.parts.len() - ci);
                    break;
                }
                (None, _) => break,
            }
        }
        let depth = current_total.saturating_sub(common_len);
        if depth == 0 {
            return 0;
        }
        (-(depth as i32)).max(-20)
    }

    // Same depth as `penalty_for`. The directory is the bytes before the filename.
    // Trailing separators trim in the last slot, stepping back one slot if the run crosses.
    pub(crate) fn penalty_for_path(
        &self,
        path: &crate::simd_path::ChunkedString,
        arena: crate::simd_path::ArenaPtr,
    ) -> i32 {
        if self.parts.is_empty() {
            return 0;
        }
        let (byte_len_u, name_off) = path.span_lens();
        let total = byte_len_u as usize;
        let mut end = (name_off as usize).min(total);
        if end == 0 {
            return 0;
        }
        let simd = crate::simd_path::SIMD_CHUNK_BYTES;
        let mut slot = (end - 1) / simd;
        let mut index = path.slot_index(arena, slot);
        loop {
            let chunk = crate::simd_path::ChunkedString::chunk_at(arena, unsafe { *index });
            let base = slot * simd;
            let mut i = end - base;
            let mut cross = false;
            while i >= 16 {
                let later = u64::from_le(unsafe {
                    core::ptr::read_unaligned(chunk.as_ptr().add(i - 8) as *const u64)
                });
                let earlier = u64::from_le(unsafe {
                    core::ptr::read_unaligned(chunk.as_ptr().add(i - 16) as *const u64)
                });
                let non_later = non_sep_mask(later);
                let non_earlier = non_sep_mask(earlier);
                if non_later != 0 {
                    let byte = ((63 - non_later.leading_zeros()) / 8) as usize;
                    end = base + (i - 8) + byte + 1;
                    break;
                }
                if non_earlier != 0 {
                    let byte = ((63 - non_earlier.leading_zeros()) / 8) as usize;
                    end = base + (i - 16) + byte + 1;
                    break;
                }
                i -= 16;
            }
            if i < 16 {
            while i >= 8 {
                let v = u64::from_le(unsafe {
                    core::ptr::read_unaligned(chunk.as_ptr().add(i - 8) as *const u64)
                });
                // High bit set in each byte that is neither separator.
                let non = non_sep_mask(v);
                if non != 0 {
                    let byte = ((63 - non.leading_zeros()) / 8) as usize;
                    end = base + (i - 8) + byte + 1;
                    break;
                }
                i -= 8;
            }
            if i < 8 {
                if i == 0 {
                    end = base;
                    cross = slot > 0;
                } else {
                    while i > 0 {
                        i -= 1;
                        let b = chunk[i];
                        if b != b'/' && b != b'\\' {
                            end = base + i + 1;
                            break;
                        }
                        if i == 0 {
                            end = base;
                            cross = slot > 0;
                        }
                    }
                }
            }
            }
            if !cross {
                break;
            }
            slot -= 1;
            index = unsafe { index.sub(1) };
        }
        if end == 0 {
            return 0;
        }
        let mut ci = 0usize;
        let mut common_len = 0usize;
        let mut current_total = 0usize;
        let mut seg_start = 0usize;
        let mut off = 0usize;
        let mut stopped = false;
        index = path.slot_index(arena, 0);
        while off < end && !stopped {
            let chunk = crate::simd_path::ChunkedString::chunk_at(arena, unsafe { *index });
            let n = simd.min(end - off);
            #[cfg(target_arch = "x86_64")]
            if off + n < end {
                let next = unsafe { *index.add(1) };
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        crate::simd_path::ChunkedString::chunk_at(arena, next).as_ptr() as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
            for i in memchr::memchr2_iter(b'/', b'\\', &chunk[..n]) {
                let abs = off + i;
                if abs > seg_start {
                    let kept = if seg_start >= off {
                        self.absorb_bytes(
                            &chunk[seg_start - off..i],
                            &mut ci,
                            &mut common_len,
                            &mut current_total,
                        )
                    } else {
                        self.absorb_segment(
                            path,
                            arena,
                            seg_start,
                            abs,
                            &mut ci,
                            &mut common_len,
                            &mut current_total,
                        )
                    };
                    if !kept {
                        stopped = true;
                        break;
                    }
                }
                seg_start = abs + 1;
            }
            off += n;
            if off < end {
                index = unsafe { index.add(1) };
            }
        }
        if !stopped
            && end > seg_start
            && !self.absorb_segment(
                path,
                arena,
                seg_start,
                end,
                &mut ci,
                &mut common_len,
                &mut current_total,
            )
        {
            stopped = true;
        }
        if !stopped && self.parts.get(ci).is_some() {
            ci += 1;
            current_total += 1 + self.parts.len() - ci;
        }
        let depth = current_total.saturating_sub(common_len);
        if depth == 0 {
            return 0;
        }
        (-(depth as i32)).max(-20)
    }

    fn absorb_segment(
        &self,
        path: &crate::simd_path::ChunkedString,
        arena: crate::simd_path::ArenaPtr,
        start: usize,
        end: usize,
        ci: &mut usize,
        common_len: &mut usize,
        current_total: &mut usize,
    ) -> bool {
        let cur = self.parts.get(*ci).copied();
        *ci += cur.is_some() as usize;
        match cur {
            Some(part) => {
                *current_total += 1;
                if path.eq_range(arena, start, end, part) {
                    *common_len += 1;
                    true
                } else {
                    *current_total += self.parts.len() - *ci;
                    false
                }
            }
            None => false,
        }
    }

    fn absorb_bytes(
        &self,
        bytes: &[u8],
        ci: &mut usize,
        common_len: &mut usize,
        current_total: &mut usize,
    ) -> bool {
        let cur = self.parts.get(*ci).copied();
        *ci += cur.is_some() as usize;
        match cur {
            Some(part) => {
                *current_total += 1;
                if bytes == part {
                    *common_len += 1;
                    true
                } else {
                    *current_total += self.parts.len() - *ci;
                    false
                }
            }
            None => false,
        }
    }
}

/// Memoized distance penalties keyed by `parent_dir_index`. Scoring loops
/// revisit the same dirs thousands of times per query; this keeps the
/// first computation and reuses it. Size once per query with the dir count.
pub struct DirPenaltyMemo {
    penalties: Vec<i32>,
}

thread_local! {
    static PENALTY_BUF: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl DirPenaltyMemo {
    pub fn new(dir_count: usize) -> Self {
        let mut penalties = PENALTY_BUF.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
        // `clear` keeps the allocation. A new search must not see old dirs.
        penalties.clear();
        if dir_count > 0 {
            penalties.resize(dir_count, i32::MAX);
        }
        Self { penalties }
    }

    pub fn penalty_for(&mut self, cur: &CurrentDir, dir_idx: usize, candidate_dir: &str) -> i32 {
        if let Some(&p) = self.penalties.get(dir_idx)
            && p != i32::MAX
        {
            return p;
        }
        let p = cur.penalty_for(candidate_dir);
        if dir_idx < self.penalties.len() {
            self.penalties[dir_idx] = p;
        }
        p
    }

    pub fn get(&self, dir_idx: usize) -> Option<i32> {
        match self.penalties.get(dir_idx) {
            Some(&p) if p != i32::MAX => Some(p),
            _ => None,
        }
    }

    pub fn penalty_for_grow(
        &mut self,
        cur: &CurrentDir,
        dir_idx: usize,
        candidate_dir: &str,
    ) -> i32 {
        if let Some(&p) = self.penalties.get(dir_idx)
            && p != i32::MAX
        {
            return p;
        }
        let p = cur.penalty_for(candidate_dir);
        if dir_idx >= self.penalties.len() {
            self.penalties.resize(dir_idx + 1, i32::MAX);
        }
        self.penalties[dir_idx] = p;
        p
    }

    pub(crate) fn penalty_for_grow_path(
        &mut self,
        cur: &CurrentDir,
        dir_idx: usize,
        path: &crate::simd_path::ChunkedString,
        arena: crate::simd_path::ArenaPtr,
    ) -> i32 {
        if let Some(&p) = self.penalties.get(dir_idx)
            && p != i32::MAX
        {
            return p;
        }
        let p = cur.penalty_for_path(path, arena);
        if dir_idx >= self.penalties.len() {
            self.penalties.resize(dir_idx + 1, i32::MAX);
        }
        self.penalties[dir_idx] = p;
        p
    }
}

impl Drop for DirPenaltyMemo {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.penalties);
        PENALTY_BUF.with(|slot| {
            let slot = &mut *slot.borrow_mut();
            if buf.capacity() > slot.capacity() {
                *slot = buf;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference implementation using `Path::components` (July / upstream
    // semantics). Used to catch byte-walk scoring drift (checksum mismatches).
    fn penalty_via_components(current_file: Option<&str>, candidate_dir: &str) -> i32 {
        let Some(current_path) = current_file else {
            return 0;
        };
        let current_dir = Path::new(current_path).parent().unwrap_or(Path::new(""));
        let candidate = Path::new(candidate_dir);
        if current_dir == candidate {
            return 0;
        }
        let mut current_parts = current_dir.components();
        let mut candidate_parts = candidate.components();
        let mut common_len = 0usize;
        let mut current_total = 0usize;
        loop {
            match (current_parts.next(), candidate_parts.next()) {
                (Some(a), Some(b)) => {
                    current_total += 1;
                    if a == b {
                        common_len += 1;
                    } else {
                        current_total += current_parts.count();
                        break;
                    }
                }
                (Some(_), None) => {
                    current_total += 1 + current_parts.count();
                    break;
                }
                (None, _) => break,
            }
        }
        let depth = current_total - common_len;
        if depth == 0 {
            0
        } else {
            (-(depth as i32)).max(-20)
        }
    }

    #[test]
    fn byte_walk_matches_path_components() {
        let cases: &[(Option<&str>, &str)] = &[
            (Some("examples/user/test/main.rs"), "examples/user/test/"),
            (
                Some("examples/user/test/subdir/file.rs"),
                "examples/user/test/",
            ),
            (Some("crates/fff-core/src/score.rs"), "crates/fff-core/src/"),
            (Some("crates/fff-core/src/score.rs"), "score/"),
            (Some("src/score.rs"), "src/"),
            (Some("score.rs"), "score/"),
            (Some("a/b/c/d/e.rs"), "a/b/"),
            (Some("main.rs"), ""),
            (Some("dir/file.rs"), "other/"),
        ];
        for (cur, cand) in cases {
            let a = calculate_distance_penalty(*cur, cand);
            let b = penalty_via_components(*cur, cand);
            assert_eq!(a, b, "penalty mismatch cur={cur:?} cand={cand:?}");
        }
    }

    #[test]
    fn dir_memo_matches_direct_and_reuses() {
        let cur_file = Some("examples/user/test/subdir/file.rs");
        let cur = CurrentDir::new(cur_file);
        let dirs = ["examples/user/test/", "examples/user/", "other/", ""];
        let mut memo = DirPenaltyMemo::new(dirs.len());
        for (i, d) in dirs.iter().enumerate() {
            let a = memo.penalty_for(&cur, i, d);
            assert_eq!(a, calculate_distance_penalty(cur_file, d));
            assert_eq!(memo.penalty_for(&cur, i, d), a);
        }
        // Out-of-range index computes without caching.
        let p = memo.penalty_for(&cur, 99, "examples/user/test/");
        assert_eq!(
            p,
            calculate_distance_penalty(cur_file, "examples/user/test/")
        );
    }

    #[test]
    #[cfg(not(target_family = "windows"))]
    fn test_calculate_distance_penalty() {
        // candidate_dir is the directory portion (with or without trailing /)
        assert_eq!(calculate_distance_penalty(None, "examples/user/test/"), 0);
        // Same directory
        assert_eq!(
            calculate_distance_penalty(Some("examples/user/test/main.rs"), "examples/user/test/"),
            0
        );
        //
        // One level apart
        assert_eq!(
            calculate_distance_penalty(
                Some("examples/user/test/subdir/file.rs"),
                "examples/user/test/"
            ),
            -1
        );
        //
        // Different subdirectories (same parent)
        assert_eq!(
            calculate_distance_penalty(
                Some("examples/user/test/dir1/file.rs"),
                "examples/user/test/dir2/"
            ),
            -1
        );

        assert_eq!(
            calculate_distance_penalty(
                Some("examples/audio-announce/src/lib/audio-announce.rs"),
                "examples/audio-announce/src/"
            ),
            -1
        );

        assert_eq!(
            calculate_distance_penalty(
                Some("examples/audio-announce/src/audio-announce.rs"),
                "examples/pixel/src/"
            ),
            -2
        );

        // Root level files (empty dir)
        assert_eq!(calculate_distance_penalty(Some("main.rs"), ""), 0);
    }

    #[test]
    #[cfg(target_family = "windows")]
    fn distance_penalty_works_on_windows() {
        assert_eq!(
            calculate_distance_penalty(None, "examples\\user\\test\\"),
            0
        );
        // Same directory
        assert_eq!(
            calculate_distance_penalty(
                Some("examples\\user\\test\\main.rs"),
                "examples\\user\\test\\"
            ),
            0
        );
        //
        // One level apart
        assert_eq!(
            calculate_distance_penalty(
                Some("examples\\user\\test\\subdir\\file.rs"),
                "examples\\user\\test\\"
            ),
            -1
        );
    }

    #[test]
    fn penalty_for_path_matches_copied_dir() {
        let current = "examples/user/test/main.rs";
        let cur = CurrentDir::new(Some(current));
        let paths = [
            "examples/user/test/main.rs",
            "examples/user/other.rs",
            "examples/audio-announce/src/lib/audio-announce.rs",
            "lib/mod.rs",
            "main.rs",
            "src/foo\\bar/baz.txt",
            "very/deeply/nested/directory/structure/with/many/levels/file.txt",
            "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q.rs",
        ];
        let files: Vec<crate::types::FileItem> = paths
            .iter()
            .map(|path| {
                let start = path
                    .rfind(std::path::is_separator)
                    .map(|i| i + 1)
                    .unwrap_or(0) as u16;
                crate::types::FileItem::new_raw(start, 0, 0, None, false)
            })
            .collect();
        let owned: Vec<String> = paths.iter().map(|p| (*p).to_string()).collect();
        let (store, strings) =
            crate::simd_path::build_chunked_path_store_from_strings(&owned, &files);
        let arena = store.as_arena_ptr();
        for (path, chunked) in paths.iter().zip(&strings) {
            let dir = match path.rfind(std::path::is_separator) {
                Some(i) => &path[..=i],
                None => "",
            };
            assert_eq!(
                cur.penalty_for_path(chunked, arena),
                calculate_distance_penalty(Some(current), dir),
                "{path}"
            );
        }
        let mut memo = DirPenaltyMemo::new(0);
        let first = memo.penalty_for_grow_path(&cur, 3, &strings[0], arena);
        assert_eq!(memo.penalty_for_grow_path(&cur, 3, &strings[1], arena), first);
    }
}
