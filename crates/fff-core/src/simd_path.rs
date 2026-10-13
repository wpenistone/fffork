use ahash::AHashMap;
use std::borrow::Cow;

// Path chunk size in bytes. Swept on sumo_fork (226k files, 87 B avg path):
//
//   bytes   heap    dedup    gate      sw
//    16     7.5MB   12.2x   4879us   6478us
//    32     8.2MB    4.6x   4223us   5783us
//    64    14.1MB    2.2x   4438us   6715us
//
// 32 wins: it halves the index table (1.33M -> 0.72M slots) and shortens
// the dependent chunk chain, which is what the gate actually costs. 64
// overshoots - dedup collapses and the chunk array (12.3 MB) no longer
// fits cache, so both gate and SW get worse again.
pub(crate) const SIMD_CHUNK_BYTES: usize = 32;

/// Read-only view of a path store: 16-byte chunk arena plus the flat table of
/// per-path chunk indices. Both point into the owning store's Vecs.
#[derive(Clone, Copy)]
pub struct ArenaPtr {
    chunks: *const u8,
    indices: *const u32,
}

// SAFETY: The arena is a read-only immutable part of file sync
unsafe impl Send for ArenaPtr {}
unsafe impl Sync for ArenaPtr {}

impl ArenaPtr {
    #[inline]
    pub fn new(chunks: *const u8, indices: *const u32) -> Self {
        Self { chunks, indices }
    }

    #[inline]
    pub fn null() -> Self {
        Self {
            chunks: std::ptr::null(),
            indices: std::ptr::null(),
        }
    }

    #[inline]
    pub fn as_ptr(self) -> *const u8 {
        self.chunks
    }

    #[inline]
    fn chunk_ptr(self, idx: u32) -> *const u8 {
        unsafe { self.chunks.add(idx as usize * SIMD_CHUNK_BYTES) }
    }
}

impl std::fmt::Debug for ArenaPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "--arena-raw-pointer-0x({:?})", self.chunks)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct SimdChunk(pub(crate) [u8; SIMD_CHUNK_BYTES]);

impl Default for SimdChunk {
    #[inline]
    fn default() -> Self {
        Self([0u8; SIMD_CHUNK_BYTES])
    }
}

impl std::fmt::Debug for SimdChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Show the actual bytes, trimming trailing zeros for readability
        let end = self.0.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        write!(f, "SimdChunk({:?})", &self.0[..end])
    }
}

pub use crate::constants::PATH_BUF_SIZE;

/// Chunk pointer capacity needed for the longest path the platform allows.
pub(crate) const MAX_PATH_CHUNKS: usize = PATH_BUF_SIZE.div_ceil(SIMD_CHUNK_BYTES);

// Pointer capacity for frizbee's resolved API, which counts 16 B units.
// Keep separate from MAX_PATH_CHUNKS (32 B arena units).
pub(crate) const MAX_FRIZBEE_CHUNKS: usize =
    PATH_BUF_SIZE.div_ceil(neo_frizbee::SIMD_CHUNK_BYTES);

/// A path stored as a run of chunk indices in the owning store's flat index
/// table. 8 bytes; every read needs the store's `ArenaPtr`.
#[derive(Clone)]
pub(crate) struct ChunkedString {
    index_offset: u32,
    pub byte_len: u16,
    /// Byte offset where the filename begins. 0 for root-level files.
    pub filename_offset: u16,
}

impl ChunkedString {
    // Stable ordering key for tie-breaks. Paths are stored in scan order, so
    // this is unique per path and orders deterministically.
    #[inline]
    pub(crate) fn rank_key(&self) -> u64 {
        (self.byte_len as u64) << 32 | self.index_offset as u64
    }

    pub fn empty() -> Self {
        Self {
            index_offset: 0,
            byte_len: 0,
            filename_offset: 0,
        }
    }

    #[inline]
    pub fn new(index_offset: u32, byte_len: u16, filename_offset: u16) -> Self {
        Self {
            index_offset,
            byte_len,
            filename_offset,
        }
    }

    #[inline]
    pub fn chunk_count(&self) -> usize {
        chunks_needed(self.byte_len as usize)
    }

    #[inline]
    fn indices<'a>(&self, arena: ArenaPtr) -> &'a [u32] {
        let count = self.chunk_count();
        if count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(arena.indices.add(self.index_offset as usize), count) }
    }

    #[inline]
    pub(crate) fn equals(&self, arena: ArenaPtr, other: &str) -> bool {
        if other.len() != self.byte_len as usize {
            return false;
        }
        self.indices(arena)
            .iter()
            .zip(other.as_bytes().chunks(SIMD_CHUNK_BYTES))
            .all(|(&index, expected)| {
                let actual =
                    unsafe { core::slice::from_raw_parts(arena.chunk_ptr(index), expected.len()) };
                actual == expected
            })
    }

    #[inline]
    pub fn resolve_ptrs<'a>(&self, arena: ArenaPtr, buf: &'a mut [*const u8]) -> &'a [*const u8] {
        let indices = self.indices(arena);
        let count = indices.len().min(buf.len());
        for (slot, &idx) in buf.iter_mut().zip(&indices[..count]) {
            *slot = arena.chunk_ptr(idx);
        }
        &buf[..count]
    }

    /// Visit each 16-byte chunk of this path in order (no full-path copy).
    /// `f(global_offset, chunk_slice)` — slice may be shorter than 16 at the end.
    #[inline]
    pub fn for_each_chunk(&self, arena: ArenaPtr, mut f: impl FnMut(usize, &[u8])) {
        let total = self.byte_len as usize;
        if total == 0 {
            return;
        }
        let indices = self.indices(arena);
        let mut off = 0usize;
        for &idx in indices.iter() {
            if off >= total {
                break;
            }
            let take = SIMD_CHUNK_BYTES.min(total - off);
            let chunk = unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), take) };
            f(off, chunk);
            off += take;
        }
    }

    #[inline]
    pub(crate) fn common_suffix_len_ignore_ascii_case(
        &self,
        arena: ArenaPtr,
        other: &[u8],
    ) -> usize {
        let indices = self.indices(arena);
        let mut end = self.byte_len as usize;
        let mut matched = 0;
        while end > 0 && matched < other.len() {
            let chunk = (end - 1) / SIMD_CHUNK_BYTES;
            let len = end - chunk * SIMD_CHUNK_BYTES;
            let bytes =
                unsafe { core::slice::from_raw_parts(arena.chunk_ptr(indices[chunk]), len) };
            for (&a, &b) in bytes
                .iter()
                .rev()
                .zip(other[..other.len() - matched].iter().rev())
            {
                if !a.eq_ignore_ascii_case(&b) {
                    return matched;
                }
                matched += 1;
            }
            end -= len;
        }
        matched
    }

    #[inline]
    pub(crate) fn has_extension(&self, arena: ArenaPtr, extension: &str) -> bool {
        let filename_len = (self.byte_len - self.filename_offset) as usize;
        if filename_len <= extension.len() + 1 {
            return false;
        }
        let dot = self.byte_len as usize - extension.len() - 1;
        let chunk = self.indices(arena)[dot / SIMD_CHUNK_BYTES];
        let byte = unsafe { *arena.chunk_ptr(chunk).add(dot % SIMD_CHUNK_BYTES) };
        byte == b'.'
            && self.common_suffix_len_ignore_ascii_case(arena, extension.as_bytes())
                == extension.len()
    }
    /// Filename as `&str`, borrowing from the arena when it fits in one chunk
    /// and otherwise copying into caller scratch. `filename_cow` heap-allocates
    /// in exactly that second case, which on a repo whose directory prefixes
    /// straddle chunk boundaries (measured: linux-kernel, a 26.6 B prefix in a
    /// 32 B chunk) happens once per candidate.
    #[inline]
    pub fn filename_ref<'a>(&self, arena: ArenaPtr, scratch: &'a mut [u8]) -> &'a str {
        let off = self.filename_offset as usize;
        let total = self.byte_len as usize;
        if off >= total {
            return "";
        }
        let len = total - off;
        let idx = self.indices(arena);
        let start_chunk = off / SIMD_CHUNK_BYTES;
        let offset_in_chunk = off % SIMD_CHUNK_BYTES;

        // Zero-copy when the filename sits inside a single chunk.
        if start_chunk < idx.len() && offset_in_chunk + len <= SIMD_CHUNK_BYTES {
            let ptr = unsafe { arena.chunk_ptr(idx[start_chunk]).add(offset_in_chunk) };
            let slice = unsafe { core::slice::from_raw_parts(ptr, len) };
            return unsafe { core::str::from_utf8_unchecked(slice) };
        }

        let n = len.min(scratch.len());
        self.copy_window_to(arena, off as u32, (off + n) as u32, scratch);
        unsafe { core::str::from_utf8_unchecked(&scratch[..n]) }
    }

    #[inline]
    fn write_slice_to_vec(
        indices: &[u32],
        arena: ArenaPtr,
        offset_in_chunk: usize,
        len: usize,
        vec: &mut Vec<u8>,
    ) {
        let mut written = 0usize;
        for (i, &idx) in indices.iter().enumerate() {
            let chunk_bytes =
                unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), SIMD_CHUNK_BYTES) };
            let start = if i == 0 { offset_in_chunk } else { 0 };
            let end = SIMD_CHUNK_BYTES.min(start + (len - written));
            vec.extend_from_slice(&chunk_bytes[start..end]);
            written += end - start;
        }
    }

    /// Return the filename portion as a `Cow<str>`.
    ///
    /// When the filename starts at a chunk boundary and fits in one chunk we
    /// borrow directly from the arena (zero-copy). Otherwise we allocate.
    /// Filenames are almost always <=16 bytes so the fast path dominates.
    #[inline]
    pub fn filename_cow<'a>(&self, arena: ArenaPtr) -> Cow<'a, str> {
        let fname_offset = self.filename_offset as usize;
        let fname_len = self.byte_len as usize - fname_offset;
        if fname_len == 0 {
            return Cow::Borrowed("");
        }

        let indices = self.indices(arena);
        let start_chunk = fname_offset / SIMD_CHUNK_BYTES;
        let offset_in_chunk = fname_offset % SIMD_CHUNK_BYTES;

        // Borrow when the filename fits in the remainder of its first chunk,
        // including the common mid-chunk start after a directory prefix.
        if offset_in_chunk + fname_len <= SIMD_CHUNK_BYTES {
            let ptr = unsafe { arena.chunk_ptr(indices[start_chunk]).add(offset_in_chunk) };
            let slice = unsafe { core::slice::from_raw_parts(ptr, fname_len) };
            return Cow::Borrowed(unsafe { core::str::from_utf8_unchecked(slice) });
        }

        let mut out = String::with_capacity(fname_len);
        let needed_chunks = chunks_needed(offset_in_chunk + fname_len);
        Self::write_slice_to_vec(
            &indices[start_chunk..start_chunk + needed_chunks],
            arena,
            offset_in_chunk,
            fname_len,
            unsafe { out.as_mut_vec() },
        );
        Cow::Owned(out)
    }

    /// Prefetch this path's arena chunks. The gate's chunk lookups are scattered
    /// across the dedup store, so issuing the next candidate's loads early
    /// hides part of the miss.
    #[inline]
    pub fn prefetch(&self, arena: ArenaPtr) {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let total = self.byte_len as usize;
            if total == 0 {
                return;
            }
            let mut off = 0usize;
            for &idx in self.indices(arena).iter() {
                if off >= total {
                    break;
                }
                unsafe {
                    _mm_prefetch(arena.chunk_ptr(idx) as *const i8, _MM_HINT_T0);
                }
                off += SIMD_CHUNK_BYTES;
            }
        }
    }

    /// Copy path bytes `[start, end)` into `out` (needs `end - start` bytes),
    /// touching only the arena chunks that cover the window. One arena pass,
    /// no full-path bounce buffer. Short/missing trailing chunks zero-fill.
    pub fn copy_window_to(&self, arena: ArenaPtr, start: u32, end: u32, out: &mut [u8]) {
        let total = self.byte_len as usize;
        let start = (start as usize).min(total);
        let end = (end as usize).min(total);
        let len = end.saturating_sub(start);
        if len == 0 {
            return;
        }
        debug_assert!(out.len() >= len);
        let indices = self.indices(arena);
        let start_chunk = start / SIMD_CHUNK_BYTES;
        let offset_in_chunk = start % SIMD_CHUNK_BYTES;
        let chunks_to_copy =
            chunks_needed(offset_in_chunk + len).min(indices.len().saturating_sub(start_chunk));

        let mut written = 0usize;
        for (i, &idx) in indices[start_chunk..start_chunk + chunks_to_copy]
            .iter()
            .enumerate()
        {
            let chunk_bytes =
                unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), SIMD_CHUNK_BYTES) };
            let s = if i == 0 { offset_in_chunk } else { 0 };
            let e = SIMD_CHUNK_BYTES.min(s + (len - written));
            out[written..written + (e - s)].copy_from_slice(&chunk_bytes[s..e]);
            written += e - s;
        }
        out[written..len].fill(0);
    }

    /// Truncates at `buf.len()` if exceeded -- use `[u8; PATH_BUF_SIZE]` to avoid.
    #[inline]
    pub fn read_to_buf<'a>(&self, arena: ArenaPtr, buf: &'a mut [u8]) -> &'a str {
        let total = (self.byte_len as usize).min(buf.len());
        let indices = self.indices(arena);
        let chunks_to_copy = total.div_ceil(SIMD_CHUNK_BYTES).min(indices.len());

        for (i, &idx) in indices[..chunks_to_copy].iter().enumerate() {
            let dst_offset = i * SIMD_CHUNK_BYTES;
            let take = SIMD_CHUNK_BYTES.min(total - dst_offset);

            unsafe {
                core::ptr::copy_nonoverlapping(
                    arena.chunk_ptr(idx),
                    buf.as_mut_ptr().add(dst_offset),
                    take,
                );
            }
        }

        unsafe { core::str::from_utf8_unchecked(&buf[..total]) }
    }

    #[inline]
    pub fn write_dir_to(&self, arena: ArenaPtr, out: &mut String) {
        out.clear();

        let dir_len = self.filename_offset as usize;
        out.reserve(dir_len);
        let indices = self.indices(arena);
        let dir_chunks = chunks_needed(dir_len).min(indices.len());
        let vec = unsafe { out.as_mut_vec() };
        for (i, &idx) in indices[..dir_chunks].iter().enumerate() {
            let take = SIMD_CHUNK_BYTES.min(dir_len - i * SIMD_CHUNK_BYTES);
            vec.extend_from_slice(unsafe {
                core::slice::from_raw_parts(arena.chunk_ptr(idx), take)
            });
        }
    }

    #[inline]
    pub fn write_filename_to(&self, arena: ArenaPtr, out: &mut String) {
        out.clear();

        let fname_offset = self.filename_offset as usize;
        let fname_len = self.byte_len as usize - fname_offset;
        out.reserve(fname_len);
        let indices = self.indices(arena);
        let start_chunk = fname_offset / SIMD_CHUNK_BYTES;
        let offset_in_chunk = fname_offset % SIMD_CHUNK_BYTES;
        let needed_chunks = chunks_needed(offset_in_chunk + fname_len);
        Self::write_slice_to_vec(
            &indices[start_chunk..start_chunk + needed_chunks],
            arena,
            offset_in_chunk,
            fname_len,
            unsafe { out.as_mut_vec() },
        );
    }

    #[inline]
    pub fn write_to_string(&self, arena: ArenaPtr, out: &mut String) {
        out.clear();

        let total = self.byte_len as usize;
        if total == 0 {
            return;
        }
        out.reserve(total);
        let vec = unsafe { out.as_mut_vec() };
        for (i, &idx) in self.indices(arena).iter().enumerate() {
            let take = SIMD_CHUNK_BYTES.min(total - i * SIMD_CHUNK_BYTES);
            vec.extend_from_slice(unsafe {
                core::slice::from_raw_parts(arena.chunk_ptr(idx), take)
            });
        }
    }
}

impl std::fmt::Debug for ChunkedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkedString")
            .field("index_offset", &self.index_offset)
            .field("chunks", &self.chunk_count())
            .field("byte_len", &self.byte_len)
            .field("filename_offset", &self.filename_offset)
            .finish()
    }
}

#[inline]
const fn chunks_needed(byte_len: usize) -> usize {
    if byte_len == 0 {
        0
    } else {
        byte_len.div_ceil(SIMD_CHUNK_BYTES)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ChunkedPathStore {
    arena: Vec<SimdChunk>,
    indices: Vec<u32>,
}

// SAFETY: arena is immutable after construction. Pointers derived from it are
// only read during scoring (no mutation, no reallocation).
unsafe impl Send for ChunkedPathStore {}
unsafe impl Sync for ChunkedPathStore {}

impl ChunkedPathStore {
    pub fn heap_bytes(&self) -> usize {
        self.arena.len() * SIMD_CHUNK_BYTES + self.indices.len() * std::mem::size_of::<u32>()
    }

    #[cfg(test)]
    fn unique_chunks(&self) -> usize {
        self.arena.len()
    }

    /// Deduplicated chunk count, for FFF_PHASE_TIMING layout stats.
    pub fn unique_chunks_for_stats(&self) -> usize {
        self.arena.len()
    }

    #[inline]
    pub fn as_arena_ptr(&self) -> ArenaPtr {
        ArenaPtr::new(self.arena.as_ptr() as *const u8, self.indices.as_ptr())
    }
}

/// At runtime the builder should be split out from the store after `finish()`.
#[derive(Clone, Debug)]
pub(crate) struct ChunkedPathStoreBuilder {
    arena: Vec<SimdChunk>,
    indices: Vec<u32>,
    chunk_dedup: AHashMap<[u8; SIMD_CHUNK_BYTES], u32>,
}

impl ChunkedPathStoreBuilder {
    pub fn new(estimated_files: usize) -> Self {
        // Most paths fit into 32 bytes = 2 chunks. Dedup keeps the arena
        // below that; do not over-reserve the HashMap table (it is the
        // largest transient allocation during scan on flat corpora).
        let est_indices = estimated_files * 2;
        Self {
            arena: Vec::with_capacity(est_indices / 2),
            indices: Vec::with_capacity(est_indices),
            chunk_dedup: AHashMap::with_capacity(est_indices / 4),
        }
    }

    pub fn finish(self) -> ChunkedPathStore {
        let Self {
            mut arena,
            mut indices,
            ..
        } = self;
        arena.shrink_to_fit();
        indices.shrink_to_fit();
        ChunkedPathStore { arena, indices }
    }

    pub fn as_arena_ptr(&self) -> ArenaPtr {
        ArenaPtr::new(self.arena.as_ptr() as *const u8, self.indices.as_ptr())
    }

    /// Like [`add_file_immediate`] but for directory paths where the entire
    /// string is the "directory" portion (filename_offset == byte_len).
    pub fn add_dir_immediate(&mut self, dir_rel_path: &str) -> ChunkedString {
        self.add_file_immediate(dir_rel_path, dir_rel_path.len() as u16)
    }

    pub fn add_file_immediate(&mut self, rel_path: &str, filename_offset: u16) -> ChunkedString {
        let path_bytes = rel_path.as_bytes();
        // Platform paths never approach u16::MAX (Linux 4K, Windows 32K),
        // so the `as u16` below is exact, never a wrap.
        debug_assert!(path_bytes.len() <= u16::MAX as usize);
        let index_offset = self.indices.len() as u32;

        for chunk in path_bytes.chunks(SIMD_CHUNK_BYTES) {
            let mut chunk_bytes = [0u8; SIMD_CHUNK_BYTES];
            chunk_bytes[..chunk.len()].copy_from_slice(chunk);

            let arena_idx = match self.chunk_dedup.get(&chunk_bytes) {
                Some(&idx) => idx,
                None => {
                    let idx = self.arena.len() as u32;
                    self.arena.push(SimdChunk(chunk_bytes));
                    self.chunk_dedup.insert(chunk_bytes, idx);
                    idx
                }
            };

            self.indices.push(arena_idx);
        }

        ChunkedString::new(index_offset, path_bytes.len() as u16, filename_offset)
    }
}

#[cfg(test)]
pub(crate) fn build_chunked_path_store_from_strings(
    rel_paths: &[String],
    files: &[crate::types::FileItem],
) -> (ChunkedPathStore, Vec<ChunkedString>) {
    assert_eq!(rel_paths.len(), files.len());
    let mut builder = ChunkedPathStoreBuilder::new(rel_paths.len());
    let strings: Vec<ChunkedString> = rel_paths
        .iter()
        .zip(files.iter())
        .map(|(rel_path, file)| builder.add_file_immediate(rel_path, file.path.filename_offset))
        .collect();
    (builder.finish(), strings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_suffix_matches_contiguous_bytes() {
        let paths = [
            "",
            "src/main.rs",
            "src/components/Controller.tsx",
            "src/日本語/éController.tsx",
            "abcdefghijklmnopABCDEFGHIJKLMNOP.rs",
        ];
        let (store, strings, _) = build_test_store(&paths);
        for (path, chunked) in paths.iter().zip(&strings) {
            for candidate in paths {
                for start in 0..=candidate.len() {
                    let suffix = &candidate.as_bytes()[start..];
                    let expected = path
                        .as_bytes()
                        .iter()
                        .rev()
                        .zip(suffix.iter().rev())
                        .take_while(|(a, b)| a.eq_ignore_ascii_case(b))
                        .count();
                    assert_eq!(
                        chunked.common_suffix_len_ignore_ascii_case(store.as_arena_ptr(), suffix),
                        expected
                    );
                }
            }
        }
    }

    fn make_file_item(path: &str) -> crate::types::FileItem {
        let filename_start = path
            .rfind(std::path::is_separator)
            .map(|i| i + 1)
            .unwrap_or(0) as u16;
        crate::types::FileItem::new_raw(filename_start, 0, 0, None, false)
    }

    fn build_test_store(
        paths: &[&str],
    ) -> (
        ChunkedPathStore,
        Vec<ChunkedString>,
        Vec<crate::types::FileItem>,
    ) {
        let mut files: Vec<crate::types::FileItem> =
            paths.iter().map(|p| make_file_item(p)).collect();
        let path_strings: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
        let (store, strings) = build_chunked_path_store_from_strings(&path_strings, &files);
        for (i, file) in files.iter_mut().enumerate() {
            file.set_path(strings[i].clone());
            file.set_presence(crate::presence::mask_of(path_strings[i].as_bytes()));
        }
        (store, strings, files)
    }

    #[test]
    fn test_chunked_store_empty() {
        let (store, strings, _files) = build_test_store(&[]);
        assert_eq!(strings.len(), 0);
        assert_eq!(store.unique_chunks(), 0);
    }

    #[test]
    fn chunked_extensions_match_contiguous_filenames() {
        let paths = [
            "",
            ".rs",
            "a.",
            "src/long_directory/main.RS",
            "src/é.日本語",
            "dir.rs/file",
            "abcdefghijklmn.rs",
            "abcdefghijklmno.rs",
            "abcdefghijklmnop.rs",
            "a.tar.gz",
        ];
        let (store, strings, _) = build_test_store(&paths);
        for (path, chunked) in paths.iter().zip(&strings) {
            let filename = path.rsplit('/').next().unwrap();
            for extension in [
                "",
                "rs",
                "RS",
                "gz",
                "tar.gz",
                "日本語",
                "語",
                "a/more.than.a.chunk",
            ] {
                assert_eq!(
                    chunked.has_extension(store.as_arena_ptr(), extension),
                    crate::index::constraints::file_has_extension(filename, extension),
                    "{path}, {extension}"
                );
            }
        }
    }

    #[test]
    fn test_chunked_store_basic() {
        let (store, strings, _files) =
            build_test_store(&["src/lib.rs", "src/main.rs", "Cargo.toml"]);
        let arena = store.as_arena_ptr();

        assert_eq!(strings.len(), 3);
        assert!(store.unique_chunks() >= 2);

        let mut buf = [0u8; 512];
        assert_eq!(
            strings[0].read_to_buf(arena, &mut buf).len(),
            "src/lib.rs".len()
        );
        assert_eq!(
            strings[2].read_to_buf(arena, &mut buf).len(),
            "Cargo.toml".len()
        );
    }

    #[test]
    fn test_chunked_string_full_path() {
        let (store, strings, _files) = build_test_store(&[
            "src/components/Button.tsx",
            "src/components/Button.test.tsx",
        ]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];

        let mut buf = [0u8; 512];
        assert_eq!(cs.read_to_buf(arena, &mut buf), "src/components/Button.tsx");
        assert_eq!(cs.byte_len, 25);
        assert_eq!(cs.filename_offset, 15);

        let cs = &strings[1];
        let mut buf = [0u8; 512];
        assert_eq!(
            cs.read_to_buf(arena, &mut buf),
            "src/components/Button.test.tsx"
        );
        assert_eq!(cs.byte_len, 30);
        assert_eq!(cs.filename_offset, 15);
    }

    #[test]
    fn test_chunked_string_dir_and_filename() {
        let (store, strings, _files) = build_test_store(&["src/components/Button.tsx"]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];

        let mut s = String::new();
        cs.write_dir_to(arena, &mut s);
        assert_eq!(s, "src/components/");
        cs.write_filename_to(arena, &mut s);
        assert_eq!(s, "Button.tsx");
    }

    #[test]
    fn test_chunked_string_root_file() {
        let (store, strings, _files) = build_test_store(&["Cargo.toml"]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];

        let mut s = String::new();
        cs.write_dir_to(arena, &mut s);
        assert_eq!(s, "");
        cs.write_filename_to(arena, &mut s);
        assert_eq!(s, "Cargo.toml");
        let mut buf = [0u8; 512];
        assert_eq!(cs.read_to_buf(arena, &mut buf), "Cargo.toml");
    }

    #[test]
    fn test_chunked_string_resolve_ptrs() {
        // Sized from the constant so it always spans several chunks.
        let long = format!("a/{}", "bcdefgh/".repeat(SIMD_CHUNK_BYTES));
        let (store, strings, _files) = build_test_store(&[long.as_str()]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];
        let total = cs.byte_len as usize;

        let mut ptrs = [std::ptr::null::<u8>(); MAX_PATH_CHUNKS];
        let resolved = cs.resolve_ptrs(arena, &mut ptrs);
        assert_eq!(resolved.len(), total.div_ceil(SIMD_CHUNK_BYTES));
        assert!(
            resolved.len() > 1,
            "fixture must span more than one chunk at SIMD_CHUNK_BYTES={SIMD_CHUNK_BYTES}"
        );

        // Verify we can read back the bytes
        let mut reconstructed = Vec::new();
        for (i, &ptr) in resolved.iter().enumerate() {
            let start = i * SIMD_CHUNK_BYTES;
            let take = SIMD_CHUNK_BYTES.min(total - start);
            reconstructed.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, take) });
        }
        assert_eq!(std::str::from_utf8(&reconstructed).unwrap(), long);
    }

    #[test]
    fn test_resolve_ptrs_path_exceeding_512_bytes() {
        // Regression: a fixed 32-ptr buffer covered only 512 bytes while
        // PATH_BUF_SIZE (libc::PATH_MAX) allows longer paths, panicking with
        // "index out of bounds: the len is 32 but the index is 32"
        let mut path = String::new();
        while path.len() < 600 {
            path.push_str("deeply_nested_directory_segment/");
        }
        path.push_str("needle_file.rs");
        assert!(path.len() > 512 && path.len() < PATH_BUF_SIZE);

        let (store, strings, _files) = build_test_store(&[path.as_str()]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];
        assert!(
            cs.chunk_count() > 1,
            "path must span more than one chunk at SIMD_CHUNK_BYTES={SIMD_CHUNK_BYTES}"
        );
        assert!(
            cs.chunk_count() <= MAX_PATH_CHUNKS,
            "PATH_MAX-legal path must fit the resolve buffer"
        );

        let mut ptrs = [std::ptr::null::<u8>(); MAX_PATH_CHUNKS];
        let resolved = cs.resolve_ptrs(arena, &mut ptrs);

        // Truncation is not acceptable either: it silently drops the tail of
        // the path (including the filename here) from fuzzy matching.
        assert_eq!(
            resolved.len(),
            cs.chunk_count(),
            "resolve_ptrs must resolve every chunk of a PATH_MAX-legal path"
        );

        let total = cs.byte_len as usize;
        let mut reconstructed = Vec::with_capacity(total);
        for (i, &ptr) in resolved.iter().enumerate() {
            let take = SIMD_CHUNK_BYTES.min(total - i * SIMD_CHUNK_BYTES);
            reconstructed.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, take) });
        }
        assert_eq!(std::str::from_utf8(&reconstructed).unwrap(), path);
    }

    #[test]
    fn test_relative_path_eq_path_exceeding_512_bytes() {
        let mut path = String::new();
        while path.len() < 600 {
            path.push_str("deeply_nested_directory_segment/");
        }
        path.push_str("needle_file.rs");
        assert!(path.len() > 512 && path.len() < PATH_BUF_SIZE);

        let (store, _strings, files) = build_test_store(&[path.as_str(), "src/lib.rs"]);
        let arena = store.as_arena_ptr();

        assert!(
            files[0].relative_path_eq(arena, &path),
            "a PATH_MAX-legal path must compare equal to itself"
        );

        // Short paths keep comparing exactly as before.
        assert!(files[1].relative_path_eq(arena, "src/lib.rs"));
        assert!(!files[1].relative_path_eq(arena, "src/main.rs"));
        assert!(!files[0].relative_path_eq(arena, &path[..path.len() - 1]));
    }

    #[test]
    fn test_filename_cow_mid_chunk() {
        let (store, strings, _files) = build_test_store(&["src/components/Button.tsx"]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];

        assert_eq!(cs.filename_offset, 15);
        assert_eq!(cs.byte_len, 25);

        let fname = cs.filename_cow(arena);
        assert_eq!(&*fname, "Button.tsx");
    }

    #[test]
    fn test_filename_cow_chunk_aligned() {
        let path = "0123456789abcdef/file.txt";
        let (store, strings, _files) = build_test_store(&[path]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];

        assert_eq!(cs.filename_offset, 17);
        let fname = cs.filename_cow(arena);
        assert_eq!(&*fname, "file.txt");
    }

    #[test]
    fn test_filename_cow_root_file() {
        let (store, strings, _files) = build_test_store(&["Cargo.toml"]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];

        assert_eq!(cs.filename_offset, 0);
        let fname = cs.filename_cow(arena);
        assert_eq!(&*fname, "Cargo.toml");
    }

    #[test]
    fn test_chunked_string_long_path() {
        let path = "very/deeply/nested/directory/structure/with/many/levels/file.txt";
        let (store, strings, _files) = build_test_store(&[path]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];

        let mut buf = [0u8; 512];
        assert_eq!(cs.read_to_buf(arena, &mut buf), path);
        assert_eq!(cs.chunk_count(), path.len().div_ceil(SIMD_CHUNK_BYTES));
    }

    #[test]
    fn test_chunked_string_clone() {
        let (store, strings, _files) = build_test_store(&["src/main.rs"]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];
        let cs2 = cs.clone();

        let mut buf1 = [0u8; 512];
        let mut buf2 = [0u8; 512];
        assert_eq!(
            cs.read_to_buf(arena, &mut buf1),
            cs2.read_to_buf(arena, &mut buf2)
        );
    }

    #[test]
    fn long_path_keeps_exact_byte_len() {
        // 5000 B exceeds PATH_BUF_SIZE but not u16::MAX: byte_len must be
        // exact and the full window readable (read_to_buf would truncate).
        let path = format!("d/{}/file.rs", "x".repeat(4990));
        assert!(path.len() > crate::constants::PATH_BUF_SIZE);
        let (store, strings, _) = build_test_store(&[path.as_str()]);
        let arena = store.as_arena_ptr();
        let cs = &strings[0];
        assert_eq!(cs.byte_len as usize, path.len());
        let mut buf = vec![0u8; path.len()];
        cs.copy_window_to(arena, 0, path.len() as u32, &mut buf);
        assert_eq!(buf, path.as_bytes());
    }

    #[test]
    fn test_chunked_string_full_path_roundtrip() {
        let paths = [
            "src/components/Button.tsx",
            "src/components/ui/DatePicker.tsx",
            "very/deeply/nested/directory/structure/file.txt",
            "Cargo.toml",
            "a.rs",
        ];
        let (store, strings, _files) = build_test_store(&paths);
        let arena = store.as_arena_ptr();

        for (i, expected) in paths.iter().enumerate() {
            let mut buf = [0u8; 512];
            let got = strings[i].read_to_buf(arena, &mut buf);
            assert_eq!(got, *expected, "full path roundtrip failed for file {i}");

            let mut ds = String::new();
            let mut fs = String::new();
            strings[i].write_dir_to(arena, &mut ds);
            strings[i].write_filename_to(arena, &mut fs);
            assert_eq!(
                format!("{ds}{fs}"),
                *expected,
                "dir+fname mismatch for file {i}"
            );
        }
    }
}
