use ahash::AHashMap;
use std::borrow::Cow;

use crate::simd_string_utils::case::eq_lowered_case;
use crate::simd_string_utils::memmem::PairPlan;

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

// One 32-byte chunk, both sides folded. Same bool as `eq_ignore_ascii_case`.
#[inline(always)]
fn eq_ignore_ascii_chunk(hay: &[u8; SIMD_CHUNK_BYTES], needle: &[u8]) -> bool {
    debug_assert!(needle.len() >= SIMD_CHUNK_BYTES);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return eq_ignore_ascii_chunk_sse(hay.as_ptr(), needle.as_ptr());
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        hay.eq_ignore_ascii_case(&needle[..SIMD_CHUNK_BYTES])
    }
}

// A word holds an ASCII `A..=Z` when this mask is nonzero.
#[inline(always)]
fn ascii_upper64(v: u64) -> u64 {
    let is_ascii = !v & 0x8080_8080_8080_8080;
    let masked = v & 0x7F7F_7F7F_7F7F_7F7F;
    let ge_a = masked.wrapping_add(0x3F3F_3F3F_3F3F_3F3F);
    let gt_z = masked.wrapping_add(0x2525_2525_2525_2525);
    ge_a & !gt_z & is_ascii
}

#[inline(always)]
fn ascii_upper32(v: u32) -> u32 {
    let is_ascii = !v & 0x8080_8080;
    let masked = v & 0x7F7F_7F7F;
    let ge_a = masked.wrapping_add(0x3F3F_3F3F);
    let gt_z = masked.wrapping_add(0x2525_2525);
    ge_a & !gt_z & is_ascii
}

// `A..=Z` becomes `a..=z`. Every other byte stays.
#[inline(always)]
fn fold_ascii64(v: u64) -> u64 {
    v | ((ascii_upper64(v) >> 2) & 0x2020_2020_2020_2020)
}

#[inline(always)]
fn fold_ascii32(v: u32) -> u32 {
    v | ((ascii_upper32(v) >> 2) & 0x2020_2020)
}

#[inline(always)]
fn u64_has_byte(v: u64, b: u8) -> bool {
    let x = v ^ (b as u64 * 0x0101_0101_0101_0101);
    (x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080) != 0
}

#[inline(always)]
fn u32_has_byte(v: u32, b: u8) -> bool {
    let x = v ^ (b as u32 * 0x0101_0101);
    (x.wrapping_sub(0x0101_0101) & !x & 0x8080_8080) != 0
}

// Reads only `bytes`. True when any byte is `/` or `\`.
#[inline(always)]
fn slice_has_sep(bytes: &[u8]) -> bool {
    let p = bytes.as_ptr();
    let n = bytes.len();
    let mut i = 0usize;
    while i + 16 <= n {
        let v0 = unsafe { core::ptr::read_unaligned(p.add(i) as *const u64) };
        let v1 = unsafe { core::ptr::read_unaligned(p.add(i + 8) as *const u64) };
        if u64_has_byte(v0, b'/')
            || u64_has_byte(v0, b'\\')
            || u64_has_byte(v1, b'/')
            || u64_has_byte(v1, b'\\')
        {
            return true;
        }
        i += 16;
    }
    while i + 8 <= n {
        let v = unsafe { core::ptr::read_unaligned(p.add(i) as *const u64) };
        if u64_has_byte(v, b'/') || u64_has_byte(v, b'\\') {
            return true;
        }
        i += 8;
    }
    let rest = n - i;
    if rest >= 4 {
        let v = unsafe { core::ptr::read_unaligned(p.add(i) as *const u32) };
        if u32_has_byte(v, b'/') || u32_has_byte(v, b'\\') {
            return true;
        }
        if rest > 4 {
            let at = n - 4;
            let v = unsafe { core::ptr::read_unaligned(p.add(at) as *const u32) };
            if u32_has_byte(v, b'/') || u32_has_byte(v, b'\\') {
                return true;
            }
        }
        return false;
    }
    while i < n {
        if is_sep(unsafe { *p.add(i) }) {
            return true;
        }
        i += 1;
    }
    false
}

// Reads only `needle`. True when any byte is ASCII `A..=Z`.
#[inline(always)]
pub(crate) fn needle_has_ascii_upper(needle: &[u8]) -> bool {
    let p = needle.as_ptr();
    let n = needle.len();
    let mut i = 0usize;
    while i + 16 <= n {
        let v0 = unsafe { core::ptr::read_unaligned(p.add(i) as *const u64) };
        let v1 = unsafe { core::ptr::read_unaligned(p.add(i + 8) as *const u64) };
        if ascii_upper64(v0) != 0 || ascii_upper64(v1) != 0 {
            return true;
        }
        i += 16;
    }
    while i + 8 <= n {
        let v = unsafe { core::ptr::read_unaligned(p.add(i) as *const u64) };
        if ascii_upper64(v) != 0 {
            return true;
        }
        i += 8;
    }
    let rest = n - i;
    if rest >= 4 {
        let v = unsafe { core::ptr::read_unaligned(p.add(i) as *const u32) };
        if ascii_upper32(v) != 0 {
            return true;
        }
        if rest > 4 {
            let at = n - 4;
            let v = unsafe { core::ptr::read_unaligned(p.add(at) as *const u32) };
            if ascii_upper32(v) != 0 {
                return true;
            }
        }
        return false;
    }
    while i < n {
        if unsafe { *p.add(i) }.is_ascii_uppercase() {
            return true;
        }
        i += 1;
    }
    false
}

// Matching bytes from the tail. `LOWER` means `want` is already ASCII-lowercase.
#[inline(always)]
fn ascii_tail_match<const LOWER: bool>(path: &[u8], want: &[u8]) -> usize {
    debug_assert_eq!(path.len(), want.len());
    let n = path.len();
    let mut end = n;
    let mut matched = 0usize;
    let hp = path.as_ptr();
    let wp = want.as_ptr();
    while end >= 16 {
        let at = end - 16;
        let hv0 = unsafe { core::ptr::read_unaligned(hp.add(at) as *const u64) };
        let hv1 = unsafe { core::ptr::read_unaligned(hp.add(at + 8) as *const u64) };
        let wv0 = unsafe { core::ptr::read_unaligned(wp.add(at) as *const u64) };
        let wv1 = unsafe { core::ptr::read_unaligned(wp.add(at + 8) as *const u64) };
        let diff1 = if LOWER {
            fold_ascii64(hv1) ^ wv1
        } else {
            fold_ascii64(hv1) ^ fold_ascii64(wv1)
        };
        if diff1 != 0 {
            return matched + trailing_zero_bytes(diff1);
        }
        let diff0 = if LOWER {
            fold_ascii64(hv0) ^ wv0
        } else {
            fold_ascii64(hv0) ^ fold_ascii64(wv0)
        };
        if diff0 != 0 {
            return matched + 8 + trailing_zero_bytes(diff0);
        }
        matched += 16;
        end = at;
    }
    while end >= 8 {
        let at = end - 8;
        let hv = unsafe { core::ptr::read_unaligned(hp.add(at) as *const u64) };
        let wv = unsafe { core::ptr::read_unaligned(wp.add(at) as *const u64) };
        let diff = if LOWER {
            fold_ascii64(hv) ^ wv
        } else {
            fold_ascii64(hv) ^ fold_ascii64(wv)
        };
        if diff == 0 {
            matched += 8;
            end = at;
            continue;
        }
        return matched + trailing_zero_bytes(diff);
    }
    if end >= 4 {
        let at = end - 4;
        let hv = unsafe { core::ptr::read_unaligned(hp.add(at) as *const u32) };
        let wv = unsafe { core::ptr::read_unaligned(wp.add(at) as *const u32) };
        let diff = if LOWER {
            fold_ascii32(hv) ^ wv
        } else {
            fold_ascii32(hv) ^ fold_ascii32(wv)
        };
        if diff == 0 {
            matched += 4;
            end = at;
        } else {
            return matched + trailing_zero_bytes32(diff);
        }
    }
    while end > 0 {
        end -= 1;
        let same = if LOWER {
            path[end].to_ascii_lowercase() == want[end]
        } else {
            path[end].eq_ignore_ascii_case(&want[end])
        };
        if !same {
            return matched;
        }
        matched += 1;
    }
    matched
}

// Little-endian: the last slice byte is the high byte, so leading zeros are the matched tail.
#[inline(always)]
fn trailing_zero_bytes(diff: u64) -> usize {
    (diff.leading_zeros() / 8) as usize
}

#[inline(always)]
fn trailing_zero_bytes32(diff: u32) -> usize {
    (diff.leading_zeros() / 8) as usize
}

// Writes `needle.len()` folded bytes. `out` is at least that long.
#[inline(always)]
pub(crate) fn fold_ascii_needle(needle: &[u8], out: &mut [u8]) {
    debug_assert!(out.len() >= needle.len());
    let p = needle.as_ptr();
    let d = out.as_mut_ptr();
    let n = needle.len();
    let mut i = 0usize;
    while i + 16 <= n {
        let v0 = unsafe { core::ptr::read_unaligned(p.add(i) as *const u64) };
        let v1 = unsafe { core::ptr::read_unaligned(p.add(i + 8) as *const u64) };
        unsafe {
            (d.add(i) as *mut u64).write_unaligned(fold_ascii64(v0));
            (d.add(i + 8) as *mut u64).write_unaligned(fold_ascii64(v1));
        }
        i += 16;
    }
    while i + 8 <= n {
        let v = unsafe { core::ptr::read_unaligned(p.add(i) as *const u64) };
        unsafe { (d.add(i) as *mut u64).write_unaligned(fold_ascii64(v)) };
        i += 8;
    }
    let rest = n - i;
    if rest >= 4 {
        let v = unsafe { core::ptr::read_unaligned(p.add(i) as *const u32) };
        unsafe { (d.add(i) as *mut u32).write_unaligned(fold_ascii32(v)) };
        if rest > 4 {
            let at = n - 4;
            let v = unsafe { core::ptr::read_unaligned(p.add(at) as *const u32) };
            unsafe { (d.add(at) as *mut u32).write_unaligned(fold_ascii32(v)) };
        }
        return;
    }
    while i < n {
        unsafe { *d.add(i) = (*p.add(i)).to_ascii_lowercase() };
        i += 1;
    }
}

// `LOWER` means `needle` is already ASCII-lowercase, so only the path byte folds.
#[inline(always)]
fn eq_piece<const LOWER: bool>(hay: &[u8], needle: &[u8]) -> bool {
    if LOWER {
        debug_assert_eq!(hay.len(), needle.len());
        return unsafe { eq_lowered_case(hay.as_ptr(), needle) };
    }
    if hay.len() == SIMD_CHUNK_BYTES {
        let chunk = unsafe { &*(hay.as_ptr() as *const [u8; SIMD_CHUNK_BYTES]) };
        return eq_ignore_ascii_chunk(chunk, needle);
    }
    eq_ignore_ascii_bytes(hay, needle)
}

// Equal lengths up to one slot. `A..=Z` folds on the needle, then the wide compare.
#[inline(always)]
pub(crate) fn eq_ignore_ascii_bytes(hay: &[u8], needle: &[u8]) -> bool {
    debug_assert_eq!(hay.len(), needle.len());
    if hay.len() < 4 || hay.len() > SIMD_CHUNK_BYTES {
        return hay.eq_ignore_ascii_case(needle);
    }
    // Already folded: the wide compare only changes A..=Z on the hay side.
    if !needle_has_ascii_upper(needle) {
        return unsafe { eq_lowered_case(hay.as_ptr(), needle) };
    }
    let mut lower = [0u8; SIMD_CHUNK_BYTES];
    fold_ascii_needle(needle, &mut lower);
    unsafe { eq_lowered_case(hay.as_ptr(), &lower[..hay.len()]) }
}

// Each 16-byte half is the same bool as `eq_ignore_ascii_case`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn eq_ignore_ascii_chunk_sse(hay: *const u8, needle: *const u8) -> bool {
    use std::arch::x86_64::*;
    unsafe {
        let add = _mm_set1_epi8((127 - b'Z') as i8);
        let thresh = _mm_set1_epi8(101);
        let bit = _mm_set1_epi8(0x20);
        let hv0 = _mm_loadu_si128(hay as *const __m128i);
        let nv0 = _mm_loadu_si128(needle as *const __m128i);
        let hv1 = _mm_loadu_si128(hay.add(16) as *const __m128i);
        let nv1 = _mm_loadu_si128(needle.add(16) as *const __m128i);
        let hu0 = _mm_cmpgt_epi8(_mm_add_epi8(hv0, add), thresh);
        let nu0 = _mm_cmpgt_epi8(_mm_add_epi8(nv0, add), thresh);
        let hu1 = _mm_cmpgt_epi8(_mm_add_epi8(hv1, add), thresh);
        let nu1 = _mm_cmpgt_epi8(_mm_add_epi8(nv1, add), thresh);
        let hf0 = _mm_or_si128(hv0, _mm_and_si128(hu0, bit));
        let nf0 = _mm_or_si128(nv0, _mm_and_si128(nu0, bit));
        let hf1 = _mm_or_si128(hv1, _mm_and_si128(hu1, bit));
        let nf1 = _mm_or_si128(nv1, _mm_and_si128(nu1, bit));
        let m0 = _mm_movemask_epi8(_mm_cmpeq_epi8(hf0, nf0));
        let m1 = _mm_movemask_epi8(_mm_cmpeq_epi8(hf1, nf1));
        (m0 & m1) == 0xFFFF
    }
}

// The next slot, fetched while the current one is walked. A prefetch does not fault.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_chunk(arena: ArenaPtr, idx: u32) {
    unsafe {
        std::arch::x86_64::_mm_prefetch(
            arena.chunk_ptr(idx) as *const i8,
            std::arch::x86_64::_MM_HINT_T0,
        );
    }
}

// Current slot index. When another slot follows, its index is the high half.
#[inline(always)]
fn load_slot_idx(index: *const u32, slot: usize, arena: ArenaPtr, more: bool) -> u32 {
    if more {
        let pair = unsafe { core::ptr::read_unaligned(index.add(slot) as *const u64) };
        let pair = u64::from_le(pair);
        #[cfg(target_arch = "x86_64")]
        prefetch_chunk(arena, (pair >> 32) as u32);
        #[cfg(not(target_arch = "x86_64"))]
        let _ = arena;
        pair as u32
    } else {
        let _ = arena;
        unsafe { *index.add(slot) }
    }
}

// Two adjacent slot indexes. The low half is `index`.
#[inline(always)]
pub(crate) fn load_two_slots(index: *const u32) -> (u32, u32) {
    let pair = unsafe { core::ptr::read_unaligned(index as *const u64) };
    let pair = u64::from_le(pair);
    (pair as u32, (pair >> 32) as u32)
}

impl std::fmt::Debug for ArenaPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "--arena-raw-pointer-0x({:?})", self.chunks)
    }
}

// 32B chunks stay inside one cache line; split 32B loads cost
// extra fill-buffer entries on misses.
#[repr(C, align(32))]
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
pub(crate) const MAX_FRIZBEE_CHUNKS: usize = PATH_BUF_SIZE.div_ceil(neo_frizbee::SIMD_CHUNK_BYTES);

/// A path stored as a run of chunk indices in the owning store's flat index
/// table. 8 bytes; every read needs the store's `ArenaPtr`.
#[repr(C)]
#[derive(Clone)]
pub(crate) struct ChunkedString {
    index_offset: u32,
    pub byte_len: u16,
    /// Byte offset where the filename begins. 0 for root-level files.
    pub filename_offset: u16,
}

impl ChunkedString {
    // One load of the path header. Low word is the index, then length, then the filename.
    #[inline(always)]
    fn header(&self) -> u64 {
        const _: () = assert!(core::mem::size_of::<ChunkedString>() == 8);
        unsafe { core::ptr::read_unaligned(self as *const Self as *const u64) }
    }

    // Stable ordering key for tie-breaks. Paths are stored in scan order, so
    // this is unique per path and orders deterministically.
    #[inline(always)]
    pub(crate) fn rank_key(&self) -> u64 {
        let h = self.header();
        let index_offset = h as u32 as u64;
        let byte_len = (h >> 32) as u16 as u64;
        debug_assert_eq!(index_offset, self.index_offset as u64);
        debug_assert_eq!(byte_len, self.byte_len as u64);
        byte_len << 32 | index_offset
    }

    /// Path length and filename start. Both live in the header word after the index.
    #[inline(always)]
    pub(crate) fn span_lens(&self) -> (u16, u16) {
        let h = self.header();
        let byte_len = (h >> 32) as u16;
        let filename_offset = (h >> 48) as u16;
        debug_assert_eq!(byte_len, self.byte_len);
        debug_assert_eq!(filename_offset, self.filename_offset);
        (byte_len, filename_offset)
    }

    // Index, length, and filename start from the one header word.
    #[inline(always)]
    pub(crate) fn header_parts(&self) -> (u32, u16, u16) {
        let h = self.header();
        let index_offset = h as u32;
        let byte_len = (h >> 32) as u16;
        let filename_offset = (h >> 48) as u16;
        debug_assert_eq!(index_offset, self.index_offset);
        debug_assert_eq!(byte_len, self.byte_len);
        debug_assert_eq!(filename_offset, self.filename_offset);
        (index_offset, byte_len, filename_offset)
    }

    // First slot pointer for an index already read from the header.
    #[inline(always)]
    pub(crate) fn slot_base(arena: ArenaPtr, index_offset: u32) -> *const u32 {
        unsafe { arena.indices.add(index_offset as usize) }
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
        if other.is_empty() {
            return true;
        }
        let bytes = other.as_bytes();
        let index = self.slot_index(arena, 0);
        let mut off = 0usize;
        let mut slot = 0usize;
        while off + SIMD_CHUNK_BYTES * 2 <= bytes.len() {
            let (i0, i1) = unsafe { load_two_slots(index.add(slot)) };
            let (c0, c1) = (Self::chunk_at(arena, i0), Self::chunk_at(arena, i1));
            if c0 != &bytes[off..off + SIMD_CHUNK_BYTES]
                || c1 != &bytes[off + SIMD_CHUNK_BYTES..off + SIMD_CHUNK_BYTES * 2]
            {
                return false;
            }
            off += SIMD_CHUNK_BYTES * 2;
            slot += 2;
        }
        while off < bytes.len() {
            let n = (bytes.len() - off).min(SIMD_CHUNK_BYTES);
            let idx = unsafe { *index.add(slot) };
            let actual = unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), n) };
            if actual != &bytes[off..off + n] {
                return false;
            }
            off += n;
            slot += 1;
        }
        true
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

    // Pointers at every `unit` inside each slot, in path order. Stops at `buf.len()`.
    #[inline]
    pub(crate) fn fill_unit_ptrs(&self, arena: ArenaPtr, unit: usize, buf: &mut [*const u8]) -> usize {
        let total = self.byte_len as usize;
        if total == 0 || unit == 0 || buf.is_empty() {
            return 0;
        }
        let n = total.div_ceil(unit).min(buf.len());
        let index = self.slot_index(arena, 0);
        let mut filled = 0usize;
        let mut off = 0usize;
        let mut slot = 0usize;
        // Frizbee reads 16-byte pieces. A 32-byte slot holds two of them.
        if unit == 16 {
            while filled < n && off < total {
                let take = SIMD_CHUNK_BYTES.min(total - off);
                let idx = load_slot_idx(index, slot, arena, off + take < total);
                let base = arena.chunk_ptr(idx);
                buf[filled] = base;
                filled += 1;
                if take > 16 && filled < n {
                    buf[filled] = unsafe { base.add(16) };
                    filled += 1;
                }
                off += take;
                slot += 1;
            }
            return filled;
        }
        while filled < n && off < total {
            let take = SIMD_CHUNK_BYTES.min(total - off);
            let idx = load_slot_idx(index, slot, arena, off + take < total);
            let base = arena.chunk_ptr(idx);
            let per_slot = SIMD_CHUNK_BYTES / unit;
            let need = take.div_ceil(unit).min(per_slot);
            for k in 0..need {
                if filled >= n {
                    break;
                }
                buf[filled] = unsafe { base.add(k * unit) };
                filled += 1;
            }
            off += take;
            slot += 1;
        }
        filled
    }

    // Chunks that miss `[lo, hi)` are not loaded. `f` returns false to stop.
    // An overlapping chunk is the same 32B entry a full visit would hand over.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn for_each_chunk32_span(
        &self,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        mut f: impl FnMut(usize, usize, &[u8; SIMD_CHUNK_BYTES]) -> bool,
    ) {
        let total = self.byte_len as usize;
        let hi = hi.min(total);
        if lo >= hi {
            return;
        }
        let indices = self.indices(arena);
        let start_chunk = lo / SIMD_CHUNK_BYTES;
        if start_chunk >= indices.len() {
            return;
        }
        let mut off = start_chunk * SIMD_CHUNK_BYTES;
        let slots = &indices[start_chunk..];
        for (i, &idx) in slots.iter().enumerate() {
            if off >= hi {
                break;
            }
            let take = SIMD_CHUNK_BYTES.min(total - off);
            // The next slot is fetched only when this span still reads it.
            #[cfg(target_arch = "x86_64")]
            if off + take < hi {
                if let Some(&next) = slots.get(i + 1) {
                    prefetch_chunk(arena, next);
                }
            }
            // SAFETY: arena entries are alive and immutable for the search.
            let chunk = unsafe { &*(arena.chunk_ptr(idx) as *const [u8; SIMD_CHUNK_BYTES]) };
            if !f(off, take, chunk) {
                break;
            }
            off += take;
        }
    }

    // Same span as `for_each_chunk32_span`; the slice is the valid `take` bytes.
    #[inline]
    pub(crate) fn for_each_chunk_span(
        &self,
        arena: ArenaPtr,
        lo: usize,
        hi: usize,
        mut f: impl FnMut(usize, &[u8]) -> bool,
    ) {
        let total = self.byte_len as usize;
        let hi = hi.min(total);
        if lo >= hi {
            return;
        }
        let start_chunk = lo / SIMD_CHUNK_BYTES;
        let index = self.slot_index(arena, start_chunk);
        let mut off = start_chunk * SIMD_CHUNK_BYTES;
        let mut slot = 0usize;
        while off < hi {
            let take = SIMD_CHUNK_BYTES.min(total - off);
            let idx = load_slot_idx(index, slot, arena, off + take < hi);
            let chunk = unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), take) };
            if !f(off, chunk) {
                break;
            }
            off += take;
            slot += 1;
        }
    }

    #[inline]
    pub(crate) fn common_suffix_len_ignore_ascii_case(
        &self,
        arena: ArenaPtr,
        other: &[u8],
    ) -> usize {
        self.suffix_len::<false>(arena, other)
    }

    // `other` is already ASCII-lowercase. The path side still folds.
    #[inline]
    pub(crate) fn common_suffix_len_lower(&self, arena: ArenaPtr, other: &[u8]) -> usize {
        self.suffix_len::<true>(arena, other)
    }

    // ASCII case-insensitive shared suffix length.
    // A mismatch returns the length matched from the tail.
    #[inline]
    fn suffix_len<const LOWER: bool>(&self, arena: ArenaPtr, other: &[u8]) -> usize {
        if other.is_empty() || self.byte_len == 0 {
            return 0;
        }
        let total = self.byte_len as usize;
        let mut slot = (total - 1) / SIMD_CHUNK_BYTES;
        let mut index = self.slot_index(arena, slot);
        let mut matched = 0usize;
        let mut avail = total - slot * SIMD_CHUNK_BYTES;
        loop {
            let idx = unsafe { *index };
            let chunk = unsafe { &*(arena.chunk_ptr(idx) as *const [u8; SIMD_CHUNK_BYTES]) };
            let need = other.len() - matched;
            let n = avail.min(need);
            let path = &chunk[avail - n..avail];
            let want = &other[need - n..need];
            // One pass from the tail. A mismatch stops without a second compare.
            let k = ascii_tail_match::<LOWER>(path, want);
            if k < n {
                return matched + k;
            }
            matched += n;
            if matched >= other.len() || slot == 0 {
                return matched;
            }
            slot -= 1;
            index = unsafe { index.sub(1) };
            avail = SIMD_CHUNK_BYTES;
        }
    }

    /// True when the filename ends in `.extension`, ASCII case-insensitive.
    /// The dot must sit inside the filename, so `.rs` and `file.rsx` miss.
    #[inline]
    pub(crate) fn has_extension(&self, arena: ArenaPtr, extension: &str) -> bool {
        let ext = extension.as_bytes();
        let total = self.byte_len as usize;
        let name_len = self.byte_len.wrapping_sub(self.filename_offset) as usize;
        let need = ext.len() + 1;
        if name_len <= need {
            return false;
        }
        let (chunk, within) = self.chunk_slot(arena, total - 1);
        if within + 1 >= need {
            let start = within + 1 - need;
            let end = &chunk[start..=within];
            return end[0] == b'.' && eq_ignore_ascii_bytes(&end[1..], ext);
        }
        // The slot above is the tail. Keep it; only a short extension crosses it.
        let mut index = unsafe {
            arena
                .indices
                .add(self.index_offset as usize + (total - 1) / SIMD_CHUNK_BYTES)
        };
        let mut chunk = chunk;
        let mut slot_last = within;
        let mut end_at = total;
        let mut matched = 0usize;
        loop {
            let n = slot_last + 1;
            let remain = ext.len() - matched;
            let take = n.min(remain);
            if take >= 4 {
                let end = ext.len() - matched;
                if !eq_ignore_ascii_bytes(&chunk[n - take..n], &ext[end - take..end]) {
                    return false;
                }
                matched += take;
                if matched == ext.len() {
                    if take == n {
                        if end_at <= n {
                            return false;
                        }
                        let prev = unsafe {
                            &*(arena.chunk_ptr(*index.sub(1)) as *const [u8; SIMD_CHUNK_BYTES])
                        };
                        return prev[SIMD_CHUNK_BYTES - 1] == b'.';
                    }
                    return chunk[n - take - 1] == b'.';
                }
            } else {
                for i in (0..n).rev() {
                    let a = chunk[i];
                    if matched < ext.len() {
                        if !a.eq_ignore_ascii_case(&ext[ext.len() - 1 - matched]) {
                            return false;
                        }
                        matched += 1;
                    } else {
                        return a == b'.';
                    }
                }
            }
            if end_at <= n {
                return false;
            }
            end_at -= n;
            index = unsafe { index.sub(1) };
            chunk = unsafe { &*(arena.chunk_ptr(*index) as *const [u8; SIMD_CHUNK_BYTES]) };
            slot_last = SIMD_CHUNK_BYTES - 1;
        }
    }
    // Filename `&str`. `off` and `total` are this path's filename offset and byte length.
    // One chunk is borrowed; a crossing name is copied into `scratch`.
    #[inline]
    pub(crate) fn filename_at<'a>(
        &self,
        arena: ArenaPtr,
        scratch: &'a mut [u8],
        off: usize,
        total: usize,
    ) -> &'a str {
        if off >= total {
            return "";
        }
        let len = total - off;
        let offset_in_chunk = off % SIMD_CHUNK_BYTES;

        if offset_in_chunk + len <= SIMD_CHUNK_BYTES {
            let chunk = self.chunk_for_slot(arena, off / SIMD_CHUNK_BYTES);
            let slice = &chunk[offset_in_chunk..offset_in_chunk + len];
            return unsafe { core::str::from_utf8_unchecked(slice) };
        }

        let n = len.min(scratch.len());
        if n == 0 {
            return "";
        }
        // The one-chunk borrow already missed, so the copy starts at the split.
        let room = SIMD_CHUNK_BYTES - offset_in_chunk;
        if offset_in_chunk + n <= SIMD_CHUNK_BYTES * 2 {
            let first = room.min(n);
            let index = unsafe {
                arena
                    .indices
                    .add(self.index_offset as usize + off / SIMD_CHUNK_BYTES)
            };
            let (i0, i1) = load_two_slots(index);
            let chunk = unsafe {
                &*(arena.chunk_ptr(i0) as *const [u8; SIMD_CHUNK_BYTES])
            };
            unsafe {
                Self::copy_slot_bytes(
                    chunk.as_ptr().add(offset_in_chunk),
                    scratch.as_mut_ptr(),
                    first,
                );
            }
            if n > first {
                // `first` ends on the slot boundary, so the next slot starts at byte 0.
                let next = unsafe {
                    &*(arena.chunk_ptr(i1) as *const [u8; SIMD_CHUNK_BYTES])
                };
                unsafe {
                    Self::copy_slot_bytes(
                        next.as_ptr(),
                        scratch.as_mut_ptr().add(first),
                        n - first,
                    );
                }
            }
        } else {
            self.copy_span(arena, off as u32, (off + n) as u32, scratch);
        }
        unsafe { core::str::from_utf8_unchecked(&scratch[..n]) }
    }

    /// Filename `&str` when those bytes sit in one arena chunk.
    /// `None` when the name crosses a chunk; the caller copies that name.
    #[inline]
    pub(crate) fn filename_in_chunk_at<'a>(
        &self,
        arena: ArenaPtr,
        off: usize,
        total: usize,
    ) -> Option<&'a str> {
        if off >= total {
            return Some("");
        }
        let len = total - off;
        let offset_in_chunk = off % SIMD_CHUNK_BYTES;
        if offset_in_chunk + len > SIMD_CHUNK_BYTES {
            return None;
        }
        let chunk = self.chunk_for_slot(arena, off / SIMD_CHUNK_BYTES);
        let slice = &chunk[offset_in_chunk..offset_in_chunk + len];
        Some(unsafe { core::str::from_utf8_unchecked(slice) })
    }

    // Exact window compare. Skips the scratch copy when the caller only needs equality.
    #[inline]
    pub(crate) fn window_eq_ignore_ascii(&self, arena: ArenaPtr, off: u32, needle: &[u8]) -> bool {
        self.window_eq::<false>(arena, off, needle)
    }

    // `needle` is already ASCII-lowercase. The path side still folds.
    #[inline]
    pub(crate) fn window_eq_lower(&self, arena: ArenaPtr, off: u32, needle: &[u8]) -> bool {
        self.window_eq::<true>(arena, off, needle)
    }

    // Caller proved `off + needle.len()` is the path length.
    #[inline]
    pub(crate) fn window_eq_ignore_ascii_ready(
        &self,
        arena: ArenaPtr,
        off: u32,
        needle: &[u8],
    ) -> bool {
        self.window_eq_ready::<false>(arena, off, needle)
    }

    // Caller proved `off + needle.len()` is the path length. `needle` is lowercase.
    #[inline]
    pub(crate) fn window_eq_lower_ready(&self, arena: ArenaPtr, off: u32, needle: &[u8]) -> bool {
        self.window_eq_ready::<true>(arena, off, needle)
    }

    #[inline(always)]
    fn window_eq<const LOWER: bool>(&self, arena: ArenaPtr, off: u32, needle: &[u8]) -> bool {
        let total = self.byte_len as usize;
        let pos = off as usize;
        if pos > total || total - pos != needle.len() {
            return false;
        }
        self.window_eq_ready::<LOWER>(arena, off, needle)
    }

    #[inline(always)]
    fn window_eq_ready<const LOWER: bool>(&self, arena: ArenaPtr, off: u32, needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        let pos = off as usize;
        let within = pos % SIMD_CHUNK_BYTES;
        if within + needle.len() <= SIMD_CHUNK_BYTES {
            let chunk = self.chunk_for_slot(arena, pos / SIMD_CHUNK_BYTES);
            return eq_piece::<LOWER>(&chunk[within..within + needle.len()], needle);
        }
        // The name crosses one slot boundary. Both slots are inside the path.
        if within + needle.len() <= SIMD_CHUNK_BYTES * 2 {
            let first = SIMD_CHUNK_BYTES - within;
            let index = self.slot_index(arena, pos / SIMD_CHUNK_BYTES);
            let (i0, i1) = load_two_slots(index);
            let (chunk, next) = (Self::chunk_at(arena, i0), Self::chunk_at(arena, i1));
            let head = &needle[..first];
            let tail = &needle[first..];
            let head_ok = if first == SIMD_CHUNK_BYTES {
                eq_piece::<LOWER>(chunk, head)
            } else {
                eq_piece::<LOWER>(&chunk[within..], head)
            };
            return head_ok
                && if tail.len() == SIMD_CHUNK_BYTES {
                    eq_piece::<LOWER>(next, tail)
                } else {
                    eq_piece::<LOWER>(&next[..tail.len()], tail)
                };
        }
        self.eq_indexed_span::<true>(arena, pos, pos + needle.len(), needle)
    }

    // True when the path begins with `prefix`. Compared in the slots it occupies.
    #[inline]
    pub(crate) fn starts_with_bytes(&self, arena: ArenaPtr, prefix: &[u8]) -> bool {
        let total = self.byte_len as usize;
        if prefix.len() > total {
            return false;
        }
        if prefix.is_empty() {
            return true;
        }
        if prefix.len() <= SIMD_CHUNK_BYTES {
            let (chunk, _) = self.chunk_slot(arena, 0);
            return &chunk[..prefix.len()] == prefix;
        }
        // The prefix fits in the path, so these slots exist.
        let index = self.slot_index(arena, 0);
        if prefix.len() <= SIMD_CHUNK_BYTES * 2 {
            let rest = prefix.len() - SIMD_CHUNK_BYTES;
            let (i0, i1) = load_two_slots(index);
            let (c0, c1) = (Self::chunk_at(arena, i0), Self::chunk_at(arena, i1));
            return &c0[..] == &prefix[..SIMD_CHUNK_BYTES]
                && &c1[..rest] == &prefix[SIMD_CHUNK_BYTES..];
        }
        let mut pos = 0usize;
        let mut slot = 0usize;
        while pos + SIMD_CHUNK_BYTES * 2 <= prefix.len() {
            let (i0, i1) = unsafe { load_two_slots(index.add(slot)) };
            let (c0, c1) = (Self::chunk_at(arena, i0), Self::chunk_at(arena, i1));
            if c0 != &prefix[pos..pos + SIMD_CHUNK_BYTES]
                || c1 != &prefix[pos + SIMD_CHUNK_BYTES..pos + SIMD_CHUNK_BYTES * 2]
            {
                return false;
            }
            pos += SIMD_CHUNK_BYTES * 2;
            slot += 2;
        }
        while pos < prefix.len() {
            let chunk = unsafe { Self::chunk_at(arena, *index.add(slot)) };
            let take = SIMD_CHUNK_BYTES.min(prefix.len() - pos);
            if &chunk[..take] != &prefix[pos..pos + take] {
                return false;
            }
            pos += take;
            slot += 1;
        }
        true
    }

    // `full` ends with this path's bytes. Same as `full.ends_with(path)`.
    // The last byte is compared first.
    #[inline]
    pub(crate) fn is_suffix_of(&self, arena: ArenaPtr, full: &[u8]) -> bool {
        let n = self.byte_len as usize;
        if full.len() < n {
            return false;
        }
        if n == 0 {
            return true;
        }
        let tail = &full[full.len() - n..];
        // The last slot is compared first. Each earlier slot is the previous entry.
        let slot = (n - 1) / SIMD_CHUNK_BYTES;
        let mut index = self.slot_index(arena, slot);
        let mut end = n;
        let mut avail = n - slot * SIMD_CHUNK_BYTES;
        loop {
            let idx = unsafe { *index };
            let chunk = unsafe { &*(arena.chunk_ptr(idx) as *const [u8; SIMD_CHUNK_BYTES]) };
            let start = end - avail;
            let path = &chunk[..avail];
            let want = &tail[start..end];
            // The filename sits at the end of the last slot. A mismatch stops there.
            if avail >= 16 {
                let at = avail - 16;
                let hv0 = unsafe { core::ptr::read_unaligned(path.as_ptr().add(at) as *const u64) };
                let hv1 = unsafe { core::ptr::read_unaligned(path.as_ptr().add(at + 8) as *const u64) };
                let wv0 = unsafe { core::ptr::read_unaligned(want.as_ptr().add(at) as *const u64) };
                let wv1 = unsafe { core::ptr::read_unaligned(want.as_ptr().add(at + 8) as *const u64) };
                if hv0 != wv0 || hv1 != wv1 || (at > 0 && &path[..at] != &want[..at]) {
                    return false;
                }
            } else if avail >= 8 {
                let at = avail - 8;
                let hv = unsafe { core::ptr::read_unaligned(path.as_ptr().add(at) as *const u64) };
                let wv = unsafe { core::ptr::read_unaligned(want.as_ptr().add(at) as *const u64) };
                if hv != wv || (at > 0 && &path[..at] != &want[..at]) {
                    return false;
                }
            } else if path != want {
                return false;
            }
            if start == 0 {
                return true;
            }
            end = start;
            index = unsafe { index.sub(1) };
            avail = SIMD_CHUNK_BYTES;
        }
    }

    // Any `needle` byte in the path. Stops at the first hit.
    #[inline]
    pub(crate) fn contains_byte(&self, arena: ArenaPtr, needle: u8) -> bool {
        self.contains_byte_from(arena, 0, needle)
    }

    // Any `needle` byte at or after `start`. Stops at the first hit.
    #[inline]
    fn contains_byte_from(&self, arena: ArenaPtr, start: usize, needle: u8) -> bool {
        let total = self.byte_len as usize;
        if start >= total {
            return false;
        }
        let mut pos = (start / SIMD_CHUNK_BYTES) * SIMD_CHUNK_BYTES;
        let mut index = unsafe {
            arena
                .indices
                .add(self.index_offset as usize + start / SIMD_CHUNK_BYTES)
        };
        while pos < total {
            let chunk = unsafe {
                &*(arena.chunk_ptr(*index) as *const [u8; SIMD_CHUNK_BYTES])
            };
            let from = start.max(pos) - pos;
            let end = SIMD_CHUNK_BYTES.min(total - pos);
            if memchr::memchr(needle, &chunk[from..end]).is_some() {
                return true;
            }
            pos += SIMD_CHUNK_BYTES;
            if pos < total {
                index = unsafe { index.add(1) };
            }
        }
        false
    }

    // Parent directory before the last `sep` equals `dir`, ASCII case-insensitive.
    #[inline]
    pub(crate) fn parent_eq_ignore_ascii_case(
        &self,
        arena: ArenaPtr,
        sep: u8,
        dir: &[u8],
    ) -> bool {
        if dir.is_empty() {
            return false;
        }
        let total = self.byte_len as usize;
        let mut end = total;
        let mut split = None;
        while end > 0 {
            let (chunk, within) = self.chunk_slot(arena, end - 1);
            let len = within + 1;
            let start = end - len;
            if let Some(rel) = memchr::memrchr(sep, &chunk[..len]) {
                split = Some(start + rel);
                break;
            }
            end = start;
        }
        let Some(pos) = split else {
            return false;
        };
        pos == dir.len() && self.prefix_eq_ignore_ascii_case(arena, pos, dir)
    }

    // `path[..pos]` equals `dir`, ASCII case-insensitive. `pos` is `dir.len()`.
    #[inline]
    fn prefix_eq<const LOWER: bool>(&self, arena: ArenaPtr, pos: usize, dir: &[u8]) -> bool {
        if pos == 0 {
            return dir.is_empty();
        }
        // The path starts at the beginning of its first slot.
        if pos <= SIMD_CHUNK_BYTES {
            let (chunk, at) = self.chunk_slot(arena, 0);
            if pos == SIMD_CHUNK_BYTES {
                return eq_piece::<LOWER>(chunk, dir);
            }
            return eq_piece::<LOWER>(&chunk[at..at + pos], dir);
        }
        if pos <= SIMD_CHUNK_BYTES * 2 {
            let index = self.slot_index(arena, 0);
            let (i0, i1) = load_two_slots(index);
            let (first, second) = (Self::chunk_at(arena, i0), Self::chunk_at(arena, i1));
            if !eq_piece::<LOWER>(first, &dir[..SIMD_CHUNK_BYTES]) {
                return false;
            }
            let rest = pos - SIMD_CHUNK_BYTES;
            if rest == SIMD_CHUNK_BYTES {
                return eq_piece::<LOWER>(second, &dir[SIMD_CHUNK_BYTES..]);
            }
            return eq_piece::<LOWER>(&second[..rest], &dir[SIMD_CHUNK_BYTES..]);
        }
        let mut off = 0usize;
        let mut index = unsafe { arena.indices.add(self.index_offset as usize) };
        while off + SIMD_CHUNK_BYTES * 2 <= pos {
            let (i0, i1) = load_two_slots(index);
            let (c0, c1) = unsafe {
                (
                    &*(arena.chunk_ptr(i0) as *const [u8; SIMD_CHUNK_BYTES]),
                    &*(arena.chunk_ptr(i1) as *const [u8; SIMD_CHUNK_BYTES]),
                )
            };
            let ok0 = eq_piece::<LOWER>(c0, &dir[off..off + SIMD_CHUNK_BYTES]);
            let ok1 = eq_piece::<LOWER>(c1, &dir[off + SIMD_CHUNK_BYTES..off + SIMD_CHUNK_BYTES * 2]);
            if !ok0 || !ok1 {
                return false;
            }
            off += SIMD_CHUNK_BYTES * 2;
            index = unsafe { index.add(2) };
        }
        while off < pos {
            let chunk = unsafe {
                &*(arena.chunk_ptr(*index) as *const [u8; SIMD_CHUNK_BYTES])
            };
            let n = SIMD_CHUNK_BYTES.min(pos - off);
            if !eq_piece::<LOWER>(&chunk[..n], &dir[off..off + n]) {
                return false;
            }
            off += n;
            if off < pos {
                index = unsafe { index.add(1) };
            }
        }
        true
    }

    fn prefix_eq_ignore_ascii_case(&self, arena: ArenaPtr, pos: usize, dir: &[u8]) -> bool {
        self.prefix_eq::<false>(arena, pos, dir)
    }

    // Parent before the last `\` when the path has one, otherwise before the last `/`.
    #[inline]
    pub(crate) fn same_parent_dir(
        &self,
        arena: ArenaPtr,
        slash_dir: &[u8],
        back_dir: &[u8],
    ) -> bool {
        self.same_parent::<false>(arena, slash_dir, back_dir)
    }

    // Both directory strings are already ASCII-lowercase.
    #[inline]
    pub(crate) fn same_parent_dir_lower(
        &self,
        arena: ArenaPtr,
        slash_dir: &[u8],
        back_dir: &[u8],
    ) -> bool {
        self.same_parent::<true>(arena, slash_dir, back_dir)
    }

    fn same_parent<const LOWER: bool>(
        &self,
        arena: ArenaPtr,
        slash_dir: &[u8],
        back_dir: &[u8],
    ) -> bool {
        if slash_dir.is_empty() && back_dir.is_empty() {
            return false;
        }
        // The byte before the filename is the last separator. A different parent
        // length cannot match, so the path walk only runs when the length matches.
        let (byte_len_u, name_off) = self.span_lens();
        let total = byte_len_u as usize;
        let off = name_off as usize;
        if off > 0 && off <= total {
            let (chunk, at) = self.chunk_slot(arena, off - 1);
            match chunk[at] {
                b'\\' => {
                    let pos = off - 1;
                    return pos == back_dir.len()
                        && self.prefix_eq::<LOWER>(arena, pos, back_dir);
                }
                b'/' if back_dir.is_empty() => {
                    let pos = off - 1;
                    if pos != slash_dir.len()
                        || !self.prefix_eq::<LOWER>(arena, pos, slash_dir)
                    {
                        return false;
                    }
                    // A backslash in the filename still selects the backslash parent.
                    return !self.contains_byte_from(arena, off, b'\\');
                }
                _ => {}
            }
        }
        let mut end = total;
        let mut slash_at = None;
        while end > 0 {
            let (chunk, within) = self.chunk_slot(arena, end - 1);
            let len = within + 1;
            let start = end - len;
            let window = &chunk[..len];
            if let Some(rel) = memchr::memrchr(b'\\', window) {
                let pos = start + rel;
                return !back_dir.is_empty()
                    && pos == back_dir.len()
                    && self.prefix_eq::<LOWER>(arena, pos, back_dir);
            }
            if slash_at.is_none() {
                if let Some(rel) = memchr::memrchr(b'/', window) {
                    slash_at = Some(start + rel);
                }
            }
            end = start;
        }
        let Some(pos) = slash_at else {
            return false;
        };
        !slash_dir.is_empty()
            && pos == slash_dir.len()
            && self.prefix_eq::<LOWER>(arena, pos, slash_dir)
    }

    /// Return the filename portion as a `Cow<str>`.
    ///
    /// When the filename starts at a chunk boundary and fits in one chunk we
    /// borrow directly from the arena (zero-copy). Otherwise we allocate.
    /// Filenames are almost always <=16 bytes so the fast path dominates.
    #[inline]
    pub fn filename_cow<'a>(&self, arena: ArenaPtr) -> Cow<'a, str> {
        let (byte_len_u, name_off) = self.span_lens();
        let fname_offset = name_off as usize;
        let fname_len = byte_len_u as usize - fname_offset;
        if fname_len == 0 {
            return Cow::Borrowed("");
        }

        let start_chunk = fname_offset / SIMD_CHUNK_BYTES;
        let offset_in_chunk = fname_offset % SIMD_CHUNK_BYTES;

        // Borrow when the filename fits in the remainder of its first chunk,
        // including the common mid-chunk start after a directory prefix.
        if offset_in_chunk + fname_len <= SIMD_CHUNK_BYTES {
            let idx = unsafe { *self.slot_index(arena, start_chunk) };
            let ptr = unsafe { arena.chunk_ptr(idx).add(offset_in_chunk) };
            let slice = unsafe { core::slice::from_raw_parts(ptr, fname_len) };
            return Cow::Borrowed(unsafe { core::str::from_utf8_unchecked(slice) });
        }

        let mut out = String::with_capacity(fname_len);
        let end = fname_offset + fname_len;
        // The copy writes every new byte, then the length is published.
        unsafe {
            self.copy_trimmed_ptr(
                arena,
                fname_offset as u32,
                end as u32,
                out.as_mut_vec().as_mut_ptr(),
            );
            out.as_mut_vec().set_len(fname_len);
        }
        Cow::Owned(out)
    }

    /// Prefetch this path's arena chunks. The gate's chunk lookups are scattered
    /// across the dedup store, so issuing the next candidate's loads early
    /// hides part of the miss.
    #[inline]
    #[allow(dead_code)]
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

    // Byte at `index`. Caller guarantees `index` < byte_len.
    #[inline]
    pub(crate) fn byte_at(&self, arena: ArenaPtr, index: usize) -> u8 {
        let (chunk, within) = self.chunk_slot(arena, index);
        chunk[within]
    }

    // `[start, end)` against `expect`. One index base; each later slot is the next entry.
    // `ASCII` folds case. Returns at the first slot that differs.
    #[inline]
    fn eq_indexed_span<const ASCII: bool>(
        &self,
        arena: ArenaPtr,
        start: usize,
        end: usize,
        expect: &[u8],
    ) -> bool {
        let mut index = unsafe {
            arena
                .indices
                .add(self.index_offset as usize + start / SIMD_CHUNK_BYTES)
        };
        let mut at = start % SIMD_CHUNK_BYTES;
        let mut off = start;
        let mut npos = 0usize;
        while off < end {
            if at == 0 && off + SIMD_CHUNK_BYTES * 2 <= end {
                let (i0, i1) = load_two_slots(index);
                let (c0, c1) = unsafe {
                    (
                        &*(arena.chunk_ptr(i0) as *const [u8; SIMD_CHUNK_BYTES]),
                        &*(arena.chunk_ptr(i1) as *const [u8; SIMD_CHUNK_BYTES]),
                    )
                };
                if ASCII {
                    let ok0 = eq_ignore_ascii_chunk(c0, &expect[npos..npos + SIMD_CHUNK_BYTES]);
                    let ok1 = eq_ignore_ascii_chunk(
                        c1,
                        &expect[npos + SIMD_CHUNK_BYTES..npos + SIMD_CHUNK_BYTES * 2],
                    );
                    if !ok0 || !ok1 {
                        return false;
                    }
                } else if c0 != &expect[npos..npos + SIMD_CHUNK_BYTES]
                    || c1 != &expect[npos + SIMD_CHUNK_BYTES..npos + SIMD_CHUNK_BYTES * 2]
                {
                    return false;
                }
                off += SIMD_CHUNK_BYTES * 2;
                npos += SIMD_CHUNK_BYTES * 2;
                index = unsafe { index.add(2) };
                continue;
            }
            let idx = unsafe { *index };
            let chunk = unsafe { &*(arena.chunk_ptr(idx) as *const [u8; SIMD_CHUNK_BYTES]) };
            let n = (SIMD_CHUNK_BYTES - at).min(end - off);
            let got = &chunk[at..at + n];
            let want = &expect[npos..npos + n];
            if ASCII {
                if n == SIMD_CHUNK_BYTES {
                    if !eq_ignore_ascii_chunk(chunk, want) {
                        return false;
                    }
                } else if !eq_ignore_ascii_bytes(got, want) {
                    return false;
                }
            } else if got != want {
                return false;
            }
            off += n;
            npos += n;
            if off >= end {
                break;
            }
            at = 0;
            index = unsafe { index.add(1) };
        }
        true
    }

    // Exact `[start, end)`. `end` stays within the path.
    #[inline]
    pub(crate) fn eq_range(&self, arena: ArenaPtr, start: usize, end: usize, expect: &[u8]) -> bool {
        if end < start || end - start != expect.len() || end > self.byte_len as usize {
            return false;
        }
        let len = end - start;
        if len == 0 {
            return true;
        }
        let within = start % SIMD_CHUNK_BYTES;
        if within + len <= SIMD_CHUNK_BYTES {
            let chunk = self.chunk_for_slot(arena, start / SIMD_CHUNK_BYTES);
            return &chunk[within..within + len] == expect;
        }
        if within + len <= SIMD_CHUNK_BYTES * 2 {
            let first = SIMD_CHUNK_BYTES - within;
            let index = self.slot_index(arena, start / SIMD_CHUNK_BYTES);
            let (i0, i1) = load_two_slots(index);
            let (chunk, next) = (Self::chunk_at(arena, i0), Self::chunk_at(arena, i1));
            return &chunk[within..] == &expect[..first] && &next[..len - first] == &expect[first..];
        }
        self.eq_indexed_span::<false>(arena, start, end, expect)
    }

    // Bytes of `[start, end)` when they sit in one arena chunk.
    // One index load. `false` when the range crosses a chunk or leaves the path.
    #[inline]
    pub(crate) fn copy_in_chunk(
        &self,
        arena: ArenaPtr,
        start: u32,
        end: u32,
        out: &mut [u8],
    ) -> bool {
        let total = self.byte_len as usize;
        let start = start as usize;
        let end = end as usize;
        if end < start || end > total {
            return false;
        }
        let len = end - start;
        if len == 0 {
            return true;
        }
        if out.len() < len {
            return false;
        }
        let within = start % SIMD_CHUNK_BYTES;
        if within + len > SIMD_CHUNK_BYTES {
            return false;
        }
        let chunk = self.chunk_for_slot(arena, start / SIMD_CHUNK_BYTES);
        unsafe {
            Self::copy_slot_bytes(chunk.as_ptr().add(within), out.as_mut_ptr(), len);
        }
        true
    }

    // `[start, end)` in one or two chunks. `false` when the range needs a third chunk.
    #[inline]
    pub(crate) fn copy_short(
        &self,
        arena: ArenaPtr,
        start: u32,
        end: u32,
        out: &mut [u8],
    ) -> bool {
        let total = self.byte_len as usize;
        let start_us = start as usize;
        let end_us = end as usize;
        if end_us < start_us || end_us > total {
            return false;
        }
        let len = end_us - start_us;
        if out.len() < len {
            return false;
        }
        if self.copy_in_chunk(arena, start, end, out) {
            return true;
        }
        let within = start_us % SIMD_CHUNK_BYTES;
        if within + len > SIMD_CHUNK_BYTES * 2 {
            return false;
        }
        let first = SIMD_CHUNK_BYTES - within;
        let index = self.slot_index(arena, start_us / SIMD_CHUNK_BYTES);
        let (i0, i1) = load_two_slots(index);
        let (chunk, next) = (Self::chunk_at(arena, i0), Self::chunk_at(arena, i1));
        // The next slot starts at the boundary, so its offset is 0.
        unsafe {
            Self::copy_slot_bytes(chunk.as_ptr().add(within), out.as_mut_ptr(), first);
            Self::copy_slot_bytes(next.as_ptr(), out.as_mut_ptr().add(first), len - first);
        }
        true
    }

    // Head and tail moves of a fixed width. Lengths inside a slot stay in 1..=32.
    #[inline(always)]
    pub(crate) unsafe fn copy_slot_bytes(src: *const u8, dst: *mut u8, len: usize) {
        unsafe {
            if len >= 16 {
                let a = core::ptr::read_unaligned(src as *const u128);
                if len == 16 {
                    core::ptr::write_unaligned(dst as *mut u128, a);
                    return;
                }
                let b = core::ptr::read_unaligned(src.add(len - 16) as *const u128);
                core::ptr::write_unaligned(dst as *mut u128, a);
                core::ptr::write_unaligned(dst.add(len - 16) as *mut u128, b);
                return;
            }
            if len >= 8 {
                let a = core::ptr::read_unaligned(src as *const u64);
                if len == 8 {
                    core::ptr::write_unaligned(dst as *mut u64, a);
                    return;
                }
                let b = core::ptr::read_unaligned(src.add(len - 8) as *const u64);
                core::ptr::write_unaligned(dst as *mut u64, a);
                core::ptr::write_unaligned(dst.add(len - 8) as *mut u64, b);
                return;
            }
            if len >= 4 {
                let a = core::ptr::read_unaligned(src as *const u32);
                if len == 4 {
                    core::ptr::write_unaligned(dst as *mut u32, a);
                    return;
                }
                let b = core::ptr::read_unaligned(src.add(len - 4) as *const u32);
                core::ptr::write_unaligned(dst as *mut u32, a);
                core::ptr::write_unaligned(dst.add(len - 4) as *mut u32, b);
                return;
            }
            core::ptr::copy_nonoverlapping(src, dst, len);
        }
    }

    /// Copies `len` bytes at `start` when they sit in one slot.
    #[inline(always)]
    pub(crate) unsafe fn copy_in_slot(
        &self,
        arena: ArenaPtr,
        start: u32,
        len: usize,
        out: *mut u8,
    ) -> bool {
        if len == 0 {
            return true;
        }
        let start = start as usize;
        let within = start % SIMD_CHUNK_BYTES;
        if within + len > SIMD_CHUNK_BYTES {
            return false;
        }
        let idx = unsafe { *self.slot_index(arena, start / SIMD_CHUNK_BYTES) };
        unsafe {
            Self::copy_slot_bytes(Self::chunk_at(arena, idx).as_ptr().add(within), out, len);
        }
        true
    }

    // `[start, end)` is already inside the path and `out` holds it.
    #[inline]
    pub(crate) fn copy_trimmed(&self, arena: ArenaPtr, start: u32, end: u32, out: &mut [u8]) {
        let len = (end as usize).saturating_sub(start as usize);
        debug_assert!(out.len() >= len);
        unsafe { self.copy_trimmed_ptr(arena, start, end, out.as_mut_ptr()) }
    }

    /// Writes `end - start` bytes at `out`. Those bytes need not be initialized yet.
    pub(crate) unsafe fn copy_trimmed_ptr(
        &self,
        arena: ArenaPtr,
        start: u32,
        end: u32,
        out: *mut u8,
    ) {
        let start_u = start;
        let start = start as usize;
        let end = end as usize;
        let len = end - start;
        debug_assert!(end >= start && end <= self.byte_len as usize);
        if len == 0 || unsafe { self.copy_in_slot(arena, start_u, len, out) } {
            return;
        }
        unsafe { self.copy_crossing(arena, start, end, out) };
    }

    /// `[start, end)` crosses a slot. The caller already tried the one-slot copy.
    pub(crate) unsafe fn copy_crossing(
        &self,
        arena: ArenaPtr,
        start: usize,
        end: usize,
        out: *mut u8,
    ) {
        #[cfg(target_arch = "x86_64")]
        if crate::simd_string_utils::memmem::has_avx2() {
            unsafe { self.copy_window_avx2(arena, start, end, out) };
            return;
        }
        let len = end - start;
        let within = start % SIMD_CHUNK_BYTES;
        // One slot already returned. The next slots stay inside the path.
        if within + len <= SIMD_CHUNK_BYTES * 2 {
            let first = SIMD_CHUNK_BYTES - within;
            // The span stays inside the path, so the next slot exists.
            let index = unsafe {
                arena
                    .indices
                    .add(self.index_offset as usize + start / SIMD_CHUNK_BYTES)
            };
            // The next slot starts at the boundary, so its offset is 0.
            unsafe {
                let (i0, i1) = load_two_slots(index);
                #[cfg(target_arch = "x86_64")]
                prefetch_chunk(arena, i1);
                Self::copy_slot_bytes(arena.chunk_ptr(i0).add(within), out, first);
                Self::copy_slot_bytes(arena.chunk_ptr(i1), out.add(first), len - first);
            }
            return;
        }
        if within + len <= SIMD_CHUNK_BYTES * 3 {
            let index = unsafe {
                arena
                    .indices
                    .add(self.index_offset as usize + start / SIMD_CHUNK_BYTES)
            };
            let first = SIMD_CHUNK_BYTES - within;
            let tail = len - first - SIMD_CHUNK_BYTES;
            unsafe {
                let (i0, i1) = load_two_slots(index);
                let i2 = *index.add(2);
                #[cfg(target_arch = "x86_64")]
                {
                    prefetch_chunk(arena, i1);
                    prefetch_chunk(arena, i2);
                }
                Self::copy_slot_bytes(arena.chunk_ptr(i0).add(within), out, first);
                Self::copy_slot_bytes(arena.chunk_ptr(i1), out.add(first), SIMD_CHUNK_BYTES);
                Self::copy_slot_bytes(
                    arena.chunk_ptr(i2),
                    out.add(first + SIMD_CHUNK_BYTES),
                    tail,
                );
            }
            return;
        }
        unsafe { self.copy_span_ptr(arena, start as u32, end as u32, out) };
    }

    // `[start, end)` is already inside the path. `None` when it crosses a chunk.
    #[inline]
    pub(crate) fn trimmed_in_chunk<'a>(
        &self,
        arena: ArenaPtr,
        start: u32,
        end: u32,
    ) -> Option<&'a [u8]> {
        let start = start as usize;
        let end = end as usize;
        debug_assert!(end <= self.byte_len as usize);
        if start >= end {
            return None;
        }
        let len = end - start;
        let within = start % SIMD_CHUNK_BYTES;
        if within + len > SIMD_CHUNK_BYTES {
            return None;
        }
        let chunk = self.chunk_for_slot(arena, start / SIMD_CHUNK_BYTES);
        Some(&chunk[within..within + len])
    }

    // Bytes of `[start, end)` when they sit in one arena chunk.
    #[inline]
    pub(crate) fn window_in_chunk<'a>(
        &self,
        arena: ArenaPtr,
        start: u32,
        end: u32,
    ) -> Option<&'a [u8]> {
        let total = self.byte_len as usize;
        let start = start as usize;
        let end = end as usize;
        if start >= end || end > total {
            return None;
        }
        let len = end - start;
        let within = start % SIMD_CHUNK_BYTES;
        if within + len > SIMD_CHUNK_BYTES {
            return None;
        }
        // `end <= byte_len` keeps this slot inside the path's index.
        let chunk = self.chunk_for_slot(arena, start / SIMD_CHUNK_BYTES);
        Some(&chunk[within..within + len])
    }

    /// Copy `[start, end)` into `out`. One or two chunks load their slots;
    /// a longer span walks the index, and a short tail is zero-filled.
    pub fn copy_window_to(&self, arena: ArenaPtr, start: u32, end: u32, out: &mut [u8]) {
        let total = self.byte_len as usize;
        let start = (start as usize).min(total);
        let end = (end as usize).min(total);
        let len = end.saturating_sub(start);
        if len == 0 {
            return;
        }
        debug_assert!(out.len() >= len);
        if self.copy_short(arena, start as u32, end as u32, out) {
            return;
        }
        self.copy_span(arena, start as u32, end as u32, out);
    }

    // Three or more chunks. The caller already rejected a one- or two-chunk copy.
    #[inline]
    pub(crate) fn copy_span(&self, arena: ArenaPtr, start: u32, end: u32, out: &mut [u8]) {
        let len = (end as usize).saturating_sub(start as usize);
        debug_assert!(out.len() >= len);
        unsafe { self.copy_span_ptr(arena, start, end, out.as_mut_ptr()) }
    }

    /// Writes `end - start` bytes at `out`, zeroing any short tail. `out` may be uninitialized.
    // A full slot is one 32-byte move. A short piece stays on the 16-byte copy.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn copy_window_avx2(
        &self,
        arena: ArenaPtr,
        start: usize,
        end: usize,
        out: *mut u8,
    ) {
        use core::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256, __m256i};
        let len = end - start;
        if len == 0 {
            return;
        }
        let offset_in_chunk = start % SIMD_CHUNK_BYTES;
        let chunks_to_copy = chunks_needed(offset_in_chunk + len);
        let index = self.slot_index(arena, start / SIMD_CHUNK_BYTES);
        macro_rules! move_full {
            ($src:expr, $dst:expr) => {{
                let v = _mm256_loadu_si256($src as *const __m256i);
                _mm256_storeu_si256($dst as *mut __m256i, v);
            }};
        }
        macro_rules! move_n {
            ($src:expr, $dst:expr, $n:expr) => {
                if $n == SIMD_CHUNK_BYTES {
                    move_full!($src, $dst);
                } else if $n != 0 {
                    Self::copy_slot_bytes($src, $dst, $n);
                }
            };
        }
        unsafe {
            // One to three slots are the common path lengths. Longer paths use the loop.
            if chunks_to_copy == 1 {
                let idx = *index;
                move_n!(arena.chunk_ptr(idx).add(offset_in_chunk), out, len);
                return;
            }
            if chunks_to_copy == 2 {
                let (i0, i1) = load_two_slots(index);
                prefetch_chunk(arena, i1);
                if offset_in_chunk == 0 {
                    let s0 = arena.chunk_ptr(i0);
                    let s1 = arena.chunk_ptr(i1);
                    let v0 = _mm256_loadu_si256(s0 as *const __m256i);
                    if len == SIMD_CHUNK_BYTES * 2 {
                        let v1 = _mm256_loadu_si256(s1 as *const __m256i);
                        _mm256_storeu_si256(out as *mut __m256i, v0);
                        _mm256_storeu_si256(out.add(SIMD_CHUNK_BYTES) as *mut __m256i, v1);
                    } else {
                        _mm256_storeu_si256(out as *mut __m256i, v0);
                        Self::copy_slot_bytes(
                            s1,
                            out.add(SIMD_CHUNK_BYTES),
                            len - SIMD_CHUNK_BYTES,
                        );
                    }
                } else {
                    let first = SIMD_CHUNK_BYTES - offset_in_chunk;
                    let rest = len - first;
                    let p0 = arena.chunk_ptr(i0).add(offset_in_chunk);
                    let p1 = arena.chunk_ptr(i1);
                    if first >= 16 && rest >= 16 {
                        let a0 = core::ptr::read_unaligned(p0 as *const u128);
                        let b0 = core::ptr::read_unaligned(p1 as *const u128);
                        let a1 = if first > 16 {
                            core::ptr::read_unaligned(p0.add(first - 16) as *const u128)
                        } else {
                            0
                        };
                        let b1 = if rest > 16 {
                            core::ptr::read_unaligned(p1.add(rest - 16) as *const u128)
                        } else {
                            0
                        };
                        core::ptr::write_unaligned(out as *mut u128, a0);
                        if first > 16 {
                            core::ptr::write_unaligned(out.add(first - 16) as *mut u128, a1);
                        }
                        core::ptr::write_unaligned(out.add(first) as *mut u128, b0);
                        if rest > 16 {
                            core::ptr::write_unaligned(out.add(first + rest - 16) as *mut u128, b1);
                        }
                    } else {
                        move_n!(p0, out, first);
                        move_n!(p1, out.add(first), rest);
                    }
                }
                return;
            }
            if chunks_to_copy == 3 {
                let (i0, i1) = load_two_slots(index);
                let i2 = *index.add(2);
                prefetch_chunk(arena, i1);
                prefetch_chunk(arena, i2);
                if offset_in_chunk == 0 {
                    let s0 = arena.chunk_ptr(i0);
                    let s1 = arena.chunk_ptr(i1);
                    let s2 = arena.chunk_ptr(i2);
                    let v0 = _mm256_loadu_si256(s0 as *const __m256i);
                    let v1 = _mm256_loadu_si256(s1 as *const __m256i);
                    let tail = len - SIMD_CHUNK_BYTES * 2;
                    if tail == SIMD_CHUNK_BYTES {
                        let v2 = _mm256_loadu_si256(s2 as *const __m256i);
                        _mm256_storeu_si256(out as *mut __m256i, v0);
                        _mm256_storeu_si256(out.add(SIMD_CHUNK_BYTES) as *mut __m256i, v1);
                        _mm256_storeu_si256(out.add(SIMD_CHUNK_BYTES * 2) as *mut __m256i, v2);
                    } else {
                        _mm256_storeu_si256(out as *mut __m256i, v0);
                        _mm256_storeu_si256(out.add(SIMD_CHUNK_BYTES) as *mut __m256i, v1);
                        Self::copy_slot_bytes(s2, out.add(SIMD_CHUNK_BYTES * 2), tail);
                    }
                } else {
                    let first = SIMD_CHUNK_BYTES - offset_in_chunk;
                    let tail = len - first - SIMD_CHUNK_BYTES;
                    let p0 = arena.chunk_ptr(i0).add(offset_in_chunk);
                    let p1 = arena.chunk_ptr(i1);
                    let p2 = arena.chunk_ptr(i2);
                    if first >= 16 && tail >= 16 {
                        let a0 = core::ptr::read_unaligned(p0 as *const u128);
                        let mid = _mm256_loadu_si256(p1 as *const __m256i);
                        let c0 = core::ptr::read_unaligned(p2 as *const u128);
                        let a1 = if first > 16 {
                            core::ptr::read_unaligned(p0.add(first - 16) as *const u128)
                        } else {
                            0
                        };
                        let c1 = if tail > 16 {
                            core::ptr::read_unaligned(p2.add(tail - 16) as *const u128)
                        } else {
                            0
                        };
                        core::ptr::write_unaligned(out as *mut u128, a0);
                        if first > 16 {
                            core::ptr::write_unaligned(out.add(first - 16) as *mut u128, a1);
                        }
                        _mm256_storeu_si256(out.add(first) as *mut __m256i, mid);
                        core::ptr::write_unaligned(out.add(first + SIMD_CHUNK_BYTES) as *mut u128, c0);
                        if tail > 16 {
                            core::ptr::write_unaligned(
                                out.add(first + SIMD_CHUNK_BYTES + tail - 16) as *mut u128,
                                c1,
                            );
                        }
                    } else {
                        move_n!(p0, out, first);
                        move_n!(p1, out.add(first), SIMD_CHUNK_BYTES);
                        move_n!(p2, out.add(first + SIMD_CHUNK_BYTES), tail);
                    }
                }
                return;
            }
            if offset_in_chunk == 0 {
                // Every slot but the tail is a full 32 bytes.
                prefetch_chunk(arena, *index.add(1));
                let full = len / SIMD_CHUNK_BYTES;
                let tail = len % SIMD_CHUNK_BYTES;
                let mut i = 0usize;
                while i + 1 < full {
                    if i + 3 < chunks_to_copy {
                        let (p0, p1) = load_two_slots(index.add(i + 2));
                        prefetch_chunk(arena, p0);
                        prefetch_chunk(arena, p1);
                    } else if i + 2 < chunks_to_copy {
                        prefetch_chunk(arena, *index.add(i + 2));
                    }
                    let (a, b) = load_two_slots(index.add(i));
                    let src0 = arena.chunk_ptr(a);
                    let src1 = arena.chunk_ptr(b);
                    let v0 = _mm256_loadu_si256(src0 as *const __m256i);
                    let v1 = _mm256_loadu_si256(src1 as *const __m256i);
                    _mm256_storeu_si256(out.add(i * SIMD_CHUNK_BYTES) as *mut __m256i, v0);
                    _mm256_storeu_si256(
                        out.add((i + 1) * SIMD_CHUNK_BYTES) as *mut __m256i,
                        v1,
                    );
                    i += 2;
                }
                while i < full {
                    if i + 2 < chunks_to_copy {
                        prefetch_chunk(arena, *index.add(i + 2));
                    }
                    let src = arena.chunk_ptr(*index.add(i));
                    let dst = out.add(i * SIMD_CHUNK_BYTES);
                    let v = _mm256_loadu_si256(src as *const __m256i);
                    _mm256_storeu_si256(dst as *mut __m256i, v);
                    i += 1;
                }
                if tail != 0 {
                    Self::copy_slot_bytes(
                        arena.chunk_ptr(*index.add(full)),
                        out.add(full * SIMD_CHUNK_BYTES),
                        tail,
                    );
                }
                return;
            }
            let (i0, i1) = load_two_slots(index);
            prefetch_chunk(arena, i1);
            let n = SIMD_CHUNK_BYTES - offset_in_chunk;
            Self::copy_slot_bytes(
                arena.chunk_ptr(i0).add(offset_in_chunk),
                out,
                n,
            );
            let mut written = n;
            let last = chunks_to_copy - 1;
            let mut i = 1usize;
            while i + 1 < last {
                if i + 3 < chunks_to_copy {
                    let (p0, p1) = load_two_slots(index.add(i + 2));
                    prefetch_chunk(arena, p0);
                    prefetch_chunk(arena, p1);
                }
                let (a, b) = load_two_slots(index.add(i));
                let src0 = arena.chunk_ptr(a);
                let src1 = arena.chunk_ptr(b);
                let v0 = _mm256_loadu_si256(src0 as *const __m256i);
                let v1 = _mm256_loadu_si256(src1 as *const __m256i);
                _mm256_storeu_si256(out.add(written) as *mut __m256i, v0);
                _mm256_storeu_si256(out.add(written + SIMD_CHUNK_BYTES) as *mut __m256i, v1);
                written += SIMD_CHUNK_BYTES * 2;
                i += 2;
            }
            while i < last {
                if i + 2 < chunks_to_copy {
                    prefetch_chunk(arena, *index.add(i + 2));
                }
                let src = arena.chunk_ptr(*index.add(i));
                let v = _mm256_loadu_si256(src as *const __m256i);
                _mm256_storeu_si256(out.add(written) as *mut __m256i, v);
                written += SIMD_CHUNK_BYTES;
                i += 1;
            }
            let n = len - written;
            let src = arena.chunk_ptr(*index.add(last));
            if n == SIMD_CHUNK_BYTES {
                let v = _mm256_loadu_si256(src as *const __m256i);
                _mm256_storeu_si256(out.add(written) as *mut __m256i, v);
            } else if n != 0 {
                Self::copy_slot_bytes(src, out.add(written), n);
            }
        }
    }

    unsafe fn copy_span_ptr(&self, arena: ArenaPtr, start: u32, end: u32, out: *mut u8) {
        let total = self.byte_len as usize;
        let start = (start as usize).min(total);
        let end = (end as usize).min(total);
        let len = end.saturating_sub(start);
        if len == 0 {
            return;
        }
        let start_chunk = start / SIMD_CHUNK_BYTES;
        let offset_in_chunk = start % SIMD_CHUNK_BYTES;
        // `end` is inside the path, so these slots exist. No full index slice.
        let chunks_to_copy = chunks_needed(offset_in_chunk + len);
        let index = unsafe { arena.indices.add(self.index_offset as usize + start_chunk) };

        let mut written = 0usize;
        #[cfg(target_arch = "x86_64")]
        if chunks_to_copy > 1 {
            prefetch_chunk(arena, unsafe { *index.add(1) });
        }
        for i in 0..chunks_to_copy {
            let idx = unsafe { *index.add(i) };
            #[cfg(target_arch = "x86_64")]
            if i + 2 < chunks_to_copy {
                prefetch_chunk(arena, unsafe { *index.add(i + 2) });
            }
            let s = if i == 0 { offset_in_chunk } else { 0 };
            let n = SIMD_CHUNK_BYTES.min(s + (len - written)) - s;
            // SAFETY: `idx` is an initialized chunk; `s..s+n` stays in it,
            // and `out` holds `len` bytes starting at index 0.
            unsafe {
                Self::copy_slot_bytes(arena.chunk_ptr(idx).add(s), out.add(written), n);
            }
            written += n;
        }
        if written < len {
            unsafe { out.add(written).write_bytes(0, len - written) };
        }
    }

    // Non-empty path components, skipping `.`. A component inside one chunk is borrowed.
    pub(crate) fn for_each_segment(
        &self,
        arena: ArenaPtr,
        scratch: &mut [u8],
        mut f: impl FnMut(&[u8]) -> bool,
    ) -> bool {
        let total = self.byte_len as usize;
        if total == 0 {
            return true;
        }
        let index = self.slot_index(arena, 0);
        let mut start = 0usize;
        let mut pos = 0usize;
        let mut slot = 0usize;
        while pos < total {
            let idx = unsafe { *index.add(slot) };
            let take = SIMD_CHUNK_BYTES.min(total - pos);
            let chunk = unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), take) };
            for i in memchr::memchr2_iter(b'/', b'\\', chunk) {
                let abs = pos + i;
                if start < abs && !self.emit_segment(arena, scratch, start, abs, &mut f) {
                    return false;
                }
                start = abs + 1;
            }
            pos += take;
            slot += 1;
        }
        if start < total {
            return self.emit_segment(arena, scratch, start, total, &mut f);
        }
        true
    }

    fn emit_segment(
        &self,
        arena: ArenaPtr,
        scratch: &mut [u8],
        start: usize,
        end: usize,
        f: &mut impl FnMut(&[u8]) -> bool,
    ) -> bool {
        if let Some(slice) = self.window_in_chunk(arena, start as u32, end as u32) {
            if slice == b"." {
                return true;
            }
            return f(slice);
        }
        let len = end - start;
        if len > scratch.len() {
            return true;
        }
        self.copy_window_to(arena, start as u32, end as u32, &mut scratch[..len]);
        if &scratch[..len] == b"." {
            return true;
        }
        f(&scratch[..len])
    }

    /// Truncates at `buf.len()` if exceeded -- use `[u8; PATH_BUF_SIZE]` to avoid.
    #[inline]
    pub fn read_to_buf<'a>(&self, arena: ArenaPtr, buf: &'a mut [u8]) -> &'a str {
        let total = (self.byte_len as usize).min(buf.len());
        if total == 0 {
            return unsafe { core::str::from_utf8_unchecked(&buf[..0]) };
        }
        // Full chunks copy a constant 32B. The index pointer starts at slot 0.
        let full_chunks = total / SIMD_CHUNK_BYTES;
        let slots = total.div_ceil(SIMD_CHUNK_BYTES);
        let index = self.slot_index(arena, 0);
        #[cfg(target_arch = "x86_64")]
        if slots > 1 {
            prefetch_chunk(arena, unsafe { *index.add(1) });
        }
        for i in 0..full_chunks {
            let idx = unsafe { *index.add(i) };
            #[cfg(target_arch = "x86_64")]
            if i + 2 < slots {
                prefetch_chunk(arena, unsafe { *index.add(i + 2) });
            }
            unsafe {
                Self::copy_slot_bytes(
                    arena.chunk_ptr(idx),
                    buf.as_mut_ptr().add(i * SIMD_CHUNK_BYTES),
                    SIMD_CHUNK_BYTES,
                );
            }
        }
        if full_chunks * SIMD_CHUNK_BYTES < total {
            let dst_offset = full_chunks * SIMD_CHUNK_BYTES;
            let take = total - dst_offset;
            let idx = unsafe { *index.add(full_chunks) };
            unsafe {
                Self::copy_slot_bytes(
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
        if dir_len == 0 {
            return;
        }
        out.reserve(dir_len);
        // The directory is a prefix. The copy writes it, then the length is published.
        unsafe {
            let vec = out.as_mut_vec();
            self.copy_trimmed_ptr(arena, 0, dir_len as u32, vec.as_mut_ptr());
            vec.set_len(dir_len);
        }
    }

    // Same order as `file_name(arena).cmp`, reading arena bytes in place.
    #[inline]
    pub(crate) fn cmp_filename(&self, arena: ArenaPtr, other: &Self) -> std::cmp::Ordering {
        let (a_len_u, a_off) = self.span_lens();
        let (b_len_u, b_off) = other.span_lens();
        let a_len = (a_len_u as usize).saturating_sub(a_off as usize);
        let b_len = (b_len_u as usize).saturating_sub(b_off as usize);
        let n = a_len.min(b_len);
        let mut done = 0usize;
        while done < n {
            let a = self.bytes_from(arena, a_off as usize + done);
            let b = other.bytes_from(arena, b_off as usize + done);
            let take = a.len().min(b.len()).min(n - done);
            match a[..take].cmp(&b[..take]) {
                std::cmp::Ordering::Equal => done += take,
                ord => return ord,
            }
        }
        a_len.cmp(&b_len)
    }

    // One slot, the one that holds `abs`. The path header supplies the length and the index.
    #[inline(always)]
    pub(crate) fn prefetch_at(&self, arena: ArenaPtr, abs: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            let h = self.header();
            let total = (h >> 32) as u16 as usize;
            if abs >= total {
                return;
            }
            let idx = unsafe {
                *arena
                    .indices
                    .add((h as u32 as usize) + abs / SIMD_CHUNK_BYTES)
            };
            prefetch_chunk(arena, idx);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (arena, abs);
    }

    // Sibling walks the filename. A new directory starts at byte 0 and may also need the name.
    #[inline(always)]
    pub(crate) fn prefetch_for_gate(&self, arena: ArenaPtr, sibling: bool) {
        #[cfg(target_arch = "x86_64")]
        {
            let h = self.header();
            let total = (h >> 32) as u16 as usize;
            if total == 0 {
                return;
            }
            let base = h as u32 as usize;
            let name = ((h >> 48) as u16 as usize).min(total - 1);
            let idx_at = |abs: usize| unsafe { *arena.indices.add(base + abs / SIMD_CHUNK_BYTES) };
            if sibling {
                prefetch_chunk(arena, idx_at(name));
            } else {
                prefetch_chunk(arena, idx_at(0));
                if name >= SIMD_CHUNK_BYTES {
                    prefetch_chunk(arena, idx_at(name));
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (arena, sibling);
    }

    // `abs < byte_len`. One index load and the 32B slot that contains that byte.
    #[inline]
    pub(crate) fn chunk_slot<'a>(
        &self,
        arena: ArenaPtr,
        abs: usize,
    ) -> (&'a [u8; SIMD_CHUNK_BYTES], usize) {
        let chunk_i = abs / SIMD_CHUNK_BYTES;
        let within = abs % SIMD_CHUNK_BYTES;
        let idx = unsafe { *arena.indices.add(self.index_offset as usize + chunk_i) };
        let chunk = unsafe { &*(arena.chunk_ptr(idx) as *const [u8; SIMD_CHUNK_BYTES]) };
        (chunk, within)
    }

    /// The slot that contains byte `slot * 32`. The caller already has the offset inside it.
    #[inline(always)]
    fn chunk_for_slot<'a>(&self, arena: ArenaPtr, slot: usize) -> &'a [u8; SIMD_CHUNK_BYTES] {
        unsafe { Self::chunk_at(arena, *self.slot_index(arena, slot)) }
    }

    /// Index entry of slot `chunk_i`. The following slots are `add(1)`.
    #[inline(always)]
    pub(crate) fn slot_index(&self, arena: ArenaPtr, chunk_i: usize) -> *const u32 {
        unsafe { arena.indices.add(self.index_offset as usize + chunk_i) }
    }

    #[inline(always)]
    pub(crate) fn chunk_at<'a>(arena: ArenaPtr, idx: u32) -> &'a [u8; SIMD_CHUNK_BYTES] {
        unsafe { &*(arena.chunk_ptr(idx) as *const [u8; SIMD_CHUNK_BYTES]) }
    }

    #[inline]
    fn bytes_from<'a>(&self, arena: ArenaPtr, abs: usize) -> &'a [u8] {
        let total = self.byte_len as usize;
        if abs >= total {
            return &[];
        }
        let (chunk, at) = self.chunk_slot(arena, abs);
        let take = (SIMD_CHUNK_BYTES - at).min(total - abs);
        &chunk[at..at + take]
    }

    #[inline]
    pub fn write_filename_to(&self, arena: ArenaPtr, out: &mut String) {
        out.clear();

        let (byte_len_u, name_off) = self.span_lens();
        let fname_offset = name_off as usize;
        let fname_len = byte_len_u as usize - fname_offset;
        if fname_len == 0 {
            return;
        }
        out.reserve(fname_len);
        let end = fname_offset + fname_len;
        unsafe {
            let vec = out.as_mut_vec();
            self.copy_trimmed_ptr(arena, fname_offset as u32, end as u32, vec.as_mut_ptr());
            vec.set_len(fname_len);
        }
    }

    /// Stored path bytes, in order. Same sequence `write_to_string` writes.
    #[inline]
    pub fn append_bytes(&self, arena: ArenaPtr, out: &mut Vec<u8>) {
        let total = self.byte_len as usize;
        if total == 0 {
            return;
        }
        let at = out.len();
        out.reserve(total);
        // The copy writes every new byte, then the length is published.
        unsafe {
            self.copy_trimmed_ptr(arena, 0, total as u32, out.as_mut_ptr().add(at));
            out.set_len(at + total);
        }
    }

    #[inline]
    pub fn write_to_string(&self, arena: ArenaPtr, out: &mut String) {
        out.clear();
        self.append_bytes(arena, unsafe { out.as_mut_vec() });
    }

    // Case-sensitive substring. Hits inside a chunk are in place; a crossing
    // match uses a stack window of at most two chunks.
    pub(crate) fn contains_needle(
        &self,
        arena: ArenaPtr,
        finder: &memchr::memmem::Finder<'_>,
    ) -> bool {
        let n = finder.needle().len();
        if n == 0 {
            return true;
        }
        let total = self.byte_len as usize;
        if n > total {
            return false;
        }
        if n > SIMD_CHUNK_BYTES {
            return self.contains_needle_copied(arena, finder, total);
        }
        let overlap = n - 1;
        let mut prev_ptr: *const u8 = core::ptr::null();
        let mut prev_len = 0usize;
        let index = self.slot_index(arena, 0);
        let mut off = 0usize;
        let mut slot = 0usize;
        while off < total {
            let take = SIMD_CHUNK_BYTES.min(total - off);
            let idx = load_slot_idx(index, slot, arena, off + take < total);
            // Slots stay immutable for the scan. `prev` is the prior slot.
            let chunk = unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), take) };
            if prev_len > 0 {
                let prev = unsafe { core::slice::from_raw_parts(prev_ptr, prev_len) };
                if crossing_contains(prev, chunk, overlap, finder) {
                    return true;
                }
            }
            if finder.find(chunk).is_some() {
                return true;
            }
            prev_ptr = chunk.as_ptr();
            prev_len = chunk.len();
            off += take;
            slot += 1;
        }
        false
    }

    fn contains_needle_copied(
        &self,
        arena: ArenaPtr,
        finder: &memchr::memmem::Finder<'_>,
        total: usize,
    ) -> bool {
        if total <= 512 {
            let mut buf = [0u8; 512];
            let s = self.read_to_buf(arena, &mut buf[..total]);
            return finder.find(s.as_bytes()).is_some();
        }
        let mut buf = vec![0u8; total];
        let s = self.read_to_buf(arena, &mut buf);
        finder.find(s.as_bytes()).is_some()
    }

    // Case-insensitive substring over arena chunks. `needle` is already folded.
    // Both pair bytes must hit before any full compare. No path copy.
    pub(crate) fn contains_folded(
        &self,
        arena: ArenaPtr,
        needle: &[u8],
        plan: PairPlan,
    ) -> bool {
        let n = needle.len();
        let total = self.byte_len as usize;
        if n == 0 {
            return true;
        }
        if n > total {
            return false;
        }
        let index = self.slot_index(arena, 0);
        if n == 1 {
            let b = needle[0];
            let alt = if b.is_ascii_lowercase() { b ^ 0x20 } else { b };
            return chunks_contain_byte(index, arena, total, b, alt);
        }
        let (rare_pos, rare, rare_alt, other_pos, other, other_alt) = plan.pair_ends();
        debug_assert!(rare_pos < n && other_pos < n);
        let rare_lo = rare_pos;
        let rare_hi = total - n + rare_pos;
        let mut off = 0usize;
        let mut slot = 0usize;
        while off <= rare_hi && off < total {
            let take = SIMD_CHUNK_BYTES.min(total - off);
            let idx = load_slot_idx(index, slot, arena, off + take <= rare_hi);
            let chunk_end = off + take;
            let lo = off.max(rare_lo);
            let hi = chunk_end.min(rare_hi + 1);
            if lo < hi {
                // Slot is immutable for the scan. Only `take` bytes belong to the path.
                let chunk = unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), take) };
                let region = &chunk[lo - off..hi - off];
                for rel in memchr::memchr2_iter(rare, rare_alt, region) {
                    let start = lo + rel - rare_pos;
                    let other_at = start + other_pos;
                    let ob = if other_at >= off && other_at < chunk_end {
                        chunk[other_at - off]
                    } else {
                        chunk_byte(index, arena, other_at)
                    };
                    if (ob == other || ob == other_alt)
                        && folded_eq_at(index, arena, start, needle)
                    {
                        return true;
                    }
                }
            }
            off += take;
            slot += 1;
        }
        false
    }

    // Segment at byte 0 or after a separator, with a separator after it.
    // `seg` is ASCII-folded once. `/` and `\` compare equal. No path copy.
    pub(crate) fn contains_segment(&self, arena: ArenaPtr, seg: &[u8]) -> bool {
        let n = seg.len();
        let total = self.byte_len as usize;
        if total <= n {
            return false;
        }
        let index = self.slot_index(arena, 0);
        if sep_at(index, arena, total, n) && seg_eq_at(index, arena, 0, seg) {
            return true;
        }
        if total < n + 2 {
            return false;
        }
        // `i + 1 + n` must stay inside the path, so separators stop here.
        let sep_end = total - n - 1;
        let mut off = 0usize;
        let mut slot = 0usize;
        while off < sep_end {
            let take = SIMD_CHUNK_BYTES.min(total - off);
            let limit = take.min(sep_end - off);
            let idx = unsafe { *index.add(slot) };
            // Slot is immutable. `limit` stays inside this path's bytes.
            let chunk = unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), take) };
            for j in memchr::memchr2_iter(b'/', b'\\', &chunk[..limit]) {
                let start = off + j + 1;
                let end = start + n;
                // The component and the separator after it are already in this slot.
                if end < off + take {
                    let at = start - off;
                    if (chunk[at] & 0xC0) != 0x80
                        && is_sep(chunk[end - off])
                        && seg_eq_bytes(&chunk[at..at + n], seg)
                    {
                        return true;
                    }
                    continue;
                }
                let end_sep = sep_at(index, arena, total, end);
                if end_sep && on_boundary(index, arena, start) && seg_eq_at(index, arena, start, seg)
                {
                    return true;
                }
            }
            off += take;
            slot += 1;
        }
        false
    }
}

fn crossing_contains(
    prev: &[u8],
    cur: &[u8],
    overlap: usize,
    finder: &memchr::memmem::Finder<'_>,
) -> bool {
    let take_prev = overlap.min(prev.len());
    let take_cur = overlap.min(cur.len());
    if take_prev + take_cur <= overlap {
        return false;
    }
    let mut window = [0u8; SIMD_CHUNK_BYTES * 2];
    let end = take_prev + take_cur;
    window[..take_prev].copy_from_slice(&prev[prev.len() - take_prev..]);
    window[take_prev..end].copy_from_slice(&cur[..take_cur]);
    finder.find(&window[..end]).is_some()
}

#[inline(always)]
fn is_sep(b: u8) -> bool {
    b == b'/' || b == b'\\'
}

#[inline(always)]
fn chunk_byte(index: *const u32, arena: ArenaPtr, abs: usize) -> u8 {
    // `abs` is inside the path, so this slot holds that byte.
    let idx = unsafe { *index.add(abs / SIMD_CHUNK_BYTES) };
    unsafe { *arena.chunk_ptr(idx).add(abs % SIMD_CHUNK_BYTES) }
}

#[inline(always)]
fn sep_at(index: *const u32, arena: ArenaPtr, total: usize, abs: usize) -> bool {
    abs < total && is_sep(chunk_byte(index, arena, abs))
}

// Same as `str::is_char_boundary`: a continuation byte is not a segment start.
#[inline(always)]
fn on_boundary(index: *const u32, arena: ArenaPtr, start: usize) -> bool {
    if start == 0 {
        return true;
    }
    (chunk_byte(index, arena, start) & 0xC0) != 0x80
}

#[inline]
fn chunks_contain_byte(
    index: *const u32,
    arena: ArenaPtr,
    total: usize,
    b: u8,
    alt: u8,
) -> bool {
    let mut off = 0usize;
    let mut slot = 0usize;
    while off < total {
        let take = SIMD_CHUNK_BYTES.min(total - off);
        let idx = load_slot_idx(index, slot, arena, off + take < total);
        let chunk = unsafe { core::slice::from_raw_parts(arena.chunk_ptr(idx), take) };
        if memchr::memchr2(b, alt, chunk).is_some() {
            return true;
        }
        off += take;
        slot += 1;
    }
    false
}

// Each piece stays inside one slot. The compare reads exactly that piece.
#[inline]
// One divide for the first slot. Each later piece is the next index entry.
fn walk_bytes(
    index: *const u32,
    arena: ArenaPtr,
    start: usize,
    len: usize,
    mut f: impl FnMut(*const u8, usize) -> bool,
) -> bool {
    if len == 0 {
        return true;
    }
    let mut abs = start;
    let end = start + len;
    let mut cursor = unsafe { index.add(abs / SIMD_CHUNK_BYTES) };
    let mut within = abs % SIMD_CHUNK_BYTES;
    while abs < end {
        let idx = unsafe { *cursor };
        let take = (SIMD_CHUNK_BYTES - within).min(end - abs);
        let ptr = unsafe { arena.chunk_ptr(idx).add(within) };
        if !f(ptr, take) {
            return false;
        }
        abs += take;
        within = 0;
        if abs < end {
            cursor = unsafe { cursor.add(1) };
        }
    }
    true
}

fn folded_eq_at(index: *const u32, arena: ArenaPtr, start: usize, needle: &[u8]) -> bool {
    let mut done = 0usize;
    walk_bytes(index, arena, start, needle.len(), |ptr, take| {
        let ok = unsafe { eq_lowered_case(ptr, &needle[done..done + take]) };
        done += take;
        ok
    })
}

#[inline(always)]
fn seg_eq_bytes(hay: &[u8], seg: &[u8]) -> bool {
    if hay.len() != seg.len() {
        return false;
    }
    // `seg` is already folded. A separator inside it still needs the slash rule.
    if hay.len() >= 4 && !slice_has_sep(seg) {
        return unsafe { eq_lowered_case(hay.as_ptr(), seg) };
    }
    hay.iter().zip(seg).all(|(&h, &s)| {
        h == s || (is_sep(h) && is_sep(s)) || h.to_ascii_lowercase() == s
    })
}

#[inline]
fn seg_eq_at(index: *const u32, arena: ArenaPtr, start: usize, seg: &[u8]) -> bool {
    let mut done = 0usize;
    walk_bytes(index, arena, start, seg.len(), |ptr, take| {
        let hay = unsafe { core::slice::from_raw_parts(ptr, take) };
        let ok = seg_eq_bytes(hay, &seg[done..done + take]);
        done += take;
        ok
    })
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

            // One hash probe: entry holds the chunk while the miss path pushes.
            let arena_idx = *self.chunk_dedup.entry(chunk_bytes).or_insert_with(|| {
                let idx = self.arena.len() as u32;
                self.arena.push(SimdChunk(chunk_bytes));
                idx
            });

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
        let arena = store.as_arena_ptr();
        for (path, chunked) in paths.iter().zip(&strings) {
            let bytes = path.as_bytes();
            assert!(chunked.is_suffix_of(arena, bytes));
            if bytes.is_empty() {
                continue;
            }
            let mut longer = b"repo/".to_vec();
            longer.extend_from_slice(bytes);
            assert!(chunked.is_suffix_of(arena, &longer));
            assert!(!chunked.is_suffix_of(arena, &longer[..longer.len().saturating_sub(1)]));
            if !bytes.is_empty() {
                let mut miss = longer.clone();
                let last = miss.len() - 1;
                miss[last] ^= 1;
                assert!(!chunked.is_suffix_of(arena, &miss));
            }
        }
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
                    let lower: Vec<u8> = suffix.iter().copied().map(|b| b.to_ascii_lowercase()).collect();
                    assert_eq!(
                        chunked.common_suffix_len_lower(store.as_arena_ptr(), &lower),
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
            "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.rs",
            "abcdefghijklmnopqrstuvwxyz.tar.gz",
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
    fn folded_text_matches_a_lowercase_scan() {
        let paths = [
            "src/lib.rs",
            "SRC/LIB.RS",
            "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/Handler.rs",
            "nope",
        ];
        let (store, strings, _) = build_test_store(&paths);
        let arena = store.as_arena_ptr();
        for (path, chunked) in paths.iter().zip(&strings) {
            let folded_path = path.to_ascii_lowercase();
            for needle in ["lib", "handler", "xxx", "rs", "h", "zzz", "Handler"] {
                let lower = needle.to_ascii_lowercase();
                let plan = crate::simd_string_utils::memmem::plan_for(lower.as_bytes());
                assert_eq!(
                    chunked.contains_folded(arena, lower.as_bytes(), plan),
                    folded_path.contains(&lower),
                    "{path} {needle}"
                );
            }
        }
    }

    #[test]
    fn chunked_segment_matches_the_string_rule() {
        let paths = [
            "src/lib.rs",
            "SRC/lib.rs",
            "app/SRC/lib.rs",
            "source/lib.rs",
            "lib/src",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/src/lib.rs",
            "foo/LibSwscale/AArch64/input.S",
            "src\\lib.rs",
        ];
        let (store, strings, _) = build_test_store(&paths);
        let arena = store.as_arena_ptr();
        for (path, chunked) in paths.iter().zip(&strings) {
            for segment in ["src", "SRC", "lib", "LibSwscale/AArch64", "aaaa"] {
                let lower = segment.to_ascii_lowercase();
                assert_eq!(
                    chunked.contains_segment(arena, lower.as_bytes()),
                    crate::index::constraints::path_contains_segment(path, segment),
                    "{path} {segment}"
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
        assert!(cs.equals(arena, path));
        let mut miss = path.to_string();
        miss.replace_range(40..41, "X");
        assert!(!cs.equals(arena, &miss));
        assert!(cs.starts_with_bytes(arena, path.as_bytes()));
        assert!(!cs.starts_with_bytes(arena, miss.as_bytes()));
        assert_eq!(cs.chunk_count(), path.len().div_ceil(SIMD_CHUNK_BYTES));
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        assert!(cs.copy_in_chunk(arena, 0, 16, &mut a));
        cs.copy_window_to(arena, 0, 16, &mut b);
        assert_eq!(a, b);
        assert!(cs.copy_in_chunk(arena, 16, 32, &mut a));
        cs.copy_window_to(arena, 16, 32, &mut b);
        assert_eq!(a, b);
        assert!(!cs.copy_in_chunk(arena, 24, 40, &mut a));
        let mut wide_a = [0u8; 64];
        let mut wide_b = [0u8; 64];
        assert!(cs.copy_short(arena, 24, 40, &mut wide_a[..16]));
        cs.copy_window_to(arena, 24, 40, &mut wide_b[..16]);
        assert_eq!(&wide_a[..16], &wide_b[..16]);
        assert!(cs.copy_short(arena, 0, path.len() as u32, &mut wide_a));
        cs.copy_window_to(arena, 0, path.len() as u32, &mut wide_b);
        assert_eq!(wide_a.as_slice(), path.as_bytes());
        assert_eq!(wide_a, wide_b);
        let (store, strings, _) = build_test_store(&["src/components/Controller.tsx", "src\\components\\Controller.tsx"]);
        let arena = store.as_arena_ptr();
        assert!(!strings[0].contains_byte(arena, b'\\'));
        assert!(strings[1].contains_byte(arena, b'\\'));
        assert!(strings[0].parent_eq_ignore_ascii_case(arena, b'/', b"src/components"));
        assert!(!strings[0].parent_eq_ignore_ascii_case(arena, b'/', b"src/component"));
        assert!(strings[1].parent_eq_ignore_ascii_case(arena, b'\\', b"src\\components"));
        assert!(strings[0].same_parent_dir(arena, b"src/components", b""));
        assert!(strings[0].same_parent_dir_lower(arena, b"src/components", b""));
        assert!(!strings[0].same_parent_dir_lower(arena, b"SRC/components", b""));
        assert!(!strings[0].same_parent_dir(arena, b"src/component", b"src/components"));
        assert!(strings[1].same_parent_dir(arena, b"src/components", b"src\\components"));
        assert!(!strings[1].same_parent_dir(arena, b"src\\components", b"src/components"));
        let (store, strings, _) = build_test_store(&[
            "src/foo\\bar/baz.txt",
            "very/deeply/nested\\directory/structure/with/many/levels/file.txt",
            "file.txt",
        ]);
        let arena = store.as_arena_ptr();
        assert!(strings[0].same_parent_dir(arena, b"src/foo/bar", b"src/foo"));
        assert!(strings[1].same_parent_dir(arena, b"very/deeply/nested/directory", b"very/deeply/nested"));
        assert!(!strings[1].same_parent_dir(arena, b"very/deeply/nested", b"very/deeply"));
        assert!(!strings[2].same_parent_dir(arena, b"file.txt", b"file.txt"));
        assert!(!strings[0].same_parent_dir(arena, b"", b""));
        let parent = "abcdefghijklmnopqrstuvwxyz012345/nested";
        let path = format!("{parent}/file.txt");
        let (store, strings, _) = build_test_store(&[&path]);
        let arena = store.as_arena_ptr();
        assert!(parent.len() > SIMD_CHUNK_BYTES);
        assert!(strings[0].same_parent_dir(arena, parent.as_bytes(), b""));
        assert!(!strings[0].same_parent_dir(arena, b"abcdefghijklmnopqrstuvwxyz012345", b""));
    }

    #[test]
    fn eq_range_across_one_boundary_matches_the_bytes() {
        let path = "0123456789012345678901234567src/lib.rs";
        let (store, strings, _) = build_test_store(&[path]);
        let arena = store.as_arena_ptr();
        let start = 28usize;
        let end = start + "src/lib.rs".len();
        assert!(start % SIMD_CHUNK_BYTES + (end - start) > SIMD_CHUNK_BYTES);
        assert!(strings[0].eq_range(arena, start, end, b"src/lib.rs"));
        assert!(!strings[0].eq_range(arena, start, end, b"src/lib.rx"));
        assert!(strings[0].eq_range(arena, 0, 4, b"0123"));
    }

    #[test]
    fn starts_with_bytes_matches_the_string_rule() {
        let path = "01234567890123456789012345678901234/file.rs";
        assert!(path.len() > SIMD_CHUNK_BYTES);
        let (store, strings, _) = build_test_store(&[path]);
        let arena = store.as_arena_ptr();
        assert!(strings[0].starts_with_bytes(arena, b"0123456789"));
        assert!(strings[0].starts_with_bytes(arena, b"01234567890123456789012345678901234"));
        assert!(!strings[0].starts_with_bytes(arena, b"0123456789012345678901234567890123X"));
        assert!(strings[0].starts_with_bytes(arena, b""));
        assert!(!strings[0].starts_with_bytes(arena, b"nope"));
    }

    #[test]
    fn window_eq_across_one_boundary_ignores_case() {
        let path = "012345678901234567890123456/File.RS";
        assert_eq!(path.len(), 35);
        let (store, strings, _) = build_test_store(&[path]);
        let arena = store.as_arena_ptr();
        let off = strings[0].filename_offset;
        assert!(off > 0 && (off as usize) % SIMD_CHUNK_BYTES + 7 > SIMD_CHUNK_BYTES);
        assert!(strings[0].window_eq_ignore_ascii(arena, off as u32, b"file.rs"));
        assert!(!strings[0].window_eq_ignore_ascii(arena, off as u32, b"file.rx"));
    }

    #[test]
    fn copy_trimmed_three_slots_matches_the_bytes() {
        let path = "0123456789".repeat(8);
        assert!(path.len() > SIMD_CHUNK_BYTES * 2 && path.len() <= SIMD_CHUNK_BYTES * 3);
        let (store, strings, _) = build_test_store(&[path.as_str()]);
        let arena = store.as_arena_ptr();
        let mut out = [0u8; 80];
        strings[0].copy_trimmed(arena, 0, 80, &mut out);
        assert_eq!(&out[..], path.as_bytes());
        let mut mid = [0u8; 70];
        strings[0].copy_trimmed(arena, 10, 80, &mut mid);
        assert_eq!(&mid[..], &path.as_bytes()[10..80]);
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
        let long = "abcdefghij".repeat(12);
        let (store, strings, _) = build_test_store(&[&long]);
        let arena = store.as_arena_ptr();
        let mut buf = [0u8; 512];
        assert_eq!(strings[0].read_to_buf(arena, &mut buf), long);
        let mut wide = [0u8; 160];
        strings[0].copy_window_to(arena, 1, long.len() as u32, &mut wide);
        assert_eq!(&wide[..long.len() - 1], &long.as_bytes()[1..]);
    }
}
