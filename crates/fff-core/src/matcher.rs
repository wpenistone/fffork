use crate::simd_path::{ArenaPtr, ChunkedString, MAX_FRIZBEE_CHUNKS};
use crate::types::FileItem;
use neo_frizbee::Matcher;

thread_local! {
    static RESOLVE_SCRATCH: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// 16 B pointers into immutable 32 B arena chunks; frizbee copies before return.
// A short visit copies the path, same bytes as the previous contiguous buffer.
#[inline]
pub(crate) fn resolve_frizbee_bytes(
    path: &ChunkedString,
    arena: ArenaPtr,
    buf: &mut [*const u8; MAX_FRIZBEE_CHUNKS],
) -> Option<(usize, u16)> {
    let len = path.byte_len as usize;
    if len == 0 {
        return Some((0, 0));
    }
    let unit = neo_frizbee::SIMD_CHUNK_BYTES;
    let n = len.div_ceil(unit);
    if n > buf.len() {
        return None;
    }
    let room = crate::simd_path::SIMD_CHUNK_BYTES / unit;
    if room == 0 {
        return resolve_frizbee_bytes_copied(path, arena, buf, len, n, unit);
    }
    let filled = path.fill_unit_ptrs(arena, unit, &mut buf[..n]);
    if filled == n {
        return Some((n, len as u16));
    }
    resolve_frizbee_bytes_copied(path, arena, buf, len, n, unit)
}

#[derive(Clone, Copy)]
pub(crate) struct MatchWindow {
    pub(crate) file_idx: u32,
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) include_prefix: bool,
    pub(crate) include_exact: bool,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Trim {
    pub file_idx: u32,
    pub start: u16,
    pub end: u16,
    // bit0 = include_prefix, bit1 = include_exact
    pub flags: u8,
}

impl Trim {
    #[inline]
    pub(crate) fn from_window(w: MatchWindow) -> Self {
        Self {
            file_idx: w.file_idx,
            start: w.start.min(u16::MAX as u32) as u16,
            end: w.end.min(u16::MAX as u32) as u16,
            flags: (w.include_prefix as u8) | ((w.include_exact as u8) << 1),
        }
    }
    #[inline]
    pub(crate) fn include_prefix(&self) -> bool {
        self.flags & 1 != 0
    }
    #[inline]
    pub(crate) fn include_exact(&self) -> bool {
        self.flags & 2 != 0
    }
}

#[inline]
pub(crate) fn decode_match_window(m: &neo_frizbee::Match, hay_len: usize) -> MatchWindow {
    // The gate already clamped `end` to the path. `hay_len` is only the full-path check.
    let start_adj = m.score as usize;
    let end_pos = m.end_col as usize;
    let end = end_pos.min(hay_len);
    let start = start_adj.min(end);
    MatchWindow {
        file_idx: m.index,
        start: start as u32,
        end: end as u32,
        include_prefix: start == 0,
        include_exact: start == 0 && end == hay_len,
    }
}

// One compiled matcher per needle and config on this thread.
// A prior run returns the same hits: SW row 0 and column 0 stay zero.
#[inline]
pub(crate) fn with_cached_matcher<R>(
    needle: &str,
    options: &neo_frizbee::Config,
    run: impl FnOnce(&mut Matcher) -> R,
) -> R {
    thread_local! {
        static CACHE: std::cell::RefCell<Vec<(String, neo_frizbee::Config, Matcher)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(i) = c.iter().position(|(n, o, _)| n == needle && o == options) {
            return run(&mut c[i].2);
        }
        let mut inst = Matcher::new(needle, options);
        let r = run(&mut inst);
        if c.len() >= 4 {
            c.remove(0);
        }
        c.push((needle.to_string(), *options, inst));
        r
    })
}

thread_local! {
    static HIT_CAPTURE: std::cell::RefCell<Option<Vec<u32>>> =
        const { std::cell::RefCell::new(None) };
}

// `offset` maps slice indices into the corpus. `append` keeps the other arena.
pub(crate) fn capture_hit_indices(
    iter: impl Iterator<Item = u32>,
    offset: u32,
    append: bool,
) {
    HIT_CAPTURE.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            if !append {
                v.clear();
            }
            let mapped = iter.map(|i| i.wrapping_add(offset));
            let (lower, _) = mapped.size_hint();
            v.reserve(lower);
            v.extend(mapped);
        }
    });
}

pub(crate) fn with_hit_capture<T>(sink: &mut Vec<u32>, f: impl FnOnce() -> T) -> T {
    HIT_CAPTURE.with(|c| {
        *c.borrow_mut() = Some(std::mem::take(sink));
    });
    let t = f();
    HIT_CAPTURE.with(|c| {
        if let Some(v) = c.borrow_mut().take() {
            *sink = v;
        }
    });
    t
}

// Test helper. Callers clamp at 512, matching `FileItems::max_path_len`.
#[cfg(test)]
#[inline]
pub(crate) fn max_path_len_of(files: &[FileItem]) -> usize {
    let mut m = 0usize;
    for f in files {
        // Callers clamp to 512; a longer exact max would not change dispatch.
        m = m.max(f.path.byte_len as usize);
        if m >= 512 {
            return 512;
        }
    }
    m
}

thread_local! {
    static OWNED_IDX: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

// Index buffer for a filtered hit list. Drop of `CandList::Owned` returns it.
pub(crate) fn take_idx_buf() -> Vec<u32> {
    let mut v = OWNED_IDX.with(|c| std::mem::take(&mut *c.borrow_mut()));
    v.clear();
    v
}

pub(crate) fn reclaim_idx_buf(mut v: Vec<u32>) {
    v.clear();
    OWNED_IDX.with(|c| {
        let mut slot = c.borrow_mut();
        if v.capacity() > slot.capacity() {
            *slot = v;
        }
    });
}

// A filtered subset owns its indices. A repeated hit list is the caller's `Arc`.
pub(crate) enum CandList {
    Owned(Vec<u32>),
    Shared(std::sync::Arc<Vec<u32>>),
}

impl Drop for CandList {
    fn drop(&mut self) {
        if let CandList::Owned(v) = self {
            reclaim_idx_buf(std::mem::take(v));
        }
    }
}

impl std::ops::Deref for CandList {
    type Target = [u32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            CandList::Owned(v) => v,
            CandList::Shared(v) => v.as_slice(),
        }
    }
}

pub(crate) enum FileItems<'a> {
    All(&'a [FileItem]),
    // Positions address `base`.
    Filtered { base: &'a [FileItem], idx: CandList },
}

impl<'a> FileItems<'a> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match self {
            FileItems::All(s) => s.len(),
            FileItems::Filtered { idx, .. } => idx.len(),
        }
    }

    #[inline]
    pub(crate) fn index(&self, index: usize) -> &'a FileItem {
        // Positions come from this working set. Debug builds still check them.
        match self {
            FileItems::All(s) => Self::at_all(s, index),
            FileItems::Filtered { base, idx } => Self::at_filtered(base, idx, index),
        }
    }

    // Full-index load. The caller already matched `FileItems::All`.
    #[inline(always)]
    pub(crate) fn at_all(files: &'a [FileItem], index: usize) -> &'a FileItem {
        debug_assert!(index < files.len());
        unsafe { files.get_unchecked(index) }
    }

    // Narrowed load. The caller already matched `FileItems::Filtered`.
    #[inline(always)]
    pub(crate) fn at_filtered(
        base: &'a [FileItem],
        idx: &CandList,
        index: usize,
    ) -> &'a FileItem {
        debug_assert!(index < idx.len());
        let file_idx = unsafe { *idx.get_unchecked(index) } as usize;
        debug_assert!(file_idx < base.len());
        unsafe { base.get_unchecked(file_idx) }
    }

}

// `FFF_MATCH_THREADS` forces width and is still capped by `max_threads`. Read once.
static THREAD_OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

pub(crate) fn effective_match_threads(
    item_count: usize,
    needle_len: usize,
    max_haystack_len: usize,
    max_typos: Option<u16>,
    max_threads: usize,
) -> usize {
    let forced = *THREAD_OVERRIDE.get_or_init(|| {
        std::env::var("FFF_MATCH_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n >= 1)
    });
    if let Some(n) = forced {
        return n.min(max_threads.max(1));
    }
    if max_threads <= 1 || item_count == 0 {
        return 1;
    }
    let per_file = crate::dispatch::match_bytes_per_file(needle_len, max_haystack_len, max_typos);
    crate::dispatch::plan_files(item_count, per_file, max_threads.max(1)).threads
}

// Writes `len` path bytes to `dst`. Does not read `dst`.
unsafe fn copy_path_raw(path: &ChunkedString, arena: ArenaPtr, dst: *mut u8, len: usize) {
    let total = path.byte_len as usize;
    let end = len.min(total);
    #[cfg(target_arch = "x86_64")]
    if crate::simd_string_utils::memmem::has_avx2() {
        unsafe { path.copy_window_avx2(arena, 0, end, dst) };
        if end < len {
            unsafe { std::ptr::write_bytes(dst.add(end), 0, len - end) };
        }
        return;
    }
    let mut written = 0usize;
    // The copy starts at byte 0, so every piece begins on a slot.
    let index = path.slot_index(arena, 0);
    let mut slot = 0usize;
    let simd = crate::simd_path::SIMD_CHUNK_BYTES;
    #[cfg(target_arch = "x86_64")]
    if end > simd {
        let next = unsafe { *index.add(1) };
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                ChunkedString::chunk_at(arena, next).as_ptr() as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }
    while written < end {
        let n = simd.min(end - written);
        let idx = unsafe { *index.add(slot) };
        #[cfg(target_arch = "x86_64")]
        if written + n + simd < end {
            let ahead = unsafe { *index.add(slot + 2) };
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    ChunkedString::chunk_at(arena, ahead).as_ptr() as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        unsafe {
            ChunkedString::copy_slot_bytes(
                ChunkedString::chunk_at(arena, idx).as_ptr(),
                dst.add(written),
                n,
            );
        }
        written += n;
        slot += 1;
    }
    if written < len {
        unsafe {
            std::ptr::write_bytes(dst.add(written), 0, len - written);
        }
    }
}

// Contiguous copy when the arena visit does not cover `n` frizbee chunks.
// Growth zeros `[max(old_len, len), padded)`; `copy_window_to` writes `[0, len)`.
fn resolve_frizbee_bytes_copied(
    path: &ChunkedString,
    arena: ArenaPtr,
    buf: &mut [*const u8; MAX_FRIZBEE_CHUNKS],
    len: usize,
    n: usize,
    unit: usize,
) -> Option<(usize, u16)> {
    RESOLVE_SCRATCH.with(|c| {
        let mut s = c.borrow_mut();
        let padded = n * unit;
        let old_len = s.len();
        if old_len < padded {
            if old_len < len && len <= padded {
                s.reserve(padded - old_len);
                unsafe {
                    let base = s.as_mut_ptr();
                    // Spare capacity is uninitialized. Write it before set_len.
                    copy_path_raw(path, arena, base, len);
                    if len < padded {
                        std::ptr::write_bytes(base.add(len), 0, padded - len);
                    }
                    s.set_len(padded);
                }
            } else {
                s.resize(padded, 0);
                path.copy_window_to(arena, 0, len as u32, &mut s[..len]);
            }
        } else {
            path.copy_window_to(arena, 0, len as u32, &mut s[..len]);
        }
        let base = s.as_ptr();
        for (slot, k) in buf.iter_mut().zip(0..n) {
            *slot = unsafe { base.add(k * unit) };
        }
        Some((n, len as u16))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_width_caps_plan() {
        let (n, max_path) = (200_000, 120);
        for w in [1usize, 2, 3, 4, 8] {
            let t = effective_match_threads(n, 3, max_path, Some(2), w);
            assert!(t >= 1 && t <= w, "threads {t} exceed request {w}");
        }
        // Tiny item counts stay serial regardless of a wide request.
        assert_eq!(effective_match_threads(1, 3, max_path, Some(2), 20), 1);
    }

    #[test]
    fn cached_matcher_agrees_with_fresh() {
        let paths = ["src/score.rs", "src/gate.rs", "README.md", "src/main.rs"];
        for needle in ["abc", "score", "src", "UserName", "item.rs"] {
            for typos in [Some(0u16), Some(2), Some(6)] {
                let cfg = neo_frizbee::Config {
                    max_typos: typos,
                    sort: neo_frizbee::SortStrategy::Unsorted,
                    ..Default::default()
                };
                // Warm the TLS template, then compare a cached run vs fresh.
                let cached_first: Vec<(u32, u16)> = with_cached_matcher(needle, &cfg, |m| {
                    m.match_list(&paths)
                        .into_iter()
                        .map(|x| (x.index, x.score))
                        .collect()
                });
                let cached_second: Vec<(u32, u16)> = with_cached_matcher(needle, &cfg, |m| {
                    m.match_list(&paths)
                        .into_iter()
                        .map(|x| (x.index, x.score))
                        .collect()
                });
                let mut fresh = Matcher::new(needle, &cfg);
                let fresh_out: Vec<(u32, u16)> = fresh
                    .match_list(&paths)
                    .into_iter()
                    .map(|x| (x.index, x.score))
                    .collect();
                assert_eq!(cached_first, fresh_out, "needle={needle:?} typos={typos:?}");
                assert_eq!(cached_second, fresh_out, "reuse needle={needle:?}");
            }
        }
    }

    #[test]
    fn capped_max_path_len_matches_exact_below_cap() {
        let mk = |len: usize| {
            let s = "a".repeat(len);
            let item = FileItem::new_raw(0, 0, 0, None, false);
            let (store, strings) =
                crate::simd_path::build_chunked_path_store_from_strings(&[s.clone()], &[item]);
            let mut file = FileItem::new_raw(0, 0, 0, None, false);
            file.set_path(strings[0].clone());
            std::mem::forget(store);
            (file, len)
        };
        let (a, _) = mk(10);
        let (b, _) = mk(100);
        let files = [a, b];
        assert_eq!(max_path_len_of(&files), 100);
        let (c, _) = mk(600);
        let files = [c];
        assert_eq!(max_path_len_of(&files), 512);
    }
}
