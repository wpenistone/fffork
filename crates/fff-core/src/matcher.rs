//! Match-set plumbing: windows from frizbee skip_sw, path resolve,
//! candidate gates, and thread dispatch. Score/rank lives in `score`.

use crate::simd_path::{ArenaPtr, ChunkedString, MAX_FRIZBEE_CHUNKS};
use crate::types::FileItem;
use neo_frizbee::Matcher;

thread_local! {
    static RESOLVE_SCRATCH: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// Copy a chunked path into 16 B-unit pointers for frizbee's resolved API.
// Arena chunks are 32 B deduped, so their sub-slices are not contiguous.
// Frizbee gathers synchronously before the next resolve on this thread,
// one TLS buffer is sound.
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
    RESOLVE_SCRATCH.with(|c| {
        let mut s = c.borrow_mut();
        let padded = n * unit;
        if s.len() < padded {
            s.resize(padded, 0);
        }
        path.copy_window_to(arena, 0, len as u32, &mut s[..len]);
        let base = s.as_ptr();
        for (slot, k) in buf.iter_mut().zip(0..n) {
            *slot = unsafe { base.add(k * unit) };
        }
        Some((n, len as u16))
    })
}

#[derive(Clone, Copy)]
pub(crate) struct MatchWindow {
    pub(crate) file_idx: u32,
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) include_prefix: bool,
    pub(crate) include_exact: bool,
}

// Packed trim window: 12 B vs 16 B tuple. Paths fit u16 (ChunkedString).
#[derive(Clone, Copy, Default)]
pub(crate) struct Trim {
    pub file_idx: u32,
    pub start: u16,
    pub end: u16,
    /// bit0 = include_prefix, bit1 = include_exact
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

pub(crate) fn decode_match_window(m: &neo_frizbee::Match, hay_len: usize) -> MatchWindow {
    // skip_sw packs (start_pos_adj, end_pos) into (score, end_col).
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

pub(crate) fn resolve_file_chunks_prefiltered(
    file: &FileItem,
    arena: ArenaPtr,
    buf: &mut [*const u8; MAX_FRIZBEE_CHUNKS],
    np: &crate::presence::NeedlePresence,
    needle: &[u8],
) -> Option<(usize, u16)> {
    // Sound length reject: LCS ≤ path_len, so shorter paths cannot match.
    if (file.path.byte_len as usize) < np.min_path_len() {
        return None;
    }
    let mask = file.presence_mask();
    if mask != 0 && np.should_check() && !np.allows(mask, needle) {
        return None;
    }
    resolve_file_chunks(file, arena, buf)
}

// Build a frizbee `Matcher` for one search. Construction parses the
// pattern, so a thread-local pristine-template cache clones one per call.
// Every call still runs on a fresh-equivalent instance (`new().clone()`),
// never on a previously-run matcher, which keeps results identical.
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
        if let Some((_, _, t)) = c.iter().find(|(n, o, _)| n == needle && o == options) {
            let mut inst = t.clone();
            return run(&mut inst);
        }
        let template = Matcher::new(needle, options);
        let mut inst = template.clone();
        let r = run(&mut inst);
        if c.len() >= 4 {
            c.remove(0);
        }
        c.push((needle.to_string(), *options, template));
        r
    })
}

thread_local! {
    static HIT_CAPTURE: std::cell::RefCell<Option<Vec<u32>>> =
        const { std::cell::RefCell::new(None) };
}

// Collect match indices from the next `fuzzy_search` (typing live-narrow).
pub(crate) fn capture_hit_indices(iter: impl Iterator<Item = u32>) {
    HIT_CAPTURE.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            v.clear();
            v.extend(iter);
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

// Max path byte length over a plain slice. Test helper; production uses
// the incremental `PathLenCache`.
#[cfg(test)]
#[inline]
pub(crate) fn max_path_len_of(files: &[FileItem]) -> usize {
    let mut m = 0usize;
    for f in files {
        // Callers clamp to 512, so stop at the cap instead of walking 227k.
        m = m.max(f.path.byte_len as usize);
        if m >= 512 {
            return 512;
        }
    }
    m
}

pub(crate) enum FileItems<'a> {
    All(&'a [FileItem]),
    /// Compact candidate list over a base slice — indices avoid a
    /// `Vec<&FileItem>` pointer-chase on the live-narrow typing path.
    /// `idx` is owned so it can carry candidates without borrowing the caller.
    Filtered {
        base: &'a [FileItem],
        idx: Vec<u32>,
    },
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
    pub(crate) fn max_path_len(&self) -> usize {
        // `effective_match_threads` clamps to 512, so the exact max above
        // the cap never changes dispatch. Early-out keeps this O(1)-ish.
        let mut m = 0usize;
        match self {
            FileItems::All(s) => {
                for f in s.iter() {
                    m = m.max(f.path.byte_len as usize);
                    if m >= 512 {
                        return 512;
                    }
                }
            }
            FileItems::Filtered { base, idx } => {
                for &i in idx.iter() {
                    m = m.max(base[i as usize].path.byte_len as usize);
                    if m >= 512 {
                        return 512;
                    }
                }
            }
        }
        m
    }

    #[inline]
    pub(crate) fn index(&self, index: usize) -> &'a FileItem {
        match self {
            FileItems::All(s) => &s[index],
            FileItems::Filtered { base, idx } => &base[idx[index] as usize],
        }
    }

    // Corpus/file index for a working-set position (live-narrow hits).
    #[inline]
    pub(crate) fn file_index(&self, index: usize) -> u32 {
        match self {
            FileItems::All(_) => index as u32,
            FileItems::Filtered { idx, .. } => idx[index],
        }
    }
}

// Resolve a FileItem's chunked path into frizbee's pointer buffer.
// Returns `Some((chunk_count, byte_len))` or `None` for deleted files.
#[inline]
pub(crate) fn resolve_file_chunks(
    file: &FileItem,
    arena: ArenaPtr,
    buf: &mut [*const u8; MAX_FRIZBEE_CHUNKS],
) -> Option<(usize, u16)> {
    if file.is_deleted() {
        return None;
    }
    resolve_frizbee_bytes(&file.path, arena, buf)
}

// Test-only override so thread scaling is actually measurable. Without it
// the estimate below overwrites the caller's request, which silently makes
// every "1T / 4T" comparison the same thread count. Read once.
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
