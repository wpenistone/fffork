//! Match-set plumbing: windows from frizbee skip_sw, path resolve,
//! candidate gates, and thread dispatch. Score/rank lives in `score`.

use neo_frizbee::Matcher;
use crate::types::FileItem;
use crate::simd_path::{ArenaPtr, MAX_PATH_CHUNKS};

#[derive(Clone, Copy)]
pub(crate) struct MatchWindow {
    pub(crate) file_idx: u32,
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) include_prefix: bool,
    pub(crate) include_exact: bool,
}

/// Packed trim window: 12 B vs 16 B tuple. Paths fit u16 (ChunkedString).
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
    buf: &mut [*const u8; MAX_PATH_CHUNKS],
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

/// Build a frizbee `Matcher` for one search. Construction is not free
/// (prefilter + SW kernels); a thread-local cache would be a typing win but
/// 0.13.3 matcher reuse changed `total_matched` on linux `score` (checksum
/// gate). Keep fresh construction until reuse is proven result-identical.
#[inline]
pub(crate) fn with_cached_matcher<R>(
    needle: &str,
    options: &neo_frizbee::Config,
    run: impl FnOnce(&mut Matcher) -> R,
) -> R {
    let mut matcher = Matcher::new(needle, options);
    run(&mut matcher)
}

thread_local! {
    static HIT_CAPTURE: std::cell::RefCell<Option<Vec<u32>>> =
        const { std::cell::RefCell::new(None) };
}

/// Collect match indices from the next `fuzzy_search` (typing live-narrow).
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

/// Max path byte length over a plain slice (tests and non-cached callers).
#[inline]
pub(crate) fn max_path_len_of(files: &[FileItem]) -> usize {
    files.iter().map(|f| f.path.byte_len as usize).max().unwrap_or(0)
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
        let mut m = 0usize;
        match self {
            FileItems::All(s) => {
                for f in s.iter() {
                    m = m.max(f.path.byte_len as usize);
                }
            }
            FileItems::Filtered { base, idx } => {
                for &i in idx.iter() {
                    m = m.max(base[i as usize].path.byte_len as usize);
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

    /// Corpus/file index for a working-set position (live-narrow hits).
    #[inline]
    pub(crate) fn file_index(&self, index: usize) -> u32 {
        match self {
            FileItems::All(_) => index as u32,
            FileItems::Filtered { idx, .. } => idx[index],
        }
    }
}

/// Resolve a FileItem's chunked path into frizbee's pointer buffer.
/// Returns `Some((chunk_count, byte_len))` or `None` for deleted files.
#[inline]
pub(crate) fn resolve_file_chunks(
    file: &FileItem,
    arena: ArenaPtr,
    buf: &mut [*const u8; MAX_PATH_CHUNKS],
) -> Option<(usize, u16)> {
    if file.is_deleted() {
        return None;
    }
    let ptrs = file.path.resolve_ptrs(arena, buf);
    Some((ptrs.len(), file.path.byte_len))
}

/// Test-only override so thread scaling is actually measurable. Without it
/// the estimate below overwrites the caller's request, which silently makes
/// every "1T / 4T" comparison the same thread count. Read once.
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
    if item_count < 1024 {
        return 1;
    }
    if max_threads <= 1 {
        return 1;
    }
    let selectivity = match max_typos {
        Some(t) if t >= 3 => 25,
        Some(_) => 50,
        None => 100,
    };
    let work = (item_count as u64)
        .saturating_mul(needle_len.max(1) as u64)
        .saturating_mul(max_haystack_len.min(512).div_ceil(16).max(1) as u64)
        .saturating_mul(selectivity)
        / 100;
    (work.div_ceil(crate::machine::match_work_per_thread() as u64) as usize).clamp(1, max_threads.max(1))
}
