//! Query intent + evidence model for file finding.
//!
//! Classifies the job from query shape and context; exposes evidence
//! the ranker mixes per intent.

/// What the user is trying to do. One query → one primary intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntent {
    // Empty or whitespace — show working set (MRU, buffers, current dir).
    WorkingSet,
    // `foo.rs` / `user_service.ts` — name + optional extension.
    FileName,
    // `src/auth/login` / `../x/y` — path structure.
    Path,
    // `handler`, `router`, `login` — concept / role / word.
    Concept,
    // Looks like a typo of a name (`filw_pikcer`).
    Typo,
}

/// Parsed name/ext split for filename-shaped queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameExt<'a> {
    pub stem: &'a str,
    /// Including the dot, e.g. `".rs"`. Empty if the query has no ext.
    pub ext: &'a str,
}

/// Path segment list (no empty segments).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathTokens<'a> {
    pub segments: Vec<&'a str>,
    /// Query ended with `/` — user wants a directory-ish hit.
    pub ends_with_sep: bool,
}

/// Structural role of a path, cheaply derived from the filename.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileRole {
    Entrypoint,
    Test,
    Type,
    Config,
    Style,
    Doc,
    Other,
}

/// Everything the ranker needs to score one hit for a given intent.
#[derive(Debug, Clone, Copy, Default)]
pub struct Evidence {
    /// Exact filename (case-insensitive) match with the query stem+ext.
    pub exact_filename: bool,
    /// Filename contains the stem as a contiguous run (word-ish).
    pub filename_substring: bool,
    /// Extension equals the query extension.
    pub ext_match: bool,
    /// Query ext present but this file's ext differs.
    pub ext_mismatch: bool,
    /// Last path segment equals the last query segment.
    pub last_segment_match: bool,
    /// All query path segments appear in order as whole path components.
    pub segment_sequence: bool,
    /// Fraction of query segments that land on path boundaries (0..=100).
    pub segment_boundary_pct: u8,
    /// Word/camel tokens from the filename cover the query words (0..=100).
    pub word_coverage: u8,
    /// Same directory as the current file.
    pub same_dir: bool,
    /// Sibling / related-name hit (test, css, module).
    pub related: bool,
    /// File role matches the query's implied role (e.g. `test` query).
    pub role_match: bool,
}

/// Classify the query. `current_file` is the buffer path if any.
pub fn detect_intent(raw: &str, current_file: Option<&str>, _has_mru: bool) -> QueryIntent {
    let q = raw.trim();
    if q.is_empty() {
        return QueryIntent::WorkingSet;
    }
    if q.contains('/') || q.contains('\\') {
        return QueryIntent::Path;
    }
    // `name.ext` — a single dot, not leading, plausible extension.
    if let Some(_ne) = split_name_ext(q) {
        // Long random strings with a dot still count as name.
        return QueryIntent::FileName;
    }
    // Typo heuristic: high oddness (rare double letters, qwerty near-misses
    // are expensive; use a cheap proxy — long token, no separators, few vowels
    // relative to length, or non-dictionary-ish mix). Keep it conservative.
    if looks_like_typo(q) {
        return QueryIntent::Typo;
    }
    // Short common words and identifiers are concept/role queries.
    if current_file.is_some() && q.len() <= 3 {
        // Tiny query next to a file — prefer working-set-ish boosts via Concept
        // (recency already applies).
        return QueryIntent::Concept;
    }
    QueryIntent::Concept
}

/// Split `foo.bar.rs` into (`foo.bar`, `".rs"`). Returns None when there is
/// no extension (leading dot, trailing dot, or no dot).
pub fn split_name_ext(q: &str) -> Option<NameExt<'_>> {
    if !q.contains('.') || q.starts_with('.') || q.ends_with('.') || q.contains('/') {
        return None;
    }
    let i = q.rfind('.')?;
    if i == 0 || i + 1 >= q.len() {
        return None;
    }
    let (stem, ext_dot) = q.split_at(i);
    if stem.is_empty() {
        return None;
    }
    // Extension should be short-ish (1–8 chars) and alphanumeric.
    let ext = &ext_dot[1..];
    if ext.is_empty() || ext.len() > 8 || !ext.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    Some(NameExt { stem, ext: ext_dot })
}

/// Path tokens: `src/auth/login/` → ["src","auth","login"], ends_with_sep=true.
pub fn split_path_tokens(q: &str) -> PathTokens<'_> {
    let sep = if q.contains('\\') { '\\' } else { '/' };
    let ends_with_sep = q.ends_with(sep);
    let segments: Vec<&str> = q
        .split(sep)
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    PathTokens {
        segments,
        ends_with_sep,
    }
}

/// Filename word tokens: `user_service` / `UserService` / `user-service`
/// → ["user","service"].
pub fn filename_words(name: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for_each_word(name, |w| out.push(w));
    out
}

/// Visit word tokens without allocating.
pub fn for_each_word<'a>(name: &'a str, mut f: impl FnMut(&'a str)) {
    let _ = for_each_word_until(name, |w| {
        f(w);
        true
    });
}

// One load: separator, `A..=Z`, `a..=z`, or digit.
const WORD_SEP: u8 = 1;
const WORD_UPPER: u8 = 2;
const WORD_LOWER: u8 = 4;
const WORD_DIGIT: u8 = 8;
const WORD_CLASS: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let b = i as u8;
        let class = if b == b'_' || b == b'-' || b == b'.' || b == b' ' {
            WORD_SEP
        } else if b >= b'A' && b <= b'Z' {
            WORD_UPPER
        } else if b >= b'a' && b <= b'z' {
            WORD_LOWER
        } else if b >= b'0' && b <= b'9' {
            WORD_DIGIT
        } else {
            0
        };
        t[i as usize] = class;
        i += 1;
    }
    t
};

// `f` returns false to stop. Tokens match `for_each_word`.
fn for_each_word_until<'a>(name: &'a str, mut f: impl FnMut(&'a str) -> bool) -> bool {
    let mut start = 0usize;
    let bytes = name.as_bytes();
    let mut prev_lower_or_digit = false;
    let mut prev_upper = false;
    for (i, &b) in bytes.iter().enumerate() {
        let class = WORD_CLASS[b as usize];
        let next_class = if i + 1 < bytes.len() {
            WORD_CLASS[bytes[i + 1] as usize]
        } else {
            0
        };
        let is_sep = class & WORD_SEP != 0;
        let is_upper = class & WORD_UPPER != 0;
        let is_lower_or_digit = class & (WORD_LOWER | WORD_DIGIT) != 0;
        let camel_break = prev_lower_or_digit && is_upper;
        let acronym_break = prev_upper && is_upper && next_class & WORD_LOWER != 0;
        if i > start && (is_sep || camel_break || acronym_break) {
            if !is_sep {
                if i > start && !f(&name[start..i]) {
                    return false;
                }
                start = i;
            } else {
                if i > start && !f(&name[start..i]) {
                    return false;
                }
                start = i + 1;
            }
        }
        if is_sep {
            prev_lower_or_digit = false;
            prev_upper = false;
        } else {
            prev_lower_or_digit = is_lower_or_digit;
            prev_upper = is_upper;
        }
    }
    if start < name.len() && !f(&name[start..]) {
        return false;
    }
    true
}

// Cheap typo proxy: long identifier, mixed case noise, or uncommon bigrams.
// Conservative — only fires when the string looks "wrong".
fn looks_like_typo(q: &str) -> bool {
    if q.len() < 4 || q.len() > 40 {
        return false;
    }
    if q.chars().any(|c| c.is_whitespace()) {
        return false;
    }
    // Vowel ratio far from English-ish.
    let vowels = q
        .chars()
        .filter(|c| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U'))
        .count();
    let ratio = vowels as f32 / q.len() as f32;
    #[allow(clippy::manual_range_contains)]
    if ratio < 0.08 || ratio > 0.55 {
        // Still allow `xkcd`-style; require also no separators/underscore.
        return !q.contains('_') && !q.contains('-');
    }
    false
}
// `lower` is ASCII-lowercase query bytes. Haystack folds; the needle does not.
#[inline]
fn eq_ascii_lower(hay: &[u8], lower: &[u8]) -> bool {
    hay.len() == lower.len()
        && unsafe { crate::simd_string_utils::case::eq_lowered_case(hay.as_ptr(), lower) }
}

#[inline]
fn contains_ascii_lower(hay: &str, needle_lower: &str) -> bool {
    let h = hay.as_bytes();
    let n = needle_lower.as_bytes();
    if n.is_empty() {
        return true;
    }
    if n.len() > h.len() {
        return false;
    }
    let lo = n[0];
    let up = lo.to_ascii_uppercase();
    let mut rest = h;
    while rest.len() >= n.len() {
        let span = &rest[..rest.len() - n.len() + 1];
        let rel = if lo == up {
            memchr::memchr(lo, span)
        } else {
            memchr::memchr2(lo, up, span)
        };
        let Some(rel) = rel else {
            return false;
        };
        if eq_ascii_lower(&rest[rel..rel + n.len()], n) {
            return true;
        }
        rest = &rest[rel + 1..];
    }
    false
}

// Needle is ASCII-lowercase. Haystack folds; the needle does not.
#[inline]
pub(crate) fn contains_ignore_ascii(hay: &str, needle: &str) -> bool {
    contains_ascii_lower(hay, needle)
}

// Suffix is ASCII-lowercase. Haystack folds; the suffix does not.
#[inline]
pub(crate) fn ends_with_ignore_ascii(hay: &str, suffix: &str) -> bool {
    let h = hay.as_bytes();
    let s = suffix.as_bytes();
    h.len() >= s.len() && eq_ascii_lower(&h[h.len() - s.len()..], s)
}

// `options` are ASCII-lowercase.
#[inline]
fn matches_ignore_ascii(hay: &str, options: &[&str]) -> bool {
    let h = hay.as_bytes();
    options.iter().any(|o| eq_ascii_lower(h, o.as_bytes()))
}

/// File role from a filename (not the full path).
pub fn role_of_filename(fname: &str) -> FileRole {
    if is_test_filename(fname) {
        FileRole::Test
    } else if is_entrypoint_filename(fname) {
        FileRole::Entrypoint
    } else if is_type_filename(fname) {
        FileRole::Type
    } else if is_config_filename(fname) {
        FileRole::Config
    } else if is_style_filename(fname) {
        FileRole::Style
    } else if is_doc_filename(fname) {
        FileRole::Doc
    } else {
        FileRole::Other
    }
}

fn role_is(fname: &str, want: FileRole) -> bool {
    let hit = match want {
        FileRole::Test => return is_test_filename(fname),
        FileRole::Entrypoint => is_entrypoint_filename(fname),
        FileRole::Type => is_type_filename(fname),
        FileRole::Config => is_config_filename(fname),
        FileRole::Style => is_style_filename(fname),
        FileRole::Doc => is_doc_filename(fname),
        FileRole::Other => return role_of_filename(fname) == FileRole::Other,
    };
    if !hit {
        return false;
    }
    match want {
        FileRole::Entrypoint => !is_test_filename(fname),
        FileRole::Type => !is_test_filename(fname) && !is_entrypoint_filename(fname),
        FileRole::Config => {
            !is_test_filename(fname) && !is_entrypoint_filename(fname) && !is_type_filename(fname)
        }
        FileRole::Style => {
            !is_test_filename(fname)
                && !is_entrypoint_filename(fname)
                && !is_type_filename(fname)
                && !is_config_filename(fname)
        }
        FileRole::Doc => {
            !is_test_filename(fname)
                && !is_entrypoint_filename(fname)
                && !is_type_filename(fname)
                && !is_config_filename(fname)
                && !is_style_filename(fname)
        }
        FileRole::Test | FileRole::Other => true,
    }
}

fn is_test_filename(fname: &str) -> bool {
    let n = fname.len();
    if n < 6 {
        return false;
    }
    if n >= 8 {
        let hit = match fname.as_bytes().last() {
            Some(b's' | b'S') => {
                ends_with_ignore_ascii(fname, "_test.rs")
                    || ends_with_ignore_ascii(fname, ".test.ts")
                    || ends_with_ignore_ascii(fname, ".test.js")
                    || ends_with_ignore_ascii(fname, ".spec.ts")
                    || ends_with_ignore_ascii(fname, ".spec.js")
                    || ends_with_ignore_ascii(fname, ".spec.rs")
            }
            Some(b'x' | b'X') => ends_with_ignore_ascii(fname, ".test.tsx"),
            _ => false,
        };
        if hit {
            return true;
        }
    }
    // `_` and `/` have no case pair; one pass covers both markers.
    contains_test_marker(fname.as_bytes())
}

fn contains_test_marker(hay: &[u8]) -> bool {
    let mut rest = hay;
    let mut base = 0usize;
    while let Some(rel) = memchr::memchr2(b'_', b'/', rest) {
        let i = base + rel;
        if rest[rel] == b'_' && i + 6 <= hay.len() && eq_ascii_lower(&hay[i..i + 6], b"_test.") {
            return true;
        }
        if rest[rel] == b'/' && i + 7 <= hay.len() && eq_ascii_lower(&hay[i..i + 7], b"/tests/") {
            return true;
        }
        let next = rel + 1;
        if next > rest.len() {
            break;
        }
        rest = &rest[next..];
        base += next;
    }
    false
}

fn is_entrypoint_filename(fname: &str) -> bool {
    let b = fname.as_bytes();
    let Some(&first) = b.first() else {
        return false;
    };
    // The open set is grouped by the first byte so other names miss in one load.
    match first {
        b'm' | b'M' => matches_ignore_ascii(
            fname,
            &[
                "mod.rs", "main.rs", "main.go", "main.py", "main.cpp", "main.c", "main.ts",
                "main.js",
            ],
        ),
        b'i' | b'I' => {
            matches_ignore_ascii(fname, &["index.ts", "index.tsx", "index.js", "index.jsx"])
        }
        b'a' | b'A' => matches_ignore_ascii(fname, &["app.rs"]),
        b'l' | b'L' => matches_ignore_ascii(fname, &["lib.rs"]),
        b'_' => matches_ignore_ascii(fname, &["__init__.py"]),
        _ => false,
    }
}

fn is_type_filename(fname: &str) -> bool {
    let suffix = fname
        .as_bytes()
        .last()
        .is_some_and(|b| matches!(b, b's' | b'S'));
    (suffix
        && (ends_with_ignore_ascii(fname, ".d.ts")
            || ends_with_ignore_ascii(fname, "_pb.rs")
            || ends_with_ignore_ascii(fname, "type.ts")
            || ends_with_ignore_ascii(fname, "type.rs")))
        || contains_ignore_ascii(fname, "types")
}

fn is_config_filename(fname: &str) -> bool {
    match fname.as_bytes().last() {
        Some(b'l' | b'L') => {
            ends_with_ignore_ascii(fname, ".toml")
                || ends_with_ignore_ascii(fname, ".yaml")
                || ends_with_ignore_ascii(fname, ".yml")
        }
        Some(b'n' | b'N') => ends_with_ignore_ascii(fname, ".json"),
        Some(b'i' | b'I') => ends_with_ignore_ascii(fname, ".ini"),
        Some(b'e' | b'E') => matches_ignore_ascii(fname, &["makefile", "dockerfile"]),
        _ => false,
    }
}

fn is_style_filename(fname: &str) -> bool {
    match fname.as_bytes().last() {
        Some(b's' | b'S') => {
            ends_with_ignore_ascii(fname, ".css")
                || ends_with_ignore_ascii(fname, ".scss")
                || ends_with_ignore_ascii(fname, ".less")
        }
        Some(b'l' | b'L') => ends_with_ignore_ascii(fname, ".styl"),
        _ => false,
    }
}

fn is_doc_filename(fname: &str) -> bool {
    match fname.as_bytes().last() {
        Some(b'd' | b'D') => ends_with_ignore_ascii(fname, ".md"),
        Some(b't' | b'T') => {
            ends_with_ignore_ascii(fname, ".rst") || ends_with_ignore_ascii(fname, ".txt")
        }
        _ => false,
    }
}

/// Implied role from a concept query (e.g. "test", "config", "types").
pub fn implied_role_from_query(q: &str) -> Option<FileRole> {
    if matches_ignore_ascii(q, &["test", "tests", "spec", "specs"]) {
        Some(FileRole::Test)
    } else if matches_ignore_ascii(q, &["config", "conf", "settings"]) {
        Some(FileRole::Config)
    } else if matches_ignore_ascii(q, &["types", "type", "defs"]) {
        Some(FileRole::Type)
    } else if matches_ignore_ascii(q, &["css", "style", "styles"]) {
        Some(FileRole::Style)
    } else if matches_ignore_ascii(q, &["doc", "docs", "readme"]) {
        Some(FileRole::Doc)
    } else if matches_ignore_ascii(q, &["main", "index", "mod", "app", "lib"]) {
        Some(FileRole::Entrypoint)
    } else {
        None
    }
}

/// Segment-boundary coverage: how many query segments match path components.
pub fn segment_boundary_evidence(path: &str, q: &PathTokens<'_>) -> (bool, u8, bool) {
    segment_boundary_with(path, &q.segments, |c, s| {
        c.len() == s.len() && crate::simd_path::eq_ignore_ascii_bytes(c.as_bytes(), s.as_bytes())
    })
}

fn segment_boundary_with(
    path: &str,
    segments: &[&str],
    mut eq_seg: impl FnMut(&str, &str) -> bool,
) -> (bool, u8, bool) {
    if segments.is_empty() {
        return (false, 0, false);
    }
    let bytes = path.as_bytes();
    let last_q = segments.last().copied().unwrap_or("");
    let mut qi = 0usize;
    let mut last = "";
    let mut start = 0usize;
    let mut rest = bytes;
    let mut base = 0usize;
    loop {
        let Some(rel) = memchr::memchr2(b'/', b'\\', rest) else {
            if start < bytes.len() {
                let c = &path[start..];
                last = c;
                if qi < segments.len() && eq_seg(c, segments[qi]) {
                    qi += 1;
                }
            }
            break;
        };
        let i = base + rel;
        if bytes[i] == b'\\' {
            return segment_boundary_sep(path, segments, b'\\', &mut eq_seg);
        }
        if start < i {
            let c = &path[start..i];
            last = c;
            if qi < segments.len() && eq_seg(c, segments[qi]) {
                qi += 1;
            }
        }
        start = i + 1;
        let next = rel + 1;
        rest = &rest[next..];
        base += next;
    }
    let last_match = !last.is_empty() && eq_seg(last, last_q);
    let sequence = qi == segments.len();
    let pct = (qi * 100 / segments.len()) as u8;
    (last_match, pct, sequence)
}

// A backslash selects `\` only, including components already seen as slashes.
fn segment_boundary_sep(
    path: &str,
    segments: &[&str],
    sep: u8,
    eq_seg: &mut impl FnMut(&str, &str) -> bool,
) -> (bool, u8, bool) {
    let last_q = segments.last().copied().unwrap_or("");
    let mut qi = 0usize;
    let mut last = "";
    let bytes = path.as_bytes();
    let mut start = 0usize;
    let mut rest = bytes;
    let mut base = 0usize;
    loop {
        let Some(rel) = memchr::memchr(sep, rest) else {
            if start < bytes.len() {
                let c = &path[start..];
                last = c;
                if qi < segments.len() && eq_seg(c, segments[qi]) {
                    qi += 1;
                }
            }
            break;
        };
        let i = base + rel;
        if start < i {
            let c = &path[start..i];
            last = c;
            if qi < segments.len() && eq_seg(c, segments[qi]) {
                qi += 1;
            }
        }
        start = i + 1;
        let next = rel + 1;
        rest = &rest[next..];
        base += next;
    }
    let last_match = !last.is_empty() && eq_seg(last, last_q);
    let sequence = qi == segments.len();
    let pct = (qi * 100 / segments.len()) as u8;
    (last_match, pct, sequence)
}

// `part` is a subslice of `query`. `lower` is that query ASCII-lowercased.
#[inline]
pub(crate) fn query_part_lower<'a>(query: &str, lower: &'a str, part: &str) -> &'a [u8] {
    let start = (part.as_ptr() as usize).wrapping_sub(query.as_ptr() as usize);
    &lower.as_bytes()[start..start + part.len()]
}

/// Word coverage: percent of query tokens found in filename words.
/// No heap allocs — query tokens fit a small stack array.
/// Test-only helper: production uses the prepared-query variant below.
#[cfg(test)]
pub fn word_coverage(fname: &str, query: &str) -> u8 {
    let mut qwords: [&str; 12] = [""; 12];
    let mut qn = 0usize;
    for_each_word(query, |w| {
        if qn < qwords.len() {
            qwords[qn] = w;
            qn += 1;
        }
    });
    word_coverage_prepared(fname, &qwords[..qn])
}

fn word_coverage_one_lower(fname: &str, q_lower: &[u8]) -> u8 {
    let mut hit = false;
    for_each_word_until(fname, |fw| {
        let h = fw.as_bytes();
        if h.len() >= q_lower.len() && eq_ascii_lower(&h[..q_lower.len()], q_lower) {
            hit = true;
            return false;
        }
        true
    });
    if hit { 100 } else { 0 }
}

pub(crate) fn word_coverage_prepared(fname: &str, qwords: &[&str]) -> u8 {
    let n = qwords.len();
    if n == 0 {
        return 0;
    }
    // One filename walk for every query word. Above 128 words, rescan.
    if n > 128 {
        return word_coverage_each_word(fname, qwords);
    }
    let mut seen = 0u128;
    let mut hits = 0usize;
    for_each_word_until(fname, |fw| {
        for (i, qw) in qwords.iter().enumerate() {
            let bit = 1u128 << (i as u32);
            if seen & bit != 0 {
                continue;
            }
            // Exact equality is a prefix match.
            if starts_with_ignore_ascii(fw, qw) {
                seen |= bit;
                hits += 1;
                if hits == n {
                    return false;
                }
            }
        }
        true
    });
    // At most one hit per query word, so this percent is already <= 100.
    (hits * 100 / n) as u8
}

// Query words are already ASCII-lowercase. Folds filename bytes only.
fn word_coverage_folded(fname: &str, qwords: &[&[u8]]) -> u8 {
    let n = qwords.len();
    if n == 0 {
        return 0;
    }
    let mut seen = 0u128;
    let mut hits = 0usize;
    for_each_word_until(fname, |fw| {
        let h = fw.as_bytes();
        for (i, qw) in qwords.iter().enumerate() {
            let bit = 1u128 << (i as u32);
            if seen & bit != 0 {
                continue;
            }
            if qw.len() <= h.len() && eq_ascii_lower(&h[..qw.len()], qw) {
                seen |= bit;
                hits += 1;
                if hits == n {
                    return false;
                }
            }
        }
        true
    });
    (hits * 100 / n) as u8
}

fn word_coverage_each_word(fname: &str, qwords: &[&str]) -> u8 {
    let mut hits = 0usize;
    for qw in qwords {
        let mut found = false;
        for_each_word(fname, |fw| {
            if found {
                return;
            }
            if starts_with_ignore_ascii(fw, qw) {
                found = true;
            }
        });
        if found {
            hits += 1;
        }
    }
    (hits * 100 / qwords.len()) as u8
}

// Case-insensitive prefix test without allocating.
fn starts_with_ignore_ascii(hay: &str, prefix: &str) -> bool {
    let h = hay.as_bytes();
    let p = prefix.as_bytes();
    if p.len() > h.len() {
        return false;
    }
    crate::simd_path::eq_ignore_ascii_bytes(&h[..p.len()], p)
}

/// Mix evidence into a score delta for a given intent. `base` is the SW
/// score — deltas are relative so they scale with match strength.
pub fn evidence_bonus(intent: QueryIntent, ev: &Evidence, base: i32) -> i32 {
    let b = base.max(1);
    let mut x = 0i32;
    match intent {
        QueryIntent::FileName => {
            if ev.exact_filename {
                x += b; // 100%
            } else if ev.filename_substring {
                x += b / 3;
            }
            if ev.ext_match {
                x += b / 5;
            }
            if ev.ext_mismatch {
                x -= b / 4;
            }
            x += (ev.word_coverage as i32 * b) / 400;
        }
        QueryIntent::Path => {
            if ev.segment_sequence {
                x += b / 2;
            }
            x += (ev.segment_boundary_pct as i32 * b) / 200;
            if ev.last_segment_match {
                x += b / 5;
            }
            if ev.ext_match {
                x += b / 10;
            }
        }
        QueryIntent::Concept | QueryIntent::Typo => {
            if ev.exact_filename {
                x += b / 2;
            }
            x += (ev.word_coverage as i32 * b) / 250;
            if ev.role_match {
                x += b / 4;
            }
            if ev.filename_substring {
                x += b / 6;
            }
            if ev.related {
                x += b / 10;
            }
            if ev.same_dir {
                x += b / 10;
            }
            if ev.ext_match {
                x += b / 20;
            }
        }
        QueryIntent::WorkingSet => {
            if ev.same_dir {
                x += b / 5;
            }
            if ev.related {
                x += b / 8;
            }
        }
    }
    x
}

/// Query-side state built once per search. Per-file evidence borrows it
/// and does not allocate.
pub struct PreparedQuery<'a> {
    pub query: &'a str,
    pub lower: String,
    pub name_ext: Option<NameExt<'a>>,
    pub wants_role: Option<FileRole>,
    pub qwords: [&'a str; 12],
    /// Start of `qwords[i]` in both `query` and `lower`.
    pub qword_at: [u32; 12],
    pub qword_count: usize,
    pub path_tokens: Option<PathTokens<'a>>,
}

// The folded query. The next search reuses this allocation.
thread_local! {
    static LOWER_BUF: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

impl<'a> PreparedQuery<'a> {
    pub fn new(query: &'a str) -> Self {
        let query = query.trim();
        let name_ext = split_name_ext(query);
        let mut lower = LOWER_BUF.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
        lower.clear();
        lower.push_str(query);
        lower.make_ascii_lowercase();
        let mut qwords: [&'a str; 12] = [""; 12];
        let mut qword_at = [0u32; 12];
        let mut qword_count = 0usize;
        let qptr = query.as_ptr() as usize;
        for_each_word(query, |w| {
            if qword_count < qwords.len() {
                qwords[qword_count] = w;
                qword_at[qword_count] = (w.as_ptr() as usize).wrapping_sub(qptr) as u32;
                qword_count += 1;
            }
        });
        let path_tokens = if query.contains('/') || query.contains('\\') {
            Some(split_path_tokens(query))
        } else {
            None
        };
        Self {
            query,
            lower,
            name_ext,
            wants_role: implied_role_from_query(query),
            qwords,
            qword_at,
            qword_count,
            path_tokens,
        }
    }
}

impl Drop for PreparedQuery<'_> {
    fn drop(&mut self) {
        let mut lower = std::mem::take(&mut self.lower);
        lower.clear();
        LOWER_BUF.with(|slot| {
            let slot = &mut *slot.borrow_mut();
            if lower.capacity() > slot.capacity() {
                *slot = lower;
            }
        });
    }
}

/// Full evidence bundle for one file vs one query + intent.
pub fn evidence_for(
    relative_path: &str,
    filename: &str,
    query: &str,
    intent: QueryIntent,
    current_file: Option<&str>,
) -> Evidence {
    let pq = PreparedQuery::new(query);
    evidence_prepared(&pq, relative_path, filename, intent, current_file)
}

/// [`evidence_for`] against a pre-computed query. Allocation-free per call.
pub fn evidence_prepared(
    pq: &PreparedQuery<'_>,
    relative_path: &str,
    filename: &str,
    intent: QueryIntent,
    current_file: Option<&str>,
) -> Evidence {
    let mut ev = Evidence::default();
    let q = pq.query;
    if q.is_empty() && intent != QueryIntent::WorkingSet {
        return ev;
    }

    // Proximity / related work even with no query text (working set).
    // Only Concept/Typo/WorkingSet bonuses read these; FileName and Path
    // never do, so skip the rsplit+compare per candidate there.
    if let Some(cur) = current_file
        && matches!(
            intent,
            QueryIntent::Concept | QueryIntent::Typo | QueryIntent::WorkingSet
        )
    {
        let sep = if relative_path.contains('\\') {
            '\\'
        } else {
            '/'
        };
        let cur_dir = cur.rsplit_once(sep).map(|(d, _)| d).unwrap_or("");
        let hit_dir = relative_path.rsplit_once(sep).map(|(d, _)| d).unwrap_or("");
        ev.same_dir = !cur_dir.is_empty()
            && cur_dir.len() == hit_dir.len()
            && crate::simd_path::eq_ignore_ascii_bytes(cur_dir.as_bytes(), hit_dir.as_bytes());
        let cur_fname = cur.rsplit(['/', '\\']).next().unwrap_or(cur);
        ev.related = is_related_filename(filename, cur_fname);
    }
    if q.is_empty() {
        return ev;
    }

    // Name / ext. Path bonuses read ext_match only; exact_filename and
    // filename_substring are dead there, so skip those scans.
    let want_name = intent != QueryIntent::Path;
    if let Some(ne) = pq.name_ext.as_ref() {
        let q_stem_len = ne.stem.len();
        let q_stem = &pq.lower.as_bytes()[..q_stem_len];
        let q_ext = &pq.lower.as_bytes()[q_stem_len + 1..];
        let (stem, ext, has_dot) = stem_and_ext(filename);
        // One dot split. Stem plus extension is the whole basename.
        ev.ext_match = has_dot && eq_ascii_lower(ext.as_bytes(), q_ext);
        ev.ext_mismatch = !ev.ext_match && has_dot;
        if want_name {
            let stem_matches = eq_ascii_lower(stem.as_bytes(), q_stem);
            ev.exact_filename = stem_matches && ev.ext_match;
            let stem_l = &pq.lower[..q_stem_len];
            ev.filename_substring = stem_matches || contains_ascii_lower(filename, stem_l);
        }
    } else if want_name {
        // Bare name: exact stem or basename is a substring, so a miss skips both compares.
        let sub = contains_ascii_lower(filename, &pq.lower);
        ev.filename_substring = sub;
        if sub {
            let q_lower = pq.lower.as_bytes();
            ev.exact_filename = if eq_ascii_lower(filename.as_bytes(), q_lower) {
                true
            } else {
                let (stem, _, has_dot) = stem_and_ext(filename);
                has_dot && eq_ascii_lower(stem.as_bytes(), q_lower)
            };
        }
    }

    // Path structure
    #[allow(clippy::collapsible_if)]
    if intent == QueryIntent::Path {
        if let Some(pt) = pq.path_tokens.as_ref() {
            let query = pq.query;
            let lower = pq.lower.as_str();
            let (last_match, pct, sequence) =
                segment_boundary_with(relative_path, &pt.segments, |c, seg| {
                    eq_ascii_lower(c.as_bytes(), query_part_lower(query, lower, seg))
                });
            ev.last_segment_match = last_match;
            ev.segment_boundary_pct = pct;
            ev.segment_sequence = sequence;
        }
    }

    // Words / camel. Only FileName/Concept/Typo bonuses read this.
    if matches!(
        intent,
        QueryIntent::FileName | QueryIntent::Concept | QueryIntent::Typo
    ) {
        let n = pq.qword_count;
        let words = &pq.qwords[..n];
        // That token is the whole query, so a substring miss has no word prefix.
        let whole = n == 1 && words[0].len() == pq.lower.len();
        ev.word_coverage = if whole && !ev.filename_substring {
            0
        } else if whole {
            word_coverage_one_lower(filename, pq.lower.as_bytes())
        } else {
            let mut lows: [&[u8]; 12] = [&[]; 12];
            let lb = pq.lower.as_bytes();
            for i in 0..n {
                let s = pq.qword_at[i] as usize;
                let len = words[i].len();
                lows[i] = &lb[s..s + len];
            }
            word_coverage_folded(filename, &lows[..n])
        };
    }

    if let Some(want) = pq.wants_role {
        // Asked pattern first. Higher roles override it; lower roles do not.
        ev.role_match = role_is(filename, want);
    }

    ev
}

fn stem_and_ext(filename: &str) -> (&str, &str, bool) {
    match filename.rsplit_once('.') {
        Some((stem, ext)) => (stem, ext, true),
        None => (filename, "", false),
    }
}

fn filename_stem(filename: &str) -> &str {
    stem_and_ext(filename).0
}

/// True when `cand` is a conventional sibling of `cur` (same stem, different role).
pub fn is_related_filename(cand: &str, cur_fname: &str) -> bool {
    is_related_key(cand, &related_key(cur_fname))
}

// Lowercased stem of the open file, before and after role suffixes.
pub(crate) struct RelatedKey {
    full: Vec<u8>,
    stripped: Vec<u8>,
}

pub(crate) fn related_key(cur_fname: &str) -> RelatedKey {
    let stem = filename_stem(cur_fname);
    let mut full = vec![0u8; stem.len()];
    crate::simd_path::fold_ascii_needle(stem.as_bytes(), &mut full);
    let stripped_src = strip_sibling_suffix(stem);
    let mut stripped = vec![0u8; stripped_src.len()];
    crate::simd_path::fold_ascii_needle(stripped_src.as_bytes(), &mut stripped);
    RelatedKey { full, stripped }
}

pub(crate) fn is_related_key(cand: &str, key: &RelatedKey) -> bool {
    if key.full.is_empty() {
        return false;
    }
    let stem = filename_stem(cand);
    if eq_ascii_lower(stem.as_bytes(), &key.full) {
        return false;
    }
    eq_ascii_lower(strip_sibling_suffix(stem).as_bytes(), &key.stripped)
}

fn strip_sibling_suffix(mut s: &str) -> &str {
    for pat in [".test", ".spec", ".module", "_test", ".stories"] {
        while s.len() >= pat.len() {
            let n = s.len() - pat.len();
            // Byte cuts can land inside a multibyte char (CJK stems);
            // the ASCII suffix cannot match there, so keep looking.
            if !s.is_char_boundary(n) || !eq_ascii_lower(s[n..].as_bytes(), pat.as_bytes()) {
                break;
            }
            s = &s[..n];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_shapes() {
        assert_eq!(detect_intent("", None, true), QueryIntent::WorkingSet);
        assert_eq!(detect_intent("src/auth", None, true), QueryIntent::Path);
        assert_eq!(
            detect_intent("user_service.rs", None, true),
            QueryIntent::FileName
        );
        assert_eq!(detect_intent("handler", None, true), QueryIntent::Concept);
    }

    #[test]
    fn name_ext_split() {
        let ne = split_name_ext("user_service.rs").unwrap();
        assert_eq!(ne.stem, "user_service");
        assert_eq!(ne.ext, ".rs");
        assert!(split_name_ext("README").is_none());
        assert!(split_name_ext(".gitignore").is_none());
        assert!(split_name_ext("src/a.rs").is_none());
    }

    #[test]
    fn filename_words_camel_and_snake() {
        assert_eq!(filename_words("user_service"), vec!["user", "service"]);
        assert_eq!(filename_words("UserService"), vec!["User", "Service"]);
        assert_eq!(
            filename_words("XMLHttpRequest"),
            vec!["XML", "Http", "Request"]
        );
    }

    #[test]
    fn roles() {
        assert_eq!(role_of_filename("mod.rs"), FileRole::Entrypoint);
        assert_eq!(role_of_filename("index.ts"), FileRole::Entrypoint);
        assert_eq!(role_of_filename("user_test.rs"), FileRole::Test);
        assert_eq!(role_of_filename("user.spec.ts"), FileRole::Test);
    }

    #[test]
    fn related_siblings() {
        assert!(is_related_filename("user.test.ts", "user.ts"));
        assert!(is_related_filename("user.module.css", "user.ts"));
        assert!(!is_related_filename("other.ts", "user.ts"));
    }

    #[test]
    fn sibling_suffix_cut_inside_multibyte_char_keeps_stem() {
        // "controller_é_100": naive cuts land inside é; must not panic.
        assert_eq!(strip_sibling_suffix("controller_é_100"), "controller_é_100");
        assert_eq!(strip_sibling_suffix("user_test"), "user");
    }

    #[test]
    fn related_siblings_case_and_repeated_suffix() {
        assert!(is_related_filename("User.TEST.ts", "user.ts"));
        assert!(is_related_filename("user.test.test.ts", "user.ts"));
        assert!(is_related_filename("user.spec.ts", "user.ts"));
        assert!(is_related_filename("user_test.test.ts", "user.ts"));
        assert!(!is_related_filename("user.ts", "user.ts"));
        assert!(!is_related_filename("user.ts", "USER.ts"));
        assert!(!is_related_filename("other.test.ts", "user.ts"));
        assert!(!is_related_filename("user.stories.ts", "other.ts"));
        assert!(is_related_filename("user.stories.ts", "user.ts"));
    }

    #[test]
    fn prepared_query_state_matches_inline() {
        for q in [
            "user_service",
            "src/auth/login",
            "UserService",
            "a-b-c",
            "",
            "x",
        ] {
            let pq = PreparedQuery::new(q);
            assert_eq!(
                word_coverage_prepared("user_service.rs", &pq.qwords[..pq.qword_count]),
                word_coverage("user_service.rs", q.trim())
            );
            assert_eq!(
                word_coverage_prepared("OTHER", &pq.qwords[..pq.qword_count]),
                word_coverage("OTHER", q.trim())
            );
            let inline: Option<PathTokens<'_>> = if q.contains('/') || q.contains('\\') {
                Some(split_path_tokens(q))
            } else {
                None
            };
            assert_eq!(pq.path_tokens, inline);
        }
    }

    #[test]
    fn working_set_related_and_same_dir() {
        let ev = evidence_for(
            "src/api/user.test.ts",
            "user.test.ts",
            "",
            QueryIntent::WorkingSet,
            Some("src/api/user.ts"),
        );
        assert!(ev.related);
        assert!(ev.same_dir);
        let ev2 = evidence_for(
            "other/user.test.ts",
            "user.test.ts",
            "",
            QueryIntent::WorkingSet,
            Some("src/api/user.ts"),
        );
        assert!(ev2.related);
        assert!(!ev2.same_dir);
    }

    #[test]
    fn segment_evidence() {
        let q = split_path_tokens("src/auth");
        let (last, pct, seq) = segment_boundary_evidence("src/auth/login.rs", &q);
        assert!(seq);
        assert_eq!(pct, 100);
        assert!(!last);
    }

    #[test]
    fn segment_evidence_edges() {
        let q = split_path_tokens("src/auth");
        assert_eq!(segment_boundary_evidence("", &q), (false, 0, false));
        assert_eq!(segment_boundary_evidence("/", &q), (false, 0, false));
        assert_eq!(segment_boundary_evidence("src/auth", &q), (true, 100, true));
        assert_eq!(
            segment_boundary_evidence("SRC/AUTH/x.rs", &q),
            (false, 100, true)
        );
        assert_eq!(
            segment_boundary_evidence("src/other/x.rs", &q),
            (false, 50, false)
        );
        let win = split_path_tokens("src\\auth");
        assert_eq!(
            segment_boundary_evidence("src\\auth\\x.rs", &win),
            (false, 100, true)
        );
        let empty = split_path_tokens("/");
        assert_eq!(
            segment_boundary_evidence("src/auth", &empty),
            (false, 0, false)
        );
    }
}
