//! Query intent + evidence model for file finding.
//!
//! Fuzzy path score is a weak proxy for "the file I mean". This module
//! classifies the user's job from the query shape and editor context, and
//! exposes structured evidence (filename, ext, path segments, role) that
//! the ranker mixes per intent.

/// What the user is trying to do. One query → one primary intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntent {
    /// Empty or whitespace — show working set (MRU, buffers, current dir).
    WorkingSet,
    /// `foo.rs` / `user_service.ts` — name + optional extension.
    FileName,
    /// `src/auth/login` / `../x/y` — path structure.
    Path,
    /// `handler`, `router`, `login` — concept / role / word.
    Concept,
    /// Looks like a typo of a name (`filw_pikcer`).
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
        // Very long random strings with a dot still count as name.
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
    let mut start = 0usize;
    let bytes = name.as_bytes();
    let mut prev_lower_or_digit = false;
    let mut prev_upper = false;
    for (i, &b) in bytes.iter().enumerate() {
        let is_sep = b == b'_' || b == b'-' || b == b'.' || b == b' ';
        let is_upper = b.is_ascii_uppercase();
        let is_lower_or_digit = b.is_ascii_lowercase() || b.is_ascii_digit();
        let camel_break = prev_lower_or_digit && is_upper;
        let acronym_break =
            prev_upper && is_upper && i + 1 < bytes.len() && bytes[i + 1].is_ascii_lowercase();
        if i > start && (is_sep || camel_break || acronym_break) {
            if !is_sep {
                if i > start {
                    f(&name[start..i]);
                }
                start = i;
            } else {
                if i > start {
                    f(&name[start..i]);
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
    if start < name.len() {
        f(&name[start..]);
    }
}

/// Cheap typo proxy: long identifier, mixed case noise, or uncommon bigrams.
/// Conservative — only fires when the string looks "wrong".
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
    if ratio < 0.08 || ratio > 0.55 {
        // Still allow `xkcd`-style; require also no separators/underscore.
        return !q.contains('_') && !q.contains('-');
    }
    false
}
/// Case-insensitive substring test without allocating.
///
/// The naive `h.windows(n).any(eq_ignore_ascii_case)` pays a full slice
/// comparison at every offset. Screening on the first byte first turns the
/// common case (most offsets mismatch immediately) into one compare, which
/// matters because this runs once per candidate on the non-collapsed path.
#[inline]
pub(crate) fn contains_ignore_ascii(hay: &str, needle: &str) -> bool {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() {
        return true;
    }
    if n.len() > h.len() {
        return false;
    }
    let lo = n[0].to_ascii_lowercase();
    let up = n[0].to_ascii_uppercase();
    let last = h.len() - n.len();
    for i in 0..=last {
        let c = h[i];
        if c != lo && c != up {
            continue;
        }
        if h[i..i + n.len()].eq_ignore_ascii_case(n) {
            return true;
        }
    }
    false
}

#[inline]
pub(crate) fn ends_with_ignore_ascii(hay: &str, suffix: &str) -> bool {
    let h = hay.as_bytes();
    let s = suffix.as_bytes();
    h.len() >= s.len() && h[h.len() - s.len()..].eq_ignore_ascii_case(s)
}

#[inline]
fn matches_ignore_ascii(hay: &str, options: &[&str]) -> bool {
    options.iter().any(|o| hay.eq_ignore_ascii_case(o))
}

/// File role from a filename (not the full path). Allocation-free: this runs
/// once per candidate, and the lowercase copy it used to build showed up as
/// ~20ns of the per-candidate evidence cost.
pub fn role_of_filename(fname: &str) -> FileRole {
    if ends_with_ignore_ascii(fname, "_test.rs")
        || ends_with_ignore_ascii(fname, ".test.ts")
        || ends_with_ignore_ascii(fname, ".test.tsx")
        || ends_with_ignore_ascii(fname, ".test.js")
        || ends_with_ignore_ascii(fname, ".spec.ts")
        || ends_with_ignore_ascii(fname, ".spec.js")
        || ends_with_ignore_ascii(fname, ".spec.rs")
        || contains_ignore_ascii(fname, "_test.")
        || contains_ignore_ascii(fname, "/tests/")
    {
        return FileRole::Test;
    }
    if matches_ignore_ascii(
        fname,
        &[
            "mod.rs",
            "index.ts",
            "index.tsx",
            "index.js",
            "index.jsx",
            "main.rs",
            "main.go",
            "main.py",
            "app.rs",
            "lib.rs",
            "__init__.py",
            "main.cpp",
            "main.c",
            "main.ts",
            "main.js",
        ],
    ) {
        return FileRole::Entrypoint;
    }
    if ends_with_ignore_ascii(fname, ".d.ts")
        || ends_with_ignore_ascii(fname, "_pb.rs")
        || contains_ignore_ascii(fname, "types")
        || ends_with_ignore_ascii(fname, "type.ts")
        || ends_with_ignore_ascii(fname, "type.rs")
    {
        return FileRole::Type;
    }
    if ends_with_ignore_ascii(fname, ".toml")
        || ends_with_ignore_ascii(fname, ".yaml")
        || ends_with_ignore_ascii(fname, ".yml")
        || ends_with_ignore_ascii(fname, ".json")
        || ends_with_ignore_ascii(fname, ".ini")
        || matches_ignore_ascii(fname, &["makefile", "dockerfile"])
    {
        return FileRole::Config;
    }
    if ends_with_ignore_ascii(fname, ".css")
        || ends_with_ignore_ascii(fname, ".scss")
        || ends_with_ignore_ascii(fname, ".less")
        || ends_with_ignore_ascii(fname, ".styl")
    {
        return FileRole::Style;
    }
    if ends_with_ignore_ascii(fname, ".md")
        || ends_with_ignore_ascii(fname, ".rst")
        || ends_with_ignore_ascii(fname, ".txt")
    {
        return FileRole::Doc;
    }
    FileRole::Other
}

/// Implied role from a concept query (e.g. "test", "config", "types").
pub fn implied_role_from_query(q: &str) -> Option<FileRole> {
    let s = q.to_ascii_lowercase();
    match s.as_str() {
        "test" | "tests" | "spec" | "specs" => Some(FileRole::Test),
        "config" | "conf" | "settings" => Some(FileRole::Config),
        "types" | "type" | "defs" => Some(FileRole::Type),
        "css" | "style" | "styles" => Some(FileRole::Style),
        "doc" | "docs" | "readme" => Some(FileRole::Doc),
        "main" | "index" | "mod" | "app" | "lib" => Some(FileRole::Entrypoint),
        _ => None,
    }
}

/// Segment-boundary coverage: how many query segments match path components.
pub fn segment_boundary_evidence(path: &str, q: &PathTokens<'_>) -> (bool, u8, bool) {
    if q.segments.is_empty() {
        return (false, 0, false);
    }
    let sep = if path.contains('\\') { '\\' } else { '/' };
    let last_q = q.segments.last().copied().unwrap_or("");
    // One pass, no component vec: track the subsequence cursor and the
    // last component for the exact-last-segment test.
    let mut qi = 0usize;
    let mut hits = 0usize;
    let mut last = "";
    for c in path.split(sep).filter(|s| !s.is_empty()) {
        last = c;
        if qi < q.segments.len() && c.eq_ignore_ascii_case(q.segments[qi]) {
            qi += 1;
            hits += 1;
        }
    }
    let last_match = !last.is_empty() && last.eq_ignore_ascii_case(last_q);
    let sequence = qi == q.segments.len();
    let pct = (hits * 100 / q.segments.len()).min(100) as u8;
    (last_match, pct, sequence)
}

/// Word coverage: percent of query tokens found in filename words.
/// No heap allocs — query tokens fit a small stack array.
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

pub(crate) fn word_coverage_prepared(fname: &str, qwords: &[&str]) -> u8 {
    if qwords.is_empty() {
        return 0;
    }
    let mut hits = 0usize;
    for qw in qwords.iter() {
        let mut found = false;
        for_each_word(fname, |fw| {
            if found {
                return;
            }
            if fw.eq_ignore_ascii_case(qw) || starts_with_ignore_ascii(fw, qw) {
                found = true;
            }
        });
        if found {
            hits += 1;
        }
    }
    (hits * 100 / qwords.len()).min(100) as u8
}

/// Case-insensitive prefix test without allocating.
fn starts_with_ignore_ascii(hay: &str, prefix: &str) -> bool {
    let h = hay.as_bytes();
    let p = prefix.as_bytes();
    if p.len() > h.len() {
        return false;
    }
    h[..p.len()].eq_ignore_ascii_case(p)
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

/// Filename-only evidence (no relative path walk). Same as `evidence_for`
/// for fields that don't need the path; path/segment/same_dir stay default.
pub fn evidence_filename_only(filename: &str, query: &str, intent: QueryIntent) -> Evidence {
    evidence_for("", filename, query, intent, None)
}

/// Query-side state that is identical for every candidate. Built once per
/// query so the per-candidate evidence pass does no heap allocation: the
/// lowercased query and its stem used to be rebuilt for every file, which on
/// a 100k-match query was most of the scoring cost.
pub struct PreparedQuery<'a> {
    pub query: &'a str,
    pub lower: String,
    pub name_ext: Option<NameExt<'a>>,
    pub stem_lower: Option<String>,
    pub wants_role: Option<FileRole>,
    pub qwords: [&'a str; 12],
    pub qword_count: usize,
    pub path_tokens: Option<PathTokens<'a>>,
}

impl<'a> PreparedQuery<'a> {
    pub fn new(query: &'a str) -> Self {
        let query = query.trim();
        let name_ext = split_name_ext(query);
        let stem_lower = name_ext.as_ref().map(|ne| ne.stem.to_ascii_lowercase());
        let mut qwords: [&'a str; 12] = [""; 12];
        let mut qword_count = 0usize;
        for_each_word(query, |w| {
            if qword_count < qwords.len() {
                qwords[qword_count] = w;
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
            lower: query.to_ascii_lowercase(),
            name_ext,
            stem_lower,
            wants_role: implied_role_from_query(query),
            qwords,
            qword_count,
            path_tokens,
        }
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
        ev.same_dir = !cur_dir.is_empty() && cur_dir.eq_ignore_ascii_case(hit_dir);
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
        ev.ext_match = filename
            .rsplit_once('.')
            .map(|(_, e)| e.eq_ignore_ascii_case(&ne.ext[1..]))
            .unwrap_or(false);
        ev.ext_mismatch = !ev.ext_match && filename.contains('.');
        if want_name {
            let stem_matches = filename_stem(filename).eq_ignore_ascii_case(ne.stem);
            ev.exact_filename = filename.eq_ignore_ascii_case(q) || (stem_matches && ev.ext_match);
            ev.filename_substring =
                contains_ignore_ascii(filename, pq.stem_lower.as_deref().unwrap_or_default());
        }
    } else if want_name {
        // Bare name query: stem equality is "this is the file", not just contains.
        let stem = filename_stem(filename);
        ev.exact_filename = filename.eq_ignore_ascii_case(q) || stem.eq_ignore_ascii_case(q);
        ev.filename_substring = contains_ignore_ascii(filename, &pq.lower);
    }

    // Path structure
    if intent == QueryIntent::Path {
        if let Some(pt) = pq.path_tokens.as_ref() {
            let (last_match, pct, sequence) = segment_boundary_evidence(relative_path, pt);
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
        ev.word_coverage = word_coverage_prepared(filename, &pq.qwords[..pq.qword_count]);
    }

    // Role is ~15 substring scans; only pay when the query wants one.
    if let Some(want) = pq.wants_role {
        ev.role_match = role_of_filename(filename) == want;
    } else if intent == QueryIntent::FileName || intent == QueryIntent::Typo {
        ev.role_match = false;
    } else {
        ev.role_match = false;
    }

    ev
}

fn filename_stem(filename: &str) -> &str {
    filename
        .rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or(filename)
}

/// True when `cand` is a conventional sibling of `cur` (same stem, different role).
pub fn is_related_filename(cand: &str, cur_fname: &str) -> bool {
    let a = filename_stem(cur_fname);
    let b = filename_stem(cand);
    if a.is_empty() || a.eq_ignore_ascii_case(b) {
        return false;
    }
    strip_sibling_suffix(a).eq_ignore_ascii_case(strip_sibling_suffix(b))
}

fn strip_sibling_suffix(mut s: &str) -> &str {
    for pat in [".test", ".spec", ".module", "_test", ".stories"] {
        while s.len() >= pat.len() && s[s.len() - pat.len()..].eq_ignore_ascii_case(pat) {
            s = &s[..s.len() - pat.len()];
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
