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
pub fn detect_intent(
    raw: &str,
    current_file: Option<&str>,
    _has_mru: bool,
) -> QueryIntent {
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
    Some(NameExt {
        stem,
        ext: ext_dot,
    })
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
    let mut start = 0usize;
    let bytes = name.as_bytes();
    let mut prev_lower_or_digit = false;
    let mut prev_upper = false;
    for (i, &b) in bytes.iter().enumerate() {
        let is_sep = b == b'_' || b == b'-' || b == b'.' || b == b' ';
        let is_upper = b.is_ascii_uppercase();
        let is_lower_or_digit = b.is_ascii_lowercase() || b.is_ascii_digit();
        // Break on separator, or camelCase boundary (lower→Upper).
        let camel_break = prev_lower_or_digit && is_upper;
        let acronym_break = prev_upper && is_upper && i + 1 < bytes.len() && bytes[i + 1].is_ascii_lowercase();
        if i > start && (is_sep || camel_break || acronym_break) {
            if !is_sep {
                out.push(&name[start..i]);
                start = i;
            } else {
                if i > start {
                    out.push(&name[start..i]);
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
        out.push(&name[start..]);
    }
    out.into_iter().filter(|s| !s.is_empty()).collect()
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

/// File role from a filename (not the full path).
pub fn role_of_filename(fname: &str) -> FileRole {
    let lower = fname.to_ascii_lowercase();
    let stem_ext = lower.as_str();
    if lower.ends_with("_test.rs")
        || lower.ends_with(".test.ts")
        || lower.ends_with(".test.tsx")
        || lower.ends_with(".test.js")
        || lower.ends_with(".spec.ts")
        || lower.ends_with(".spec.js")
        || lower.ends_with(".spec.rs")
        || lower.contains("_test.")
        || lower.contains("/tests/")
    {
        return FileRole::Test;
    }
    if matches!(
        stem_ext,
        "mod.rs" | "index.ts" | "index.tsx" | "index.js" | "index.jsx" | "main.rs"
            | "main.go" | "main.py" | "app.rs" | "lib.rs" | "__init__.py" | "main.cpp"
            | "main.c" | "main.ts" | "main.js"
    ) || stem_ext == "mod.rs"
    {
        return FileRole::Entrypoint;
    }
    if lower.ends_with(".d.ts")
        || lower.ends_with("_pb.rs")
        || lower.contains("types")
        || lower.ends_with("type.ts")
        || lower.ends_with("type.rs")
    {
        return FileRole::Type;
    }
    if lower.ends_with(".toml")
        || lower.ends_with(".yaml")
        || lower.ends_with(".yml")
        || lower.ends_with(".json")
        || lower.ends_with(".ini")
        || lower == "makefile"
        || lower == "dockerfile"
    {
        return FileRole::Config;
    }
    if lower.ends_with(".css")
        || lower.ends_with(".scss")
        || lower.ends_with(".less")
        || lower.ends_with(".styl")
    {
        return FileRole::Style;
    }
    if lower.ends_with(".md") || lower.ends_with(".rst") || lower.ends_with(".txt") {
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
    let comps: Vec<&str> = path.split(sep).filter(|s| !s.is_empty()).collect();
    let last_q = q.segments.last().copied().unwrap_or("");
    let last_match = comps
        .last()
        .map(|c| c.eq_ignore_ascii_case(last_q))
        .unwrap_or(false);

    // Subsequence over components (case-insensitive).
    let mut qi = 0usize;
    let mut hits = 0usize;
    for c in &comps {
        if qi < q.segments.len() && c.eq_ignore_ascii_case(q.segments[qi]) {
            qi += 1;
            hits += 1;
        }
    }
    let sequence = qi == q.segments.len();
    let pct = if q.segments.is_empty() {
        0
    } else {
        (hits * 100 / q.segments.len()).min(100) as u8
    };
    (last_match, pct, sequence)
}

/// Word coverage: percent of query tokens found in filename words.
pub fn word_coverage(fname: &str, query: &str) -> u8 {
    let fwords = filename_words(fname);
    let qwords = filename_words(query);
    if qwords.is_empty() {
        return 0;
    }
    let mut hits = 0usize;
    for qw in &qwords {
        if fwords
            .iter()
            .any(|fw| fw.eq_ignore_ascii_case(qw) || fw.to_ascii_lowercase().starts_with(&qw.to_ascii_lowercase()))
        {
            hits += 1;
        }
    }
    (hits * 100 / qwords.len()).min(100) as u8
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

/// Full evidence bundle for one file vs one query + intent.
pub fn evidence_for(
    relative_path: &str,
    filename: &str,
    query: &str,
    intent: QueryIntent,
    current_file: Option<&str>,
) -> Evidence {
    let mut ev = Evidence::default();
    let q = query.trim();
    if q.is_empty() {
        return ev;
    }

    // Name / ext
    if let Some(ne) = split_name_ext(q) {
        ev.ext_match = filename
            .rsplit_once('.')
            .map(|(_, e)| e.eq_ignore_ascii_case(&ne.ext[1..]))
            .unwrap_or(false);
        ev.ext_mismatch = !ev.ext_match && filename.contains('.');
        let stem_matches = filename_stem(filename).eq_ignore_ascii_case(ne.stem);
        ev.exact_filename = filename.eq_ignore_ascii_case(q) || (stem_matches && ev.ext_match);
        ev.filename_substring = filename
            .to_ascii_lowercase()
            .contains(&ne.stem.to_ascii_lowercase());
    } else {
        // Bare name query: stem equality is "this is the file", not just contains.
        let stem = filename_stem(filename);
        ev.exact_filename = filename.eq_ignore_ascii_case(q) || stem.eq_ignore_ascii_case(q);
        ev.filename_substring = filename.to_ascii_lowercase().contains(&q.to_ascii_lowercase());
    }

    // Path structure
    if intent == QueryIntent::Path {
        let pt = split_path_tokens(q);
        let (last_match, pct, sequence) = segment_boundary_evidence(relative_path, &pt);
        ev.last_segment_match = last_match;
        ev.segment_boundary_pct = pct;
        ev.segment_sequence = sequence;
    }

    // Words / camel
    ev.word_coverage = word_coverage(filename, q);

    // Proximity
    if let Some(cur) = current_file {
        let sep = if relative_path.contains('\\') { '\\' } else { '/' };
        let cur_dir = cur.rsplit_once(sep).map(|(d, _)| d).unwrap_or("");
        let hit_dir = relative_path.rsplit_once(sep).map(|(d, _)| d).unwrap_or("");
        ev.same_dir = !cur_dir.is_empty() && cur_dir.eq_ignore_ascii_case(hit_dir);
    }

    // Role
    let role = role_of_filename(filename);
    if let Some(want) = implied_role_from_query(q) {
        ev.role_match = role == want;
    } else if intent == QueryIntent::FileName || intent == QueryIntent::Typo {
        ev.role_match = false;
    } else {
        ev.role_match = false;
    }

    // Related names (foo.ts ↔ foo.test.ts / foo.module.css)
    if let Some(cur) = current_file {
        ev.related = is_related_filename(filename, cur.rsplit(['/', '\\']).next().unwrap_or(cur));
    }

    ev
}

fn filename_stem(filename: &str) -> &str {
    filename.rsplit_once('.').map(|(s, _)| s).unwrap_or(filename)
}

/// True when `cand` is a conventional sibling of `cur` (same stem, different role).
pub fn is_related_filename(cand: &str, cur_fname: &str) -> bool {
    let a = filename_stem(cur_fname).to_ascii_lowercase();
    let b = filename_stem(cand).to_ascii_lowercase();
    if a.is_empty() || a == b {
        return false;
    }
    // Strip common sibling suffixes from cand stem.
    let b2 = b
        .trim_end_matches(".test")
        .trim_end_matches(".spec")
        .trim_end_matches(".module")
        .trim_end_matches("_test")
        .trim_end_matches(".stories")
        .to_string();
    let a2 = a
        .trim_end_matches(".test")
        .trim_end_matches(".spec")
        .trim_end_matches(".module")
        .trim_end_matches("_test")
        .trim_end_matches(".stories")
        .to_string();
    a2 == b2
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
        assert_eq!(filename_words("XMLHttpRequest"), vec!["XML", "Http", "Request"]);
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
    fn segment_evidence() {
        let q = split_path_tokens("src/auth");
        let (last, pct, seq) = segment_boundary_evidence("src/auth/login.rs", &q);
        assert!(seq);
        assert_eq!(pct, 100);
        assert!(!last);
    }
}
