//! Definition and import line classification (vibe coded POC)
//!
//! Byte-level heuristics that tag a matched line as a code definition
//! (`struct`, `fn`, `class`, …) or an import/use statement. Used to
//! rank/annotate grep results for AI/MCP consumers. Gated behind the
//! `definitions` feature since only such consumers need it.

/// Detect if a line looks like a code definition (struct, fn, class, etc.)
#[inline]
pub fn is_definition_line(line: &str) -> bool {
    let s = line.trim_start().as_bytes();
    let s = skip_modifiers(s);
    is_definition_keyword(s)
}

/// Modifier keywords that can precede a definition keyword.
/// Each must be followed by whitespace to be consumed.
const MODIFIERS: &[&[u8]] = &[
    b"pub",
    b"export",
    b"default",
    b"async",
    b"abstract",
    b"unsafe",
    b"static",
    b"protected",
    b"private",
    b"public",
];


/// Skip zero or more modifier keywords (including `pub(crate)` style visibility).
fn skip_modifiers(mut s: &[u8]) -> &[u8] {
    loop {
        // Handle `pub(...)` — e.g. `pub(crate)`, `pub(super)`
        if s.starts_with(b"pub(")
            && let Some(end) = s.iter().position(|&b| b == b')')
        {
            s = skip_ws(&s[end + 1..]);
            continue;
        }
        let mut matched = false;
        for &kw in MODIFIERS {
            if s.starts_with(kw) {
                let rest = &s[kw.len()..];
                if rest.first().is_some_and(|b| b.is_ascii_whitespace()) {
                    s = skip_ws(rest);
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            return s;
        }
    }
}

/// Check if `s` starts with a definition keyword followed by a word boundary.
#[inline]
fn is_definition_keyword(s: &[u8]) -> bool {
    let Some(&first) = s.first() else { return false };
    let candidates: &[&[u8]] = match first {
        b's' => &[b"struct"],
        b'f' => &[b"fn", b"function", b"func"],
        b'e' => &[b"enum"],
        b't' => &[b"trait", b"type"],
        b'i' => &[b"impl", b"interface"],
        b'c' => &[b"class"],
        b'd' => &[b"def"],
        b'm' => &[b"module"],
        b'o' => &[b"object"],
        _ => return false,
    };
    for &kw in candidates {
        if s.starts_with(kw) {
            let after = s.get(kw.len());
            // Word boundary: end of input, or next byte is not alphanumeric/underscore
            if after.is_none_or(|b| !b.is_ascii_alphanumeric() && *b != b'_') {
                return true;
            }
        }
    }
    false
}

/// Skip ASCII whitespace.
#[inline]
fn skip_ws(s: &[u8]) -> &[u8] {
    let n = s
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(s.len());
    &s[n..]
}

/// Detect import/use lines — lower value than definitions or usages.
///
/// Checks if the line (after leading whitespace) starts with a common
/// import statement prefix. Pure byte-level checks, no regex.
pub fn is_import_line(line: &str) -> bool {
    let s = line.trim_start().as_bytes();
    s.starts_with(b"import ")
        || s.starts_with(b"import\t")
        || (s.starts_with(b"from ") && s.get(5).is_some_and(|&b| b == b'\'' || b == b'"'))
        || s.starts_with(b"use ")
        || s.starts_with(b"use\t")
        || starts_with_require(s)
        || starts_with_include(s)
}

/// Match `require(` or `require (`.
#[inline]
fn starts_with_require(s: &[u8]) -> bool {
    if !s.starts_with(b"require") {
        return false;
    }
    let rest = &s[b"require".len()..];
    rest.first() == Some(&b'(') || (rest.first() == Some(&b' ') && rest.get(1) == Some(&b'('))
}

/// Match `# include ` (with optional spaces after `#`).
#[inline]
fn starts_with_include(s: &[u8]) -> bool {
    if s.first() != Some(&b'#') {
        return false;
    }
    let rest = skip_ws(&s[1..]);
    rest.starts_with(b"include ") || rest.starts_with(b"include\t")
}
