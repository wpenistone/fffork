//! public-v1 fixture materialization (FFF_REFERENCE_PACK Phase 4).

use crate::identity;
use std::fs;
use std::path::{Path, PathBuf};

pub const FIXTURE_ID: &str = "public-v1";
pub const QUERY_BANK_VERSION: &str = "query-bank/v1";

/// Materialize a content-addressable mini corpus for gates/mutation.
pub fn materialize_public_v1(cache_root: &Path) -> Result<(PathBuf, String), String> {
    let root = cache_root.join("fixtures").join(FIXTURE_ID);
    fs::create_dir_all(&root).map_err(|e| e.to_string())?;

    write_tree(&root)?;

    let digest = semantic_digest(&root)?;
    let manifest = serde_json::json!({
        "fixture_id": FIXTURE_ID,
        "semantic_digest": format!("sha256:{digest}"),
        "query_bank": QUERY_BANK_VERSION,
        "files": list_rel_files(&root)?,
    });
    fs::write(
        root.join("fixture.manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .map_err(|e| e.to_string())?;

    Ok((root, format!("sha256:{digest}")))
}

fn write_tree(root: &Path) -> Result<(), String> {
    // Exact / near-collision names
    write_file(root, "README.md", b"# public-v1 fixture\n")?;
    write_file(root, "Cargo.toml", b"[package]\nname = \"fixture\"\n")?;
    write_file(root, "src/main.rs", b"fn main() {}\n")?;
    write_file(root, "src/lib.rs", b"pub fn lib() {}\n")?;
    write_file(root, "src/controller.rs", b"pub struct Controller;\n")?;
    write_file(root, "src/file_picker.rs", b"pub struct FilePicker;\n")?;
    write_file(root, "src/score.rs", b"pub fn score() -> i32 { 0 }\n")?;
    write_file(root, "src/path_utils.rs", b"pub fn join() {}\n")?;
    write_file(
        root,
        "src/background_watcher.rs",
        b"pub fn watch() {}\n",
    )?;
    // Naming styles
    write_file(root, "src/CamelCaseModule.rs", b"// camel\n")?;
    write_file(root, "src/snake_case_mod.rs", b"// snake\n")?;
    write_file(root, "src/kebab-case-mod.rs", b"// kebab\n")?;
    write_file(root, "src/dotted.name.rs", b"// dotted\n")?;
    // Deep / long
    write_file(
        root,
        "deep/a/b/c/d/e/long_file_name_for_path_depth.rs",
        b"// deep\n",
    )?;
    // Hidden
    write_file(root, ".hidden_config", b"secret=1\n")?;
    // Grep content (plain / regex goldens)
    write_file(
        root,
        "src/grep_target.rs",
        b"// UNIQUE_GREP_TOKEN_ALPHA\npub fn unique_grep_token_alpha() {}\nstruct FffAdapter;\n",
    )?;
    write_file(
        root,
        "src/grep_regex.rs",
        b"error[E0308]: mismatched types\nfn foo() {}\n",
    )?;
    write_file(
        root,
        "src/grep_lines.rs",
        b"line1_plain\nline2_UNIQUE_LINE_MARKER\nline3_plain\nTODO: fixture note\nFIXME: later\n",
    )?;
    write_file(
        root,
        "src/score_components.rs",
        b"// score audit fixture\npub fn exact_match_target() {}\n",
    )?;
    // Collision pair
    write_file(root, "pkg/mod.rs", b"// pkg mod\n")?;
    write_file(root, "other/mod.rs", b"// other mod\n")?;
    // Non-ASCII filename (NFC café)
    write_file(root, "src/unicode_cafe_\u{00e9}.rs", b"pub fn cafe() {}\n")?;
    // Mutation sandbox marker
    write_file(root, "mutation/.keep", b"")?;
    Ok(())
}

fn write_file(root: &Path, rel: &str, body: &[u8]) -> Result<(), String> {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(p, body).map_err(|e| e.to_string())
}

fn list_rel_files(root: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<(), String> {
    for e in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let e = e.map_err(|e| e.to_string())?;
        let p = e.path();
        if p.file_name().and_then(|s| s.to_str()) == Some("fixture.manifest.json") {
            continue;
        }
        if p.is_dir() {
            walk(root, &p, out)?;
        } else {
            let rel = p
                .strip_prefix(root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(rel);
        }
    }
    Ok(())
}

pub fn semantic_digest(root: &Path) -> Result<String, String> {
    let files = list_rel_files(root)?;
    let mut parts = Vec::new();
    for rel in files {
        let data = fs::read(root.join(&rel)).map_err(|e| e.to_string())?;
        let h = identity::sha256_hex(&data);
        parts.push(format!("{rel}:{h}"));
    }
    Ok(identity::sha256_hex(parts.join("\n").as_bytes()))
}

/// Versioned query bank used by golden gates on public-v1.
pub fn query_bank_v1() -> Vec<QueryCase> {
    vec![
        QueryCase {
            class: "exact_filename",
            query: "Cargo.toml",
            expect: "Cargo.toml",
            must_hit: true,
            kind: QueryKind::Search,
            expect_line: None,
        },
        QueryCase {
            class: "exact_path_segment",
            query: "src/file_picker",
            expect: "file_picker.rs",
            must_hit: true,
            kind: QueryKind::Search,
            expect_line: None,
        },
        QueryCase {
            class: "medium_multipart",
            query: "src/background_watcher",
            expect: "background_watcher.rs",
            must_hit: true,
            kind: QueryKind::Search,
            expect_line: None,
        },
        QueryCase {
            class: "camel_case",
            query: "CamelCaseModule",
            expect: "CamelCaseModule.rs",
            must_hit: true,
            kind: QueryKind::Search,
            expect_line: None,
        },
        QueryCase {
            class: "snake_case",
            query: "snake_case_mod",
            expect: "snake_case_mod.rs",
            must_hit: true,
            kind: QueryKind::Search,
            expect_line: None,
        },
        QueryCase {
            class: "short_prefix",
            query: "cont",
            expect: "controller.rs",
            must_hit: false,
            kind: QueryKind::Search,
            expect_line: None,
        },
        QueryCase {
            class: "ambiguous_collision",
            query: "mod",
            expect: "mod.rs",
            must_hit: false,
            kind: QueryKind::Search,
            expect_line: None,
        },
        QueryCase {
            class: "exact_score_target",
            query: "score_components",
            expect: "score_components.rs",
            must_hit: true,
            kind: QueryKind::Search,
            expect_line: None,
        },
        QueryCase {
            class: "no_result",
            query: "zzznomatchzzz",
            expect: "",
            must_hit: false,
            kind: QueryKind::Search,
            expect_line: None,
        },
        // Grep bank
        QueryCase {
            class: "grep_plain_unique",
            query: "UNIQUE_GREP_TOKEN_ALPHA",
            expect: "grep_target.rs",
            must_hit: true,
            kind: QueryKind::GrepPlain,
            expect_line: Some(1),
        },
        QueryCase {
            class: "grep_plain_line",
            query: "UNIQUE_LINE_MARKER",
            expect: "grep_lines.rs",
            must_hit: true,
            kind: QueryKind::GrepPlain,
            expect_line: Some(2), // line2_UNIQUE_LINE_MARKER
        },
        QueryCase {
            class: "grep_todo",
            query: "TODO:",
            expect: "grep_lines.rs",
            must_hit: true,
            kind: QueryKind::GrepPlain,
            expect_line: Some(4),
        },
        QueryCase {
            class: "grep_regex_error",
            query: r"error\[E[0-9]+\]",
            expect: "grep_regex.rs",
            must_hit: true,
            kind: QueryKind::GrepRegex,
            expect_line: Some(1),
        },
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    Search,
    GrepPlain,
    GrepRegex,
}

#[derive(Debug, Clone)]
pub struct QueryCase {
    pub class: &'static str,
    pub query: &'static str,
    pub expect: &'static str,
    pub must_hit: bool,
    pub kind: QueryKind,
    pub expect_line: Option<u32>,
}

/// Isolated mutable copy for mutation scenarios.
pub fn clone_for_mutation(src: &Path) -> Result<PathBuf, String> {
    let dest = std::env::temp_dir().join(format!(
        "fff-bench-mut-{}-{}",
        std::process::id(),
        identity::now_unix()
    ));
    copy_dir(src, &dest)?;
    Ok(dest)
}

fn copy_dir(src: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    for e in fs::read_dir(src).map_err(|e| e.to_string())? {
        let e = e.map_err(|e| e.to_string())?;
        let p = e.path();
        let name = e.file_name();
        let t = dest.join(name);
        if p.is_dir() {
            copy_dir(&p, &t)?;
        } else {
            fs::copy(&p, &t).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materialize_is_deterministic() {
        let base = std::env::temp_dir().join(format!("fff-fx-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let (a, da) = materialize_public_v1(&base).unwrap();
        let (_b, db) = materialize_public_v1(&base).unwrap();
        assert_eq!(da, db);
        assert!(a.join("src/file_picker.rs").exists());
        let _ = fs::remove_dir_all(&base);
    }
}
