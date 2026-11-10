use crate::constraints::Constraint;
use crate::glob_detect::has_wildcards;

/// Check if a token looks like a filename/path for use as a `FilePath` constraint.
#[deprecated(note = "use `Constraint::is_filename_constraint_token`")]
pub fn is_filename_constraint_token(token: &str) -> bool {
    Constraint::is_filename_constraint_token(token)
}

/// Per-picker switches for which tokens become constraints.
pub trait ParserConfig {
    fn enable_glob(&self) -> bool {
        true
    }

    /// Extension shortcuts (`*.rs`). On by default.
    fn enable_extension(&self) -> bool {
        true
    }

    /// Exclusions (`!test`, `!*.rs`). On by default.
    fn enable_exclude(&self) -> bool {
        true
    }

    /// Path segments (`/src/`, `www/`). On by default.
    fn enable_path_segments(&self) -> bool {
        true
    }

    /// `type:rust`. On by default.
    fn enable_type_filter(&self) -> bool {
        true
    }

    /// `status:`, `st:`, `g:`, and `git:`. On by default.
    fn enable_git_status(&self) -> bool {
        true
    }

    /// `file:12` and `file:12:4`. Off for grep, so `localhost:8080` stays text.
    fn enable_location(&self) -> bool {
        true
    }

    /// Glob constraint. Default: `*`, `?`, `[`, `{`. Grep overrides this.
    fn is_glob_pattern(&self, token: &str) -> bool {
        has_wildcards(token)
    }

    /// A lone PathSegment is fuzzy text, not a directory filter.
    fn treat_lone_path_as_text(&self) -> bool {
        true
    }

    /// `score.rs` and `src/main.rs` become FilePath. Off by default.
    /// `vite.conf` would otherwise hide `vite.config.ts`.
    fn enable_filename_constraint(&self) -> bool {
        false
    }

    /// Picker-specific constraint. Default: none.
    #[inline(always)]
    fn parse_custom<'a>(&self, _input: &'a str) -> Option<Constraint<'a>> {
        None
    }
}

/// File picker. Filename constraints stay off unless a config opts in.
#[derive(Debug, Clone, Copy, Default)]
pub struct FileSearchConfig;

impl ParserConfig for FileSearchConfig {}

/// Content search. Globs need `/` or `{a,b}`. `?`, `[`, and bare `*` stay text.
#[derive(Debug, Clone, Copy, Default)]
pub struct GrepConfig;

impl ParserConfig for GrepConfig {
    fn enable_path_segments(&self) -> bool {
        true
    }

    fn enable_git_status(&self) -> bool {
        false
    }

    fn enable_location(&self) -> bool {
        false
    }

    /// Path glob (`src/**/*.rs`) or brace list (`{src,lib}`). `?` and `[` stay text.
    fn is_glob_pattern(&self, token: &str) -> bool {
        if !has_wildcards(token) {
            return false;
        }

        let bytes = token.as_bytes();

        if bytes.contains(&b'/') {
            return true;
        }

        // `{src,lib}` needs a comma and a letter. `}{` must not slice backwards.
        if let Some(open) = bytes.iter().position(|&b| b == b'{')
            && let Some(close) = bytes.iter().rposition(|&b| b == b'}')
            && open < close
        {
            let inner = &bytes[open + 1..close];
            if inner.contains(&b',') && inner.iter().any(|b| b.is_ascii_alphabetic()) {
                return true;
            }
        }

        false
    }
}

/// Grep plus FilePath. `*text*` is a glob; bare `*` and `**` are not.
#[derive(Debug, Clone, Copy, Default)]
pub struct AiGrepConfig;

/// Directories: no path segments, extensions, types, or git status.
/// A trailing `/` stays fuzzy text.
#[derive(Debug, Clone, Copy, Default)]
pub struct DirSearchConfig;

impl ParserConfig for AiGrepConfig {
    fn enable_path_segments(&self) -> bool {
        true
    }

    fn enable_git_status(&self) -> bool {
        false
    }

    fn enable_location(&self) -> bool {
        false
    }

    fn enable_filename_constraint(&self) -> bool {
        true
    }

    fn is_glob_pattern(&self, token: &str) -> bool {
        if GrepConfig.is_glob_pattern(token) {
            return true;
        }

        // `*text*` is a glob. Bare `*` and `**` are not. No extra `*` inside.
        if !has_wildcards(token) {
            return false;
        }
        let bytes = token.as_bytes();
        if bytes.len() >= 3
            && bytes[0] == b'*'
            && bytes[bytes.len() - 1] == b'*'
            && bytes[1..bytes.len() - 1].iter().all(|&b| b != b'*')
        {
            return true;
        }

        false
    }
}

impl ParserConfig for DirSearchConfig {
    fn enable_path_segments(&self) -> bool {
        false
    }

    fn enable_extension(&self) -> bool {
        false
    }

    fn enable_type_filter(&self) -> bool {
        false
    }

    fn enable_git_status(&self) -> bool {
        false
    }
}

/// Files and directories. No path segments; extensions and git status stay on.
#[derive(Debug, Clone, Copy, Default)]
pub struct MixedSearchConfig;

impl ParserConfig for MixedSearchConfig {
    fn enable_path_segments(&self) -> bool {
        false
    }
}
