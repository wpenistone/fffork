//! FFF subject adapter — public API only, smart config.

use crate::cost::{OpResultData, ScoreBreakdown, TopHit};
use crate::manifest::AdapterManifest;
use fff::file_picker::{FFFMode, FilePicker};
use fff::grep::{parse_grep_query, GrepMode, GrepSearchOptions};
use fff::{
    DirSearchConfig, FileSearchConfig, FuzzySearchOptions, MixedSearchConfig, PaginationArgs,
    QueryParser, SharedFilePicker, SharedFrecency,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub enable_content_indexing: bool,
    pub enable_mmap_cache: bool,
    pub score_breakdown: bool,
    pub top_k_log: usize,
    pub timeout_secs: u64,
    pub mode: String,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            enable_content_indexing: false,
            enable_mmap_cache: false,
            score_breakdown: true,
            top_k_log: 10,
            // Mega-corpus (sumo_fork ~227k) cold scan exceeds 120s on Windows.
            timeout_secs: 600,
            mode: "neovim".into(),
        }
    }
}

impl From<&AdapterManifest> for AdapterConfig {
    fn from(m: &AdapterManifest) -> Self {
        Self {
            enable_content_indexing: m.enable_content_indexing,
            enable_mmap_cache: m.enable_mmap_cache,
            score_breakdown: m.score_breakdown,
            top_k_log: m.top_k_log.max(1),
            timeout_secs: m.timeout_secs.unwrap_or(600),
            mode: "neovim".into(),
        }
    }
}

/// Smart defaults: small repos can enable content index; large skip for open speed.
pub fn smart_config_for_path(path: &Path, base: AdapterConfig) -> AdapterConfig {
    let mut cfg = base;
    // Heuristic: if path looks like a huge monorepo name, keep indexing off
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if matches!(
        name.as_str(),
        "linux" | "rust" | "chromium" | "webkit" | "llvm-project"
    ) {
        cfg.enable_content_indexing = false;
        cfg.enable_mmap_cache = false;
    }
    cfg
}

pub struct Caps {
    pub search: bool,
    pub grep: bool,
    pub scan: bool,
    pub score_breakdown: bool,
}

impl Default for Caps {
    fn default() -> Self {
        Self {
            search: true,
            grep: true,
            scan: true,
            score_breakdown: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CallKind {
    #[default]
    File,
    Dir,
    Mixed,
    GrepPlain,
    GrepFuzzy,
    GrepRegex,
}

#[derive(Debug, Clone)]
pub struct SearchSpec {
    pub query: String,
    pub limit: usize,
    pub offset: usize,
    pub threads: usize,
    pub call: CallKind,
}

#[derive(Debug, Clone)]
pub struct GrepSpec {
    pub query: String,
    pub mode: GrepMode,
    pub page_limit: usize,
    pub file_offset: usize,
    pub threads_hint: usize,
}

pub struct FffAdapter {
    shared_picker: SharedFilePicker,
    #[allow(dead_code)]
    shared_frecency: SharedFrecency,
    parser: QueryParser<FileSearchConfig>,
    dir_parser: QueryParser<DirSearchConfig>,
    mixed_parser: QueryParser<MixedSearchConfig>,
    base_path: PathBuf,
    pub config: AdapterConfig,
}

impl FffAdapter {
    fn ordered_digest<'a>(
        paths: impl IntoIterator<Item = &'a str>,
        scores: impl IntoIterator<Item = i32>,
    ) -> String {
        let mut digest = Sha256::new();
        for (path, score) in paths.into_iter().zip(scores) {
            digest.update((path.len() as u64).to_le_bytes());
            digest.update(path.as_bytes());
            digest.update(score.to_le_bytes());
        }
        format!("sha256:{}", hex::encode(digest.finalize()))
    }

    pub fn id() -> &'static str {
        "fff"
    }

    pub fn capabilities() -> Caps {
        Caps::default()
    }

    pub fn open_corpus(path: &Path, cfg: AdapterConfig) -> Result<Self, String> {
        Self::open_corpus_watch(path, cfg, false)
    }

    pub fn open_corpus_watch(
        path: &Path,
        cfg: AdapterConfig,
        watch: bool,
    ) -> Result<Self, String> {
        let path = fff::path_utils::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if !path.exists() {
            return Err(format!("corpus path does not exist: {}", path.display()));
        }
        let cfg = smart_config_for_path(&path, cfg);

        let shared_picker = SharedFilePicker::default();
        let shared_frecency = SharedFrecency::default();

        FilePicker::new_with_shared_state(
            shared_picker.clone(),
            shared_frecency.clone(),
            fff::FilePickerOptions {
                base_path: path.to_string_lossy().to_string(),
                enable_mmap_cache: cfg.enable_mmap_cache,
                enable_content_indexing: cfg.enable_content_indexing,
                mode: FFFMode::Neovim,
                watch,
                // Worktree search only. .git is not a search corpus.
                skip_git: true,
                git_recency: fff::git_recency::GitRecencyConfig {
                    enabled: false,
                    max_commits: 0,
                    max_files_per_commit: 0,
                },
                ..Default::default()
            },
        )
        .map_err(|e| format!("FilePicker init: {e}"))?;

        std::thread::sleep(Duration::from_millis(50));

        if !shared_picker.wait_for_scan(Duration::from_secs(cfg.timeout_secs)) {
            return Err(format!("scan timed out after {}s", cfg.timeout_secs));
        }
        if watch {
            let _ = shared_picker.wait_for_watcher(Duration::from_secs(2));
        }

        Ok(Self {
            shared_picker,
            shared_frecency,
            parser: QueryParser::default(),
            dir_parser: QueryParser::new(DirSearchConfig),
            mixed_parser: QueryParser::new(MixedSearchConfig),
            base_path: path,
            config: cfg,
        })
    }

    /// Apply a create/modify notification to the live index (mutation gates).
    pub fn notify_path_changed(&self, path: &Path) {
        if let Ok(mut g) = self.shared_picker.write() {
            if let Some(p) = g.as_mut() {
                let _ = p.handle_create_or_modify(path);
            }
        }
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub fn file_count(&self) -> usize {
        self.shared_picker
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(|p| p.get_files().len()))
            .unwrap_or(0)
    }

    pub fn path_list(&self) -> Vec<String> {
        let guard = match self.shared_picker.read() {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        let Some(picker) = guard.as_ref() else {
            return vec![];
        };
        picker
            .get_files()
            .iter()
            .filter(|f| !f.is_deleted())
            .map(|f| f.relative_path(picker))
            .collect()
    }

    fn fuzzy_opts<'a>(&'a self, spec: &SearchSpec) -> FuzzySearchOptions<'a> {
        FuzzySearchOptions {
            max_threads: spec.threads,
            current_file: None,
            project_path: Some(self.base_path.as_path()),
            combo_boost_score_multiplier: 100,
            min_combo_count: 3,
            pagination: PaginationArgs {
                offset: spec.offset,
                limit: spec.limit,
            },
        }
    }

    /// Dispatch by call kind: file / dir / mixed fuzzy, or grep modes.
    pub fn call(&self, spec: &SearchSpec) -> OpResultData {
        match spec.call {
            CallKind::File => self.search_files(spec),
            CallKind::Dir => self.search_dirs(spec),
            CallKind::Mixed => self.search_mixed(spec),
            CallKind::GrepPlain => self.grep(&GrepSpec {
                query: spec.query.clone(),
                mode: GrepMode::PlainText,
                page_limit: spec.limit,
                file_offset: spec.offset,
                threads_hint: spec.threads,
            }),
            CallKind::GrepFuzzy => self.grep(&GrepSpec {
                query: spec.query.clone(),
                mode: GrepMode::Fuzzy,
                page_limit: spec.limit,
                file_offset: spec.offset,
                threads_hint: spec.threads,
            }),
            CallKind::GrepRegex => self.grep(&GrepSpec {
                query: spec.query.clone(),
                mode: GrepMode::Regex,
                page_limit: spec.limit,
                file_offset: spec.offset,
                threads_hint: spec.threads,
            }),
        }
    }

    pub fn search(&self, spec: &SearchSpec) -> OpResultData {
        self.call(spec)
    }

    pub fn search_files(&self, spec: &SearchSpec) -> OpResultData {
        let guard = match self.shared_picker.read() {
            Ok(g) => g,
            Err(e) => {
                return OpResultData {
                    error: Some(format!("lock: {e}")),
                    ..Default::default()
                };
            }
        };
        let Some(picker) = guard.as_ref() else {
            return OpResultData {
                error: Some("picker missing".into()),
                ..Default::default()
            };
        };

        let parsed = self.parser.parse(&spec.query);
        let result = picker.fuzzy_search(&parsed, None, self.fuzzy_opts(spec));
        let paths: Vec<String> = result
            .items
            .iter()
            .map(|item| item.relative_path(picker))
            .collect();
        let ordered_digest = Self::ordered_digest(
            paths.iter().map(String::as_str),
            result.scores.iter().map(|score| score.total),
        );

        let k = self.config.top_k_log.min(spec.limit);
        let top_k: Vec<TopHit> = paths
            .iter()
            .zip(result.scores.iter())
            .take(k)
            .map(|(path, score)| TopHit {
                path: path.clone(),
                score: if self.config.score_breakdown {
                    ScoreBreakdown::from(score)
                } else {
                    ScoreBreakdown {
                        total: score.total,
                        ..Default::default()
                    }
                },
                line: None,
            })
            .collect();

        OpResultData {
            total_matched: result.total_matched,
            returned: result.items.len(),
            files_searched: None,
            top_k,
            rank_target: None,
            ordered_digest: Some(ordered_digest),
            error: None,
        }
    }

    pub fn search_dirs(&self, spec: &SearchSpec) -> OpResultData {
        let guard = match self.shared_picker.read() {
            Ok(g) => g,
            Err(e) => {
                return OpResultData {
                    error: Some(format!("lock: {e}")),
                    ..Default::default()
                };
            }
        };
        let Some(picker) = guard.as_ref() else {
            return OpResultData {
                error: Some("picker missing".into()),
                ..Default::default()
            };
        };
        let parsed = self.dir_parser.parse(&spec.query);
        let result = picker.fuzzy_search_directories(&parsed, self.fuzzy_opts(spec));
        let paths: Vec<String> = result
            .items
            .iter()
            .map(|item| format!("{}/", item.relative_path(picker)))
            .collect();
        let ordered_digest = Self::ordered_digest(
            paths.iter().map(String::as_str),
            result.scores.iter().map(|score| score.total),
        );
        let k = self.config.top_k_log.min(spec.limit);
        let top_k: Vec<TopHit> = paths
            .iter()
            .zip(result.scores.iter())
            .take(k)
            .map(|(path, score)| TopHit {
                path: path.clone(),
                score: ScoreBreakdown::from(score),
                line: None,
            })
            .collect();
        OpResultData {
            total_matched: result.total_matched,
            returned: result.items.len(),
            files_searched: None,
            top_k,
            rank_target: None,
            ordered_digest: Some(ordered_digest),
            error: None,
        }
    }

    pub fn search_mixed(&self, spec: &SearchSpec) -> OpResultData {
        let guard = match self.shared_picker.read() {
            Ok(g) => g,
            Err(e) => {
                return OpResultData {
                    error: Some(format!("lock: {e}")),
                    ..Default::default()
                };
            }
        };
        let Some(picker) = guard.as_ref() else {
            return OpResultData {
                error: Some("picker missing".into()),
                ..Default::default()
            };
        };
        let parsed = self.mixed_parser.parse(&spec.query);
        let result = picker.fuzzy_search_mixed(&parsed, None, self.fuzzy_opts(spec));
        let paths: Vec<String> = result
            .items
            .iter()
            .map(|item| match item {
                fff::MixedItemRef::File(file) => file.relative_path(picker),
                fff::MixedItemRef::Dir(dir) => format!("{}/", dir.relative_path(picker)),
            })
            .collect();
        let ordered_digest = Self::ordered_digest(
            paths.iter().map(String::as_str),
            result.scores.iter().map(|score| score.total),
        );
        let k = self.config.top_k_log.min(spec.limit);
        let top_k: Vec<TopHit> = paths
            .iter()
            .zip(result.scores.iter())
            .take(k)
            .map(|(path, score)| {
                TopHit {
                    path: path.clone(),
                    score: ScoreBreakdown::from(score),
                    line: None,
                }
            })
            .collect();
        OpResultData {
            total_matched: result.total_matched,
            returned: result.items.len(),
            files_searched: None,
            top_k,
            rank_target: None,
            ordered_digest: Some(ordered_digest),
            error: None,
        }
    }

    pub fn grep(&self, spec: &GrepSpec) -> OpResultData {
        self.grep_with_cursor(spec).0
    }

    pub fn grep_page(&self, spec: &SearchSpec) -> (OpResultData, usize) {
        let mode = match spec.call {
            CallKind::GrepFuzzy => GrepMode::Fuzzy,
            CallKind::GrepRegex => GrepMode::Regex,
            _ => GrepMode::PlainText,
        };
        self.grep_with_cursor(&GrepSpec {
            query: spec.query.clone(),
            mode,
            page_limit: spec.limit,
            file_offset: spec.offset,
            threads_hint: spec.threads,
        })
    }

    fn grep_with_cursor(&self, spec: &GrepSpec) -> (OpResultData, usize) {
        let guard = match self.shared_picker.read() {
            Ok(g) => g,
            Err(e) => {
                return (
                    OpResultData {
                        error: Some(format!("lock: {e}")),
                        ..Default::default()
                    },
                    0,
                );
            }
        };
        let Some(picker) = guard.as_ref() else {
            return (
                OpResultData {
                    error: Some("picker missing".into()),
                    ..Default::default()
                },
                0,
            );
        };

        let parsed = parse_grep_query(&spec.query);
        let result = picker.grep(
            &parsed,
            &GrepSearchOptions {
                file_offset: spec.file_offset,
                page_limit: spec.page_limit,
                mode: spec.mode,
                time_budget_ms: 5_000,
                ..Default::default()
            },
        );

        let top_k: Vec<TopHit> = result
            .matches
            .iter()
            .take(self.config.top_k_log)
            .filter_map(|m| {
                let file = result.files.get(m.file_index)?;
                let rel = file.relative_path(picker);
                Some(TopHit {
                    path: format!("{}:{}", rel, m.line_number),
                    score: ScoreBreakdown {
                        total: m.fuzzy_score.unwrap_or(0) as i32,
                        ..Default::default()
                    },
                    line: Some(m.line_number as u32),
                })
            })
            .collect();

        let next_file_offset = result.next_file_offset;
        (
            OpResultData {
                total_matched: result.matches.len(),
                returned: top_k.len(),
                files_searched: Some(result.total_files_searched),
                top_k,
                rank_target: None,
                ordered_digest: None,
                error: None,
            },
            next_file_offset,
        )
    }

    pub fn rank_of(&self, result: &OpResultData, target_substr: &str) -> Option<usize> {
        let t = target_substr.replace('\\', "/").to_lowercase();
        result.top_k.iter().position(|h| {
            let p = h.path.replace('\\', "/").to_lowercase();
            p.contains(&t) || p.ends_with(&t)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_config_disables_index_for_linux_name() {
        let base = AdapterConfig {
            enable_content_indexing: true,
            enable_mmap_cache: true,
            ..Default::default()
        };
        let cfg = smart_config_for_path(Path::new("/tmp/linux"), base);
        assert!(!cfg.enable_content_indexing);
        assert!(!cfg.enable_mmap_cache);
    }

    #[test]
    fn adapter_config_from_manifest() {
        let m = AdapterManifest {
            enable_content_indexing: true,
            enable_mmap_cache: false,
            score_breakdown: true,
            top_k_log: 3,
            timeout_secs: None,
        };
        let c = AdapterConfig::from(&m);
        assert!(c.enable_content_indexing);
        assert_eq!(c.top_k_log, 3);
    }
}
