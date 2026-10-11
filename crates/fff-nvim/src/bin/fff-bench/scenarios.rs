//! Wide scenario catalog: quality gates, cold/warm, throughput, load, constraints, edges.
//! Suites compose families; `full` is the comprehensive agent matrix.

use crate::adapter::SearchSpec;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub scenario_id: String,
    pub repo_id: String,
    pub cell_key: String,
    pub axes: serde_json::Value,
    pub search: Option<SearchCell>,
    pub kind: CellKind,
    /// Workload family for score breakdown (e.g. cold, throughput, constraint).
    #[serde(default)]
    pub family: String,
    /// Optional tags for filtering / combinatorial labels.
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCell {
    pub query: String,
    pub limit: usize,
    pub offset: usize,
    pub threads: usize,
    pub expect_path_contains: Option<String>,
    /// For throughput cells: fire this many searches in one timed burst.
    #[serde(default)]
    pub burst: usize,
    /// Independent closed-loop clients for concurrent throughput cells.
    #[serde(default = "one")]
    pub clients: usize,
    /// Number of cursor-linked grep pages to fetch in one session.
    #[serde(default = "one")]
    pub pages: usize,
    /// file | dir | mixed | grep_plain | grep_fuzzy | grep_regex
    #[serde(default)]
    pub call_kind: Option<String>,
    /// Typing / multi-query stream (back-to-back calls under one wall).
    #[serde(default)]
    pub stream: Vec<String>,
}

impl Default for SearchCell {
    fn default() -> Self {
        Self {
            query: String::new(),
            limit: 100,
            offset: 0,
            threads: 4,
            expect_path_contains: None,
            burst: 0,
            clients: 1,
            pages: 1,
            call_kind: None,
            stream: vec![],
        }
    }
}

const fn one() -> usize {
    1
}

fn sc(query: &str, limit: usize, threads: usize) -> SearchCell {
    SearchCell {
        query: query.into(),
        limit,
        threads,
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CellKind {
    Search,
    GateGolden,
    /// Time a fresh corpus open (cold launch).
    ColdOpen,
    /// Multi-fire burst: N sequential searches under one wall clock.
    ThroughputBurst,
    /// N closed-loop clients issuing searches concurrently.
    ThroughputConcurrent,
    /// Grep content search.
    Grep,
    /// Cursor-linked grep page sequence.
    GrepPagination,
}

impl SearchCell {
    pub fn to_spec(&self) -> SearchSpec {
        use crate::adapter::CallKind;
        let call = self
            .call_kind
            .as_deref()
            .map(|s| match s {
                "dir" => CallKind::Dir,
                "mixed" => CallKind::Mixed,
                "grep_plain" | "grep" => CallKind::GrepPlain,
                "grep_fuzzy" => CallKind::GrepFuzzy,
                "grep_regex" => CallKind::GrepRegex,
                _ => CallKind::File,
            })
            .unwrap_or(CallKind::File);
        SearchSpec {
            query: self.query.clone(),
            limit: self.limit,
            offset: self.offset,
            threads: self.threads,
            call,
        }
    }
}

pub fn cell_key(scenario_id: &str, repo_id: &str, axes: &serde_json::Value) -> String {
    format!("{scenario_id}\0{repo_id}\0{axes}")
}

fn cell(
    scenario_id: &str,
    repo_id: &str,
    family: &str,
    kind: CellKind,
    axes: serde_json::Value,
    search: Option<SearchCell>,
    tags: &[&str],
) -> Cell {
    Cell {
        cell_key: cell_key(scenario_id, repo_id, &axes),
        scenario_id: scenario_id.into(),
        repo_id: repo_id.into(),
        axes,
        search,
        kind,
        family: family.into(),
        tags: tags.iter().map(|s| (*s).into()).collect(),
    }
}

/// Expand named suite into concrete cells for one repo.
pub fn expand_suite(suite: &str, repo_id: &str) -> Vec<Cell> {
    // Canonical pack names + aliases (FFF_REFERENCE_PACK / OPERATIONS).
    match suite {
        "ci-gates" | "gates" => family_gates(repo_id),
        // Quality floor: labeled queries whose top-1 must be the intended file.
        "ranking-gates" | "intent-gates" | "quality" => family_ranking_intent(repo_id),
        // Latency floor probe: same queries across thread counts so a
        // per-query overhead regression on small corpora becomes visible.
        "size-sweep" | "sweep" => family_size_sweep(repo_id),
        "public-v1" | "fixture-gates" => compose_public_fixture(repo_id),
        // Fast rejection: critical gates + representative search + one grep family.
        "agent-screen" | "agent-loop" | "screen" => compose_agent_screen(repo_id),
        // Decision-grade: full required gates + fixed search/grep/cold cells.
        "agent-confirm" | "confirm" => compose_agent_confirm(repo_id),
        // Controlled interference probe (run under each load profile separately).
        "resilience-confirm" | "resilience" => compose_resilience(repo_id),
        // Full matrix / deep profiles.
        "peak-diagnose" | "peak-frontier" | "peak" | "diagnose" => compose_peak(repo_id, false),
        "full" | "comprehensive" | "all" => compose_full(repo_id),
        "throughput" => family_throughput(repo_id, true),
        "latency" => {
            let mut cells = family_search_core(repo_id, true);
            cells.extend(family_query_length_boundaries(repo_id));
            cells.extend(family_pagination_representative(repo_id));
            cells
        }
        "pagination" => {
            let mut cells = family_pagination_representative(repo_id);
            cells.extend(family_grep_pagination(repo_id));
            cells
        }
        "typing" => family_typing(repo_id),
        "calls" => family_call_kinds(repo_id),
        "constraints" => family_constraints(repo_id),
        "folder_edge" | "folder-edge" => family_folder_edge_cases(repo_id),
        "edges" => family_edges(repo_id),
        "cold" => family_cold_warm(repo_id),
        "load" => family_load(repo_id),
        "b2b" => family_back_to_back(repo_id),
        "stress" => family_stress(repo_id),
        "stability" => family_stability(repo_id),
        _ => compose_agent_screen(repo_id),
    }
}

/// Small representative probe for resilience-confirm (FFF_REFERENCE_PACK).
fn compose_resilience(repo_id: &str) -> Vec<Cell> {
    let mut c = family_gates(repo_id);
    // Tight search probe: short + medium queries at threads 1 and 4.
    for q in ["mod", "file_picker", "score"] {
        for thr in [1usize, 4] {
            let axes = json!({
                "query": q, "limit": 50, "offset": 0, "threads": thr,
                "state_id": "search_reused_index_warm",
                "load_role": "foreground_probe",
            });
            c.push(cell(
                "profile.interference.search",
                repo_id,
                "search",
                CellKind::Search,
                axes,
                Some(SearchCell {
                    query: q.into(),
                    limit: 50,
                    threads: thr,
                    call_kind: Some("file".into()),
                    ..Default::default()
                }),
                &["resilience", "search", "probe"],
            ));
        }
    }
    // One plain grep probe.
    c.extend(
        family_grep(repo_id)
            .into_iter()
            .filter(|cell| {
                cell.axes
                    .get("mode")
                    .and_then(|m| m.as_str())
                    .map(|m| m == "plain")
                    .unwrap_or(false)
            })
            .take(1),
    );
    c
}

fn compose_agent_screen(repo_id: &str) -> Vec<Cell> {
    let mut c = family_gates(repo_id);
    c.extend(family_search_core(repo_id, true));
    c.extend(
        family_query_length_boundaries(repo_id)
            .into_iter()
            .filter(|cell| cell.search.as_ref().is_some_and(|s| s.threads == 1))
            .filter(|cell| {
                cell.axes
                    .get("query_len")
                    .and_then(|v| v.as_u64())
                    .is_some_and(|len| len == 11 || len == 12)
            }),
    );
    c.extend(
        family_pagination_representative(repo_id)
            .into_iter()
            .filter(|cell| cell.search.as_ref().is_some_and(|s| s.threads == 1))
            .take(2),
    );
    // One grep mode for screen (plain only — pack: one grep cell class).
    c.extend(
        family_grep(repo_id)
            .into_iter()
            .filter(|cell| {
                cell.axes
                    .get("mode")
                    .and_then(|m| m.as_str())
                    .map(|m| m == "plain")
                    .unwrap_or(true)
            })
            .take(2),
    );
    c.extend(family_call_kinds(repo_id).into_iter().take(4));
    c.extend(family_throughput(repo_id, false).into_iter().take(2));
    c
}

fn compose_agent_confirm(repo_id: &str) -> Vec<Cell> {
    let mut c = family_gates(repo_id);
    c.extend(family_search_core(repo_id, true));
    c.extend(family_query_length_boundaries(repo_id));
    c.extend(family_pagination_representative(repo_id));
    c.extend(family_call_kinds(repo_id));
    c.extend(family_grep(repo_id));
    c.extend(family_grep_pagination(repo_id));
    c.extend(family_throughput(repo_id, false));
    c.extend(family_typing(repo_id));
    c.extend(family_cold_warm(repo_id));
    c.extend(family_stability(repo_id).into_iter().take(2));
    // Mutation is executed as a gate phase (isolated fixture), not cell matrix.
    c
}

/// Gates against the content-addressed public-v1 fixture query bank.
fn compose_public_fixture(_repo_id: &str) -> Vec<Cell> {
    // Cells use repo_id "public-v1"; runner must resolve that corpus.
    let repo_id = "public-v1";
    let mut out = Vec::new();
    for q in crate::fixture::query_bank_v1() {
        if q.expect.is_empty() {
            continue;
        }
        let (kind, call_kind, scenario) = match q.kind {
            crate::fixture::QueryKind::Search => (
                CellKind::GateGolden,
                Some("file".into()),
                "gate.golden.public",
            ),
            crate::fixture::QueryKind::GrepPlain => (
                CellKind::Grep,
                Some("grep_plain".into()),
                "gate.grep.public",
            ),
            crate::fixture::QueryKind::GrepRegex => (
                CellKind::Grep,
                Some("grep_regex".into()),
                "gate.grep.public",
            ),
        };
        let mut axes = json!({
            "query": q.query,
            "limit": 50,
            "offset": 0,
            "threads": 4,
            "expect": q.expect,
            "class": q.class,
            "must_hit": q.must_hit,
            "state_id": "search_reused_index_warm",
            "query_bank": crate::fixture::QUERY_BANK_VERSION,
            "mode": match q.kind {
                crate::fixture::QueryKind::GrepRegex => "regex",
                crate::fixture::QueryKind::GrepPlain => "plain",
                _ => "file",
            },
        });
        if let Some(ln) = q.expect_line {
            axes.as_object_mut()
                .unwrap()
                .insert("expect_line".into(), json!(ln));
        }
        let tags: Vec<&str> = if q.must_hit {
            vec!["gate", "must_hit", q.class]
        } else {
            vec!["gate", q.class]
        };
        out.push(cell(
            scenario,
            repo_id,
            if matches!(kind, CellKind::Grep) {
                "grep"
            } else {
                "quality"
            },
            kind,
            axes,
            Some(SearchCell {
                query: q.query.into(),
                limit: 50,
                offset: 0,
                threads: 4,
                expect_path_contains: Some(q.expect.into()),
                call_kind,
                ..Default::default()
            }),
            &tags,
        ));
    }
    out
}

fn compose_peak(repo_id: &str, heavy: bool) -> Vec<Cell> {
    let mut c = family_gates(repo_id);
    c.extend(family_search_core(repo_id, false));
    c.extend(family_query_length_boundaries(repo_id));
    c.extend(family_call_kinds(repo_id));
    c.extend(family_pagination(repo_id));
    c.extend(family_constraints(repo_id));
    c.extend(family_folder_edge_cases(repo_id));
    c.extend(family_grep(repo_id));
    c.extend(family_grep_pagination(repo_id));
    c.extend(family_throughput(repo_id, heavy));
    c.extend(family_typing(repo_id));
    c.extend(family_back_to_back(repo_id));
    c.extend(family_interleave(repo_id));
    c.extend(family_cold_warm(repo_id));
    c.extend(family_edges(repo_id));
    c.extend(family_stability(repo_id));
    if heavy {
        c.extend(family_load(repo_id));
        c.extend(family_combo(repo_id));
        c.extend(family_unique(repo_id));
        c.extend(family_stress(repo_id));
    }
    c
}

fn compose_full(repo_id: &str) -> Vec<Cell> {
    compose_peak(repo_id, true)
}

// ─── Quality ───────────────────────────────────────────────────────────────

fn family_gates(repo_id: &str) -> Vec<Cell> {
    // Prefer path-anchored queries so ranking goldens stay stable under noise files.
    let cases = [
        ("Cargo.toml", "Cargo.toml"),
        ("fff-core/src/file_picker", "file_picker.rs"),
        ("fff-core/src/score", "score.rs"),
        ("fuzzy_search_bench", "fuzzy_search_bench"),
        ("readme", "README.md"),
        ("fff-core/src/grep", "grep.rs"),
        ("fff-core/src/background_watcher", "background_watcher.rs"),
        ("fff-core/src/path_utils", "path_utils.rs"),
        ("fff-query-parser", "fff-query-parser"),
        ("AGENTS.md", "AGENTS.md"),
    ];
    cases
        .into_iter()
        .map(|(query, expect)| {
            let axes = json!({
                "query": query, "limit": 50, "offset": 0, "threads": 4, "expect": expect,
            });
            cell(
                "gate.golden",
                repo_id,
                "quality",
                CellKind::GateGolden,
                axes,
                Some(SearchCell {
                    query: query.into(),
                    limit: 50,
                    offset: 0,
                    threads: 4,
                    expect_path_contains: Some(expect.into()),
                    burst: 0,
                    clients: 1,
                    pages: 1,
                    call_kind: None,
                    stream: vec![],
                }),
                &["quality", "gate"],
            )
        })
        .collect()
}

/// Quality floor: labeled user queries, each pinned to the file the user
/// actually wants at rank 0 — the regressions that matter when typing a
/// filename, because fuzzy scoring alone buries them under test fixtures
/// and vendored copies. Expectations are substrings so corpus copies of
/// the same file cannot break them.
fn family_ranking_intent(repo_id: &str) -> Vec<Cell> {
    // (query, expected path substring, must_be_top1). Queries whose top-1 is
    // genuinely ambiguous on a corpus that also contains bench patches /
    // vendored copies (e.g. `frecncy` matches `patches/*frecency*.patch`)
    // stay top-5 gates instead of top-1 gates.
    let cases = [
        ("score.rs", "fff-core/src/score.rs", true),
        ("file_picker", "fff-core/src/file_picker.rs", true),
        ("lua_types", "nvim/src/lua_types.rs", true),
        ("picker_ui", "picker_ui/picker_ui.lua", true),
        ("frecncy", "dbs/frecency.rs", false),
        ("src/dbs", "dbs/mod.rs", true),
        ("fuzzy_search_bench", "benches/fuzzy_search_bench.rs", true),
        ("conf", "lua/fff/conf.lua", true),
        ("README", "README.md", true),
        ("path_utils", "path_utils.rs", true),
    ];
    cases
        .into_iter()
        .map(|(query, expect, must_hit)| {
            let axes = json!({
                "query": query, "limit": 50, "offset": 0, "threads": 4, "expect": expect,
                "must_hit": must_hit,
            });
            let tags: &[&str] = if must_hit {
                &["quality", "gate", "must_hit", "ranking"]
            } else {
                &["quality", "gate", "ranking"]
            };
            cell(
                "gate.ranking",
                repo_id,
                "quality",
                CellKind::GateGolden,
                axes,
                Some(SearchCell {
                    query: query.into(),
                    limit: 50,
                    offset: 0,
                    threads: 4,
                    expect_path_contains: Some(expect.into()),
                    burst: 0,
                    clients: 1,
                    pages: 1,
                    call_kind: None,
                    stream: vec![],
                }),
                tags,
            )
        })
        .collect()
}

/// Latency floor probe: representative queries at every thread count the
/// picker can be asked for. Reported, not gated — a per-query overhead
/// regression on small corpora shows up here before it reaches the gates.
fn family_size_sweep(repo_id: &str) -> Vec<Cell> {
    let mut cells = Vec::new();
    for threads in [1usize, 4] {
        for query in ["score", "file_picker", "readme", "mod"] {
            let axes = json!({
                "query": query, "limit": 100, "offset": 0, "threads": threads,
            });
            cells.push(cell(
                "sweep.search",
                repo_id,
                "sweep",
                CellKind::Search,
                axes,
                Some(SearchCell {
                    query: query.into(),
                    limit: 100,
                    offset: 0,
                    threads,
                    burst: 0,
                    clients: 1,
                    pages: 1,
                    call_kind: Some("file".into()),
                    stream: vec![],
                    ..Default::default()
                }),
                &["sweep", "latency"],
            ));
        }
    }
    cells
}

// ─── Core search matrix (selectivity × threads) ────────────────────────────

fn family_search_core(repo_id: &str, agent_small: bool) -> Vec<Cell> {
    let queries: &[&str] = if agent_small {
        &["mod", "controller", "file_picker", "score"]
    } else {
        &[
            "a",
            "s",
            "t",
            "mod",
            "test",
            "main",
            "lib",
            "init",
            "controller",
            "file_picker",
            "src/lib",
            "crates/fff",
            "contrlr",
            "fuzze",
            "fuzzy_search",
            "background_watcher",
            "QueryParser",
            "GrepMode",
            "SharedFilePicker",
            "path_utils",
        ]
    };
    let threads: &[usize] = if agent_small {
        &[1, 4]
    } else {
        &[1, 2, 4, 8, 16]
    };
    let limits: &[usize] = if agent_small { &[100] } else { &[10, 50, 100] };
    let mut out = Vec::new();
    for q in queries {
        for &thr in threads {
            for &limit in limits {
                // keep matrix from exploding: only multi-limit for short queries
                if limits.len() > 1 && thr > 4 && q.len() > 4 {
                    continue;
                }
                let axes = json!({
                    "query": q, "limit": limit, "offset": 0, "threads": thr,
                    "cache_state": "cold_query",
                    "state_id": "search_reused_index_cold_query",
                });
                out.push(cell(
                    "profile.search.matrix",
                    repo_id,
                    "search",
                    CellKind::Search,
                    axes,
                    Some(SearchCell {
                        query: (*q).into(),
                        limit,
                        offset: 0,
                        threads: thr,
                        call_kind: Some("file".into()),
                        ..Default::default()
                    }),
                    &["search", "selectivity", "threads"],
                ));
            }
        }
    }
    out
}

// ─── Query-length algorithm boundaries ────────────────────────────────────

fn family_query_length_boundaries(repo_id: &str) -> Vec<Cell> {
    let cases = [
        ("file_picke", 10usize),
        ("file_picker", 11),
        ("file_picker_", 12),
        ("file_picker_r", 13),
    ];
    let mut out = Vec::new();
    for (query, query_len) in cases {
        for threads in [1usize, 4] {
            let axes = json!({
                "query": query,
                "query_len": query_len,
                "limit": 50,
                "offset": 0,
                "threads": threads,
                "boundary": "max_typos_transition",
                "cache_state": "cold_query",
                "state_id": "search_reused_index_cold_query",
            });
            out.push(cell(
                "profile.search.query_length_boundary",
                repo_id,
                "latency",
                CellKind::Search,
                axes,
                Some(SearchCell {
                    query: query.into(),
                    limit: 50,
                    threads,
                    call_kind: Some("file".into()),
                    ..Default::default()
                }),
                &["search", "latency", "query_length", "algorithm_boundary"],
            ));
        }
    }
    out
}

fn family_pagination_representative(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    for (page, offset) in [("first", 0usize), ("next", 50), ("deep", 200)] {
        for threads in [1usize, 4] {
            let axes = json!({
                "query": "a",
                "page": page,
                "limit": 50,
                "offset": offset,
                "threads": threads,
                "cache_state": "exact_repeat_warm",
                "state_id": "search_reused_index_exact_query_warm",
            });
            out.push(cell(
                "profile.search.pagination",
                repo_id,
                "pagination",
                CellKind::Search,
                axes,
                Some(SearchCell {
                    query: "a".into(),
                    limit: 50,
                    offset,
                    threads,
                    call_kind: Some("file".into()),
                    ..Default::default()
                }),
                &["search", "pagination", page],
            ));
        }
    }
    out
}

// ─── Pagination depth ──────────────────────────────────────────────────────

fn family_pagination(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    for q in ["test", "a", "mod", "file"] {
        for offset in [0usize, 25, 50, 100, 200, 400] {
            for limit in [5usize, 10, 25, 50, 100] {
                if offset > 100 && limit > 50 {
                    continue;
                }
                let axes = json!({
                    "query": q, "limit": limit, "offset": offset, "threads": 4,
                });
                out.push(cell(
                    "profile.search.pagination",
                    repo_id,
                    "pagination",
                    CellKind::Search,
                    axes,
                    Some(SearchCell {
                        query: q.into(),
                        limit,
                        offset,
                        threads: 4,
                        call_kind: Some("file".into()),
                        ..Default::default()
                    }),
                    &["search", "pagination"],
                ));
            }
        }
    }
    out
}

// ─── Query constraints (extension / path / git-ish syntax) ─────────────────

fn family_constraints(repo_id: &str) -> Vec<Cell> {
    let cases = [
        ("mod .rs", "ext_rs"),
        ("FilePicker .rs", "ext_name"),
        ("picker path:crates", "path_segment"),
        ("score path:fff-core", "path_core"),
        ("test !.md", "exclude_md"),
        ("main .toml", "toml"),
        ("search .lua", "ext_lua"),
        ("bench .rs", "ext_bench"),
        ("*.rs score", "glob_rs"),
        ("**/*picker*", "glob_star"),
        ("init path:lua", "lua_path"),
        ("grep path:fff-core", "grep_path"),
        ("FileItem .rs path:types", "multi_constraint"),
        ("test !.rs", "exclude_rs"),
        ("mod .rs !.toml", "ext_and_exclude"),
        ("watcher path:background", "watcher_path"),
    ];
    let mut out = Vec::new();
    for (query, tag) in cases {
        for thr in [1usize, 4] {
            let axes = json!({
                "query": query, "limit": 50, "offset": 0, "threads": thr, "constraint": tag,
            });
            out.push(cell(
                "profile.search.constraints",
                repo_id,
                "constraint",
                CellKind::Search,
                axes,
                Some(SearchCell {
                    query: query.into(),
                    limit: 50,
                    threads: thr,
                    call_kind: Some("file".into()),
                    ..Default::default()
                }),
                &["search", "constraint", tag],
            ));
        }
    }
    out
}

// ─── Call kinds matrix (file / dir / mixed) ────────────────────────────────

fn family_call_kinds(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    for (call, query) in [
        ("file", "mod"),
        ("file", "FilePicker"),
        ("file", "score"),
        ("dir", "src"),
        ("dir", "crates"),
        ("dir", "lua"),
        ("mixed", "mod"),
        ("mixed", "src/"),
        ("mixed", "crates/fff"),
    ] {
        let axes = json!({
            "query": query, "call": call, "limit": 50, "threads": 4,
        });
        out.push(cell(
            "profile.call.kinds",
            repo_id,
            "call",
            CellKind::Search,
            axes,
            Some(SearchCell {
                query: query.into(),
                limit: 50,
                threads: 4,
                call_kind: Some(call.into()),
                ..Default::default()
            }),
            &["call", call],
        ));
    }
    out
}

// ─── Typing simulation (keystroke prefixes) ────────────────────────────────

fn family_typing(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    // Simulate typing "file_picker" one char at a time
    let prefixes_fp: Vec<&str> = vec![
        "f",
        "fi",
        "fil",
        "file",
        "file_",
        "file_p",
        "file_pi",
        "file_pic",
        "file_pick",
        "file_picke",
        "file_picker",
    ];
    let stream_fp: Vec<String> = prefixes_fp.iter().map(|p| format!("file:{p}")).collect();
    out.push(cell(
        "profile.typing.prefix_stream",
        repo_id,
        "typing",
        CellKind::ThroughputBurst,
        json!({"prefix": "file_picker", "steps": stream_fp.len(), "direction": "forward"}),
        Some(SearchCell {
            query: "file_picker".into(),
            limit: 50,
            threads: 4,
            burst: stream_fp.len(),
            stream: stream_fp,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["typing", "prefix", "forward"],
    ));
    // Simulate typing "score" then backspace then retype
    let bounce: Vec<String> = vec![
        "file:s".into(),
        "file:sc".into(),
        "file:sco".into(),
        "file:scor".into(),
        "file:score".into(),
        "file:scor".into(),
        "file:score".into(),
    ];
    out.push(cell(
        "profile.typing.bounce_edit",
        repo_id,
        "typing",
        CellKind::ThroughputBurst,
        json!({"prefix": "score", "steps": bounce.len(), "direction": "bounce"}),
        Some(SearchCell {
            query: "score".into(),
            limit: 50,
            threads: 4,
            burst: bounce.len(),
            stream: bounce,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["typing", "bounce", "edit"],
    ));
    // Short prefix latency sweep
    for (q, thr) in [
        ("m", 1usize),
        ("mo", 2),
        ("mod", 4),
        ("mod.", 4),
        ("mod.r", 4),
        ("mod.rs", 4),
    ] {
        let axes = json!({"query": q, "threads": thr, "typing_sim": true});
        out.push(cell(
            "profile.typing.prefix_stream",
            repo_id,
            "typing",
            CellKind::Search,
            axes,
            Some(SearchCell {
                query: q.into(),
                limit: 50,
                threads: thr,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["typing", "latency", "short"],
        ));
    }
    out
}

// ─── Grep modes ────────────────────────────────────────────────────────────

fn family_grep(repo_id: &str) -> Vec<Cell> {
    let modes = [
        ("plain", "FilePicker"),
        ("fuzzy", "FilePicker"),
        ("regex", "FilePicker|FileItem"),
        ("plain", "struct FffAdapter"),
        ("plain", "fn fuzzy_search"),
        ("plain", "impl FilePicker"),
        ("plain", "SharedFilePicker"),
        ("fuzzy", "background_watcher"),
        ("regex", "pub fn [a-z_]+"),
        ("plain", "TODO"),
        ("plain", "FIXME"),
        ("regex", "error\\[E[0-9]+\\]"),
        ("plain", "use crate::"),
        ("fuzzy", "QueryParser"),
        ("plain", "max_threads"),
    ];
    modes
        .into_iter()
        .map(|(mode, q)| {
            let call = match mode {
                "fuzzy" => "grep_fuzzy",
                "regex" => "grep_regex",
                _ => "grep_plain",
            };
            let axes = json!({ "query": q, "mode": mode, "page_limit": 50 });
            cell(
                "profile.grep.modes",
                repo_id,
                "grep",
                CellKind::Grep,
                axes,
                Some(SearchCell {
                    query: q.into(),
                    limit: 50,
                    threads: 4,
                    call_kind: Some(call.into()),
                    ..Default::default()
                }),
                &["grep", mode],
            )
        })
        .collect()
}

fn family_grep_pagination(repo_id: &str) -> Vec<Cell> {
    [1usize, 2, 3]
        .into_iter()
        .map(|pages| {
            let axes = json!({
                "query": "FilePicker",
                "mode": "plain",
                "page_limit": 10,
                "pages": pages,
                "pagination": "cursor_linked",
            });
            cell(
                "profile.grep.pagination",
                repo_id,
                "pagination",
                CellKind::GrepPagination,
                axes,
                Some(SearchCell {
                    query: "FilePicker".into(),
                    limit: 10,
                    threads: 4,
                    pages,
                    call_kind: Some("grep_plain".into()),
                    ..Default::default()
                }),
                &["grep", "pagination", "cursor_linked"],
            )
        })
        .collect()
}

// ─── Throughput / multi-fire ───────────────────────────────────────────────

fn family_throughput(repo_id: &str, heavy: bool) -> Vec<Cell> {
    let bursts: &[(usize, &str)] = if heavy {
        &[
            (5, "mod"),
            (10, "mod"),
            (25, "test"),
            (50, "test"),
            (100, "a"),
            (200, "a"),
            (50, "file_picker"),
            (25, "controller"),
            (40, "score"),
            (30, "grep"),
        ]
    } else {
        &[(10, "mod"), (25, "test"), (50, "a"), (20, "file_picker")]
    };
    let mut out = Vec::new();
    for &(burst, q) in bursts {
        for thr in [1usize, 4, 8] {
            if !heavy && thr == 8 {
                continue;
            }
            let axes = json!({
                "query": q,
                "operations": burst,
                "threads_per_request": thr,
                "client_concurrency": 1,
                "arrival_model": "closed_loop",
                "worker_lifecycle": "caller_thread",
                "cache_state": "exact_repeat_warm",
                "state_id": "search_reused_index_exact_query_warm",
                "limit": 50,
            });
            out.push(cell(
                "profile.throughput.burst",
                repo_id,
                "throughput",
                CellKind::ThroughputBurst,
                axes,
                Some(SearchCell {
                    query: q.into(),
                    limit: 50,
                    threads: thr,
                    burst,
                    call_kind: Some("file".into()),
                    ..Default::default()
                }),
                &["throughput", "closed_loop", "single_client"],
            ));
        }
    }
    for (call, q, burst) in [
        ("mixed", "mod", 20usize),
        ("dir", "src", 15),
        ("grep_plain", "FilePicker", 10),
        ("grep_fuzzy", "score", 10),
    ] {
        let axes = json!({
            "call": call,
            "query": q,
            "operations": burst,
            "threads_per_request": 4,
            "client_concurrency": 1,
            "arrival_model": "closed_loop",
            "worker_lifecycle": "caller_thread",
            "cache_state": "exact_repeat_warm",
            "state_id": "search_reused_index_exact_query_warm",
        });
        out.push(cell(
            "profile.throughput.burst_call",
            repo_id,
            "throughput",
            CellKind::ThroughputBurst,
            axes,
            Some(SearchCell {
                query: q.into(),
                limit: 50,
                threads: 4,
                burst,
                call_kind: Some(call.into()),
                ..Default::default()
            }),
            &["throughput", "closed_loop", "single_client", call],
        ));
    }

    let clients: &[usize] = if heavy { &[2, 4, 8] } else { &[2, 4] };
    let operations = if heavy { 512 } else { 96 };
    for query in ["file_picker", "file_picker_"] {
        for &client_concurrency in clients {
            for threads_per_request in if heavy { vec![1usize, 4] } else { vec![1] } {
                let axes = json!({
                    "query": query,
                    "operations": operations,
                    "threads_per_request": threads_per_request,
                    "client_concurrency": client_concurrency,
                    "arrival_model": "closed_loop",
                    "worker_lifecycle": "per_sample",
                    "cache_state": "exact_repeat_warm",
                    "state_id": "search_reused_index_exact_query_warm",
                    "limit": 50,
                });
                out.push(cell(
                    "profile.throughput.concurrent",
                    repo_id,
                    "throughput",
                    CellKind::ThroughputConcurrent,
                    axes,
                    Some(SearchCell {
                        query: query.into(),
                        limit: 50,
                        threads: threads_per_request,
                        burst: operations,
                        clients: client_concurrency,
                        call_kind: Some("file".into()),
                        ..Default::default()
                    }),
                    &["throughput", "closed_loop", "concurrent_clients"],
                ));
            }
        }
    }
    out
}

// ─── Heavy load (large fanout + deep page + high threads) ──────────────────

fn family_load(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    for q in ["a", "s", "t", "e"] {
        for thr in [4usize, 8, 16] {
            for limit in [100usize, 500] {
                let axes = json!({
                    "query": q, "limit": limit, "threads": thr, "load": "fanout",
                });
                out.push(cell(
                    "profile.load.fanout",
                    repo_id,
                    "load",
                    CellKind::Search,
                    axes,
                    Some(SearchCell {
                        query: q.into(),
                        limit,
                        threads: thr,
                        call_kind: Some("file".into()),
                        ..Default::default()
                    }),
                    &["load", "fanout", "heavy"],
                ));
            }
        }
    }
    for offset in [200usize, 500, 1000] {
        let axes = json!({
            "query": "test", "limit": 100, "offset": offset, "threads": 8, "load": "deep_page",
        });
        out.push(cell(
            "profile.load.deep_page",
            repo_id,
            "load",
            CellKind::Search,
            axes,
            Some(SearchCell {
                query: "test".into(),
                limit: 100,
                offset,
                threads: 8,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["load", "pagination", "heavy"],
        ));
    }
    for (burst, thr) in [(100usize, 4usize), (200, 8), (500, 8), (100, 16)] {
        let axes = json!({ "query": "a", "burst": burst, "threads": thr, "load": "sustained" });
        out.push(cell(
            "profile.load.sustained_burst",
            repo_id,
            "load",
            CellKind::ThroughputBurst,
            axes,
            Some(SearchCell {
                query: "a".into(),
                limit: 100,
                threads: thr,
                burst,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["load", "throughput", "heavy"],
        ));
    }
    // Mixed-call load
    let axes = json!({ "load": "mixed_burst", "burst": 40 });
    let mut stream = Vec::new();
    for i in 0..40 {
        stream.push(if i % 3 == 0 {
            "file:a".into()
        } else if i % 3 == 1 {
            "mixed:mod".into()
        } else {
            "grep_plain:FilePicker".into()
        });
    }
    out.push(cell(
        "profile.load.mixed_call_burst",
        repo_id,
        "load",
        CellKind::ThroughputBurst,
        axes,
        Some(SearchCell {
            query: "a".into(),
            limit: 50,
            threads: 8,
            burst: stream.len(),
            stream,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["load", "b2b", "heavy"],
    ));
    out
}

// ─── Cold launch / warm open ───────────────────────────────────────────────

fn family_cold_warm(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    // Cold open (timed re-open)
    let axes = json!({ "phase": "cold_open" });
    out.push(cell(
        "profile.cold.open",
        repo_id,
        "cold",
        CellKind::ColdOpen,
        axes,
        None,
        &["cold", "launch"],
    ));
    // First search after open (warm index)
    let axes = json!({ "query": "file_picker", "phase": "first_search", "threads": 4 });
    out.push(cell(
        "profile.cold.first_search",
        repo_id,
        "cold",
        CellKind::Search,
        axes,
        Some(SearchCell {
            query: "file_picker".into(),
            limit: 50,
            offset: 0,
            threads: 4,
            expect_path_contains: None,
            burst: 0,
            clients: 1,
            pages: 1,
            call_kind: None,
            stream: vec![],
        }),
        &["cold", "warm_index"],
    ));
    // Second search (hot caches)
    let axes = json!({ "query": "file_picker", "phase": "hot_search", "threads": 4 });
    out.push(cell(
        "profile.cold.hot_search",
        repo_id,
        "cold",
        CellKind::Search,
        axes,
        Some(SearchCell {
            query: "file_picker".into(),
            limit: 50,
            offset: 0,
            threads: 4,
            expect_path_contains: None,
            burst: 0,
            clients: 1,
            pages: 1,
            call_kind: None,
            stream: vec![],
        }),
        &["cold", "hot"],
    ));
    out
}

// ─── Edge cases ────────────────────────────────────────────────────────────

fn family_edges(repo_id: &str) -> Vec<Cell> {
    let owned: Vec<(String, &str)> = vec![
        ("".into(), "empty_query"),
        (" ".into(), "whitespace"),
        ("   ".into(), "whitespace3"),
        ("x".into(), "single_char"),
        ("zz".into(), "two_char_rare"),
        ("zzzzznomatch999".into(), "zero_match"),
        ("nomatch_xyzzy_plugh".into(), "zero_match_long"),
        ("a".repeat(64), "long_query_64"),
        ("a".repeat(128), "long_query_128"),
        ("../../etc/passwd".into(), "path_traversal_noise"),
        ("..\\..\\windows".into(), "backslash_noise"),
        ("mod OR test".into(), "boolean_ish"),
        ("mod AND test".into(), "boolean_and"),
        ("文件".into(), "unicode_cjk"),
        ("файл".into(), "unicode_cyrillic"),
        ("🚀".into(), "emoji"),
        ("File_Picker".into(), "snake_camel"),
        ("FILEPICKER".into(), "all_caps"),
        ("filePicker".into(), "camel"),
        ("***".into(), "glob_noise"),
        (".rs".into(), "ext_only"),
        (".toml".into(), "ext_toml"),
        ("///".into(), "slashes"),
        ("@@@".into(), "at_signs"),
        ("\"quoted\"".into(), "quotes"),
        ("'sq'".into(), "single_quotes"),
        ("mod\ttab".into(), "tab_char"),
        ("mod\nline".into(), "newline_noise"),
        (
            String::from_utf8_lossy(&[0xC3, 0x28]).into_owned(),
            "invalid_utf8_ish",
        ),
    ];
    let mut out = Vec::new();
    for (query, tag) in owned {
        for call in ["file", "mixed"] {
            let axes = json!({
                "query": query, "edge": tag, "call": call, "threads": 4, "limit": 50,
            });
            out.push(cell(
                "profile.edge.cases",
                repo_id,
                "edge",
                CellKind::Search,
                axes,
                Some(SearchCell {
                    query: query.clone(),
                    limit: 50,
                    threads: 4,
                    call_kind: Some(call.into()),
                    ..Default::default()
                }),
                &["edge", tag, call],
            ));
        }
    }
    out
}

// ─── Stress / pathological ─────────────────────────────────────────────────

fn family_stress(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    // Pathological fanout + deep page + max threads
    out.push(cell(
        "profile.stress.result_fanout",
        repo_id,
        "stress",
        CellKind::Search,
        json!({
            "query": "a",
            "limit": 1000,
            "threads_per_request": 16,
            "corpus_scale": "supplied_corpus",
            "stress_kind": "result_fanout",
        }),
        Some(SearchCell {
            query: "a".into(),
            limit: 1000,
            threads: 16,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["stress", "fanout"],
    ));
    out.push(cell(
        "profile.stress.single_client_saturation",
        repo_id,
        "stress",
        CellKind::ThroughputBurst,
        json!({
            "query": "a",
            "operations": 1000,
            "threads_per_request": 8,
            "client_concurrency": 1,
            "arrival_model": "closed_loop",
            "corpus_scale": "supplied_corpus",
        }),
        Some(SearchCell {
            query: "a".into(),
            limit: 100,
            threads: 8,
            burst: 1000,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["stress", "throughput"],
    ));
    for (clients, threads, operations) in [(8usize, 1usize, 1000usize), (4, 4, 512)] {
        let axes = json!({
            "query": "file_picker_",
            "operations": operations,
            "threads_per_request": threads,
            "client_concurrency": clients,
            "arrival_model": "closed_loop",
            "worker_lifecycle": "per_sample",
            "stress_kind": if threads == 1 { "client_saturation" } else { "nested_parallelism" },
        });
        out.push(cell(
            "profile.stress.concurrent_saturation",
            repo_id,
            "stress",
            CellKind::ThroughputConcurrent,
            axes,
            Some(SearchCell {
                query: "file_picker_".into(),
                limit: 50,
                threads,
                burst: operations,
                clients,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["stress", "throughput", "concurrent_clients"],
        ));
    }

    // Rapid full alphabet thrash
    let stream: Vec<String> = ('a'..='z').map(|c| format!("file:{c}")).collect();
    out.push(cell(
        "profile.stress.alphabet_thrash",
        repo_id,
        "stress",
        CellKind::ThroughputBurst,
        json!({"pattern": "alphabet", "n": 26}),
        Some(SearchCell {
            query: "z".into(),
            limit: 50,
            threads: 4,
            burst: stream.len(),
            stream,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["stress", "thrash"],
    ));
    // Alternating empty and full
    let mut stream = Vec::new();
    for _ in 0..20 {
        stream.push("file:".into());
        stream.push("file:mod".into());
    }
    out.push(cell(
        "profile.stress.empty_full_alt",
        repo_id,
        "stress",
        CellKind::ThroughputBurst,
        json!({"pattern": "empty_full"}),
        Some(SearchCell {
            query: "mod".into(),
            limit: 50,
            threads: 4,
            burst: stream.len(),
            stream,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["stress", "empty"],
    ));
    out
}

// ─── Combinatorial: constraint × threads × load-ish ────────────────────────

fn family_combo(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    let queries = ["mod .rs", "test path:crates", "score .rs"];
    for q in queries {
        for thr in [1usize, 4, 8] {
            for limit in [10usize, 100] {
                let axes = json!({
                    "query": q, "threads": thr, "limit": limit, "offset": 0, "combo": true,
                });
                out.push(cell(
                    "profile.combo.constraint_threads_limit",
                    repo_id,
                    "combo",
                    CellKind::Search,
                    axes,
                    Some(SearchCell {
                        query: q.into(),
                        limit,
                        offset: 0,
                        threads: thr,
                        call_kind: Some("file".into()),
                        ..Default::default()
                    }),
                    &["combo", "constraint", "threads"],
                ));
            }
        }
    }
    let axes =
        json!({ "query": "mod .rs", "burst": 30, "threads": 4, "combo": "throughput_constraint" });
    out.push(cell(
        "profile.combo.throughput_constraint",
        repo_id,
        "combo",
        CellKind::ThroughputBurst,
        axes,
        Some(SearchCell {
            query: "mod .rs".into(),
            limit: 50,
            threads: 4,
            burst: 30,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["combo", "throughput", "constraint"],
    ));
    out
}

// ─── Unique / exotic scenarios ─────────────────────────────────────────────

fn family_unique(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    out.push(cell(
        "profile.unique.empty_then_query",
        repo_id,
        "unique",
        CellKind::ThroughputBurst,
        json!({"pattern": "empty_then_query"}),
        Some(SearchCell {
            query: "mod".into(),
            limit: 50,
            threads: 4,
            burst: 2,
            stream: vec!["file:".into(), "file:mod".into()],
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["unique", "empty"],
    ));
    for thr in [1usize, 16] {
        out.push(cell(
            "profile.unique.rare_token_threads",
            repo_id,
            "unique",
            CellKind::Search,
            json!({"query": "background_watcher", "threads": thr}),
            Some(SearchCell {
                query: "background_watcher".into(),
                limit: 20,
                threads: thr,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["unique", "rare"],
        ));
    }
    out.push(cell(
        "profile.unique.dir_slash",
        repo_id,
        "unique",
        CellKind::Search,
        json!({"query": "src/", "call": "mixed"}),
        Some(SearchCell {
            query: "src/".into(),
            limit: 50,
            threads: 4,
            call_kind: Some("mixed".into()),
            ..Default::default()
        }),
        &["unique", "dir_slash"],
    ));
    out.push(cell(
        "profile.unique.repeat_identical",
        repo_id,
        "unique",
        CellKind::ThroughputBurst,
        json!({"pattern": "repeat", "n": 20}),
        Some(SearchCell {
            query: "FilePicker".into(),
            limit: 50,
            threads: 4,
            burst: 20,
            stream: (0..20).map(|_| "file:FilePicker".into()).collect(),
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["unique", "repeat"],
    ));
    for off in [0usize, 1, 25, 99] {
        out.push(cell(
            "profile.unique.offset_thrash",
            repo_id,
            "unique",
            CellKind::Search,
            json!({"query": "test", "offset": off, "limit": 10}),
            Some(SearchCell {
                query: "test".into(),
                limit: 10,
                offset: off,
                threads: 4,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["unique", "offset"],
        ));
    }
    out
}

// ─── Back-to-back sessions ─────────────────────────────────────────────────

fn family_back_to_back(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    let stream: Vec<String> = vec![
        "file:FilePicker".into(),
        "grep_plain:struct FilePicker".into(),
        "dir:src".into(),
        "file:score".into(),
        "mixed:mod".into(),
    ];
    out.push(cell(
        "profile.b2b.session_mix",
        repo_id,
        "b2b",
        CellKind::ThroughputBurst,
        json!({ "pattern": "session_mix", "steps": stream.len() }),
        Some(SearchCell {
            query: "FilePicker".into(),
            limit: 50,
            threads: 4,
            burst: stream.len(),
            stream,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["b2b", "session", "mixed_calls"],
    ));
    let axes = json!({ "pattern": "alt_file_mixed", "query": "mod", "reps": 10 });
    let mut stream = Vec::new();
    for _ in 0..10 {
        stream.push("file:mod".into());
        stream.push("mixed:mod".into());
    }
    out.push(cell(
        "profile.b2b.alt_file_mixed",
        repo_id,
        "b2b",
        CellKind::ThroughputBurst,
        axes,
        Some(SearchCell {
            query: "mod".into(),
            limit: 50,
            threads: 4,
            burst: stream.len(),
            stream,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["b2b", "alternate"],
    ));
    let switches = ["a", "mod", "test", "score", "picker", "a", "mod", "test"];
    let stream: Vec<String> = switches.iter().map(|s| format!("file:{s}")).collect();
    let axes = json!({ "pattern": "query_thrash", "stream": switches });
    out.push(cell(
        "profile.b2b.query_thrash",
        repo_id,
        "b2b",
        CellKind::ThroughputBurst,
        axes,
        Some(SearchCell {
            query: "test".into(),
            limit: 50,
            threads: 4,
            burst: stream.len(),
            stream,
            call_kind: Some("file".into()),
            ..Default::default()
        }),
        &["b2b", "thrash"],
    ));
    out
}

// ─── Interleave search+grep under one burst ────────────────────────────────

fn family_interleave(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    let patterns: &[(&str, &[&str])] = &[
        (
            "search_grep_search",
            &[
                "file:FilePicker",
                "grep_plain:struct FilePicker",
                "file:FilePicker",
            ],
        ),
        (
            "triple_alt",
            &[
                "file:mod",
                "grep_plain:mod",
                "mixed:mod",
                "file:mod",
                "grep_fuzzy:mod",
            ],
        ),
        (
            "type_then_grep",
            &[
                "file:f",
                "file:fi",
                "file:fil",
                "file:file",
                "grep_plain:file",
            ],
        ),
        (
            "dir_then_file",
            &["dir:src", "file:src/lib", "mixed:src/", "file:lib"],
        ),
    ];
    for (name, steps) in patterns {
        let stream: Vec<String> = steps.iter().map(|s| (*s).into()).collect();
        let axes = json!({ "pattern": name, "stream": stream, "threads": 4 });
        out.push(cell(
            "profile.interleave.mixed_api",
            repo_id,
            "interleave",
            CellKind::ThroughputBurst,
            axes,
            Some(SearchCell {
                query: name.to_string(),
                limit: 50,
                threads: 4,
                burst: stream.len(),
                stream,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["interleave", "b2b", name],
        ));
    }
    out
}

// ─── Stability: same cell many times (variance / regression) ───────────────

fn family_stability(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();
    for q in ["mod", "FilePicker", "a", "score"] {
        for thr in [1usize, 4] {
            let axes = json!({
                "query": q, "threads": thr, "pattern": "stability_repeat", "reps": 15,
            });
            let stream: Vec<String> = (0..15).map(|_| format!("file:{q}")).collect();
            out.push(cell(
                "profile.stability.repeat",
                repo_id,
                "stability",
                CellKind::ThroughputBurst,
                axes,
                Some(SearchCell {
                    query: q.into(),
                    limit: 50,
                    threads: thr,
                    burst: stream.len(),
                    stream,
                    call_kind: Some("file".into()),
                    ..Default::default()
                }),
                &["stability", "repeat"],
            ));
        }
    }
    out
}

// ─── Folder & Directory Path Edge Cases ──────────────────────────────────

fn family_folder_edge_cases(repo_id: &str) -> Vec<Cell> {
    let mut out = Vec::new();

    // Deep nested paths: 5-6 directory levels (like linux kernel / game engines)
    for (query, tag) in [
        (
            "drivers/net/ethernet/intel/e1000e/netdev.c",
            "deep_6_driver",
        ),
        ("arch/x86/kvm/vmx/nested.c", "deep_5_arch"),
        ("crates/fff-core/src/score.rs", "deep_4_score"),
        (
            "vendor/neo_frizbee/src/smith_waterman/simd/algo.rs",
            "deep_5_vendor",
        ),
        ("lua/fff/file_picker/preview.lua", "deep_4_lua"),
        ("lib/x86/insn-eval.c", "deep_3_lib"),
    ] {
        let axes = json!({"query": query, "tag": tag, "depth": "deep_nested", "call": "file"});
        out.push(cell(
            "profile.folder.deep_nested",
            repo_id,
            "folder",
            CellKind::Search,
            axes,
            Some(SearchCell {
                query: query.into(),
                limit: 50,
                threads: 4,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["folder", "deep_nested", tag],
        ));
    }

    // Pure directory queries with trailing slash
    for (query, tag) in [
        ("crates/fff-core/src/", "dir_trailing_slash_src"),
        ("vendor/neo_frizbee/", "dir_trailing_slash_vendor"),
        ("lua/fff/", "dir_trailing_slash_lua"),
        ("arch/x86/kvm/", "dir_trailing_slash_arch"),
        ("src/", "dir_trailing_slash_top"),
        ("drivers/", "dir_trailing_slash_drivers"),
    ] {
        let axes = json!({"query": query, "tag": tag, "depth": "dir_only", "call": "mixed"});
        out.push(cell(
            "profile.folder.dir_only",
            repo_id,
            "folder",
            CellKind::Search,
            axes,
            Some(SearchCell {
                query: query.into(),
                limit: 50,
                threads: 4,
                call_kind: Some("mixed".into()),
                ..Default::default()
            }),
            &["folder", "dir_only", tag],
        ));
    }

    // Mixed Windows backslash / forward slash paths
    for (query, tag) in [
        ("crates\\fff-core\\src\\score.rs", "windows_backslash"),
        ("lua/fff\\file_picker/init.lua", "mixed_slash"),
        ("crates\\fff-core\\src", "windows_dir"),
    ] {
        let axes = json!({"query": query, "tag": tag, "slash": "mixed"});
        out.push(cell(
            "profile.folder.mixed_slashes",
            repo_id,
            "folder",
            CellKind::Search,
            axes,
            Some(SearchCell {
                query: query.into(),
                limit: 50,
                threads: 4,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["folder", "mixed_slash", tag],
        ));
    }

    // Relative path traversals
    for (query, tag) in [
        ("./crates/fff-core/src/lib.rs", "relative_dot_slash"),
        ("../fff-main/crates/fff-core", "relative_parent"),
        ("./lua/fff/", "relative_dir_slash"),
    ] {
        let axes = json!({"query": query, "tag": tag, "traversal": "relative"});
        out.push(cell(
            "profile.folder.relative_traversal",
            repo_id,
            "folder",
            CellKind::Search,
            axes,
            Some(SearchCell {
                query: query.into(),
                limit: 50,
                threads: 4,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["folder", "relative", tag],
        ));
    }

    // Asymmetric path ratios: long dir prefix + short filename vs short dir + long filename
    for (query, tag) in [
        // Very long directory, short filename (6 chars)
        (
            "crates/fff-core/src/dbs/frecency/mod.rs",
            "long_dir_short_name",
        ),
        // Very short directory, very long filename (60+ chars)
        (
            "src/very_long_file_name_with_detailed_description_of_component.rs",
            "short_dir_long_name",
        ),
        // Max depth with common short name
        (
            "drivers/net/ethernet/intel/e1000e/hw.h",
            "deep_dir_short_name",
        ),
    ] {
        let axes = json!({"query": query, "tag": tag, "ratio": "asymmetric"});
        out.push(cell(
            "profile.folder.asymmetric_ratio",
            repo_id,
            "folder",
            CellKind::Search,
            axes,
            Some(SearchCell {
                query: query.into(),
                limit: 50,
                threads: 4,
                call_kind: Some("file".into()),
                ..Default::default()
            }),
            &["folder", "asymmetric", tag],
        ));
    }

    out
}

/// Catalog for `list` command.
pub fn list_scenarios() -> Vec<(&'static str, &'static str)> {
    vec![
        ("gate.golden", "quality — golden rank gates"),
        ("profile.search.matrix", "search — selectivity × threads"),
        (
            "profile.search.query_length_boundary",
            "search — query-length algorithm transitions",
        ),
        ("profile.call.kinds", "call types — file/dir/mixed/grep"),
        ("profile.search.pagination", "search — deep offset/limit"),
        ("profile.search.constraints", "search — query constraints"),
        (
            "profile.folder.edge_cases",
            "folder — deep nested / slashes / relative edge cases",
        ),
        ("profile.grep.modes", "grep — plain/fuzzy/regex"),
        (
            "profile.grep.pagination",
            "grep — cursor-linked first/next pages",
        ),
        (
            "profile.throughput.burst",
            "single-client closed-loop throughput",
        ),
        (
            "profile.throughput.concurrent",
            "concurrent-client closed-loop throughput",
        ),
        (
            "profile.typing.prefix_stream",
            "typing keystroke prefixes b2b",
        ),
        ("profile.typing.bounce_edit", "type then backspace stream"),
        ("profile.b2b.session_mix", "file↔grep↔dir session sequence"),
        ("profile.b2b.alt_file_mixed", "alternate call kinds rapidly"),
        ("profile.b2b.query_thrash", "rapid query switching"),
        ("profile.load.*", "heavy fanout / deep page / sustained"),
        ("profile.cold.*", "cold open + first/hot search"),
        ("profile.edge.cases", "edge queries"),
        ("profile.combo.*", "combinatorial axes"),
        ("profile.unique.*", "exotic / stability / slash / repeat"),
        ("profile.interleave.*", "search↔grep↔dir sequences"),
        ("profile.stability.*", "repeat identical for variance"),
        ("profile.stress.*", "pathological fanout/burst/thrash"),
        ("ci-gates", "suite: required public gates only"),
        (
            "ranking-gates",
            "suite: labeled ranking goldens (top-1 quality floor)",
        ),
        (
            "size-sweep",
            "suite: latency floor probe across thread counts",
        ),
        ("public-v1", "suite: content-addressed fixture query bank"),
        (
            "agent-screen",
            "suite: gates + representative search/grep (fast reject)",
        ),
        (
            "agent-confirm",
            "suite: decision-grade + mutation gate phase",
        ),
        (
            "resilience-confirm",
            "suite: probe under quiet/cpu/memory load profiles",
        ),
        ("peak-diagnose", "suite: wide profiles for diagnosis"),
        ("agent-loop", "alias → agent-screen"),
        ("peak-frontier", "alias → peak-diagnose"),
        ("full", "suite: everything including load/combo/stress"),
        (
            "latency",
            "suite: individual warm-search latency and boundaries",
        ),
        (
            "pagination",
            "suite: fuzzy offsets and cursor-linked grep pages",
        ),
        ("typing", "suite: typing streams only"),
        ("calls", "suite: call-kind matrix only"),
        ("load", "suite: heavy load only"),
        ("stress", "suite: pathological only"),
        ("stability", "suite: repeatability only"),
        ("b2b", "suite: back-to-back sequences only"),
    ]
}

pub fn list_scenario_ids() -> Vec<&'static str> {
    list_scenarios().into_iter().map(|(id, _)| id).collect()
}

/// Count cells for a suite (for list --json meta).
pub fn suite_cell_count(suite: &str, repo_id: &str) -> usize {
    expand_suite(suite, repo_id).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn full_suite_is_wide() {
        let cells = expand_suite("full", "self");
        assert!(
            cells.len() >= 250,
            "expected wide full suite, got {}",
            cells.len()
        );
        let families: HashSet<_> = cells.iter().map(|c| c.family.as_str()).collect();
        for need in [
            "quality",
            "search",
            "throughput",
            "load",
            "cold",
            "edge",
            "constraint",
            "grep",
            "combo",
            "pagination",
            "call",
            "typing",
            "b2b",
            "unique",
            "interleave",
            "stability",
            "stress",
        ] {
            assert!(
                families.contains(need),
                "missing family {need} in {families:?}"
            );
        }
    }

    #[test]
    fn agent_loop_smaller_than_full() {
        let a = expand_suite("agent-loop", "r").len();
        let f = expand_suite("full", "r").len();
        assert!(a < f);
        assert!(a >= 10);
    }

    #[test]
    fn pack_suite_aliases() {
        let screen = expand_suite("agent-screen", "r").len();
        let loop_alias = expand_suite("agent-loop", "r").len();
        assert_eq!(screen, loop_alias);
        let confirm = expand_suite("agent-confirm", "r").len();
        assert!(confirm > screen, "confirm should be broader than screen");
        let peak = expand_suite("peak-diagnose", "r").len();
        let peak_alias = expand_suite("peak-frontier", "r").len();
        assert_eq!(peak, peak_alias);
        assert!(peak > confirm);
    }

    #[test]
    fn cell_keys_unique_in_full() {
        let cells = expand_suite("full", "self");
        let mut seen = HashSet::new();
        for c in &cells {
            assert!(seen.insert(c.cell_key.clone()), "dup key {}", c.cell_key);
        }
    }

    #[test]
    fn peak_has_throughput_and_cold() {
        let cells = expand_suite("peak-frontier", "r");
        assert!(cells.iter().any(|c| c.family == "throughput"));
        assert!(cells.iter().any(|c| c.family == "cold"));
        assert!(cells.iter().any(|c| c.kind == CellKind::ColdOpen));
    }

    #[test]
    fn confirm_covers_latency_pagination_and_real_throughput() {
        let cells = expand_suite("agent-confirm", "r");
        assert!(cells.iter().any(|c| c.family == "latency"));
        assert!(
            cells
                .iter()
                .any(|c| c.scenario_id == "profile.search.pagination")
        );
        assert!(
            cells
                .iter()
                .any(|c| c.scenario_id == "profile.grep.pagination")
        );
        assert!(cells.iter().any(|c| c.kind == CellKind::ThroughputBurst));
        assert!(
            cells
                .iter()
                .any(|c| c.kind == CellKind::ThroughputConcurrent)
        );
    }

    #[test]
    fn throughput_axes_separate_clients_from_subject_threads() {
        for cell in family_throughput("r", false) {
            assert_eq!(
                cell.axes.get("arrival_model").and_then(|v| v.as_str()),
                Some("closed_loop")
            );
            assert!(cell.axes.get("client_concurrency").is_some());
            assert!(cell.axes.get("threads_per_request").is_some());
            assert!(cell.axes.get("operations").is_some());
        }
    }

    #[test]
    fn focused_latency_and_pagination_suites_are_available() {
        let latency = expand_suite("latency", "r");
        assert!(latency.iter().any(|c| c.family == "latency"));
        let pagination = expand_suite("pagination", "r");
        assert!(pagination.iter().any(|c| c.kind == CellKind::Search));
        assert!(
            pagination
                .iter()
                .any(|c| c.kind == CellKind::GrepPagination)
        );
    }

    #[test]
    fn screen_checks_first_and_next_fuzzy_pages() {
        let pages: Vec<_> = expand_suite("agent-screen", "r")
            .into_iter()
            .filter(|c| c.scenario_id == "profile.search.pagination")
            .filter_map(|c| {
                c.axes
                    .get("page")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            })
            .collect();
        assert_eq!(pages, ["first", "next"]);
    }
}
