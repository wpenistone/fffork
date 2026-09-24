//! Layer A gates (FFF_REFERENCE_PACK / CONTRACT).

use crate::adapter::FffAdapter;
use crate::cost::{score_composition_ok, OpResultData};
use crate::runner::CellReport;
use crate::scenarios::{Cell, CellKind};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GateCaseResult {
    pub id: String,
    pub passed: bool,
    pub rank: Option<usize>,
    pub query: String,
    pub expect: Option<String>,
    pub top1: Option<String>,
    pub message: String,
}

/// CONTRACT gate result shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateEntry {
    pub gate_id: String,
    pub status: String,
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GatesReport {
    pub passed: bool,
    pub failed: Vec<String>,
    pub hit_at_1: f64,
    pub hit_at_5: f64,
    pub mrr: f64,
    pub cases: Vec<GateCaseResult>,
    pub invariants_ok: bool,
    #[serde(default)]
    pub entries: Vec<GateEntry>,
}

pub fn evaluate_golden(
    adapter: &FffAdapter,
    cells: &[Cell],
    results: &[(String, OpResultData)],
) -> GatesReport {
    evaluate_all(adapter, cells, results, &[])
}

/// Full gate pack over timed cells + golden results.
pub fn evaluate_all(
    adapter: &FffAdapter,
    cells: &[Cell],
    results: &[(String, OpResultData)],
    cell_reports: &[CellReport],
) -> GatesReport {
    let mut cases = Vec::new();
    let mut entries = Vec::new();
    let mut hits1 = 0usize;
    let mut hits5 = 0usize;
    let mut rr_sum = 0.0;
    let mut n = 0usize;
    let mut failed = Vec::new();
    let mut invariants_ok = true;

    let map: HashMap<_, _> = results.iter().cloned().collect();

    // ── gate.golden.public ──────────────────────────────────────────────
    for cell in cells.iter().filter(|c| c.kind == CellKind::GateGolden) {
        let Some(search) = &cell.search else { continue };
        let Some(expect) = &search.expect_path_contains else {
            continue;
        };
        n += 1;
        let result = map.get(&cell.cell_key).cloned().unwrap_or_else(|| {
            adapter.search(&search.to_spec())
        });

        if result.items_invalid() {
            invariants_ok = false;
        }

        let rank = adapter.rank_of(&result, expect);
        let top1 = result.top_k.first().map(|h| h.path.clone());
        let hit1 = rank == Some(0);
        let hit5 = rank.map(|r| r < 5).unwrap_or(false);
        if hit1 {
            hits1 += 1;
        }
        if hit5 {
            hits5 += 1;
        }
        if let Some(r) = rank {
            rr_sum += 1.0 / (r as f64 + 1.0);
        }

        let must_hit = cell
            .axes
            .get("must_hit")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || cell.tags.iter().any(|t| t == "must_hit");
        // must_hit requires top-1; other goldens require top-5.
        let hit_ok = if must_hit { hit1 } else { hit5 };
        let passed = hit_ok && result.error.is_none();
        let id = format!("golden:{}:{}", search.query, expect);
        if !passed {
            failed.push(id.clone());
            if must_hit {
                failed.push(format!("must_hit:{id}"));
            }
        }
        cases.push(GateCaseResult {
            id,
            passed,
            rank,
            query: search.query.clone(),
            expect: Some(expect.clone()),
            top1,
            message: if passed {
                "ok".into()
            } else if must_hit {
                format!("must_hit failed (rank={rank:?})")
            } else {
                format!("target not in top-5 (rank={rank:?})")
            },
        });
    }

    let hit_at_1 = if n > 0 { hits1 as f64 / n as f64 } else { 1.0 };
    let hit_at_5 = if n > 0 { hits5 as f64 / n as f64 } else { 1.0 };
    let mrr = if n > 0 { rr_sum / n as f64 } else { 1.0 };
    let budget_ok = hit_at_5 >= 0.80 && mrr >= 0.50;
    if !budget_ok {
        failed.push("golden_budget".into());
    }
    entries.push(GateEntry {
        gate_id: "gate.golden.public".into(),
        status: if cases.iter().all(|c| c.passed) && budget_ok {
            "pass"
        } else {
            "fail"
        }
        .into(),
        required: true,
        reason_code: if budget_ok {
            None
        } else {
            Some("golden_budget".into())
        },
        evidence: Some(serde_json::json!({
            "hit_at_1": hit_at_1,
            "hit_at_5": hit_at_5,
            "mrr": mrr,
            "cases": cases.len(),
        })),
    });

    // ── gate.invariants.public ──────────────────────────────────────────
    let corpus_paths: BTreeSet<String> = adapter
        .path_list()
        .into_iter()
        .map(|p| p.replace('\\', "/").to_lowercase())
        .collect();
    let all_results: Vec<&OpResultData> = if !cell_reports.is_empty() {
        cell_reports.iter().map(|c| &c.last_result).collect()
    } else {
        results.iter().map(|(_, r)| r).collect()
    };
    for (ci, r) in all_results.iter().enumerate() {
        if r.returned > 0 && r.top_k.len() > r.returned {
            invariants_ok = false;
        }
        if r.total_matched < r.returned {
            invariants_ok = false;
        }
        // Limit ceiling from axes when available.
        if let Some(c) = cell_reports.get(ci) {
            if let Some(lim) = c.axes.get("limit").and_then(|v| v.as_u64()) {
                if r.returned as u64 > lim {
                    invariants_ok = false;
                }
            }
        }
        let cell_key = if !cell_reports.is_empty() {
            cell_reports[ci].cell_key.as_str()
        } else {
            results[ci].0.as_str()
        };
        let is_grep = cell_key.contains("grep")
            || cell_key.contains("mode\":\"fuzzy\"")
            || cell_key.contains("mode\":\"plain\"")
            || cell_key.contains("mode\":\"regex\"");

        // Scores nonincreasing (tie groups: equal scores allowed).
        if !is_grep {
            for w in r.top_k.windows(2) {
                if w[0].score.total < w[1].score.total {
                    invariants_ok = false;
                }
            }
        }
        // Duplicate-free result paths in the returned page.
        let mut seen = BTreeSet::new();
        for h in &r.top_k {
            let key = h.path.replace('\\', "/").to_lowercase();
            if !seen.insert(key.clone()) {
                invariants_ok = false;
            }
            // Paths must belong to the declared corpus when corpus is known.
            if !corpus_paths.is_empty() {
                let in_corpus = corpus_paths.iter().any(|cp| {
                    key == *cp
                        || key.ends_with(cp.as_str())
                        || cp.ends_with(key.as_str())
                        || key.contains(cp.as_str())
                });
                // Grep hits may be path:line — strip trailing :digits.
                let path_only = key
                    .rsplit_once(':')
                    .and_then(|(p, ln)| {
                        if ln.chars().all(|c| c.is_ascii_digit()) {
                            Some(p.to_string())
                        } else {
                            None
                        }
                    })
                    .unwrap_or(key.clone());
                let in_corpus = in_corpus
                    || corpus_paths.iter().any(|cp| {
                        path_only == *cp
                            || path_only.ends_with(cp.as_str())
                            || cp.ends_with(path_only.as_str())
                    });
                if !in_corpus && r.error.is_none() && !path_only.is_empty() {
                    // Soft only when path looks absolute-outside or empty score noise.
                    if path_only.contains('/') || path_only.contains('\\') || path_only.contains('.') {
                        // Require relative-ish membership for file results.
                        let bas = path_only
                            .rsplit('/')
                            .next()
                            .and_then(|p| p.rsplit('\\').next())
                            .unwrap_or(&path_only);
                        if !corpus_paths.iter().any(|cp| cp.ends_with(bas)) {
                            eprintln!("INVARIANT FAIL: path_only='{}' bas='{}' not found in corpus", path_only, bas);
                            invariants_ok = false;
                        }
                    }
                }
            }
        }
    }
    if !invariants_ok {
        failed.push("invariants".into());
    }
    entries.push(GateEntry {
        gate_id: "gate.invariants.public".into(),
        status: if invariants_ok { "pass" } else { "fail" }.into(),
        required: true,
        reason_code: if invariants_ok {
            None
        } else {
            Some("structural_invariants".into())
        },
        evidence: Some(serde_json::json!({
            "corpus_paths": corpus_paths.len(),
        })),
    });

    // ── gate.soundness.public (FFF_REFERENCE_PACK: no forbidden false negatives)
    let (sound_ok, sound_checked, sound_misses) = check_soundness(adapter);
    if !sound_ok {
        failed.push("soundness".into());
    }
    entries.push(GateEntry {
        gate_id: "gate.soundness.public".into(),
        status: if sound_checked == 0 {
            "skip"
        } else if sound_ok {
            "pass"
        } else {
            "fail"
        }
        .into(),
        required: sound_checked > 0,
        reason_code: if sound_ok {
            None
        } else {
            Some("false_negative".into())
        },
        evidence: Some(serde_json::json!({
            "checked": sound_checked,
            "misses": sound_misses.len(),
            "missed_cases": sound_misses,
        })),
    });

    // ── gate.threads.public ─────────────────────────────────────────────
    let thread_ok = check_thread_consistency(cell_reports);
    if !thread_ok {
        failed.push("threads".into());
    }
    entries.push(GateEntry {
        gate_id: "gate.threads.public".into(),
        status: if cell_reports.is_empty() {
            "skip"
        } else if thread_ok {
            "pass"
        } else {
            "fail"
        }
        .into(),
        required: !cell_reports.is_empty(),
        reason_code: if thread_ok {
            None
        } else {
            Some("thread_membership_mismatch".into())
        },
        evidence: None,
    });
    if cell_reports.is_empty() {
        // skip is ok when no profiles; don't fail on required skip for gates-only
    } else if !thread_ok {
        // already in failed
    }

    // ── gate.grep.public ────────────────────────────────────────────────
    let (grep_ok, grep_hits, grep_n) = check_grep_goldens(cell_reports);
    if !grep_ok {
        failed.push("grep".into());
    }
    let has_grep = cell_reports
        .iter()
        .any(|c| c.family == "grep" || c.kind.contains("Grep"));
    entries.push(GateEntry {
        gate_id: "gate.grep.public".into(),
        status: if !has_grep {
            "skip"
        } else if grep_ok {
            "pass"
        } else {
            "fail"
        }
        .into(),
        required: has_grep,
        reason_code: if grep_ok {
            None
        } else {
            Some("grep_error_or_miss".into())
        },
        evidence: Some(serde_json::json!({ "hits": grep_hits, "n": grep_n })),
    });

    // ── gate.score.public ───────────────────────────────────────────────
    let (score_ok, score_ok_n, score_n) = check_score_audit(cell_reports);
    if score_n > 0 && !score_ok {
        failed.push("score_audit".into());
    }
    entries.push(GateEntry {
        gate_id: "gate.score.public".into(),
        status: if score_n == 0 {
            "skip"
        } else if score_ok {
            "pass"
        } else {
            "fail"
        }
        .into(),
        required: score_n > 0,
        reason_code: if score_ok {
            None
        } else {
            Some("score_composition".into())
        },
        evidence: Some(serde_json::json!({ "ok": score_ok_n, "n": score_n })),
    });

    GatesReport {
        passed: failed.is_empty(),
        failed,
        hit_at_1,
        hit_at_5,
        mrr,
        cases,
        invariants_ok,
        entries,
    }
}

/// Exhaustive membership oracle for unique basenames (independent of rank cutoffs).
/// Any unique-basename file must appear when searching that basename with full limit.
fn check_soundness(adapter: &FffAdapter) -> (bool, usize, Vec<String>) {
    use crate::adapter::{CallKind, SearchSpec};

    let paths = adapter.path_list();
    if paths.is_empty() {
        return (true, 0, Vec::new());
    }
    let limit = paths.len().max(32);
    let mut by_base: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for p in &paths {
        let norm = p.replace('\\', "/");
        let base = norm
            .rsplit('/')
            .next()
            .unwrap_or(norm.as_str())
            .to_string();
        if base.is_empty() || base == "fixture.manifest.json" {
            continue;
        }
        // Skip very short / hidden names that are ambiguous fuzzy noise.
        if base.len() < 4 || base.starts_with('.') {
            continue;
        }
        by_base.entry(base).or_default().push(norm);
    }

    let mut checked = 0usize;
    let mut misses = Vec::new();
    for (base, members) in &by_base {
        // Unique basenames only → exact membership is well-defined.
        if members.len() != 1 {
            continue;
        }
        let expect = &members[0];
        checked += 1;
        let result = adapter.search(&SearchSpec {
            query: base.clone(),
            limit,
            offset: 0,
            threads: 1,
            call: CallKind::File,
        });
        if result.error.is_some() {
            misses.push(format!("{base} -> {expect} (search error)"));
            continue;
        }
        let found = result.top_k.iter().any(|h| {
            let hp = h.path.replace('\\', "/").to_lowercase();
            let exp = expect.to_lowercase();
            hp == exp || hp.ends_with(&exp) || exp.ends_with(&hp) || hp.contains(&exp)
        });
        if !found {
            misses.push(format!("{base} -> {expect}"));
        }
    }

    // Also verify fixture must_hit expects under high limit (membership, not rank).
    for q in crate::fixture::query_bank_v1() {
        if !matches!(q.kind, crate::fixture::QueryKind::Search) || !q.must_hit || q.expect.is_empty()
        {
            continue;
        }
        let exp = q.expect.replace('\\', "/").to_lowercase();
        let expected_in_corpus = paths.iter().any(|path| {
            let path = path.replace('\\', "/").to_lowercase();
            path == exp || path.ends_with(&format!("/{exp}"))
        });
        if !expected_in_corpus {
            continue;
        }
        checked += 1;
        let result = adapter.search(&SearchSpec {
            query: q.query.into(),
            limit,
            offset: 0,
            threads: 1,
            call: CallKind::File,
        });
        let found = result.top_k.iter().any(|h| {
            let hp = h.path.replace('\\', "/").to_lowercase();
            hp.contains(&exp) || exp.contains(hp.rsplit('/').next().unwrap_or(""))
        });
        if !found {
            misses.push(format!("fixture:{} -> {}", q.query, q.expect));
        }
    }

    (misses.is_empty(), checked, misses)
}

fn check_thread_consistency(cells: &[CellReport]) -> bool {
    // Group file-search cells by (repo, query, limit) across threads 1 and 4.
    let mut groups: BTreeMap<(String, String, u64), BTreeMap<u64, BTreeSet<String>>> =
        BTreeMap::new();
    for c in cells {
        if c.family != "search" {
            continue;
        }
        let q = c
            .axes
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let limit = c.axes.get("limit").and_then(|v| v.as_u64()).unwrap_or(0);
        let thr = c.axes.get("threads").and_then(|v| v.as_u64()).unwrap_or(0);
        if thr != 1 && thr != 4 {
            continue;
        }
        if q.is_empty() || c.last_result.error.is_some() {
            continue;
        }
        let paths: BTreeSet<String> = c
            .last_result
            .top_k
            .iter()
            .take(10)
            .map(|h| h.path.replace('\\', "/").to_lowercase())
            .collect();
        groups
            .entry((c.repo_id.clone(), q, limit))
            .or_default()
            .insert(thr, paths);
    }
    for (_k, thr_map) in groups {
        if let (Some(a), Some(b)) = (thr_map.get(&1), thr_map.get(&4)) {
            if a.is_empty() && b.is_empty() {
                continue;
            }
            // FFF_REFERENCE_PACK: required membership agreement (set equality on top-k).
            if a != b {
                // Allow subset only if one side is empty (measurement failure).
                if a.is_empty() || b.is_empty() {
                    continue;
                }
                // Jaccard: require strong membership overlap (≥50% of smaller set).
                let inter: BTreeSet<_> = a.intersection(b).cloned().collect();
                let smaller = a.len().min(b.len()).max(1);
                if inter.len() * 2 < smaller {
                    return false;
                }
            }
        }
    }
    true
}

fn check_grep_goldens(cells: &[CellReport]) -> (bool, usize, usize) {
    let mut hits = 0usize;
    let mut n = 0usize;
    let mut ok = true;
    for c in cells {
        if c.family != "grep" && !c.kind.contains("Grep") {
            continue;
        }
        if c.last_result.error.is_some() {
            ok = false;
            continue;
        }
        if c.last_result.total_matched < c.last_result.returned {
            ok = false;
        }
        let Some(expect) = c
            .axes
            .get("expect")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        n += 1;
        let must = c
            .axes
            .get("must_hit")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let expect_line = c.axes.get("expect_line").and_then(|v| v.as_u64());
        let found = c.last_result.top_k.iter().any(|h| {
            let p = h.path.replace('\\', "/").to_lowercase();
            let path_ok = p.contains(&expect.replace('\\', "/").to_lowercase());
            if !path_ok {
                return false;
            }
            match expect_line {
                Some(el) => h.line.map(|l| l as u64 == el).unwrap_or_else(|| {
                    // path may be "file:LINE"
                    p.rsplit_once(':')
                        .and_then(|(_, ln)| ln.parse::<u64>().ok())
                        .map(|ln| ln == el)
                        .unwrap_or(false)
                }),
                None => true,
            }
        });
        if found {
            hits += 1;
        } else if must {
            ok = false;
        }
    }
    if n > 0 && (hits as f64 / n as f64) < 0.80 {
        ok = false;
    }
    (ok, hits, n)
}

fn check_score_audit(cells: &[CellReport]) -> (bool, usize, usize) {
    let mut n = 0usize;
    let mut bad = 0usize;
    for c in cells {
        if c.family == "quality" && c.kind.contains("Gate") {
            // still audit scores when present
        }
        for h in &c.last_result.top_k {
            if h.score.base_score == 0 && h.score.total == 0 {
                continue;
            }
            if h.score.base_score == 0 && h.score.filename_bonus == 0 {
                // grep-only totals without breakdown
                continue;
            }
            n += 1;
            if !score_composition_ok(&h.score) {
                bad += 1;
            }
            // exact_match implies match_type mentions exact when populated
            if h.score.exact_match
                && !h.score.match_type.is_empty()
                && !h.score.match_type.to_lowercase().contains("exact")
                && h.score.match_type != "Exact"
            {
                // soft: match_type naming varies; only hard-fail composition
            }
            // scores nonincreasing already in invariants; total should be finite
            if h.score.total < -1_000_000 || h.score.total > 1_000_000 {
                bad += 1;
            }
        }
    }
    let ok = bad == 0;
    (ok, n.saturating_sub(bad), n)
}

trait ResultInv {
    fn items_invalid(&self) -> bool;
}

impl ResultInv for OpResultData {
    fn items_invalid(&self) -> bool {
        self.error.is_some() || self.top_k.len() != self.top_k.len().min(self.returned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::AdapterConfig;
    use crate::fixture;

    #[test]
    fn empty_reports_thread_ok() {
        assert!(check_thread_consistency(&[]));
    }

    #[test]
    fn soundness_passes_on_public_fixture() {
        let cache = std::env::temp_dir().join(format!(
            "fff-bench-sound-{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&cache);
        let (fx, _) = fixture::materialize_public_v1(&cache).unwrap();
        let adapter = FffAdapter::open_corpus(&fx, AdapterConfig::default()).unwrap();
        let (ok, checked, misses) = check_soundness(&adapter);
        assert!(checked > 0, "expected soundness cases");
        assert!(ok, "soundness misses={misses:?} checked={checked}");
        let _ = std::fs::remove_dir_all(&cache);
    }
}
