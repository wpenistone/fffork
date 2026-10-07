use super::{BigramFilter, BigramOverlay, extract_bigrams};
use super::{fuzzy_to_bigram_query, regex_to_bigram_query};

/// Number of evenly-spaced probe bigrams used by the fuzzy candidate query.
const FUZZY_PROBE_COUNT: usize = 5;

#[inline]
fn set_bit(candidates: &mut [u64], file_idx: usize) {
    let word = file_idx / 64;
    if word < candidates.len() {
        candidates[word] |= 1u64 << (file_idx % 64);
    }
}

#[inline]
fn clear_tombstones(candidates: &mut [u64], overlay: &BigramOverlay) {
    for (r, t) in candidates.iter_mut().zip(overlay.tombstones().iter()) {
        *r &= !t;
    }
}

/// Number of base files covered by the bigram bitset; files past this
/// boundary (overflow, max 1024) are always scanned.
#[inline]
pub(crate) fn bigram_boundary(overlay: Option<&BigramOverlay>, files_len: usize) -> usize {
    overlay.map(|o| o.base_file_count()).unwrap_or(files_len)
}

/// Candidate bitset for literal patterns, OR-ed across all of them: a file is
/// a candidate when it contains the bigrams of ANY pattern. Overlay-modified
/// files are re-checked against each pattern's bigrams.
pub(crate) fn literal_candidates(
    index: Option<&BigramFilter>,
    overlay: Option<&BigramOverlay>,
    patterns: &[&str],
) -> Option<Vec<u64>> {
    let index = ready_index(index)?;

    let mut combined: Option<Vec<u64>> = None;
    for pattern in patterns {
        if let Some(candidates) = index.query(pattern.as_bytes()) {
            combined = Some(match combined {
                None => candidates,
                Some(mut acc) => {
                    acc.iter_mut()
                        .zip(candidates.iter())
                        .for_each(|(a, b)| *a |= *b);
                    acc
                }
            });
        }
    }

    let mut candidates = combined?;
    if let Some(overlay) = overlay {
        clear_tombstones(&mut candidates, overlay);
        for pattern in patterns {
            let pattern_bigrams = extract_bigrams(pattern.as_bytes());
            for file_idx in overlay.query_modified(&pattern_bigrams) {
                set_bit(&mut candidates, file_idx);
            }
        }
    }
    Some(candidates)
}

/// Candidate bitset for a regex pattern: the regex HIR is decomposed into an
/// AND/OR bigram query tree (supports alternation, optional groups, character
/// classes, and sparse-1 bigrams across single-byte wildcards). Since modified
/// file contents can't be re-checked against a regex cheaply, all
/// overlay-modified files are conservatively added.
pub(crate) fn regex_candidates(
    index: Option<&BigramFilter>,
    overlay: Option<&BigramOverlay>,
    pattern: &str,
) -> Option<Vec<u64>> {
    let index = ready_index(index)?;

    let bq = regex_to_bigram_query(pattern);
    if bq.is_any() {
        return None;
    }
    let candidates = bq.evaluate(index)?;
    Some(add_all_modified(candidates, overlay))
}

/// Candidate bitset for a fuzzy pattern: evenly-spaced probe bigrams with a
/// typo allowance (widely-spaced probes are far more selective than sliding
/// windows of adjacent bigrams). All overlay-modified files are added.
pub(crate) fn fuzzy_candidates(
    index: Option<&BigramFilter>,
    overlay: Option<&BigramOverlay>,
    pattern: &str,
) -> Option<Vec<u64>> {
    let index = ready_index(index)?;

    let bq = fuzzy_to_bigram_query(pattern, FUZZY_PROBE_COUNT);
    if bq.is_any() {
        return None;
    }
    let candidates = bq.evaluate(index)?;
    Some(add_all_modified(candidates, overlay))
}

/// Path-name fuzzy prefilter using the **search** typo budget.
///
/// Always returns `None` (no filtering): no bigram AND is a necessary
/// condition for subsequence (Smith-Waterman) matching. Haystack gaps are
/// free even at `max_typos == 0` — e.g. needle `abc` truly hits `axbxc`,
/// which contains neither `ab` nor `bc` — so AND-ing consecutive needle
/// bigrams false-rejects ~15% of brute-force hits on sampled queries.
/// Keep the hook (and the lazily built index) for a future sound filter.
pub(crate) fn path_fuzzy_candidates(
    index: Option<&BigramFilter>,
    overlay: Option<&BigramOverlay>,
    pattern: &str,
    max_typos: u16,
) -> Option<Vec<u64>> {
    let _ = (index, overlay, pattern, max_typos);
    None
}

#[inline]
fn ready_index(index: Option<&BigramFilter>) -> Option<&BigramFilter> {
    index.filter(|idx| idx.is_ready())
}

fn add_all_modified(mut candidates: Vec<u64>, overlay: Option<&BigramOverlay>) -> Vec<u64> {
    if let Some(overlay) = overlay {
        clear_tombstones(&mut candidates, overlay);
        for file_idx in overlay.modified_indices() {
            set_bit(&mut candidates, file_idx);
        }
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::bigram_filter::BigramIndexBuilder;

    fn build_path_index(paths: &[&[u8]]) -> BigramFilter {
        let n = paths.len();
        let consec = BigramIndexBuilder::new(n);
        let skip = BigramIndexBuilder::new(n);
        for (i, p) in paths.iter().enumerate() {
            consec.add_file_content(&skip, i, p);
        }
        consec.compress(None)
    }

    #[test]
    fn path_fuzzy_never_rejects_brute_hits() {
        let frags: &[&[u8]] = &[
            b"src", b"lib", b"test", b"main", b"index", b"score", b"file", b"abc", b"axbxc",
            b"axc", b"abbc", b"util", b"core",
        ];
        let mut state = 0x1234_5678u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut paths: Vec<Vec<u8>> = Vec::new();
        for _ in 0..64 {
            let mut p = Vec::new();
            for s in 0..(1 + next() % 3) {
                if s > 0 {
                    p.push(b'/');
                }
                p.extend_from_slice(frags[next() % frags.len()]);
            }
            paths.push(p);
        }
        let refs: Vec<&[u8]> = paths.iter().map(Vec::as_slice).collect();
        let index = build_path_index(&refs);
        let mut gapped_hit_seen = false;
        for needle in [
            b"abc".as_slice(),
            b"src".as_slice(),
            b"file".as_slice(),
            b"score".as_slice(),
            b"main".as_slice(),
            b"util".as_slice(),
        ] {
            let ns = std::str::from_utf8(needle).unwrap();
            for typos in [0u16, 1, 2] {
                let config = neo_frizbee::Config {
                    max_typos: Some(typos),
                    ..Default::default()
                };
                let mut matcher = neo_frizbee::Matcher::new(ns, &config);
                let cand = path_fuzzy_candidates(Some(&index), None, ns, typos);
                for (i, p) in paths.iter().enumerate() {
                    let ps = String::from_utf8_lossy(p);
                    if matcher.match_one(&*ps, 0).is_some() {
                        if needle == b"abc".as_slice() && typos == 0 && ps.contains("axbxc") {
                            gapped_hit_seen = true;
                        }
                        if let Some(bits) = &cand {
                            assert!(
                                BigramFilter::is_candidate(bits, i),
                                "false reject needle={ns:?} typos={typos} path={ps:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(gapped_hit_seen, "corpus must contain a gapped T=0 hit");
    }
}
