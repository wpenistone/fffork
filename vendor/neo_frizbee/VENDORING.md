# Vendored `neo_frizbee`

This directory is a **fork of `neo_frizbee` 0.13.3** (crates.io), vendored into
FFF while the changes below are upstreamed.

## Why it is vendored

FFF's fuzzy hot path needs a short-needle bit-parallel LCS gate that is
dramatically cheaper than running affine-gap Smith-Waterman over the full
corpus. The fork adds that gate and keeps `Match { score, index, exact,
end_col }` bit-identical to crates.io so ranking and checksums stay put.

## Changes relative to published 0.13.3

| Area | Change |
| --- | --- |
| `src/myers.rs` | Myers (1999) bit-vector Levenshtein + successor-vector LCS for `|needle| ≤ 64`, AVX2 `lcs_ge_batch8` (8 packed paths), scalar fallbacks, proptest pins vs scalar DP |
| `src/lib.rs` | `pub mod myers` |

Scoring, prefilter, matcher, sort, and the resolved/chunked APIs are
byte-identical to 0.13.3. `myers` only **rejects** candidates that cannot
satisfy `LCS(needle, path) ≥ |needle| − max_typos` (frizbee's typo model =
missing needle characters); Smith-Waterman still produces every score.

## Maintenance

- Rebase on upstream releases as they land; drop the vendor directory once a
  release contains `myers` (or an equivalent short-needle LCS gate).
- Workspace points here via `neo_frizbee = { path = "vendor/neo_frizbee" }`
  in the root `Cargo.toml`.
