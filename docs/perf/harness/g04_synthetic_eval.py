#!/usr/bin/env python3
"""G-04 synthetic filename-stem evaluation.

Builds a pooled judgment set from the union of fork/upstream top-100 paths.
Labels are synthetic and filename-derived; this is not real-user evidence.

Rule (case-insensitive, on the basename stem unless noted):
  3 = stem exactly equals the query
  2 = stem starts with the query, or the query starts with the stem (stem len >= 4)
  1 = query appears in the stem or anywhere in the relative path
  0 = unjudged (the standard pooled convention)
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ranking_metrics as rm  # noqa: E402

DUMP_DIR = r"C:\Users\kanal\AppData\Local\fff-bench\rank-eval"
CORPUS = r"C:\Users\kanal\Projects"
OUT = "g04-synthetic-ranking.json"


def load(dump_dir, name):
    path = os.path.join(dump_dir, name)
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest().upper()
    return rows, digest


def stem(path):
    return os.path.splitext(os.path.basename(path))[0].lower()


def grade(path, query):
    s = stem(path)
    q = query.lower()
    p = path.lower()
    if s == q:
        return 3
    if s.startswith(q) or (len(s) >= 4 and q.startswith(s)):
        return 2
    if q in s or q in p:
        return 1
    return 0


def build_labels(fork, upstream):
    labels = {}
    counts = {}
    for row in fork:
        key = (row["project"], row["query"])
        union = set()
        for side in (fork, upstream):
            for other in side:
                if (other["project"], other["query"]) == key:
                    union.update(r["path"] for r in other["results"])
        judged = {}
        for path in union:
            g = grade(path, row["query"])
            if g > 0:
                judged[path] = g
        labels[key] = judged
        counts[row["query"]] = {
            "pool": len(union),
            "grade3": sum(1 for g in judged.values() if g == 3),
            "grade2": sum(1 for g in judged.values() if g == 2),
            "grade1": sum(1 for g in judged.values() if g == 1),
            "relevant": sum(1 for g in judged.values() if g >= 1),
        }
    return labels, counts


def evaluate(rows, labels, k, terms):
    per = []
    for row in rows:
        key = (row["project"], row["query"])
        judged = labels.get(key)
        if not judged:
            continue
        results = [dict(r) for r in row["results"]]
        for r in results:
            r["_project"] = row["project"]
        if terms:
            results = rm.ablate_results(results, terms)
        m = rm.score_query(results, judged, k, 1)
        m["query"] = row["query"]
        per.append(m)
    summary = {
        "queries": len(per),
        "k": k,
        "ablate": terms,
        "mrr": rm.mean(m["mrr"] for m in per),
        "ndcg": rm.mean(m["ndcg"] for m in per),
        "recall": rm.mean(m["recall"] for m in per),
        "hit_rate": rm.mean(1.0 if m["first_relevant_rank"] else 0.0 for m in per),
        "per_query": per,
    }
    return summary


def overlap(rows_a, rows_b, k):
    other = {(r["project"], r["query"]): r["results"] for r in rows_b}
    vals = []
    per_query = []
    for row in rows_a:
        key = (row["project"], row["query"])
        if key not in other:
            continue
        a = [rm.norm_path(r["path"], row["project"]) for r in row["results"][:k]]
        b = [rm.norm_path(r["path"], row["project"]) for r in other[key][:k]]
        union = set(a) | set(b)
        if union:
            j = len(set(a) & set(b)) / len(union)
            vals.append(j)
            per_query.append({"query": row["query"], "jaccard": j})
    return {
        "k": k,
        "jaccard": statistics.fmean(vals) if vals else None,
        "n": len(vals),
        "per_query": per_query,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", default=DUMP_DIR, help="directory with fork/upstream top-100 JSONL")
    ap.add_argument("--corpus", default=CORPUS, help="corpus root recorded in the artifact")
    ap.add_argument("--out", default=OUT, help="output JSON artifact")
    args = ap.parse_args()

    fork, fork_sha = load(args.dump_dir, "fork-top100.jsonl")
    upstream, upstream_sha = load(args.dump_dir, "upstream-top100.jsonl")
    labels, counts = build_labels(fork, upstream)

    result = {
        "schema": "g04-synthetic-ranking-v1",
        "kind": "synthetic-filename-stem",
        "evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "corpus": args.corpus,
        "dump_date": "2026-11-05",
        "dumps": {
            "fork": {
                "file": "fork-top100.jsonl",
                "sha256": fork_sha,
                "binary_sha256": "5194357E7672E9FAB950994AC404C9E3C548173E25DA19D22CFCB493C7035417",
                "commit": "d0740a9",
            },
            "upstream": {
                "file": "upstream-top100.jsonl",
                "sha256": upstream_sha,
                "binary_sha256": "A21DD5D680106BFC01CD0F81A257BDD1A021CE2095840AF2215EEAA3CF51AF6C",
                "commit": "51649e1",
            },
        },
        "queries": [r["query"] for r in fork],
        "label_rule": {
            "grade3": "basename stem exactly equals query (case-insensitive)",
            "grade2": "basename stem starts with query, or query starts with stem (stem len >= 4)",
            "grade1": "query appears in basename stem or anywhere in the relative path",
            "grade0": "unjudged; pooled convention",
            "pool": "union of fork and upstream top-100 paths",
        },
        "label_counts": counts,
        "results": {
            "fork": {str(k): evaluate(fork, labels, k, []) for k in (10, 100)},
            "upstream": {str(k): evaluate(upstream, labels, k, []) for k in (10, 100)},
        },
        "overlap": {str(k): overlap(fork, upstream, k) for k in (10, 100)},
        "caveat": (
            "Synthetic filename-stem judgments, not real user queries or clicks. "
            "Top-k overlap is ranking-change detection, not parity or quality. "
            "Unjudged pooled candidates count as grade 0 by convention."
        ),
        "interpretation": (
            "Synthetic labels reward exact/prefix filename matches. The fork's "
            "NDCG is higher on controller/netconvert; its recall is lower on "
            "file_picke/netconvert. That is a trade-off under a synthetic rule, "
            "not evidence that the fork ranks better for users."
        ),
    }

    for k in (10, 100):
        a = result["results"]["fork"][str(k)]
        b = result["results"]["upstream"][str(k)]
        result.setdefault("deltas_fork_minus_upstream", {})[str(k)] = {
            metric: a[metric] - b[metric]
            for metric in ("mrr", "ndcg", "recall", "hit_rate")
        }

    # Per-term ablation on the fork, for every numeric component except the base.
    all_terms = set()
    terms = set()
    for row in fork:
        for r in row["results"]:
            for name, value in r["components"].items():
                if isinstance(value, bool) or name in ("base_score", "match_type"):
                    continue
                if isinstance(value, (int, float)):
                    all_terms.add(name)
                    if value != 0:
                        terms.add(name)
    result["ablation_terms_nonzero"] = sorted(terms)
    result["ablation_terms_zero"] = sorted(all_terms - terms)
    result["ablation"] = {}
    for term in sorted(terms):
        result["ablation"][term] = {
            str(k): evaluate(fork, labels, k, [term]) for k in (10, 100)
        }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")

    for side in ("fork", "upstream"):
        for k in ("10", "100"):
            s = result["results"][side][k]
            print(
                f"{side:8s} k={k:>3s} MRR={s['mrr']:.4f} "
                f"NDCG={s['ndcg']:.4f} recall={s['recall']:.4f} "
                f"hit={s['hit_rate']:.3f}"
            )
    for k in ("10", "100"):
        o = result["overlap"][k]
        print(f"overlap k={k:>3s} Jaccard={o['jaccard']:.4f} n={o['n']}")
    print("ablation terms:", ", ".join(result["ablation_terms_nonzero"]))
    for term in result["ablation_terms_nonzero"]:
        a10 = result["ablation"][term]["10"]
        a100 = result["ablation"][term]["100"]
        b10 = result["results"]["fork"]["10"]
        b100 = result["results"]["fork"]["100"]
        print(
            f"  -{term:24s} dMRR@10={a10['mrr'] - b10['mrr']:+.4f} "
            f"dNDCG@10={a10['ndcg'] - b10['ndcg']:+.4f} "
            f"dMRR@100={a100['mrr'] - b100['mrr']:+.4f} "
            f"dNDCG@100={a100['ndcg'] - b100['ndcg']:+.4f}"
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
