#!/usr/bin/env python3
"""G-04 ranking metrics: MRR/NDCG/recall over a ranked result dump.

Consumes either a real query-tracker export (see extract_query_log.py) or an
explicit judgments file. Unjudged results count as grade 0; that is the
standard pooled-judgment convention and is not a claim that they are
irrelevant.

Ranking row schema (JSONL):
  {"project": "...", "query": "...", "results": [
     {"path": "src/a.rs", "total": 512, "components": {
        "base_score": 300, "frecency_boost": 0, ...}}]}

Judgment row schema (JSONL):
  {"project": "...", "query": "...", "judgments": {"src/a.rs": 2, ...}}

Click-only rows (`clicked_file`) are accepted from extract_query_log.py; the
clicked file gets --click-grade and everything else stays unjudged.
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict


def norm_path(path, project=None, case_insensitive=True):
    p = path.replace("\\", "/")
    while "//" in p:
        p = p.replace("//", "/")
    if p.startswith("./"):
        p = p[2:]
    if project:
        root = project.replace("\\", "/").rstrip("/")
        if p.lower().startswith(root.lower() + "/"):
            p = p[len(root) + 1:]
        elif p.lower() == root.lower():
            p = ""
    if case_insensitive:
        p = p.lower()
    return p


def dcg(grades):
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_grades, ideal_grades, k):
    got = dcg(ranked_grades[:k])
    ideal = dcg(sorted(ideal_grades, reverse=True)[:k])
    return got / ideal if ideal > 0 else None


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def index_labels(rows, click_grade):
    labels = {}
    for row in rows:
        key = (row.get("project", ""), row["query"])
        judged = {}
        if "judgments" in row:
            judged.update({norm_path(p, row.get("project")): g for p, g in row["judgments"].items()})
        if row.get("clicked_file"):
            judged.setdefault(norm_path(row["clicked_file"], row.get("project")), click_grade)
        labels[key] = judged
    return labels


def ablate_results(results, terms):
    out = []
    for r in results:
        comp = r.get("components") or {}
        new_total = r["total"]
        for term in terms:
            new_total -= comp.get(term, 0)
        out.append({**r, "total": new_total, "_orig_rank": r.get("rank", len(out))})
    out.sort(key=lambda r: (-r["total"], r["_orig_rank"]))
    return out


def score_query(results, judged, k, rel_threshold):
    graded = [judged.get(norm_path(r["path"], r.get("_project")), 0) for r in results]
    relevant_total = sum(1 for g in judged.values() if g >= rel_threshold)
    first = None
    for i, g in enumerate(graded[:k]):
        if g >= rel_threshold:
            first = i + 1
            break
    mrr = 1.0 / first if first else 0.0
    recall = (sum(1 for g in graded[:k] if g >= rel_threshold) / relevant_total) if relevant_total else None
    ndcg = ndcg_at_k(graded, [g for g in judged.values() if g > 0], k) if any(
        g > 0 for g in judged.values()) else None
    return {
        "mrr": mrr,
        "first_relevant_rank": first,
        "recall": recall,
        "ndcg": ndcg,
        "n_relevant": relevant_total,
        "n_judged": len(judged),
    }


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rankings", nargs="?", help="ranked results JSONL")
    ap.add_argument("--labels", help="judgments JSONL; defaults to clicked_file in rankings")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rel-threshold", type=int, default=1)
    ap.add_argument("--click-grade", type=int, default=2)
    ap.add_argument("--case-sensitive", action="store_true")
    ap.add_argument("--ablate", default="", help="comma-separated score components to remove")
    ap.add_argument("--compare", help="second rankings JSONL for top-k overlap")
    ap.add_argument("--min-mrr", type=float)
    ap.add_argument("--out-md")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rankings = load_jsonl(args.rankings)
    labels_rows = load_jsonl(args.labels) if args.labels else rankings
    labels = index_labels(labels_rows, args.click_grade)
    terms = [t for t in args.ablate.split(",") if t]

    per_query = []
    rows_out = []
    for row in rankings:
        key = (row.get("project", ""), row["query"])
        judged = labels.get(key)
        if not judged:
            continue
        results = row["results"]
        for r in results:
            r["_project"] = row.get("project")
        if terms:
            results = ablate_results(results, terms)
        metrics = score_query(results, judged, args.k, args.rel_threshold)
        metrics.update({"project": key[0], "query": key[1]})
        per_query.append(metrics)
        rows_out.append(metrics | {"top": [norm_path(r["path"], row.get("project"), not args.case_sensitive)
                                            for r in results[:args.k]]})

    if not per_query:
        print("no ranking rows matched a judgment row", file=sys.stderr)
        return 2

    summary = {
        "queries": len(per_query),
        "k": args.k,
        "ablate": terms,
        "mrr": mean(m["mrr"] for m in per_query),
        "ndcg": mean(m["ndcg"] for m in per_query),
        "recall": mean(m["recall"] for m in per_query),
        "hit_rate": mean(1.0 if m["first_relevant_rank"] else 0.0 for m in per_query),
    }

    overlap = None
    if args.compare:
        other = {(r.get("project", ""), r["query"]): r["results"] for r in load_jsonl(args.compare)}
        overlaps = []
        shifts = []
        for row in rankings:
            key = (row.get("project", ""), row["query"])
            if key not in other or key not in labels:
                continue
            a = [norm_path(r["path"], row.get("project"), not args.case_sensitive) for r in row["results"][:args.k]]
            b = [norm_path(r["path"], row.get("project"), not args.case_sensitive) for r in other[key][:args.k]]
            inter = len(set(a) & set(b))
            union = len(set(a) | set(b))
            if union:
                overlaps.append(inter / union)
            clicked = next((p for p, g in labels[key].items() if g >= args.rel_threshold), None)
            if clicked:
                ra = a.index(clicked) + 1 if clicked in a else None
                rb = b.index(clicked) + 1 if clicked in b else None
                shifts.append((ra, rb))
        overlap = {
            "jaccard": mean(overlaps),
            "clicked_rank_a": mean(x[0] or args.k + 1 for x in shifts),
            "clicked_rank_b": mean(x[1] or args.k + 1 for x in shifts),
            "n": len(overlaps),
        }

    lines = [f"# Ranking evaluation (k={args.k})", ""]
    if terms:
        lines.append(f"- ablation: `{', '.join(terms)}`")
    lines += [
        f"- queries: {summary['queries']}",
        f"- MRR@{args.k}: **{summary['mrr']:.4f}**",
        f"- NDCG@{args.k}: {fmt(summary['ndcg'])}",
        f"- Recall@{args.k}: {fmt(summary['recall'])}",
        f"- hit rate: {fmt(summary['hit_rate'])}",
    ]
    if overlap:
        lines += [
            f"- top-{args.k} Jaccard vs compare: {fmt(overlap['jaccard'])}",
            f"- clicked rank (A/B, missing = k+1): {overlap['clicked_rank_a']:.1f} / "
            f"{overlap['clicked_rank_b']:.1f} over {overlap['n']}",
        ]
    lines += ["", "| project | query | first | MRR | NDCG | recall | relevant |", "|---|---|---:|---:|---:|---:|---:|"]
    for m in per_query:
        lines.append(
            f"| {m['project']} | {m['query']} | {m['first_relevant_rank'] or '-'} | "
            f"{m['mrr']:.3f} | {fmt(m['ndcg'], 3)} | {fmt(m['recall'], 3)} | {m['n_relevant']} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(text)

    if args.min_mrr is not None and summary["mrr"] < args.min_mrr:
        print(f"FAIL: MRR {summary['mrr']:.4f} < {args.min_mrr}", file=sys.stderr)
        return 1
    return 0


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def self_test():
    labels = [{"project": "C:\\p", "query": "abc", "judgments": {"C:\\p\\src\\a.rs": 2, "C:\\p\\src\\b.rs": 1}}]
    rankings = [{"project": "C:\\p", "query": "abc", "results": [
        {"path": "C:\\p\\src\\x.rs", "total": 100, "_project": "C:\\p"},
        {"path": "C:\\p\\src\\a.rs", "total": 90, "_project": "C:\\p"},
        {"path": "C:\\p\\src\\b.rs", "total": 80, "_project": "C:\\p"},
    ]}]
    judged = index_labels(labels, 2)
    m = score_query(rankings[0]["results"], judged[("C:\\p", "abc")], 10, 1)
    assert m["first_relevant_rank"] == 2, m
    assert abs(m["mrr"] - 0.5) < 1e-9, m
    assert abs(m["recall"] - 1.0) < 1e-9, m
    # DCG = 3/log2(3) + 1/log2(4); IDCG = 3/log2(2) + 1/log2(3)
    assert abs(m["ndcg"] - 0.6590018048024133) < 1e-9, m

    ablated = ablate_results([
        {"path": "a", "total": 100, "components": {"frecency_boost": 50}, "rank": 0},
        {"path": "b", "total": 80, "components": {"frecency_boost": 0}, "rank": 1},
    ], ["frecency_boost"])
    assert [r["path"] for r in ablated] == ["b", "a"], ablated

    print("self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
