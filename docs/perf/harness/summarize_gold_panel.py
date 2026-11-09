#!/usr/bin/env python3
"""Summarize a run_gold_panel.py JSONL into A/B ratios and exactness checks."""

import argparse
import json
import math
import statistics
from collections import defaultdict


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out-md")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    with open(args.jsonl, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    meta = lines[0]["meta"]
    rows = lines[1:]

    groups = defaultdict(list)
    for row in rows:
        key = (
            row["condition"],
            row["query"],
            int(row["threads"]),
            row["mode"],
            row["leg"],
        )
        groups[key].append(row)

    conditions = sorted({r["condition"] for r in rows})
    modes = sorted({r["mode"] for r in rows})
    threads = sorted({int(r["threads"]) for r in rows})
    queries = []
    for r in rows:
        if r["query"] not in queries:
            queries.append(r["query"])

    summary = {"meta": meta, "conditions": {}}
    md = ["# G-01 panel summary", ""]
    md.append(f"- corpus: `{meta['corpus']}`")
    md.append(f"- exe A: `{meta['exe_a']}` (`{meta.get('exe_a_sha256')}`)")
    if meta.get("exe_b"):
        md.append(f"- exe B: `{meta['exe_b']}` (`{meta.get('exe_b_sha256')}`)")
    md.append(f"- iters={meta['iters']} warmup={meta['warmup']} limit={meta['limit']}")
    md.append("")

    for condition in conditions:
        cond_out = {}
        md.append(f"## {condition}")
        md.append("")
        header = (
            "| mode | query | t | A wall | B wall | A/B wall (rep range) | A cyc | B cyc "
            "| A/B cyc (rep range) | A total | B total | A hash | B hash |"
        )
        md.append(header)
        md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for mode in modes:
            ratios_wall = []
            ratios_cyc = []
            for query in queries:
                for t in threads:
                    legs = {}
                    for leg in ("A", "B"):
                        got = groups.get((condition, query, t, mode, leg), [])
                        if not got:
                            continue
                        legs[leg] = {
                            "wall": statistics.median(r["us_per_op"] for r in got),
                            "p50": statistics.median(r["p50_us"] for r in got),
                            "p99": statistics.median(r["p99_us"] for r in got),
                            "cyc": statistics.median(r["cycles_per_op"] for r in got),
                            "total": sorted({r["total_matched"] for r in got}),
                            "checksum": sorted({r["checksum"] for r in got}),
                            "hash": sorted({r["hash_ordered"] for r in got}),
                            "n": len(got),
                        }
                    if "A" not in legs:
                        continue
                    a = legs["A"]
                    b = legs.get("B")
                    rw = rc = None
                    wall_range = cyc_range = None
                    if b:
                        rep_wall = []
                        rep_cyc = []
                        reps = sorted(
                            {int(r["rep"]) for r in groups[(condition, query, t, mode, "A")]}
                            & {int(r["rep"]) for r in groups[(condition, query, t, mode, "B")]}
                        )
                        for rep in reps:
                            a_rep = [
                                r
                                for r in groups[(condition, query, t, mode, "A")]
                                if int(r["rep"]) == rep
                            ]
                            b_rep = [
                                r
                                for r in groups[(condition, query, t, mode, "B")]
                                if int(r["rep"]) == rep
                            ]
                            aw = statistics.median(r["us_per_op"] for r in a_rep)
                            bw = statistics.median(r["us_per_op"] for r in b_rep)
                            ac = statistics.median(r["cycles_per_op"] for r in a_rep)
                            bc = statistics.median(r["cycles_per_op"] for r in b_rep)
                            if bw:
                                rep_wall.append(aw / bw)
                            if bc:
                                rep_cyc.append(ac / bc)
                        if rep_wall:
                            rw = statistics.median(rep_wall)
                            wall_range = (min(rep_wall), max(rep_wall))
                            ratios_wall.append(rw)
                        if rep_cyc:
                            rc = statistics.median(rep_cyc)
                            cyc_range = (min(rep_cyc), max(rep_cyc))
                            ratios_cyc.append(rc)
                    md.append(
                        "| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                        % (
                            mode,
                            query,
                            t,
                            fmt(a["wall"], 1),
                            fmt(b["wall"], 1) if b else "n/a",
                            (
                                "%s (%s..%s)" % (fmt(rw), fmt(wall_range[0]), fmt(wall_range[1]))
                                if wall_range
                                else fmt(rw)
                            ),
                            fmt(a["cyc"], 0),
                            fmt(b["cyc"], 0) if b else "n/a",
                            (
                                "%s (%s..%s)" % (fmt(rc), fmt(cyc_range[0]), fmt(cyc_range[1]))
                                if cyc_range
                                else fmt(rc)
                            ),
                            ",".join(map(str, a["total"])),
                            ",".join(map(str, b["total"])) if b else "n/a",
                            ",".join(a["hash"]),
                            ",".join(b["hash"]) if b else "n/a",
                        )
                    )
                    cond_out.setdefault(mode, {}).setdefault(query, {})[str(t)] = {
                        "A": a,
                        "B": b,
                        "ratio_wall": rw,
                        "ratio_cycles": rc,
                    }
            cond_out.setdefault("_geomean", {})[mode] = {
                "wall": geomean(ratios_wall),
                "cycles": geomean(ratios_cyc),
                "n_queries": len(ratios_wall),
            }
            md.append(
                "| **%s geomean** | | | | | **%s** | | | **%s** | | | | |"
                % (mode, fmt(geomean(ratios_wall)), fmt(geomean(ratios_cyc)))
            )
            md.append("")
        summary["conditions"][condition] = cond_out

    exact = []
    for row in rows:
        exact.append(
            {
                "condition": row["condition"],
                "query": row["query"],
                "threads": row["threads"],
                "mode": row["mode"],
                "leg": row["leg"],
                "total_matched": row["total_matched"],
                "items_len": row["items_len"],
                "checksum": row["checksum"],
                "hash_ordered": row["hash_ordered"],
            }
        )
    summary["rows"] = exact
    md.append("## Exactness")
    md.append("")
    unstable = 0
    unstable_totals = 0
    for condition in conditions:
        for mode in modes:
            for query in queries:
                for t in threads:
                    for leg in ("A", "B"):
                        got = groups.get((condition, query, t, mode, leg), [])
                        if len(got) > 1 and len({r["hash_ordered"] for r in got}) > 1:
                            unstable += 1
                            md.append(
                                f"- unstable same-exe hash: {condition} {mode} "
                                f"{query} t={t} leg={leg}"
                            )
                        if len(got) > 1 and (
                            len({r["total_matched"] for r in got}) > 1
                            or len({r["checksum"] for r in got}) > 1
                        ):
                            unstable_totals += 1
                            md.append(
                                f"- unstable same-exe total/checksum: {condition} "
                                f"{mode} {query} t={t} leg={leg} "
                                f"totals={sorted({r['total_matched'] for r in got})}"
                            )
    if unstable == 0:
        md.append("- Same-executable legs returned identical `hash_ordered` everywhere.")
    if unstable_totals == 0:
        md.append("- Same-executable legs returned identical `total_matched`/`checksum` everywhere.")
    else:
        md.append(
            "- Corpus drift: same-executable totals changed within the batch; "
            "this panel is invalid under the pinned-corpus rule."
        )
    cross_hash_same = []
    cross_hash_diff = []
    cross_checksum_diff = []
    for condition in conditions:
        for mode in modes:
            for query in queries:
                for t in threads:
                    a = groups.get((condition, query, t, mode, "A"), [])
                    b = groups.get((condition, query, t, mode, "B"), [])
                    if not a or not b:
                        continue
                    if {r["checksum"] for r in a} != {r["checksum"] for r in b}:
                        cross_checksum_diff.append(f"{condition} {mode} {query}")
                    if {r["hash_ordered"] for r in a} == {r["hash_ordered"] for r in b}:
                        cross_hash_same.append(f"{condition} {mode} {query}")
                    else:
                        cross_hash_diff.append(f"{condition} {mode} {query}")
    if cross_checksum_diff:
        md.append(f"- Cross-binary checksum differs: {', '.join(cross_checksum_diff)}")
    else:
        md.append("- Cross-binary checksum identical on every query/mode.")
    md.append(
        "- Cross-binary `hash_ordered` identical: %s; differs: %s."
        % (", ".join(cross_hash_same) or "none", ", ".join(cross_hash_diff) or "none")
    )
    for condition in conditions:
        for mode in modes:
            for query in queries:
                for t in threads:
                    a = groups.get((condition, query, t, mode, "A"), [])
                    b = groups.get((condition, query, t, mode, "B"), [])
                    if not a or not b:
                        continue
                    at = {r["total_matched"] for r in a}
                    bt = {r["total_matched"] for r in b}
                    if at != bt:
                        md.append(
                            f"- total mismatch: {condition} {mode} {query} "
                            f"t={t} A={sorted(at)} B={sorted(bt)}"
                        )
    summary["unstable_same_exe_hashes"] = unstable

    text = "\n".join(md) + "\n"
    print(text)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
