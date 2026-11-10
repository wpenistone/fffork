#!/usr/bin/env python3
"""G-03 perf gate: pin, ABBA panel, exactness, thresholds, post-run pin.

A gate that cannot fail is not a gate. Exit 0 only when every stage passes.
See plans/perf-gate.md for the protocol and its limitations.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_gold_panel as panel  # noqa: E402

DEFAULT_QUERIES = "score,controller,file_picke,index,netconvert"


class GateAbort(Exception):
    pass


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def fail(reasons, message):
    reasons.append(message)
    print(f"FAIL: {message}", flush=True)


def run_pin(exe, corpus, query, threads, timeout, affinity=0):
    row = panel.run_one(exe, corpus, query, 1, threads, 0, False, 0, timeout, affinity)
    return {
        "total_matched": row.get("total_matched"),
        "items_len": row.get("items_len"),
        "checksum": row.get("checksum"),
        "hash_ordered": row.get("hash_ordered"),
        "peak_rss_mb": row.get("peak_rss_mb"),
        "timed_query": row.get("timed_query"),
    }


def check_pin(label, got, expect_total, expect_hash, reasons, expect_query=None):
    ok = True
    if got["total_matched"] != got["items_len"]:
        fail(reasons, f"{label} pin total {got['total_matched']} != items_len {got['items_len']}")
        ok = False
    timed = got.get("timed_query")
    if expect_query is not None and timed not in (None, expect_query):
        fail(reasons, f"{label} timed query {timed!r} != argv {expect_query!r}")
        ok = False
    if expect_total is not None and got["total_matched"] != expect_total:
        fail(reasons, f"{label} pin total {got['total_matched']} != expected {expect_total}")
        ok = False
    if expect_hash and got["hash_ordered"] != expect_hash:
        fail(reasons, f"{label} pin hash {got['hash_ordered']} != expected {expect_hash}")
        ok = False
    return ok


def run_panel(args, queries, conditions, modes, threads, out_path):
    rows = []
    seen = {}
    started = datetime.now(timezone.utc)
    meta = {
        "tag": args.tag,
        "started_utc": started.isoformat(),
        "host": os.environ.get("COMPUTERNAME", ""),
        "cpu_count": os.cpu_count(),
        "corpus": args.corpus,
        "timed_query": "bare",
        "pin_query": args.pin_query,
        "exe_a": args.candidate,
        "exe_b": args.baseline,
        "exe_a_sha256": panel.sha256(args.candidate),
        "exe_b_sha256": panel.sha256(args.baseline),
        "iters": args.iters,
        "warmup": args.warmup,
        "limit": args.limit,
        "reps": args.reps,
        "queries": queries,
        "threads": threads,
        "modes": modes,
        "conditions": conditions,
        "cpu_load": args.cpu_load,
        "disk_load_mb": args.disk_load_mb,
        "affinity": f"0x{args.affinity:x}" if args.affinity else None,
    }
    abort = None
    try:
        for condition in conditions:
            procs = []
            if condition == "contended":
                procs = panel.start_load(args.cpu_load, args.disk_load_mb, args.disk_path)
                time.sleep(args.settle)
            try:
                for rep in range(args.reps):
                    for query in queries:
                        driver_query = query
                        for t in threads:
                            for mode in modes:
                                cold = mode == "query-cold"
                                for exe, label in panel.abba_legs(args.candidate, args.baseline):
                                    row = panel.run_one(
                                        exe, args.corpus, driver_query, args.iters, t,
                                        args.warmup, cold, args.limit, args.timeout,
                                        args.affinity,
                                    )
                                    row.update({
                                        "condition": condition,
                                        "rep": rep,
                                        "leg": label,
                                        "exe": exe,
                                        "query": query,
                                        "threads": t,
                                        "mode": mode,
                                        "tag": args.tag,
                                    })
                                    rows.append(row)
                                    print(
                                        "%-10s rep=%d %-16s t=%-2d %-10s %s "
                                        "wall=%9.1f cyc=%12s tm=%-8s oh=%s"
                                        % (
                                            condition, rep, query, t, mode, label,
                                            row.get("us_per_op", 0),
                                            row.get("cycles_per_op", 0),
                                            row.get("total_matched", ""),
                                            row.get("hash_ordered", ""),
                                        ),
                                        flush=True,
                                    )
                                    key = (condition, query, t, mode, exe)
                                    obs = (row.get("total_matched"), row.get("checksum"))
                                    prev = seen.get(key)
                                    if prev is None:
                                        seen[key] = obs
                                    elif prev != obs:
                                        raise GateAbort(
                                            f"corpus drift {condition} {mode} {query} t={t} "
                                            f"{Path(exe).name}: {prev} -> {obs} at rep {rep}"
                                        )
            finally:
                panel.stop_load(procs)
    except GateAbort as exc:
        abort = str(exc)
        print(f"ABORT: {abort}", flush=True)
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": meta}) + "\n")
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"wrote {len(rows)} rows to {out_path}", flush=True)
    return meta, rows, abort


def legs(rows, condition, mode, query, t, label):
    return [
        r for r in rows
        if r["condition"] == condition
        and r["mode"] == mode
        and r["query"] == query
        and int(r["threads"]) == t
        and r["leg"] == label
    ]


def per_rep_ratio(a_rows, b_rows, key):
    rep_a = {int(r["rep"]): r for r in a_rows}
    rep_b = {int(r["rep"]): r for r in b_rows}
    reps = sorted(set(rep_a) & set(rep_b))
    values = []
    for rep in reps:
        a_med = statistics.median(x[key] for x in a_rows if int(x["rep"]) == rep)
        b_med = statistics.median(x[key] for x in b_rows if int(x["rep"]) == rep)
        if b_med:
            values.append(a_med / b_med)
    if not values:
        return None, None
    return statistics.median(values), (min(values), max(values))


def evaluate(args, rows, conditions, modes, queries, threads, reasons):
    out = {}
    for condition in conditions:
        for mode in modes:
            walls = []
            cycles = []
            for query in queries:
                for t in threads:
                    a = legs(rows, condition, mode, query, t, "A")
                    b = legs(rows, condition, mode, query, t, "B")
                    if not a or not b:
                        continue
                    rw, wr = per_rep_ratio(a, b, "us_per_op")
                    rc, cr = per_rep_ratio(a, b, "cycles_per_op")
                    out.setdefault(condition, {}).setdefault(mode, {})[query] = {
                        "wall": rw,
                        "wall_range": wr,
                        "cycles": rc,
                        "cycles_range": cr,
                    }
                    tag = f"{condition} {mode} {query}"
                    if rw is not None:
                        walls.append(rw)
                        if rw > args.max_wall:
                            fail(reasons, f"{tag} wall {fmt(rw)} > {args.max_wall}")
                        if not args.quick and wr and wr[1] > args.max_wall_upper:
                            fail(reasons, f"{tag} wall range {fmt(wr[1])} > {args.max_wall_upper}")
                    if rc is not None:
                        cycles.append(rc)
                        if rc > args.max_cycles:
                            fail(reasons, f"{tag} cycles {fmt(rc)} > {args.max_cycles}")
            gm_w = geomean(walls)
            gm_c = geomean(cycles)
            out.setdefault(condition, {}).setdefault("_geomean", {})[mode] = {
                "wall": gm_w,
                "cycles": gm_c,
                "n_queries": len(walls),
            }
            if gm_w is not None and gm_w > args.max_geomean:
                fail(reasons, f"{condition} {mode} geomean wall {fmt(gm_w)} > {args.max_geomean}")
            if gm_c is not None and gm_c > args.max_geomean:
                fail(reasons, f"{condition} {mode} geomean cycles {fmt(gm_c)} > {args.max_geomean}")
    return out


def check_exactness(rows, conditions, modes, queries, threads, allow_result_change, reasons):
    groups = {}
    for row in rows:
        key = (row["condition"], row["query"], int(row["threads"]), row["mode"], row["exe"])
        groups.setdefault(key, []).append(row)
    for key, got in groups.items():
        if len({r["total_matched"] for r in got}) > 1 or len({r["checksum"] for r in got}) > 1:
            fail(reasons, f"unstable same-executable totals {key}: "
                          f"{sorted({r['total_matched'] for r in got})}")
        if len({r["hash_ordered"] for r in got}) > 1:
            fail(reasons, f"unstable same-executable hash {key}")
    if allow_result_change:
        return
    for condition in conditions:
        for mode in modes:
            for query in queries:
                for t in threads:
                    a = legs(rows, condition, mode, query, t, "A")
                    b = legs(rows, condition, mode, query, t, "B")
                    if not a or not b:
                        continue
                    tag = f"{condition} {mode} {query} t={t}"
                    if {r["total_matched"] for r in a} != {r["total_matched"] for r in b}:
                        fail(reasons, f"cross-binary total mismatch {tag}")
                    if {r["hash_ordered"] for r in a} != {r["hash_ordered"] for r in b}:
                        fail(reasons, f"cross-binary hash mismatch {tag}")


def write_summary(jsonl, out_dir):
    summary = out_dir / "panel.md"
    try:
        proc = subprocess.run(
            [sys.executable, str(HERE / "summarize_gold_panel.py"), str(jsonl), "--out-md", str(summary)],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout)[-1000:]
    return summary, None


def parse_csv(value, cast=str):
    return [cast(v) for v in value.split(",") if v]


def main():
    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pin-query", default="", help="full-set pin query, not part of timed searches")
    ap.add_argument("--queries", default=DEFAULT_QUERIES)
    ap.add_argument("--threads", default="4")
    ap.add_argument("--modes", default="warm,query-cold")
    ap.add_argument("--conditions", default="quiet,contended")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--cpu-load", type=int, default=4)
    ap.add_argument("--disk-load-mb", type=int, default=64)
    ap.add_argument("--disk-path", default="D:\\fff-corpus\\contention.bin")
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--expect-total", type=int)
    ap.add_argument("--expect-candidate-hash")
    ap.add_argument("--expect-baseline-hash")
    ap.add_argument("--allow-result-change", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-wall", type=float, default=1.06)
    ap.add_argument("--max-cycles", type=float, default=1.06)
    ap.add_argument("--max-geomean", type=float, default=1.03)
    ap.add_argument("--max-wall-upper", type=float, default=1.30)
    ap.add_argument("--tag", default="gate")
    ap.add_argument("--affinity", type=lambda s: int(s, 0), default=0,
                    help="Windows processor affinity mask for every driver process (e.g. 0xFFF)")
    ap.add_argument("--out-dir", default=str(local / "fff-bench" / "gates"))
    args = ap.parse_args()

    queries = parse_csv(args.queries)
    threads = parse_csv(args.threads, int)
    modes = parse_csv(args.modes)
    conditions = parse_csv(args.conditions)
    pin_query = args.pin_query or "!*.zzz"

    reasons = []
    started = datetime.now(timezone.utc)
    stamp = started.astimezone().strftime("%Y-%m-%d")
    out_dir = Path(args.out_dir) / f"{stamp}-{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "panel.jsonl"

    if args.reps < 8 and not args.quick:
        fail(reasons, f"reps={args.reps} < 8; use --quick to override")
    for name, path in (("candidate", args.candidate), ("baseline", args.baseline)):
        if not Path(path).is_file():
            fail(reasons, f"{name} missing: {path}")
    if reasons:
        verdict = {"passed": False, "reasons": reasons}
        (out_dir / "gate.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(verdict, indent=2))
        return 1

    print("== preflight", flush=True)
    pre_candidate = run_pin(args.candidate, args.corpus, pin_query, threads[0], args.timeout, args.affinity)
    pre_baseline = run_pin(args.baseline, args.corpus, pin_query, threads[0], args.timeout, args.affinity)
    print(f"candidate pin {pre_candidate}", flush=True)
    print(f"baseline  pin {pre_baseline}", flush=True)
    check_pin("candidate", pre_candidate, args.expect_total, args.expect_candidate_hash, reasons, pin_query)
    check_pin("baseline", pre_baseline, args.expect_total, args.expect_baseline_hash, reasons, pin_query)
    if not args.allow_result_change and pre_candidate["hash_ordered"] != pre_baseline["hash_ordered"]:
        fail(reasons, "preflight full-set hashes differ; declare --allow-result-change")
    if reasons:
        verdict = {"passed": False, "reasons": reasons, "preflight": {
            "candidate": pre_candidate, "baseline": pre_baseline, "pin_query": pin_query}}
        (out_dir / "gate.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(verdict, indent=2))
        return 1

    print("== panel", flush=True)
    meta, rows, abort = run_panel(
        args, queries, conditions, modes, threads, jsonl,
    )
    if abort:
        fail(reasons, abort)

    print("== exactness", flush=True)
    if rows and not abort:
        check_exactness(
            rows, conditions, modes, queries, threads, args.allow_result_change, reasons,
        )

    ratios = {}
    if rows and not abort:
        print("== ratios", flush=True)
        ratios = evaluate(args, rows, conditions, modes, queries, threads, reasons)

    print("== post-run pin", flush=True)
    post_candidate = run_pin(args.candidate, args.corpus, pin_query, threads[0], args.timeout, args.affinity)
    post_baseline = run_pin(args.baseline, args.corpus, pin_query, threads[0], args.timeout, args.affinity)
    for label, pre, post in (
        ("candidate", pre_candidate, post_candidate),
        ("baseline", pre_baseline, post_baseline),
    ):
        if pre["total_matched"] != post["total_matched"]:
            fail(reasons, f"{label} corpus moved: {pre['total_matched']} -> {post['total_matched']}")
        if pre["hash_ordered"] != post["hash_ordered"]:
            fail(reasons, f"{label} corpus order changed: "
                          f"{pre['hash_ordered']} -> {post['hash_ordered']}")

    summary, summary_err = write_summary(jsonl, out_dir)
    passed = not reasons
    verdict = {
        "passed": passed,
        "tag": args.tag,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus,
        "timed_query": "bare",
        "pin_query": pin_query,
        "reps": args.reps,
        "quick": args.quick,
        "affinity": f"0x{args.affinity:x}" if args.affinity else None,
        "allow_result_change": args.allow_result_change,
        "thresholds": {
            "max_wall": args.max_wall,
            "max_cycles": args.max_cycles,
            "max_geomean": args.max_geomean,
            "max_wall_upper": args.max_wall_upper,
        },
        "preflight": {"candidate": pre_candidate, "baseline": pre_baseline},
        "postrun": {"candidate": post_candidate, "baseline": post_baseline},
        "ratios": ratios,
        "reasons": reasons,
        "artifacts": {
            "panel_jsonl": str(jsonl),
            "panel_md": str(summary) if summary else None,
            "panel_md_error": summary_err,
        },
    }
    (out_dir / "gate.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: verdict[k] for k in ("passed", "reasons", "preflight", "postrun")}, indent=2))
    print("VERDICT: " + ("PASS" if passed else "FAIL") + f"  artifacts={out_dir}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
