"""G-01 target-scale cells: p95, startup, first-query, typing stream, watcher.

Grep lives in its own subcommand because a full-corpus pass is minutes.

Usage examples:
  python g01_matrix.py search --candidate ... --baseline ... --corpus ...
  python g01_matrix.py typing --exe ... --corpus ...
  python g01_matrix.py watch --candidate ... --baseline ... --corpus ...
  python g01_matrix.py grep --candidate ... --baseline ... --corpus ... --query netconvert

Only same-run interleaved ratios are reported. The watcher cell writes one
temporary file in the corpus root and deletes it; re-pin afterwards.
"""

import argparse
import ctypes
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote


def set_affinity(pid, mask):
    if not mask:
        return True
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = k32.OpenProcess(0x0200 | 0x0400, False, pid)
    if not handle:
        return False
    try:
        return bool(k32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)))
    finally:
        k32.CloseHandle(handle)


def run_proc(cmd, env, timeout, affinity=0):
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env)
    set_affinity(p.pid, affinity)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        stdout, stderr = p.communicate()
        raise RuntimeError(f"timeout after {timeout}s: {' '.join(cmd)}")
    wall = time.perf_counter() - t0
    return p.returncode, stdout, stderr, wall


def env_base(warmup, limit, cold=False, trace=False):
    env = os.environ.copy()
    env["FFF_DRV_WARMUP"] = str(warmup)
    env["FFF_DRV_LIMIT"] = str(limit)
    env.pop("FFF_DRV_GIT", None)
    if cold:
        env["FFF_DRV_COLD"] = "1"
    else:
        env.pop("FFF_DRV_COLD", None)
    if trace:
        env["FFF_DRV_TRACE"] = "1"
    else:
        env.pop("FFF_DRV_TRACE", None)
    return env


def run_driver(exe, corpus, query, iters, threads, env, affinity=0):
    rc, stdout, stderr, wall = run_proc(
        [exe, corpus, query, str(iters), str(threads)], env, 1800, affinity)
    if rc != 0:
        raise RuntimeError(f"driver failed rc={rc}\n{stdout[-2000:]}\n{stderr[-2000:]}")
    return stdout, stderr, wall


def field(text, name):
    m = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", text)
    return m.group(1) if m else None


def require_timed(text, argv_query):
    raw = field(text, "timed")
    if raw is None:
        return None
    got = unquote(raw)
    if got != argv_query:
        raise RuntimeError(f"timed query {got!r} != argv {argv_query!r}")
    return got


def timed_samples(stderr):
    out = []
    for line in stderr.splitlines():
        if not line.startswith("TIMED "):
            continue
        us = field(line, "us")
        cyc = field(line, "cyc")
        if us is not None and cyc is not None:
            out.append((float(us), float(cyc)))
    return out


def pct(values, q):
    if not values:
        return None
    vals = sorted(values)
    idx = max(0, math.ceil(q * len(vals)) - 1)
    return vals[idx]


def median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def ratio(a, b):
    return a / b if a and b else None


def summarize(rows, key):
    a = [r[key] for r in rows if r["leg"] == "A"]
    b = [r[key] for r in rows if r["leg"] == "B"]
    return {"A": median(a), "B": median(b), "ratio": ratio(median(a), median(b)),
            "A_range": [min(a), max(a)] if a else None,
            "B_range": [min(b), max(b)] if b else None}


def cmd_search(args):
    out = {"p95": {}, "startup": {}, "first_query": {}}
    for query in args.queries:
        rows = []
        for rep in range(args.reps):
            for leg, exe in panel_legs(args.candidate, args.baseline):
                env = env_base(args.warmup, args.limit, cold=False, trace=True)
                stdout, err, _ = run_driver(exe, args.corpus, query,
                                            args.p95_iters, args.threads, env, args.affinity)
                require_timed(stdout, query)
                samples = timed_samples(err)
                if not samples:
                    raise RuntimeError(f"no TIMED samples for {query} leg {leg}")
                us = [s[0] for s in samples]
                cyc = [s[1] for s in samples]
                rows.append({"query": query, "rep": rep, "leg": leg,
                             "us": us, "cyc": cyc,
                             "p50": pct(us, 0.50), "p95": pct(us, 0.95), "p99": pct(us, 0.99),
                             "c50": pct(cyc, 0.50), "c95": pct(cyc, 0.95), "c99": pct(cyc, 0.99)})
        a = [r for r in rows if r["leg"] == "A"]
        b = [r for r in rows if r["leg"] == "B"]
        # Per-rep p95 ratio, then median across reps (block-median style).
        rep_ratios = []
        for rep in range(args.reps):
            ra = [r["p95"] for r in a if r["rep"] == rep]
            rb = [r["p95"] for r in b if r["rep"] == rep]
            if ra and rb:
                rep_ratios.append(median(ra) / median(rb))
        out["p95"][query] = {
            "A_p95": median([r["p95"] for r in a]), "B_p95": median([r["p95"] for r in b]),
            "ratio": median(rep_ratios),
            "rep_ratios": [round(x, 4) for x in rep_ratios],
            "A_p50": median([r["p50"] for r in a]), "B_p50": median([r["p50"] for r in b]),
            "A_p99": median([r["p99"] for r in a]), "B_p99": median([r["p99"] for r in b]),
            "A_c95": median([r["c95"] for r in a]), "B_c95": median([r["c95"] for r in b]),
            "cyc_ratio": median([
                median([r["c95"] for r in a if r["rep"] == rep]) /
                median([r["c95"] for r in b if r["rep"] == rep])
                for rep in range(args.reps) if [r for r in a if r["rep"] == rep]
            ]),
            "samples_per_leg": args.p95_iters,
        }

    # Startup: zero timed iterations, external wall includes spawn + scan.
    rows = []
    for rep in range(args.reps):
        for leg, exe in panel_legs(args.candidate, args.baseline):
            env = env_base(0, 0)
            _, _, wall = run_driver(exe, args.corpus, args.pin_query, 0, args.threads, env,
                                    args.affinity)
            rows.append({"rep": rep, "leg": leg, "wall_ms": wall * 1000.0})
    rep_ratios = []
    for rep in range(args.reps):
        ra = [r["wall_ms"] for r in rows if r["leg"] == "A" and r["rep"] == rep]
        rb = [r["wall_ms"] for r in rows if r["leg"] == "B" and r["rep"] == rep]
        if ra and rb:
            rep_ratios.append(median(ra) / median(rb))
    out["startup"] = {
        "A_ms": median([r["wall_ms"] for r in rows if r["leg"] == "A"]),
        "B_ms": median([r["wall_ms"] for r in rows if r["leg"] == "B"]),
        "ratio": median(rep_ratios), "rep_ratios": [round(x, 4) for x in rep_ratios],
    }

    # First query: one timed search after scan, no warmup.
    for query in args.queries:
        rows = []
        for rep in range(args.reps):
            for leg, exe in panel_legs(args.candidate, args.baseline):
                env = env_base(0, args.limit)
                stdout, _, _ = run_driver(exe, args.corpus, query,
                                          1, args.threads, env, args.affinity)
                rows.append({"query": query, "timed_query": require_timed(stdout, query),
                             "rep": rep, "leg": leg,
                             "us": float(field(stdout, "us_per_op")),
                             "total": int(field(stdout, "total_matched")),
                             "checksum": int(field(stdout, "checksum"))})
        rep_ratios = []
        for rep in range(args.reps):
            ra = [r["us"] for r in rows if r["leg"] == "A" and r["rep"] == rep]
            rb = [r["us"] for r in rows if r["leg"] == "B" and r["rep"] == rep]
            if ra and rb:
                rep_ratios.append(median(ra) / median(rb))
        out["first_query"][query] = {
            "A_us": median([r["us"] for r in rows if r["leg"] == "A"]),
            "B_us": median([r["us"] for r in rows if r["leg"] == "B"]),
            "ratio": median(rep_ratios), "rep_ratios": [round(x, 4) for x in rep_ratios],
            "A_total": sorted({r["total"] for r in rows if r["leg"] == "A"}),
            "B_total": sorted({r["total"] for r in rows if r["leg"] == "B"}),
        }
    return out


def panel_legs(a, b):
    return [("A", a), ("B", b), ("B", b), ("A", a)]


def typing_session(exe, args, prefixes):
    env = env_base(0, args.limit)
    env["FFF_DRV_TYPING_SUFFIX"] = ""
    stdout, _, _ = run_driver(exe, args.corpus, prefixes, 0, args.threads, env,
                              args.affinity)
    keys = {}
    for line in stdout.splitlines():
        if line.startswith("KEY "):
            prefix = field(line, "prefix")
            keys[prefix] = {"us": float(field(line, "us")),
                            "total": int(field(line, "total")),
                            "timed": require_timed(line, prefix)}
        elif line.startswith("STREAM "):
            keys["_stream"] = {
                "scan_ms": float(field(line, "scan_ms")),
                "total_ms": float(field(line, "total_ms")),
                "search_us": float(field(line, "search_us")),
                "checksum": int(field(line, "checksum")),
            }
    return keys


def cmd_typing(args):
    streams = {
        "score": "s,sc,sco,scor,score",
        "netconvert": "n,ne,net,netc,netco,netcon,netconv,netconve,netconver,netconvert",
        "file_picke": "f,fi,fil,file,file_,file_p,file_pi,file_pic,file_pick,file_picke",
        "controller": "c,co,con,cont,contr,contro,control,controll,controlle,controller",
    }
    out = {}
    for name, prefixes in streams.items():
        order = prefixes.split(",")
        a_sessions, b_sessions = [], []
        if args.baseline:
            for rep in range(args.sessions):
                for leg, exe in panel_legs(args.exe, args.baseline):
                    (a_sessions if leg == "A" else b_sessions).append(
                        typing_session(exe, args, prefixes))
        else:
            for _ in range(args.sessions):
                a_sessions.append(typing_session(args.exe, args, prefixes))

        def keys_summary(sessions):
            return {
                p: {
                    "us": median([s[p]["us"] for s in sessions if p in s]),
                    "total": sorted({s[p]["total"] for s in sessions if p in s}),
                }
                for p in order
            }

        entry = {
            "keys": keys_summary(a_sessions),
            "first_us": median([s[order[0]]["us"] for s in a_sessions if order[0] in s]),
            "last_us": median([s[order[-1]]["us"] for s in a_sessions if order[-1] in s]),
            "total_ms": median([s["_stream"]["total_ms"] for s in a_sessions if "_stream" in s]),
            "scan_ms": median([s["_stream"]["scan_ms"] for s in a_sessions if "_stream" in s]),
            "sessions": len(a_sessions),
        }
        if args.baseline:
            b_keys = keys_summary(b_sessions)
            entry["B_keys"] = b_keys
            entry["B_first_us"] = median(
                [s[order[0]]["us"] for s in b_sessions if order[0] in s])
            entry["B_last_us"] = median(
                [s[order[-1]]["us"] for s in b_sessions if order[-1] in s])
            entry["B_total_ms"] = median(
                [s["_stream"]["total_ms"] for s in b_sessions if "_stream" in s])
            entry["B_scan_ms"] = median(
                [s["_stream"]["scan_ms"] for s in b_sessions if "_stream" in s])
            entry["first_ratio"] = ratio(entry["first_us"], entry["B_first_us"])
            entry["last_ratio"] = ratio(entry["last_us"], entry["B_last_us"])
            entry["total_ratio"] = ratio(entry["total_ms"], entry["B_total_ms"])
            entry["key_ratios"] = {
                p: ratio(entry["keys"][p]["us"], b_keys[p]["us"]) for p in order
            }
        out[name] = entry
    return out


def cmd_watch(args):
    rows = []
    for rep in range(args.reps):
        for leg, exe in panel_legs(args.candidate, args.baseline):
            env = env_base(0, 0)
            rc, stdout, stderr, _ = run_proc([exe, args.corpus, str(args.timeout)], env,
                                             args.timeout + 60, args.affinity)
            if rc != 0:
                raise RuntimeError(f"watch probe failed: {stderr[-2000:]}")
            line = next((x for x in stdout.splitlines() if x.startswith("WATCH ")), None)
            if line is None:
                raise RuntimeError("no WATCH line")
            rows.append({
                "rep": rep, "leg": leg,
                "scan_ms": float(field(line, "scan_ms")),
                "watch_ready_ms": float(field(line, "watch_ready_ms")),
                "add_ms": float(field(line, "add_ms")),
                "remove_ms": float(field(line, "remove_ms")),
                "added": int(field(line, "added")),
                "removed": int(field(line, "removed")),
                "initial": int(field(line, "initial")),
                "final": int(field(line, "final")),
            })
    out = {}
    for key in ("scan_ms", "watch_ready_ms", "add_ms", "remove_ms"):
        rep_ratios = []
        for rep in range(args.reps):
            ra = [r[key] for r in rows if r["leg"] == "A" and r["rep"] == rep]
            rb = [r[key] for r in rows if r["leg"] == "B" and r["rep"] == rep]
            if ra and rb:
                rep_ratios.append(median(ra) / median(rb))
        out[key] = {
            "A": median([r[key] for r in rows if r["leg"] == "A"]),
            "B": median([r[key] for r in rows if r["leg"] == "B"]),
            "ratio": median(rep_ratios),
            "A_range": [min(r[key] for r in rows if r["leg"] == "A"),
                        max(r[key] for r in rows if r["leg"] == "A")],
            "B_range": [min(r[key] for r in rows if r["leg"] == "B"),
                        max(r[key] for r in rows if r["leg"] == "B")],
        }
    out["all_added"] = all(r["added"] == 1 for r in rows)
    out["all_removed"] = all(r["removed"] == 1 for r in rows)
    out["totals_consistent"] = len({r["initial"] for r in rows}) == 1 and \
        len({r["final"] for r in rows}) == 1
    out["initial"] = sorted({r["initial"] for r in rows})
    out["final"] = sorted({r["final"] for r in rows})
    return out


def cmd_grep(args):
    rows = []
    env = env_base(0, 0)
    for rep in range(args.reps):
        for leg, exe in panel_legs(args.candidate, args.baseline):
            rc, stdout, stderr, _ = run_proc(
                [exe, "--path", args.corpus, "--query", args.query, "--iters", "1"],
                env, 1800, args.affinity)
            if rc != 0:
                raise RuntimeError(f"grep failed: {stderr[-2000:]}")
            text = stderr
            scan = re.search(r"Scanning files\.\.\. (\d+) files in ([0-9.]+)s", text)
            iter_line = re.search(
                r"iter 1: ([0-9.]+)(ms|s|us) \((\d+) matches in (\d+) files, (\d+)/(\d+) searched\)",
                text)
            if not scan or not iter_line:
                raise RuntimeError(f"cannot parse grep output:\n{text[-2000:]}")
            value, unit = float(iter_line.group(1)), iter_line.group(2)
            us = value * {"us": 1, "ms": 1e3, "s": 1e6}[unit]
            rows.append({
                "rep": rep, "leg": leg, "scan_files": int(scan.group(1)),
                "scan_s": float(scan.group(2)), "us": us,
                "matches": int(iter_line.group(3)), "files": int(iter_line.group(4)),
                "searched": int(iter_line.group(5)), "total_files": int(iter_line.group(6)),
            })
    rep_ratios = []
    for rep in range(args.reps):
        ra = [r["us"] for r in rows if r["leg"] == "A" and r["rep"] == rep]
        rb = [r["us"] for r in rows if r["leg"] == "B" and r["rep"] == rep]
        if ra and rb:
            rep_ratios.append(median(ra) / median(rb))
    return {
        "query": args.query,
        "A_us": median([r["us"] for r in rows if r["leg"] == "A"]),
        "B_us": median([r["us"] for r in rows if r["leg"] == "B"]),
        "ratio": median(rep_ratios), "rep_ratios": [round(x, 4) for x in rep_ratios],
        "A_scan_s": median([r["scan_s"] for r in rows if r["leg"] == "A"]),
        "B_scan_s": median([r["scan_s"] for r in rows if r["leg"] == "B"]),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--corpus", required=True)
        p.add_argument("--queries", default="score,controller,file_picke,index,netconvert")
        p.add_argument("--pin-query", default="!*frontier_research*")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--limit", type=int, default=100)
        p.add_argument("--reps", type=int, default=4)
        p.add_argument("--affinity", type=lambda s: int(s, 0), default=0xFFF)
        p.add_argument("--out", required=True)

    s = sub.add_parser("search")
    s.add_argument("--candidate", required=True)
    s.add_argument("--baseline", required=True)
    s.add_argument("--warmup", type=int, default=20)
    s.add_argument("--p95-iters", type=int, default=40)
    common(s)

    t = sub.add_parser("typing")
    t.add_argument("--exe", required=True)
    t.add_argument("--baseline", default=None)
    t.add_argument("--sessions", type=int, default=3)
    common(t)

    w = sub.add_parser("watch")
    w.add_argument("--candidate", required=True)
    w.add_argument("--baseline", required=True)
    w.add_argument("--timeout", type=int, default=120)
    common(w)

    g = sub.add_parser("grep")
    g.add_argument("--candidate", required=True)
    g.add_argument("--baseline", required=True)
    g.add_argument("--query", default="netconvert")
    common(g)

    args = ap.parse_args()
    args.queries = [q for q in args.queries.split(",") if q]
    fn = {"search": cmd_search, "typing": cmd_typing, "watch": cmd_watch, "grep": cmd_grep}[args.cmd]
    result = fn(args)
    text = json.dumps(result, indent=2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
