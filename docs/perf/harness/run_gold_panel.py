#!/usr/bin/env python3
"""G-01/G-02 driver panel: warm/query-cold, quiet/contended, optional A/B.

The driver's COLD=1 only evicts the query cache (9 dummy queries). It is not
an OS-cache drop; fresh-copy legs are the documented OS-cold protocol.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

CREATE_NO_WINDOW = 0x08000000
PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_INFORMATION = 0x0400

CPU_BURN = (
    "x = 0\n"
    "while True:\n"
    "    x = (x + 1) & 0xffffffff\n"
)

DISK_BURN = (
    "import os, sys, time\n"
    "path = sys.argv[1]\n"
    "mb = int(sys.argv[2])\n"
    "block = os.urandom(1024 * 1024)\n"
    "while True:\n"
    "    with open(path, 'wb', buffering=0) as f:\n"
    "        for _ in range(mb):\n"
    "            f.write(block)\n"
    "        f.flush()\n"
    "        os.fsync(f.fileno())\n"
)


def parse_line(line):
    out = {}
    for tok in line.split():
        if "=" not in tok:
            continue
        key, value = tok.split("=", 1)
        if key in ("query", "hash_ordered", "timed"):
            out[key] = value
            continue
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def start_load(cpu, disk_mb, disk_path):
    procs = []
    for _ in range(cpu):
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", CPU_BURN],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
        )
    if disk_mb > 0:
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", DISK_BURN, disk_path, str(disk_mb)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
        )
    return procs


def stop_load(procs):
    for p in procs:
        try:
            p.terminate()
        except OSError:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except OSError:
                pass


def set_process_affinity(pid, mask):
    if os.name != "nt" or not mask:
        return False
    import ctypes

    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        return bool(k32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)))
    finally:
        k32.CloseHandle(handle)


def run_one(exe, corpus, query, iters, threads, warmup, cold, limit, timeout, affinity=0):
    env = os.environ.copy()
    env["FFF_DRV_WARMUP"] = str(warmup)
    env["FFF_DRV_LIMIT"] = str(limit)
    env.pop("FFF_DRV_GIT", None)
    if cold:
        env["FFF_DRV_COLD"] = "1"
    else:
        env.pop("FFF_DRV_COLD", None)
    t0 = time.perf_counter()
    p = subprocess.Popen(
        [exe, corpus, query, str(iters), str(threads)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    set_process_affinity(p.pid, affinity)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        stdout, stderr = p.communicate()
        raise RuntimeError(f"driver timeout after {timeout}s")
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(
            f"driver failed rc={p.returncode}\nstdout={stdout[-2000:]}\nstderr={stderr[-2000:]}"
        )
    lines = [line for line in stdout.splitlines() if line.startswith("query=")]
    if not lines:
        raise RuntimeError(f"no query line\nstdout={stdout[-2000:]}\nstderr={stderr[-2000:]}")
    row = parse_line(lines[-1])
    if "timed" in row:
        timed_query = unquote(str(row["timed"]))
        if timed_query != query:
            raise RuntimeError(f"timed query {timed_query!r} != argv {query!r}")
        row["timed_query"] = timed_query
    else:
        row["timed_query"] = None
    row["external_wall_s"] = round(wall, 6)
    if affinity:
        row["affinity"] = f"0x{affinity:x}"
    return row


def abba_legs(exe_a, exe_b):
    if exe_b:
        return [(exe_a, "A"), (exe_b, "B"), (exe_b, "B"), (exe_a, "A")]
    return [(exe_a, "A")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe-a", required=True)
    ap.add_argument("--exe-b")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", default="score,controller,file_picke,index,netconvert")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--modes", default="warm,query-cold")
    ap.add_argument("--conditions", default="quiet")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--cpu-load", type=int, default=4)
    ap.add_argument("--disk-load-mb", type=int, default=64)
    ap.add_argument("--disk-path", default="D:\\fff-corpus\\contention.bin")
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--affinity", type=lambda s: int(s, 0), default=0,
                    help="Windows processor affinity mask for each driver process (e.g. 0xFFF)")
    args = ap.parse_args()

    queries = [q for q in args.queries.split(",") if q]
    thread_counts = [int(t) for t in args.threads.split(",") if t]
    modes = [m for m in args.modes.split(",") if m]
    conditions = [c for c in args.conditions.split(",") if c]

    meta = {
        "tag": args.tag,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.environ.get("COMPUTERNAME", ""),
        "cpu_count": os.cpu_count(),
        "corpus": args.corpus,
        "exe_a": args.exe_a,
        "exe_b": args.exe_b,
        "exe_a_sha256": sha256(args.exe_a),
        "exe_b_sha256": sha256(args.exe_b) if args.exe_b else None,
        "iters": args.iters,
        "warmup": args.warmup,
        "limit": args.limit,
        "reps": args.reps,
        "queries": queries,
        "timed_query": "bare",
        "threads": thread_counts,
        "modes": modes,
        "conditions": conditions,
        "cpu_load": args.cpu_load,
        "disk_load_mb": args.disk_load_mb,
        "affinity": f"0x{args.affinity:x}" if args.affinity else None,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for condition in conditions:
            procs = []
            if condition == "contended":
                procs = start_load(args.cpu_load, args.disk_load_mb, args.disk_path)
                time.sleep(args.settle)
            try:
                for rep in range(args.reps):
                    for query in queries:
                        driver_query = query
                        for threads in thread_counts:
                            for mode in modes:
                                cold = mode == "query-cold"
                                for exe, label in abba_legs(args.exe_a, args.exe_b):
                                    row = run_one(
                                        exe, args.corpus, driver_query, args.iters, threads,
                                        args.warmup, cold, args.limit, args.timeout, args.affinity,
                                    )
                                    row.update({
                                        "condition": condition,
                                        "rep": rep,
                                        "leg": label,
                                        "exe": exe,
                                        "query": query,
                                        "threads": threads,
                                        "mode": mode,
                                        "tag": args.tag,
                                    })
                                    rows.append(row)
                                    print(
                                        "%-10s rep=%d %-18s t=%-2d %-10s %s "
                                        "wall=%9.1f p50=%8.1f p99=%9.1f cyc=%12s "
                                        "tm=%-8s oh=%s"
                                        % (
                                            condition, rep, query, threads, mode, label,
                                            row.get("us_per_op", 0),
                                            row.get("p50_us", 0),
                                            row.get("p99_us", 0),
                                            row.get("cycles_per_op", 0),
                                            row.get("total_matched", ""),
                                            row.get("hash_ordered", ""),
                                        ),
                                        flush=True,
                                    )
            finally:
                stop_load(procs)
    finally:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": meta}) + "\n")
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
