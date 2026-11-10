#!/usr/bin/env python3
"""Pin the live G-01 tree with an rg path list and a bounded metadata sample."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_MOD = 1024
SAMPLE_BYTES = 32 * 1024


def iso(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


def now():
    return datetime.now(timezone.utc).isoformat()


def sample_hash(path, size):
    h = hashlib.blake2b(digest_size=16)
    h.update(str(size).encode("ascii"))
    with open(path, "rb", buffering=0) as f:
        if size <= 2 * SAMPLE_BYTES:
            while True:
                block = f.read(1024 * 1024)
                if not block:
                    break
                h.update(block)
        else:
            h.update(f.read(SAMPLE_BYTES))
            f.seek(size - SAMPLE_BYTES)
            h.update(f.read(SAMPLE_BYTES))
    return h.hexdigest()


def is_sampled(rel):
    digest = hashlib.blake2b(
        rel.encode("utf-8", "surrogateescape"), digest_size=8
    ).digest()
    return int.from_bytes(digest[:4], "big") % SAMPLE_MOD == 0


def stat_one(args):
    root, rel = args
    path = root / rel
    try:
        st = path.stat()
    except OSError as exc:
        return None, {"path": rel, "error": str(exc)}
    return (rel, st.st_size, st.st_mtime_ns, str(path)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rg", default="rg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--indexed-total", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    started = now()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [args.rg, "--files", "--null", "--no-messages"],
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    walk_s = time.perf_counter() - t0
    if proc.returncode not in (0, 1):
        raise SystemExit(
            f"rg failed rc={proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[-2000:]}"
        )
    raw = [p for p in proc.stdout.split(b"\0") if p]
    paths = sorted(
        {p.decode("utf-8", "replace").replace("\\", "/") for p in raw}
    )
    duplicate_paths = len(raw) - len(paths)

    t1 = time.perf_counter()
    files = []
    skipped = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, error in pool.map(stat_one, [(root, rel) for rel in paths], chunksize=256):
            if error:
                skipped.append(error)
            else:
                files.append(row)
    files.sort(key=lambda row: row[0])
    stat_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    sampled = 0
    sample_digest = hashlib.sha256()
    by_ext = Counter()
    by_top = Counter()
    by_bucket = Counter()
    total_bytes = 0
    root_h = hashlib.sha256()
    path_h = hashlib.sha256()
    mtimes = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for rel, size, mtime_ns, full in files:
            digest = ""
            if is_sampled(rel):
                try:
                    digest = sample_hash(full, size)
                except OSError as exc:
                    skipped.append({"path": rel, "error": str(exc)})
                    digest = "error"
                if digest != "error":
                    sampled += 1
                    sample_digest.update(rel.encode("utf-8", "surrogateescape"))
                    sample_digest.update(b"\0")
                    sample_digest.update(str(size).encode("ascii"))
                    sample_digest.update(b"\0")
                    sample_digest.update(digest.encode("ascii"))
                    sample_digest.update(b"\n")
            fh.write(f"{rel}\t{size}\t{mtime_ns}\t{digest}\n")

            root_h.update(rel.encode("utf-8", "surrogateescape"))
            root_h.update(b"\0")
            root_h.update(str(size).encode("ascii"))
            root_h.update(b"\0")
            root_h.update(str(mtime_ns).encode("ascii"))
            root_h.update(b"\n")
            path_h.update(rel.encode("utf-8", "surrogateescape"))
            path_h.update(b"\n")

            ext = Path(rel).suffix.lower() or "<none>"
            by_ext[ext] += 1
            by_top[rel.split("/", 1)[0]] += 1
            total_bytes += size
            mtimes.append(mtime_ns)
            if size < 4096:
                by_bucket["0-4096"] += 1
            elif size < 16384:
                by_bucket["4096-16384"] += 1
            elif size < 65536:
                by_bucket["16384-65536"] += 1
            elif size < 262144:
                by_bucket["65536-262144"] += 1
            elif size < 1048576:
                by_bucket["262144-1048576"] += 1
            elif size < 8388608:
                by_bucket["1048576-8388608"] += 1
            else:
                by_bucket[">=8388608"] += 1
    sample_s = time.perf_counter() - t2

    manifest_sha = hashlib.sha256()
    with open(out, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            manifest_sha.update(block)

    summary = {
        "root": str(root),
        "mode": "live-tree metadata manifest; bounded content sample",
        "mutable_tree": True,
        "fingerprint_authority": "driver full-set hash_ordered + total_matched",
        "scan_started_utc": started,
        "scan_finished_utc": now(),
        "walk_s": round(walk_s, 3),
        "stat_s": round(stat_s, 3),
        "sample_s": round(sample_s, 3),
        "rg_command": [args.rg, "--files", "--null", "--no-messages"],
        "rg_paths": len(paths),
        "duplicate_rg_paths": duplicate_paths,
        "stat_ok": len(files),
        "skipped": skipped,
        "manifest": str(out),
        "manifest_sha256": manifest_sha.hexdigest(),
        "root_sha256": root_h.hexdigest(),
        "path_set_sha256": path_h.hexdigest(),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024 ** 3), 3),
        "sample_mod": SAMPLE_MOD,
        "sample_bytes": SAMPLE_BYTES,
        "sample_files": sampled,
        "sample_sha256": sample_digest.hexdigest(),
        "by_top": dict(sorted(by_top.items())),
        "by_ext": dict(sorted(by_ext.items())),
        "by_size_bucket": dict(sorted(by_bucket.items())),
        "mtime_min": iso(min(mtimes)) if mtimes else None,
        "mtime_max": iso(max(mtimes)) if mtimes else None,
        "indexed_total": args.indexed_total,
    }
    if args.indexed_total is not None:
        summary["indexed_delta"] = len(paths) - args.indexed_total
        summary["indexed_delta_pct"] = round(
            100.0 * (len(paths) - args.indexed_total) / args.indexed_total, 4
        )
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if skipped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
