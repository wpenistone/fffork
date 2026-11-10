#!/usr/bin/env python3
"""Pin the G-01 corpus: sorted path/size/mtime/content manifest + root digest."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def hash_file(path):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def iso(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--indexed-total", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    files = []
    skipped = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if "\t" in rel or "\n" in rel or "\r" in rel:
                raise SystemExit(f"path contains a manifest delimiter: {rel!r}")
            try:
                st = p.stat()
            except OSError as exc:
                skipped.append({"path": rel, "error": str(exc)})
                continue
            files.append((rel, st.st_size, st.st_mtime_ns, str(p)))
    files.sort(key=lambda row: row[0])

    by_ext = Counter()
    by_bucket = Counter()
    by_top = Counter()
    total_bytes = 0
    root_h = hashlib.blake2b(digest_size=32)
    errors = []
    hash_workers = max(1, args.workers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        with ThreadPoolExecutor(max_workers=hash_workers) as pool:
            hashes = pool.map(hash_file, [row[3] for row in files], chunksize=64)
            for (rel, size, mtime_ns, _path), digest in zip(files, hashes):
                try:
                    fh.write(f"{rel}\t{size}\t{mtime_ns}\t{digest}\n")
                except OSError as exc:
                    errors.append({"path": rel, "error": str(exc)})
                    continue
                root_h.update(rel.encode("utf-8"))
                root_h.update(b"\0")
                root_h.update(str(size).encode("ascii"))
                root_h.update(b"\0")
                root_h.update(str(mtime_ns).encode("ascii"))
                root_h.update(b"\0")
                root_h.update(digest.encode("ascii"))
                root_h.update(b"\n")

                ext = Path(rel).suffix.lower() or "<none>"
                by_ext[ext] += 1
                by_top[rel.split("/", 1)[0]] += 1
                total_bytes += size
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
                if len(files) and len(files) % 50000 == 0:
                    pass

    manifest_sha = hashlib.sha256()
    with open(out, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            manifest_sha.update(block)

    mtimes = [row[2] for row in files]
    summary = {
        "root": str(root),
        "manifest": str(out),
        "files": len(files),
        "manifest_rows": sum(by_ext.values()),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024 ** 3), 3),
        "root_blake2b_256": root_h.hexdigest(),
        "manifest_sha256": manifest_sha.hexdigest(),
        "generated_files": by_top.get("chromium-gen", 0),
        "by_top": dict(sorted(by_top.items())),
        "by_ext": dict(sorted(by_ext.items())),
        "by_size_bucket": dict(sorted(by_bucket.items())),
        "mtime_min": iso(min(mtimes)) if mtimes else None,
        "mtime_max": iso(max(mtimes)) if mtimes else None,
        "skipped": skipped,
        "write_errors": errors,
        "workers": hash_workers,
    }
    if args.indexed_total is not None:
        summary["indexed_total"] = args.indexed_total
        summary["target"] = 500_000
        summary["indexed_delta"] = args.indexed_total - 500_000
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if skipped or errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
