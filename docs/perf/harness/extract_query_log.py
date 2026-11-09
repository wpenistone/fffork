#!/usr/bin/env python3
"""Export real FFF query-tracker rows to query_log.jsonl for G-04.

The tracker stores blake3(project_path) and blake3(project_path + "::" + query)
keys, so project roots must be supplied. Values are bincode-encoded; see
inspect_query_db.py for the decoders.

Output row:
  {"project": "...", "query": "...", "last_used": <unix>, "clicked_file": "...",
   "open_count": 1, "source": "tracker"}

`clicked_file` is the last file opened from that exact query in that project;
it is a single positive label, not a full judgment set.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import lmdb

from inspect_query_db import (
    ASSOC_DB,
    HISTORY_DBS,
    decode_assoc,
    decode_history,
)


def load_blake3():
    try:
        import blake3
        return blake3
    except ImportError:
        print(
            "blake3 is required to map tracker keys to project paths; "
            "install it with `python -m pip install blake3` or pass --project-map",
            file=sys.stderr,
        )
        return None


def project_map(roots, map_path, b3):
    mapping = {}
    if map_path:
        with open(map_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        for root, digest in raw.items():
            mapping[bytes.fromhex(digest)] = root
        return mapping
    if not b3:
        return mapping
    candidates = []
    for root in roots:
        p = Path(root)
        candidates.append(str(p))
        try:
            candidates.extend(str(d) for d in p.iterdir() if d.is_dir())
        except OSError:
            pass
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        mapping[b3.blake3(path.encode()).digest()] = path
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path.home() / "AppData" / "Local" / "nvim-data" / "fff_queries"))
    ap.add_argument("--roots", nargs="+", default=[], help="project roots to hash")
    ap.add_argument("--project-map", help="JSON {path: blake3_hex} to avoid the blake3 module")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b3 = load_blake3()
    if not b3 and not args.project_map:
        return 2
    projects = project_map(args.roots, args.project_map, b3)

    env = lmdb.open(args.db, readonly=True, lock=False, max_dbs=16)
    history = {name: {} for name in HISTORY_DBS}
    assoc = {}

    for name in HISTORY_DBS:
        try:
            db = env.open_db(name.encode(), create=False)
        except lmdb.Error:
            continue
        with env.begin(db=db) as txn:
            for key, val in txn.cursor():
                project = projects.get(key)
                if project:
                    history[name][project] = decode_history(val)

    try:
        db = env.open_db(ASSOC_DB.encode(), create=False)
        with env.begin(db=db) as txn:
            for key, val in txn.cursor():
                assoc[key] = decode_assoc(val)
    except lmdb.Error:
        pass

    rows = []
    for name, by_project in history.items():
        for project, entries in by_project.items():
            for entry in entries:
                query = entry["query"]
                row = {
                    "project": project,
                    "query": query,
                    "last_used": entry["timestamp"],
                    "source": name,
                }
                if b3:
                    key = b3.blake3(f"{project}::{query}".encode()).digest()
                    hit = assoc.get(key)
                    if hit:
                        row.update({
                            "clicked_file": hit["file_path"],
                            "open_count": hit["open_count"],
                            "last_opened": hit["last_opened"],
                        })
                rows.append(row)

    rows.sort(key=lambda r: (r["project"], r["last_used"]))
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    projects_seen = {r["project"] for r in rows}
    clicks = sum(1 for r in rows if "clicked_file" in r)
    print(f"wrote {len(rows)} queries from {len(projects_seen)} projects "
          f"({clicks} with a click label) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
