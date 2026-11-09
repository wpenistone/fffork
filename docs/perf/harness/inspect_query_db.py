#!/usr/bin/env python3
"""Inspect the FFF query-tracker LMDB (read-only).

Keys are blake3(project_path) for the history DBs and
blake3(project_path + "::" + query) for query_file_associations, so raw keys
are opaque. This dumps decoded values and, with --projects, maps keys back to
paths by hashing candidate roots.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import lmdb

HISTORY_DBS = ("query_history", "grep_query_history")
ASSOC_DB = "query_file_associations"


def rd_u64(buf, off):
    if off + 8 > len(buf):
        raise ValueError("short u64")
    return struct.unpack_from("<Q", buf, off)[0], off + 8


def rd_str(buf, off):
    n, off = rd_u64(buf, off)
    if off + n > len(buf):
        raise ValueError("short str")
    return buf[off:off + n].decode("utf-8", "replace"), off + n


def decode_history(buf):
    count, off = rd_u64(buf, 0)
    out = []
    for _ in range(count):
        query, off = rd_str(buf, off)
        ts, off = rd_u64(buf, off)
        out.append({"query": query, "timestamp": ts})
    return out


def decode_assoc(buf):
    path, off = rd_str(buf, 0)
    open_count, off = rd_u64(buf, off)
    last_opened, off = rd_u64(buf, off)
    return {"file_path": path, "open_count": open_count, "last_opened": last_opened}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path.home() / "AppData" / "Local" / "nvim-data" / "fff_queries"))
    ap.add_argument("--projects", nargs="*", default=[], help="candidate project roots to hash")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    env = lmdb.open(args.db, readonly=True, lock=False, max_dbs=16)
    names = []
    for name in HISTORY_DBS + (ASSOC_DB,):
        try:
            env.open_db(name.encode(), create=False)
            names.append(name)
        except lmdb.Error:
            pass
    result = {"path": args.db, "databases": {}}

    for name in names:
        db = env.open_db(name.encode())
        entries = []
        with env.begin(db=db) as txn:
            for key, val in txn.cursor():
                try:
                    if name in HISTORY_DBS:
                        entries.append({"key": key.hex(), "history": decode_history(val)})
                    elif name == ASSOC_DB:
                        entries.append({"key": key.hex(), **decode_assoc(val)})
                    else:
                        entries.append({"key": key.hex(), "bytes": len(val)})
                except Exception as exc:  # noqa: BLE001 - dump tool
                    entries.append({"key": key.hex(), "error": str(exc), "bytes": len(val)})
        result["databases"][name] = entries

    if args.projects:
        try:
            import blake3
        except ImportError:
            print("blake3 module missing; cannot map keys to projects", file=sys.stderr)
            blake3 = None
        if blake3:
            mapping = {}
            candidates = []
            for root in args.projects:
                p = Path(root)
                candidates.append(p)
                try:
                    candidates.extend([d for d in p.iterdir() if d.is_dir()])
                except OSError:
                    pass
            for p in candidates:
                mapping[blake3.blake3(str(p).encode()).hexdigest()] = str(p)
            result["key_map"] = {k: mapping.get(k, "?") for k in
                                 {e["key"] for v in result["databases"].values() for e in v}}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for name, entries in result["databases"].items():
            print(f"== {name}: {len(entries)} entries")
            for e in entries[:20]:
                if "history" in e:
                    qs = [h["query"] for h in e["history"]]
                    print(f"  {e['key'][:16]}.. n={len(qs)} {qs[-5:]}")
                elif "file_path" in e:
                    print(f"  {e['key'][:16]}.. count={e['open_count']} last={e['last_opened']} {e['file_path']}")
                else:
                    print(f"  {e}")
        if "key_map" in result:
            for k, v in result["key_map"].items():
                print(f"  {k[:16]}.. -> {v}")


if __name__ == "__main__":
    main()
