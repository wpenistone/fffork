#!/usr/bin/env python3
"""Dump real relative paths from a corpus as a flat binary file.

Format: u32 count, then per path: u16 len, then bytes. Used by the
standalone access-pattern benchmark so it measures real path shapes
without linking fff (a release build of fff takes ~4-9 min; rustc on
one self-contained file takes seconds).
"""
import os, struct, sys

root = sys.argv[1]
out = sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 40000

paths = []
for r, _d, names in os.walk(root):
    rel = os.path.relpath(r, root)
    pre = "" if rel == "." else rel.replace("\\", "/") + "/"
    for n in names:
        paths.append((pre + n).encode())
        if len(paths) >= limit:
            break
    if len(paths) >= limit:
        break

with open(out, "wb") as f:
    f.write(struct.pack("<I", len(paths)))
    for p in paths:
        f.write(struct.pack("<H", len(p)))
        f.write(p)
print(f"{len(paths)} paths -> {out}")