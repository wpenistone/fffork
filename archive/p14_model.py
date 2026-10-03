#!/usr/bin/env python3
"""Chunk-read amplification: does P1.4 (share dir prefixes) actually pay?

Per file the gate reads ceil(prefix/CH) + ceil(name/CH) chunks.
With prefix sharing: dirs*ceil(prefix/CH) + files*ceil(name/CH).
Also reports memory trade: index bytes saved vs arena bytes added.
"""
import os, sys

CH = int(sys.argv[2]) if len(sys.argv) > 2 else 32

files = 0
cur_reads = 0          # chunk reads today
shared_reads = 0       # with per-dir prefix sharing
dirs_seen = set()
idx_now = 0
idx_shared = 0
name_slots = 0
dir_slots = 0

for root, dnames, names in os.walk(sys.argv[1]):
    rel_root = os.path.relpath(root, sys.argv[1])
    prefix = b"" if rel_root == "." else rel_root.replace("\\", "/").encode() + b"/"
    plen_ch = -(-len(prefix) // CH)
    for n in names:
        nb = n.encode()
        n_ch = -(-len(nb) // CH)
        files += 1
        cur_reads += plen_ch + n_ch
        idx_now += plen_ch + n_ch
        shared_reads += n_ch
        idx_shared += n_ch
        name_slots += n_ch
        if plen_ch:
            if root not in dirs_seen:
                dirs_seen.add(root)
                shared_reads += plen_ch
                idx_shared += plen_ch
                dir_slots += plen_ch

print(f"corpus={sys.argv[1]} chunk={CH} files={files} dirs={len(dirs_seen)}")
print(f"  chunk reads today : {cur_reads:,}")
print(f"  chunk reads shared: {shared_reads:,}  ({cur_reads/shared_reads:.2f}x fewer)")
print(f"  index table       : {idx_now*4/1e6:.2f} MB -> {idx_shared*4/1e6:.2f} MB")
print(f"  arena slots needed: dirs {dir_slots*CH/1e6:.2f} MB (dedups) + names {name_slots*CH/1e6:.2f} MB (NO dedup)")