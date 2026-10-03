#!/usr/bin/env python3
"""P1.2 feasibility: do index entries fit in u16?

The index table is Vec<u32> (one per chunk slot). Halving it needs the
UNIQUE chunk count < 65536, not the slot count. Also reports P1.4's
read-amplification model for comparison.
"""
import os, sys
from collections import Counter

CH = int(sys.argv[2]) if len(sys.argv) > 2 else 32
root0 = sys.argv[1]

slots = 0
uniq = Counter()
files = 0
dirs_seen = set()
cur_reads = 0
shared_reads = 0

for root, _d, names in os.walk(root0):
    rel = os.path.relpath(root, root0)
    prefix = b"" if rel == "." else rel.replace("\\", "/").encode() + b"/"
    plen_ch = -(-len(prefix) // CH)
    for n in names:
        p = prefix + n.encode()
        files += 1
        n_ch = -(-len(n) // CH)
        cur_reads += plen_ch + n_ch
        shared_reads += n_ch
        if plen_ch and root not in dirs_seen:
            dirs_seen.add(root)
            shared_reads += plen_ch
        for j in range(0, len(p), CH):
            c = p[j:j + CH]
            slots += 1
            buf = bytearray(CH)
            buf[: len(c)] = c
            uniq[bytes(buf)] += 1

u = len(uniq)
print(f"corpus={root0} chunk={CH} files={files} dirs={len(dirs_seen)}")
print(f"  index slots  : {slots:,}  -> {slots*4/1e6:.2f} MB as u32")
print(f"  unique chunks: {u:,} -> needs {u.bit_length()} bits")
print(f"  u16 table    : {'FITS' if u < 65536 else 'DOES NOT FIT'} ({slots*2/1e6:.2f} MB)")
print(f"  dedup        : {slots/u:.2f}x")
print(f"  P1.4 reads   : {cur_reads:,} -> {shared_reads:,} ({cur_reads/shared_reads:.2f}x fewer)")

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