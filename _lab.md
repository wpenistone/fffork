Research: access-pattern lab (arena_lab) - hypothesis NOT confirmed

Built a standalone harness (no fff dependency; rustc in seconds vs a 4-9 min
release build) that streams every real path byte through four arena
layouts doing identical trivial work per byte, so any difference is
layout cost rather than compute.

NEGATIVE RESULT. Full sumo corpus, 227,706 paths, 19.7 MB of path bytes
(working set past this box's 24 MB L3), avg of 50 reps:

  A  u32 index + deduped chunks (today)   11.742 ms   1.00x
  B  u16 index + deduped chunks           11.429 ms   1.03x
  C  contiguous, padded, NO index/dedup   12.090 ms   0.97x
  D  first chunk stored inline per path   11.271 ms   1.04x

All four land within 4%. Layout C has no index table at all - a pure
sequential byte stream with zero dependent loads - and it is 3% SLOWER
than today's deduped indirection. So eliminating the entire dependent
load chain buys nothing.

This falsifies the "the gate is latency-bound on the 3-level index chain"
story that had driven several rounds of work. At minimum, the indirection
is not what costs.

Methodological caveat that also invalidates the harness as a gate proxy:
11.7 ms for 19.7 MB is ~1.7 bytes/ns, which is about scalar byte-loop
throughput. The harness is COMPUTE-bound, not memory-bound, so it cannot
resolve memory differences at all. An earlier run at 60k paths showed
the same 1.00-1.10x spread with a 2.2 MB working set that fits in L3 -
identical conclusion at both sizes, which is itself the tell that the
measurement is not seeing memory.

To answer the real question the harness needs per-byte work reduced to
near zero (vectorised fold, or a pointer-walk-only variant compared
against a pure sequential read of the same bytes) so the remaining
signal is memory. That is the next step, and it should be done before
any further arena work - the three layout projects already closed
(P1.4/P1.2/P1.3) were chosen on the premise this experiment just
undercut.

Files: archive/arena_lab/{arena_lab.rs,dump_paths.py}
Reproduce: rustc -O --edition 2021 arena_lab.rs -o arena_lab.exe
          python dump_paths.py <corpus> out.bin
          .\arena_lab.exe out.bin