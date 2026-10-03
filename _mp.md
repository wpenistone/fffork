Research: memory-bound probe - the indirection is NOT the bottleneck

Second harness (archive/arena_lab/mem_probe.rs) removes the per-byte work
that made arena_lab compute-bound. Each layout resolves every chunk
pointer and touches ONE byte per chunk, so the residual signal is the
dependent load chain.

Full sumo: 227,706 paths, 725,552 chunk reads, 200 reps.

  A  indexed u32 + deduped chunks (today)   0.771 ms   1.00x  941 Mchunk/s
  E  contiguous, 23.2 MB (control)          1.184 ms   0.65x
  F  flat sequential over 19.7 MB           0.512 ms   1.51x

FINDINGS:

1. The indirection is nearly free. 941 Mchunk/s is ~3 cycles per
   dependent load at ~3 GHz, which is an L2/L3 hit cost. The whole
   7.48 MB arena + index is resident in this box's 24 MB L3, so chasing
   index -> chunk -> byte costs about what any load costs.

2. Contiguous storage is 35% SLOWER, purely because it needs 23.2 MB
   and spills L3. This is the direct answer to P1.4: sharing prefixes
   would trade a 7.48 MB L3-resident arena for a ~11-23 MB one. The
   earlier +21% memory estimate understated how bad that is, because it
   ignored the cache cliff. P1.4 is not merely neutral, it is negative.

3. Therefore the gate's measured 4.2-5.5 ms on sumo is NOT pointer
   chasing (that is 0.77 ms of it). It is the ~19.6 MB of per-byte
   fold + LCS work: 19.6 MB / 4.5 ms is ~4.4 GB/s, consistent with
   per-byte compute being the dominant term.

4. This also explains why SSE2 in the gate changed nothing: the fold and
   the needle-byte test vectorize, but the Hyyro LCS update is inherently
   serial (state carries byte to byte), so only part of the per-byte work
   was ever vectorizable.

CORRECTION TO PRIOR WORK: the "the gate is latency-bound on a 3-level
dependent load chain" premise that guided several rounds (and the
rationale for closing P1.4/P1.2/P1.3) is wrong on this machine at this
corpus size. The real target is per-byte work: either touch fewer bytes
or make the serial LCS step cheaper. Neither is a layout change.

Reproduce:
  rustc -O --edition 2021 mem_probe.rs -o mem_probe.exe
  .\mem_probe.exe sumo_big.bin