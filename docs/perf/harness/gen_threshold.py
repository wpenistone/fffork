#!/usr/bin/env python3
"""Exact-count corpora for the PAR_THRESHOLD boundary sweep (G11/P0b).

Each corpus has exactly N files (1000 per subdirectory), all zero-byte
`file_<i>.txt`. Query `!*.zzz score` leaves all N files as constraint
survivors for ConstraintPlan::run; `*.zzz score` leaves zero. This
straddles PAR_THRESHOLD = 10_000 in constraints.rs.
"""

import os

ROOT = r"C:\Users\kanal\fff-corpus"
SIZES = (5000, 9500, 9999, 10000, 10001, 18000)


def main() -> None:
    for n in SIZES:
        d = os.path.join(ROOT, f"thr-{n}")
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            sub = os.path.join(d, f"{i // 1000:04d}")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, f"file_{i:06}.txt"), "w"):
                pass
        count = sum(len(fs) for _, _, fs in os.walk(d))
        print(f"{d}: {count} files", flush=True)
        assert count == n, (d, count, n)


if __name__ == "__main__":
    main()
