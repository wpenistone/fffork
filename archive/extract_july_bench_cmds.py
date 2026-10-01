import re
from pathlib import Path

p = Path(r"C:/Users/kanal/.codex/sessions/2026/07/14/rollout-2026-07-14T21-17-25-019f6359-9ee9-7493-9a7d-2baa315c040f.jsonl")
pat = re.compile(
    r"fff-bench[^\"\\]{0,200}|process-runs[^\"\\]{0,120}|sumo_fork[^\"\\]{0,80}|"
    r"cargo (build|run|test)[^\"\\]{0,160}|git am[^\"\\]{0,80}|"
    r"warm[^\"\\]{0,60}index|mega-corpus[^\"\\]{0,80}|"
    r"open_corpus|timeout_secs|fff-optimization",
    re.I,
)
seen = set()
hits = 0
with p.open("r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        for m in pat.finditer(line):
            s = m.group(0)[:240]
            if s in seen:
                continue
            seen.add(s)
            hits += 1
            print("L%d: %s" % (i, s))
            if hits >= 80:
                raise SystemExit
print("unique", hits)
