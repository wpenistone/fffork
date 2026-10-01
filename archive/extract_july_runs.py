import re
from pathlib import Path

p = Path(r"C:/Users/kanal/.codex/sessions/2026/07/14/rollout-2026-07-14T21-17-25-019f6359-9ee9-7493-9a7d-2baa315c040f.jsonl")
# actual shell / tool invocations
pat = re.compile(
    r"process-runs|sumo_fork|git am|cargo run[^\"\\]{0,200}|cargo build[^\"\\]{0,120}|"
    r"\./target/release/[^\"\\]{0,160}|fff-bench (run|check|baseline|gates|doctor|setup)[^\"\\]{0,180}|"
    r"skip_git|wait_for_scan|WalkBuilder|hidden\(",
    re.I,
)
seen = set()
hits = 0
with p.open("r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        for m in pat.finditer(line):
            s = m.group(0)[:300]
            key = s[:80]
            if key in seen:
                continue
            seen.add(key)
            hits += 1
            print("L%d: %s" % (i, s))
            if hits >= 100:
                raise SystemExit
print("unique", hits)
