from pathlib import Path
p = Path(r"C:/Users/kanal/.codex/sessions/2026/07/14/rollout-2026-07-14T21-17-25-019f6359-9ee9-7493-9a7d-2baa315c040f.jsonl")
# print surrounding text for key lines
keys = [1968, 2092, 2144, 2752, 2795, 2815, 2830, 2903, 8259, 7933]
with p.open("r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i not in keys:
            continue
        print("=" * 20, "LINE", i, "len", len(line))
        # extract readable chunks
        import re
        for m in re.finditer(r"[\x20-\x7e]{40,}", line):
            s = m.group(0)
            if any(k in s.lower() for k in ("sumo", "process-run", "zlob", "walkbuilder", "wait_for", "fff-bench", "corpus", "warm", "timeout", "git ")):
                print(s[:500])
        print()
