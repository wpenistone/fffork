import re
from pathlib import Path
p = Path(r"C:/Users/kanal/.codex/sessions/2026/07/14/rollout-2026-07-14T21-17-25-019f6359-9ee9-7493-9a7d-2baa315c040f.jsonl")
pat = re.compile(r"profile-driver|process-runs|fff-profile|warm_index|sumo_fork|linux.?95|deep_nested", re.I)
hits = 0
with p.open("r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if not pat.search(line):
            continue
        # pull shell_command inputs
        for m in re.finditer(r"shell_command\(\{command:\\\"(.{20,800}?)\\\"", line):
            s = m.group(1).encode().decode("unicode_escape", "replace")
            if pat.search(s):
                hits += 1
                print("L%d CMD: %s" % (i, s[:600]))
                print()
        if hits > 40:
            break
print("cmds", hits)
