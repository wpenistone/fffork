from pathlib import Path
import re
p = Path(r"C:/Users/kanal/.codex/sessions/2026/07/14/rollout-2026-07-14T21-17-25-019f6359-9ee9-7493-9a7d-2baa315c040f.jsonl")
# find tool outputs that contain profile-driver source or us_per_op implementation
want = re.compile(r"us_per_op|fn main\(\)|struct Args|file_picker 30000|checksum=", re.I)
hits = 0
with p.open("r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if "us_per_op" not in line and "fff-profile-driver" not in line and "profile-driver" not in line:
            continue
        s = line.replace("\\r\\n", "\n").replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        if "fn main" in s or "us_per_op=" in s or "println!" in s and "profile" in s:
            # extract a large slice around first key
            for key in ("fn main", "us_per_op", "checksum"):
                j = s.find(key)
                if j >= 0:
                    print("==== L%d around %s ====" % (i, key))
                    print(s[max(0, j-300):j+2500])
                    print()
                    hits += 1
                    break
        if hits > 15:
            break
print("hits", hits)
