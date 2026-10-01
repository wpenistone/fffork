from pathlib import Path
import re
p = Path(r"C:/Users/kanal/.codex/sessions/2026/07/14/rollout-2026-07-14T21-17-25-019f6359-9ee9-7493-9a7d-2baa315c040f.jsonl")
for target in (4566, 6085, 6253, 8259):
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i != target:
                continue
            print("=" * 30, "LINE", target)
            # unescape common sequences for readability
            s = line.replace("\\r\\n", "\n").replace("\\n", "\n").replace('\\"', '"')
            # print a large window
            if len(s) > 12000:
                # find interesting anchors
                for key in ("us_per_op", "fn main", "file_picker", "profile-driver", "checksum"):
                    j = s.find(key)
                    if j >= 0:
                        print("--- around", key, "---")
                        print(s[max(0, j-200):j+1500])
            else:
                print(s[:12000])
