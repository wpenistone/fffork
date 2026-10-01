import json
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

rows = []
for line in Path(r"C:/Users/kanal/Projects/fff-upstream/archive/paired-sumo-process-runs.jsonl").read_text().splitlines():
    if line.strip():
        rows.append(json.loads(line))

def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n//2] if n % 2 else (xs[n//2-1]+xs[n//2])/2

groups = defaultdict(lambda: defaultdict(list))
checks = defaultdict(set)
for r in rows:
    groups[(r["query"], r["threads"])][r["subject"]].append(r["us_per_op"])
    checks[(r["query"], r["threads"], r["subject"])].add(r["checksum"])

print(f"{'workload':<22} {'upstream':>10} {'keepers':>10} {'delta%':>8}  checksums")
stats = []
for key in sorted(groups):
    up = med(groups[key]["upstream"])
    kp = med(groups[key]["keepers"])
    delta = 100 * (kp / up - 1)
    q, t = key
    ck = "same" if len(checks[(q,t,"upstream")]|checks[(q,t,"keepers")]) == 1 else "DIFF"
    # more precise: compare median checksum sets
    ck = "same" if checks[(q,t,"upstream")] == checks[(q,t,"keepers")] else "DIFF"
    print(f"{q+' t='+str(t):<22} {up:10.1f} {kp:10.1f} {delta:+8.1f}%  {ck}")
    stats.append((f"{q} t={t}", up, kp, delta, ck))

# Chart
W, H = 1000, 560
img = Image.new("RGB", (W, H), "#0b1220")
d = ImageDraw.Draw(img)

def font(n):
    for p in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, n)
        except Exception:
            pass
    return ImageFont.load_default()

ft, fs, fb = font(24), font(15), font(12)
d.text((40, 24), "July-style paired process runs — sumo_fork 227k worktree", fill="#e8eef8", font=ft)
d.text((40, 58), "fff-profile-driver · warm index · 3 counterbalanced blocks · median us_per_op", fill="#9aa9bd", font=fs)
d.text((40, 84), "pink = upstream v0.11.0   green = keepers (Pareto + 4T fixes)", fill="#9aa9bd", font=fb)

# bars: for each workload, two bars
y0 = 120
bar_h = 28
gap = 18
label_w = 160
bar_x = 220
max_us = max(max(s[1], s[2]) for s in stats)
scale = 520 / max_us

for i, (name, up, kp, delta, ck) in enumerate(stats):
    y = y0 + i * (bar_h * 2 + gap + 10)
    d.text((40, y + 22), name, fill="#e8eef8", font=fs)
    # upstream
    d.rounded_rectangle((bar_x, y, bar_x + int(up*scale), y + bar_h), radius=5, fill="#f97373")
    d.text((bar_x + int(up*scale) + 10, y + 6), f"{up/1000:.1f} ms", fill="#e8eef8", font=fb)
    d.text((bar_x + int(up*scale) + 80, y + 6), "upstream", fill="#9aa9bd", font=fb)
    # keepers
    y2 = y + bar_h + 4
    col = "#35d19f" if delta < 0 else "#f59e0b"
    d.rounded_rectangle((bar_x, y2, bar_x + int(kp*scale), y2 + bar_h), radius=5, fill=col)
    d.text((bar_x + int(kp*scale) + 10, y2 + 6), f"{kp/1000:.1f} ms", fill="#e8eef8", font=fb)
    d.text((bar_x + int(kp*scale) + 80, y2 + 6), f"keepers {delta:+.1f}%  [{ck}]", fill=col, font=fb)

d.text((40, H - 48), "file_picker t=4: keepers ~6% faster · score t=4: ~3% faster · file_picker t=1: ~12% slower (noisy)", fill="#c5d0e0", font=fb)
d.text((40, H - 28), "score checksums differ (45186800 vs 45193600) — ranking/score path not bit-identical; investigate before claiming quality parity.", fill="#f59e0b", font=fb)
d.text((40, H - 10), "source: archive/paired-sumo-process-runs.jsonl · driver: fff-profile-driver.exe", fill="#667788", font=fb)

out = Path(r"C:/Users/kanal/Projects/fff-upstream/archive/charts/paired-upstream-vs-keepers.png")
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)
print("wrote", out)
