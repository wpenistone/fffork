import json
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

rows = [json.loads(l) for l in Path(r"C:/Users/kanal/Projects/fff-upstream/archive/paired-sensors.jsonl").read_text().splitlines() if l.strip()]

def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

# aggregate medians per (query, threads, subject)
G = defaultdict(lambda: defaultdict(list))
checks = defaultdict(set)
for r in rows:
    k = (r["query"], r["threads"])
    s = r["subject"]
    for f in ("us_per_op", "cpu_us_per_op", "cycles_per_op", "read_b_op", "faults_per_100k", "peak_rss_mb"):
        G[k][f"{s}.{f}"].append(r[f])
    checks[k + (s,)].add(r["checksum"])

def g(k, s, f):
    return med(G[k][f"{s}.{f}"])

keys = sorted(G.keys(), key=lambda k: (k[0], k[1]))

print(f"{'workload':<18} {'metric':<14} {'upstream':>12} {'keepers':>12} {'delta':>10}  ck")
metrics = [
    ("us_per_op", "wall us", 1),
    ("cpu_us_per_op", "cpu us", 1),
    ("cycles_per_op", "cycles", 1),
    ("read_b_op", "read B/op", 1),
    ("faults_per_100k", "faults/100k", 1),
    ("peak_rss_mb", "peak RSS MB", 1),
]
summary = []
for k in keys:
    q, t = k
    ck = "ok" if checks[k + ("upstream",)] == checks[k + ("keepers",)] else "DIFF"
    for f, label, _ in metrics:
        u, c = g(k, "upstream", f), g(k, "keepers", f)
        delta = 100 * (c / u - 1) if u else 0
        print(f"{q+' t='+str(t):<18} {label:<14} {u:12.2f} {c:12.2f} {delta:+9.1f}%  {ck}")
        summary.append((f"{q} t={t}", label, u, c, delta, f, ck))
    print()

# ===== multi-panel PNG like July atlas =====
W, H = 1100, 920
img = Image.new("RGB", (W, H), "#0b1220")
d = ImageDraw.Draw(img)

def font(n):
    for p in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, n)
        except Exception:
            pass
    return ImageFont.load_default()

ft, fs, fb, ftiny = font(22), font(14), font(11), font(10)

d.text((36, 22), "July-style paired sensors — sumo_fork 227k worktree", fill="#e8eef8", font=ft)
d.text((36, 52), "fff-profile-driver · 3 counterbalanced blocks · median · lower is better (except none here)", fill="#9aa9bd", font=fs)
d.text((36, 74), "pink = upstream v0.11.0    green = keepers    orange = keepers worse", fill="#9aa9bd", font=fb)

panels = [
    ("Wall us/op", "us_per_op", 100),
    ("CPU us/op", "cpu_us_per_op", 180),
    ("Cycles/op (M)", "cycles_per_op", 260),
    ("Peak RSS MB", "peak_rss_mb", 340),
    ("Read MB/op", "read_b_op", 420),
    ("Faults / 100k (M)", "faults_per_100k", 500),
]

# For each panel, 4 workloads (fp1, fp4, sc1, sc4)
workloads = keys  # list of (query, threads)
x0 = 200
bar_w_max = 420
row_h = 14
gap = 6

for pi, (title, field, y0) in enumerate(panels):
    d.text((36, y0 - 8), title, fill="#e8eef8", font=fs)
    vals = []
    for k in workloads:
        vals.append(g(k, "upstream", field))
        vals.append(g(k, "keepers", field))
    vmax = max(vals) if vals else 1
    scale = bar_w_max / vmax if vmax else 1
    yy = y0 + 8
    for k in workloads:
        q, t = k
        u, c = g(k, "upstream", field), g(k, "keepers", field)
        label = f"{q.replace('file_picker_','fp').replace('score','sc')} {t}T"
        d.text((36, yy + 1), label, fill="#c5d0e0", font=ftiny)
        d.rectangle((x0, yy, x0 + int(u * scale), yy + row_h), fill="#f97373")
        d.rectangle((x0, yy + row_h + 1, x0 + int(c * scale), yy + 2 * row_h + 1), fill="#35d19f" if c <= u else "#f59e0b")
        # delta label
        pct = 100 * (c / u - 1) if u else 0
        col = "#35d19f" if pct < 0 else "#f59e0b"
        d.text((x0 + max(int(u * scale), int(c * scale)) + 8, yy + row_h - 2), f"{pct:+.0f}%", fill=col, font=ftiny)
        yy += 2 * row_h + gap + 4

# footer takeaways
y = 620
d.text((36, y), "Takeaways (this corpus, warm search):", fill="#e8eef8", font=fs)
y += 26
lines = [
    "RSS: keepers ~78MB vs upstream ~160MB on every cell — half the peak working set.",
    "4T file_picker: keepers wall and CPU/cycles better (batching + tiny-query + frecency).",
    "1T file_picker wall: keepers sometimes slower; CPU/cycles often still lower.",
    "score: CPU/cycles lower on keepers; wall mixed; checksum DIFF — ranking not identical.",
    "July atlas 25-90x wins were full 14-patch+frizbee vs OLD upstream. Upstream absorbed most of that.",
    "Remaining keepers headroom is 10-30% CPU at 4T and ~2x RSS, not 10x wall.",
]
for ln in lines:
    d.text((48, y), ln, fill="#c5d0e0", font=fb)
    y += 22

d.text((36, H - 40), "Why not July-magnitude: v0.11 already has SIMD memmem, SharedDb, walker, RSS cuts, frizbee 0.13.", fill="#9aa9bd", font=fb)
d.text((36, H - 20), "source: archive/paired-sensors.jsonl · drivers: archive/bench-binaries/fff-profile-driver-{upstream,keepers}.exe", fill="#667788", font=fb)

out = Path(r"C:/Users/kanal/Projects/fff-upstream/archive/charts/paired-sensors-atlas.png")
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)
print("wrote", out)
