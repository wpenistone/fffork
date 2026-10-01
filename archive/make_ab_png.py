from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

W, H = 1100, 720
img = Image.new("RGB", (W, H), "#0b1220")
d = ImageDraw.Draw(img)

def font(size):
    for name in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

f_title = font(28)
f_sub = font(16)
f_body = font(15)
f_small = font(13)
f_bar = font(14)

# Title
d.text((40, 28), "FFF keepers vs upstream — quiet-host A/B", fill="#e8eef8", font=f_title)
d.text((40, 68), "Lower score is better (1.0 = reference). Same baseline, gates passed, integrity=clean.", fill="#9aa9bd", font=f_sub)

# Score bars
variants = [
    ("V1  all Pareto (filename_cow sort + flush 512)", 1.42, "#f97373", "6 cap fails · dir +703%"),
    ("V2  file_name sort + byte path + flush 512", 0.92, "#35d19f", "WINNER · 0.98 on confirm · 3 fails"),
    ("V3  + Path::components distance", 1.28, "#f59e0b", "dir cell fixed · cold worse"),
    ("V4  + walker flush 4096", 1.95, "#f97373", "worst · call cells explode"),
]

x0, y0 = 420, 130
bar_h = 48
gap = 22
max_score = 2.2
bar_max_w = 520

d.text((40, y0 - 28), "Aggregate check score (lower is better)", fill="#c5d0e0", font=f_body)

for i, (label, score, color, note) in enumerate(variants):
    y = y0 + i * (bar_h + gap)
    d.text((40, y + 8), label.split("  ", 1)[0], fill="#e8eef8", font=f_bar)
    d.text((40, y + 28), label.split("  ", 1)[1][:48], fill="#9aa9bd", font=f_small)
    bw = int(bar_max_w * (score / max_score))
    d.rounded_rectangle((x0, y, x0 + bw, y + bar_h), radius=8, fill=color)
    # score label
    d.text((x0 + bw + 12, y + 12), f"{score:.2f}", fill="#e8eef8", font=f_bar)
    d.text((x0 + bw + 70, y + 14), note, fill="#9aa9bd", font=f_small)

# Baseline reference line
ref_x = x0 + int(bar_max_w * (1.0 / max_score))
d.line((ref_x, y0 - 8, ref_x, y0 + 4 * (bar_h + gap) - 10), fill="#667788", width=2)
d.text((ref_x - 28, y0 + 4 * (bar_h + gap) - 4), "1.0 ref", fill="#9aa9bd", font=f_small)

# Key cell deltas from V2 confirmation
y1 = y0 + 4 * (bar_h + gap) + 36
d.line((40, y1 - 18, W - 40, y1 - 18), fill="#26344a", width=1)
d.text((40, y1), "V2 confirmation — wall p95 vs baseline (green = faster)", fill="#c5d0e0", font=f_body)

cells = [
    ("cold file_picker t=1", -0.40, "#35d19f"),
    ("warm pagination 'a' next", -0.36, "#35d19f"),
    ("cold score t=4", -0.25, "#35d19f"),
    ("cold mod t=1", -0.13, "#35d19f"),
    ("grep struct FffAdapter", -0.07, "#35d19f"),
    ("call file FilePicker t=4", +1.27, "#f97373"),
    ("call file mod t=4", +1.81, "#f97373"),
]

zero_x = 620
scale = 200  # px per 100%
row_h = 32
y1 += 36
for i, (name, rel, color) in enumerate(cells):
    y = y1 + i * row_h
    d.text((40, y + 6), name, fill="#e8eef8", font=f_small)
    bw = min(abs(rel) * scale, 380)
    if rel < 0:
        bx = zero_x - bw
    else:
        bx = zero_x
    d.rounded_rectangle((bx, y + 2, bx + bw, y + 22), radius=4, fill=color)
    label = f"{rel*100:+.0f}%"
    if rel < 0:
        d.text((zero_x - bw - 52, y + 4), label, fill="#35d19f", font=f_small)
    else:
        d.text((zero_x + bw + 10, y + 4), label, fill="#f97373", font=f_small)

d.line((zero_x, y1 - 6, zero_x, y1 + len(cells) * row_h - 6), fill="#667788", width=2)
d.text((zero_x - 12, y1 + len(cells) * row_h + 4), "0", fill="#9aa9bd", font=f_small)

d.text((40, H - 36), "Binaries: archive/bench-binaries/  ·  reports: archive/check-v*.json  ·  commit 1664b37", fill="#667788", font=f_small)
d.text((40, H - 18), "filename_cow in sort was the wall regression; byte distance penalty kept; flush 512 kept.", fill="#667788", font=f_small)

out = Path(r"C:/Users/kanal/Projects/fff-upstream/archive/charts/ab-keepers-vs-upstream.png")
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out, "PNG")
print("wrote", out)
