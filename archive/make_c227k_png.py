from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

W, H = 920, 520
img = Image.new("RGB", (W, H), "#0b1220")
d = ImageDraw.Draw(img)

def font(n):
    for p in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, n)
        except Exception:
            pass
    return ImageFont.load_default()

ft, fs, fb = font(22), font(14), font(12)

d.text((36, 22), "Paired process runs — sumo_fork 227k worktree (on C:)", fill="#e8eef8", font=ft)
d.text((36, 52), "fff-profile-driver · 3 counterbalanced blocks · median us_per_op · checksums identical", fill="#9aa9bd", font=fs)
d.text((36, 74), "pink = upstream v0.11.0    green = keepers (better)", fill="#9aa9bd", font=fb)

rows = [
    ("file_picker 1T", 114177, 112710),
    ("file_picker 4T", 33004, 33079),
    ("score 1T", 116815, 111259),
    ("score 4T", 42519, 39455),
]
maxv = max(max(u, k) for _, u, k in rows)
y = 110
for name, up, kp in rows:
    d.text((36, y + 22), name, fill="#e8eef8", font=fs)
    bw = int(520 * up / maxv)
    d.rounded_rectangle((220, y, 220 + bw, y + 28), radius=6, fill="#f97373")
    d.text((230 + bw, y + 18), f"{up/1000:.1f} ms", fill="#e8eef8", font=fb)
    y2 = y + 32
    delta = 100 * (kp / up - 1)
    col = "#35d19f" if delta < 0 else "#f59e0b"
    bw2 = int(520 * kp / maxv)
    d.rounded_rectangle((220, y2, 220 + bw2, y2 + 28), radius=6, fill=col)
    d.text((230 + bw2, y2 + 18), f"{kp/1000:.1f} ms  {delta:+.1f}%", fill=col, font=fb)
    y += 88

d.text((36, H - 48), "score t=4: keepers 7% faster · score t=1: 5% faster · file_picker at parity", fill="#c5d0e0", font=fb)
d.text((36, H - 28), "checksums match on every cell — same result sets as upstream.", fill="#35d19f", font=fb)
d.text((36, H - 10), "source: archive/paired-c227k.jsonl · drivers: archive/bench-binaries/fff-profile-driver-*.exe", fill="#667788", font=fb)

out = Path(r"C:/Users/kanal/Projects/fff-upstream/archive/charts/paired-c227k.png")
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)
print("wrote", out)
