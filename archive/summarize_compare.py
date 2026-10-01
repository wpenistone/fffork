import json
from pathlib import Path

p = Path(r"C:/Users/kanal/Projects/fff-upstream/archive/compare-quiet-keepers.json")
d = json.loads(p.read_text(encoding="utf-8-sig"))
cells = d.get("cell_deltas", [])
imp = sum(1 for c in cells if c.get("wall_us_p95_rel", 0) < -0.02)
reg = sum(1 for c in cells if c.get("wall_us_p95_rel", 0) > 0.02)
flat = len(cells) - imp - reg
rels = [c.get("wall_us_p95_rel", 0) for c in cells]
print("cells=%d improved>2%%=%d regressed>2%%=%d flat=%d" % (len(cells), imp, reg, flat))
if rels:
    print("mean_rel=%.4f min=%.4f max=%.4f" % (sum(rels) / len(rels), min(rels), max(rels)))
print("verdict", d.get("verdict"), "reasons", d.get("reason_codes"))

srt = sorted(cells, key=lambda c: c.get("wall_us_p95_rel", 0))
print("TOP improvements:")
for c in srt[:8]:
    key = (c.get("cell_key") or "")[:90]
    print("  %+.3f  %+.0fus  %s" % (c.get("wall_us_p95_rel") or 0, c.get("wall_us_p95_abs") or 0, key))
print("TOP regressions:")
for c in srt[-5:]:
    key = (c.get("cell_key") or "")[:90]
    print("  %+.3f  %+.0fus  %s" % (c.get("wall_us_p95_rel") or 0, c.get("wall_us_p95_abs") or 0, key))

rows = srt[:20] + srt[-10:] if len(srt) > 30 else srt
seen = set()
uniq = []
for c in rows:
    k = c.get("cell_key")
    if k in seen:
        continue
    seen.add(k)
    uniq.append(c)

W = 680
row_h = 22
pad_top = 50
pad_bottom = 40
H = pad_top + row_h * len(uniq) + pad_bottom
label_w = 280
bar_x = label_w + 20
bar_w = 280
zero_x = bar_x + bar_w / 2
scale = bar_w / 2 / 0.5

parts = []
parts.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="-apple-system, PingFang SC, Microsoft YaHei, sans-serif">' % (W, H))
parts.append('<text x="%d" y="28" text-anchor="middle" font-size="14" font-weight="500" fill="#222">Keepers vs upstream - wall p95 relative delta</text>' % (W // 2))
parts.append('<text x="%d" y="44" text-anchor="middle" font-size="11" fill="#666">negative = faster (keepers wins) | verdict: %s | host noise</text>' % (W // 2, d.get("verdict")))
parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#ccc" stroke-width="1"/>' % (zero_x, pad_top - 5, zero_x, H - pad_bottom + 5))

for i, c in enumerate(uniq):
    y = pad_top + i * row_h
    rel = float(c.get("wall_us_p95_rel") or 0)
    key = c.get("cell_key") or ""
    lab = key.split(" ")[-1] if " " in key else key
    lab = lab.replace("profile.", "").replace("search.", "")
    if len(lab) > 42:
        lab = lab[:39] + "..."
    lab = lab.replace("&", "&amp;").replace("<", "&lt;")
    bw = min(abs(rel) * scale, bar_w / 2)
    if rel < 0:
        bx = zero_x - bw
        color = "#35d19f"
    else:
        bx = zero_x
        color = "#f97373"
    parts.append('<text x="%d" y="%d" text-anchor="end" font-size="10" fill="#555">%s</text>' % (label_w, y + 14, lab))
    parts.append('<rect x="%.1f" y="%d" width="%.1f" height="14" rx="2" fill="%s"/>' % (bx, y + 4, bw, color))
    if rel >= 0:
        parts.append('<text x="%.1f" y="%d" text-anchor="start" font-size="10" fill="#333">%+.1f%%</text>' % (zero_x + bw + 6, y + 14, rel * 100))
    else:
        parts.append('<text x="%.1f" y="%d" text-anchor="end" font-size="10" fill="#333">%+.1f%%</text>' % (zero_x - bw - 6, y + 14, rel * 100))

parts.append('<text x="40" y="%d" font-size="11" fill="#666">green = keepers faster | red = slower | source: archive/compare-keepers-vs-upstream.json</text>' % (H - 14))
parts.append("</svg>")

out = Path(r"C:/Users/kanal/Projects/fff-upstream/archive/charts")
out.mkdir(parents=True, exist_ok=True)
svg_path = out / "keepers-vs-upstream-wall-delta.svg"
svg_path.write_text("\n".join(parts), encoding="utf-8")
print("wrote", svg_path)
