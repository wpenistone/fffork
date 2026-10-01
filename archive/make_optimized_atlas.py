import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"C:/Users/kanal/Projects/fff-upstream/archive")
JSONL = ROOT / "paired-sensors-c227k.jsonl"
HTML_OUT = ROOT / "optimized-vs-upstream-atlas.html"
PNG_OUT = ROOT / "charts" / "optimized-vs-upstream-atlas.png"
MD_OUT = ROOT / "optimized-vs-upstream-summary.md"

BG = "#0b1220"
BG2 = "#181818"
CARD = "#121a28"
TEXT = "#e8eef8"
MUTED = "#9aa9bd"
DIM = "#667788"
UP = "#f97373"
UP2 = "#f08fc0"
OPT = "#35d19f"
OPT2 = "#60a5fa"
WARN = "#f59e0b"

WORKLOADS = [
    ("file_picker_", 1, "file_picker_ 1t"),
    ("file_picker_", 4, "file_picker_ 4t"),
    ("score", 1, "score 1t"),
    ("score", 4, "score 4t"),
]
BLOCKS = list(range(1, 7))


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def load_rows():
    if not JSONL.exists():
        raise SystemExit(f"missing {JSONL}")
    rows = []
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fmt(v, kind="num"):
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "mb":
        return f"{v:.1f}"
    if kind == "us":
        if v >= 10000:
            return f"{v:,.0f}"
        return f"{v:,.1f}"
    if kind == "ratio":
        return f"{v:.2f}x"
    return f"{v:,.2f}"


def ratio_txt(before, after, lower_is_better=True):
    if after == 0 or before == 0:
        return "n/a"
    if lower_is_better:
        r = before / after
    else:
        r = after / before
    return f"{r:.2f}x"


def build_aggregates(rows):
    G = defaultdict(lambda: defaultdict(list))
    ck_by_block = defaultdict(lambda: defaultdict(set))
    subjects = set()
    for r in rows:
        subjects.add(r["subject"])
        k = (r["query"], int(r["threads"]))
        s = r["subject"]
        for f in (
            "us_per_op",
            "cpu_us_per_op",
            "cycles_per_op",
            "read_b_op",
            "faults_per_100k",
            "peak_rss_mb",
        ):
            G[k][f"{s}.{f}"].append(float(r[f]))
        G[k][f"{s}.qps"].append(1e6 / float(r["us_per_op"]))
        ck_by_block[k][int(r["block"])].add((s, int(r["checksum"])))
    return G, ck_by_block, subjects


def checksum_status(ck_by_block):
    problems = []
    ok = 0
    for k in WORKLOADS:
        key = (k[0], k[1])
        for b in BLOCKS:
            pairs = ck_by_block.get(key, {}).get(b, set())
            ups = {c for s, c in pairs if s == "upstream"}
            opts = {c for s, c in pairs if s == "optimized"}
            if not ups or not opts:
                problems.append(f"{k[2]} block {b}: missing subject (up={sorted(ups)} opt={sorted(opts)})")
                continue
            if ups != opts:
                problems.append(f"{k[2]} block {b}: DIFF up={sorted(ups)} opt={sorted(opts)}")
            else:
                ok += 1
    return ok, problems


def med_of(G, key, subject, field):
    return med(G[key][f"{subject}.{field}"])


def traces(G, subject, field="us_per_op"):
    out = {}
    for q, t, label in WORKLOADS:
        key = (q, t)
        out[label] = G[key][f"{subject}.{field}"]
    return out


# ---------------- HTML (self-contained inline SVG) ----------------

def svg_bars(title, unit, rows_data, qps=False):
    """rows_data: list of (label, before, after)"""
    w, h = 640, 250
    pad_l, pad_r, pad_t, pad_b = 130, 110, 36, 28
    plot_w = w - pad_l - pad_r
    vmax = max(max(b, a) for _, b, a in rows_data) or 1.0
    # headroom
    vmax *= 1.12
    parts = [
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="panel-svg">'
    ]
    parts.append(
        f'<text x="16" y="22" fill="{TEXT}" font-size="14" font-weight="600">{title}</text>'
    )
    parts.append(
        f'<text x="{w-16}" y="22" fill="{DIM}" font-size="11" text-anchor="end">{unit}</text>'
    )
    row_h = 22
    gap = 18
    y = pad_t
    for label, before, after in rows_data:
        parts.append(
            f'<text x="{pad_l-10}" y="{y+16}" fill="{MUTED}" font-size="11" text-anchor="end">{label}</text>'
        )
        bw = int(plot_w * (before / vmax)) if vmax else 0
        aw = int(plot_w * (after / vmax)) if vmax else 0
        # before bar
        parts.append(
            f'<rect x="{pad_l}" y="{y}" width="{bw}" height="9" fill="{UP}" rx="2"/>'
        )
        # after bar
        parts.append(
            f'<rect x="{pad_l}" y="{y+11}" width="{aw}" height="9" fill="{OPT}" rx="2"/>'
        )
        # numeric labels
        btxt = f"{before:,.0f}" if before >= 100 else f"{before:,.1f}"
        atxt = f"{after:,.0f}" if after >= 100 else f"{after:,.1f}"
        if qps:
            btxt = f"{before:,.1f}"
            atxt = f"{after:,.1f}"
        parts.append(
            f'<text x="{pad_l+bw+6}" y="{y+8}" fill="{UP2}" font-size="10">{btxt}</text>'
        )
        parts.append(
            f'<text x="{pad_l+aw+6}" y="{y+19}" fill="{OPT}" font-size="10">{atxt}</text>'
        )
        if qps:
            r = ratio_txt(before, after, lower_is_better=False)
        else:
            r = ratio_txt(before, after, lower_is_better=True)
        parts.append(
            f'<text x="{w-16}" y="{y+15}" fill="{OPT if (not qps and after<before) or (qps and after>before) else WARN}" font-size="12" text-anchor="end" font-family="ui-monospace,Consolas,monospace">{r}</text>'
        )
        y += row_h + gap
    parts.append("</svg>")
    return "\n".join(parts)


def svg_traces(label, series, title):
    """series: dict subject -> list of 6 us values (or fewer)."""
    w, h = 320, 160
    pad_l, pad_r, pad_t, pad_b = 42, 12, 28, 24
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    all_vals = []
    for vals in series.values():
        all_vals.extend([v for v in vals if v > 0])
    if not all_vals:
        all_vals = [1.0]
    vmin = min(all_vals)
    vmax = max(all_vals)
    # log scale
    lo = math.log10(vmin * 0.85)
    hi = math.log10(vmax * 1.15)
    if hi - lo < 0.1:
        hi = lo + 0.2

    def ypos(v):
        t = (math.log10(max(v, 1e-6)) - lo) / (hi - lo)
        return pad_t + plot_h * (1 - t)

    def xpos(i, n):
        if n <= 1:
            return pad_l + plot_w / 2
        return pad_l + plot_w * i / (n - 1)

    parts = [
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="panel-svg trace">'
    ]
    parts.append(
        f'<text x="10" y="16" fill="{TEXT}" font-size="11" font-weight="600">{title}</text>'
    )
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#0e1624" stroke="#243044" stroke-width="1"/>'
    )
    # grid
    for frac in (0, 0.5, 1.0):
        yy = pad_t + plot_h * (1 - frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy}" x2="{pad_l+plot_w}" y2="{yy}" stroke="#243044" stroke-width="0.6"/>'
        )
    colors = {"upstream": UP, "optimized": OPT2}
    for subj, vals in series.items():
        col = colors.get(subj, OPT)
        pts = []
        for i, v in enumerate(vals[:6]):
            x = xpos(i, 6)
            y = ypos(v)
            pts.append((x, y))
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(
            f'<polyline fill="none" stroke="{col}" stroke-width="1.8" points="{d}"/>'
        )
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="{col}"/>')
    # legend
    parts.append(
        f'<rect x="{pad_l}" y="{h-14}" width="8" height="8" fill="{UP}"/>'
        f'<text x="{pad_l+12}" y="{h-7}" fill="{MUTED}" font-size="9">upstream</text>'
        f'<rect x="{pad_l+70}" y="{h-14}" width="8" height="8" fill="{OPT2}"/>'
        f'<text x="{pad_l+82}" y="{h-7}" fill="{MUTED}" font-size="9">optimized</text>'
        f'<text x="{w-10}" y="{h-7}" fill="{DIM}" font-size="9" text-anchor="end">log wall us</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_html(G, ck_ok, ck_problems, n_rows, subjects):
    # bar panel data
    wall_rows, cpu_rows, cyc_rows, qps_rows = [], [], [], []
    for q, t, label in WORKLOADS:
        k = (q, t)
        ub, ua = med_of(G, k, "upstream", "us_per_op"), med_of(G, k, "optimized", "us_per_op")
        cb, ca = med_of(G, k, "upstream", "cpu_us_per_op"), med_of(G, k, "optimized", "cpu_us_per_op")
        yb, ya = med_of(G, k, "upstream", "cycles_per_op"), med_of(G, k, "optimized", "cycles_per_op")
        qb, qa = med_of(G, k, "upstream", "qps"), med_of(G, k, "optimized", "qps")
        wall_rows.append((label, ub, ua))
        cpu_rows.append((label, cb, ca))
        cyc_rows.append((label, yb, ya))
        qps_rows.append((label, qb, qa))

    wall_svg = svg_bars("Wall latency / op", "microseconds (lower better)", wall_rows)
    cpu_svg = svg_bars("CPU time / op", "microseconds (lower better)", cpu_rows)
    cyc_svg = svg_bars("CPU cycles / op", "cycles (lower better)", cyc_rows)
    qps_svg = svg_bars(
        "Sustained single-client QPS",
        "queries/sec = 1e6 / us_per_op (higher better)",
        qps_rows,
        qps=True,
    )

    # traces
    trace_svgs = []
    for q, t, label in WORKLOADS:
        k = (q, t)
        series = {
            "upstream": list(G[k]["upstream.us_per_op"]),
            "optimized": list(G[k]["optimized.us_per_op"]),
        }
        # pad/truncate to 6
        for s in series:
            series[s] = (series[s] + [series[s][-1]] * 6)[:6] if series[s] else [1.0] * 6
        trace_svgs.append(svg_traces(label, series, label))

    # resource tradeoffs table rows
    def trade_row(label, field, unit, kind="num"):
        cells = []
        ratios = []
        for q, t, lab in WORKLOADS:
            k = (q, t)
            b = med_of(G, k, "upstream", field)
            a = med_of(G, k, "optimized", field)
            r = ratio_txt(b, a, lower_is_better=True)
            ratios.append((b, a, r))
            cells.append((lab, b, a, r))
        return cells

    read_cells = trade_row("Read B/op", "read_b_op", "B")
    fault_cells = trade_row("Faults/100k", "faults_per_100k", "")
    rss_cells = trade_row("Peak RSS", "peak_rss_mb", "MB")

    def tr_html(cells, unit):
        rows = []
        for lab, b, a, r in cells:
            rows.append(
                f"<tr><td>{lab}</td>"
                f'<td class="num up">{fmt(b,"us")}</td>'
                f'<td class="num opt">{fmt(a,"us")}</td>'
                f'<td class="num">{r}</td>'
                f"<td>{unit}</td></tr>"
            )
        return "\n".join(rows)

    ck_class = "ok" if ck_ok == 24 and not ck_problems else "bad"
    ck_text = (
        f"all 24 (query x threads x block) pairs match"
        if ck_class == "ok"
        else f"{len(ck_problems)} problems"
    )

    problems_html = ""
    if ck_problems:
        li = "".join(f"<li>{p}</li>" for p in ck_problems[:30])
        problems_html = f'<div class="warnbox"><strong>Checksum issues</strong><ul>{li}</ul></div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>optimized vs upstream — sumo 227k atlas</title>
<style>
  :root {{
    --bg: {BG};
    --bg2: {CARD};
    --text: {TEXT};
    --muted: {MUTED};
    --dim: {DIM};
    --up: {UP};
    --up2: {UP2};
    --opt: {OPT};
    --opt2: {OPT2};
    --warn: {WARN};
    --line: #243044;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.5 "Segoe UI", system-ui, sans-serif;
    font-variant-numeric: tabular-nums;
  }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 36px 28px 64px; }}
  h1 {{ font-size: 28px; margin: 0 0 6px; letter-spacing: -0.02em; }}
  .sub {{ color: var(--muted); font-size: 13px; margin: 0 0 10px; }}
  .legend {{ display: flex; gap: 16px; color: var(--dim); font-size: 12px; margin: 12px 0 28px; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
  h2 {{ font-size: 18px; margin: 36px 0 12px; color: var(--text); }}
  h3 {{ font-size: 13px; margin: 0 0 8px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid4 {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }}
  .card {{
    background: var(--bg2);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 16px 16px 14px;
  }}
  .card .tag {{
    display: inline-block;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--opt2);
    background: rgba(96,165,250,0.12);
    border: 1px solid rgba(96,165,250,0.35);
    border-radius: 999px;
    padding: 2px 8px;
    margin-bottom: 8px;
  }}
  .card h4 {{ margin: 0 0 8px; font-size: 15px; }}
  .card p {{ margin: 0; color: var(--muted); font-size: 12.5px; }}
  .card .commits {{ margin-top: 10px; color: var(--dim); font-size: 11px; font-family: ui-monospace, Consolas, monospace; }}
  .panel {{
    background: var(--bg2);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 10px 8px 6px;
  }}
  .panel-svg {{ width: 100%; height: auto; display: block; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--bg2);
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow: hidden;
    font-size: 12.5px;
  }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--dim); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
  tr:last-child td {{ border-bottom: none; }}
  td.num, th.num {{ text-align: right; font-family: ui-monospace, Consolas, monospace; }}
  td.up {{ color: var(--up); }}
  td.opt {{ color: var(--opt); }}
  .pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
  }}
  .pill.ok {{ background: rgba(53,209,159,0.15); color: var(--opt); border: 1px solid rgba(53,209,159,0.4); }}
  .pill.bad {{ background: rgba(249,115,115,0.15); color: var(--up); border: 1px solid rgba(249,115,115,0.4); }}
  .warnbox {{
    margin: 12px 0;
    padding: 12px 14px;
    border-radius: 8px;
    background: rgba(249,115,115,0.08);
    border: 1px solid rgba(249,115,115,0.35);
    color: var(--up2);
    font-size: 12px;
  }}
  .meta {{
    margin-top: 40px;
    color: var(--dim);
    font-size: 11px;
    border-top: 1px solid var(--line);
    padding-top: 14px;
  }}
  .sigtable td:nth-child(2) {{ font-family: ui-monospace, Consolas, monospace; }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>Bundle-level before &rarr; after</h1>
  <p class="sub">Six counterbalanced process runs per workload &middot; sumo 227k worktree &middot; warm index &middot; lower is better except QPS</p>
  <div class="legend">
    <span><span class="swatch" style="background:{UP}"></span>upstream (pink)</span>
    <span><span class="swatch" style="background:{OPT}"></span>optimized (green)</span>
    <span><span class="swatch" style="background:{OPT2}"></span>optimized traces (blue)</span>
    <span>rows in capture: {n_rows} &middot; subjects: {", ".join(sorted(subjects))}</span>
  </div>
</header>

<section>
  <h2>Performance panels</h2>
  <div class="grid2">
    <div class="panel">{wall_svg}</div>
    <div class="panel">{cpu_svg}</div>
    <div class="panel">{cyc_svg}</div>
    <div class="panel">{qps_svg}</div>
  </div>
</section>

<section>
  <h2>What the bundle changed</h2>
  <div class="cards">
    <div class="card">
      <span class="tag">mechanism 1</span>
      <h4>Cheaper matcher / scoring loop</h4>
      <p>Hot-path scoring dropped per-match allocation and repeated work: zero-alloc filename checks, byte walks for the distance penalty, max_path_len computed once per fuzzy pass, memset skipped before read, and the per-match scoring loop parallelized.</p>
      <div class="commits">e85b9c1 &middot; b85c731 &middot; 6e367ec &middot; fba573a &middot; c26c04f</div>
    </div>
    <div class="card">
      <span class="tag">mechanism 2</span>
      <h4>Less IO</h4>
      <p>Binary sniff no longer reads text files all the way to EOF, and the 8M-String re-sort was removed. Less bytes touched per op shows up directly as lower Read B/op and fewer faults on the same warm index.</p>
      <div class="commits">040e1bb</div>
    </div>
    <div class="card">
      <span class="tag">mechanism 3</span>
      <h4>Parallel work-estimate dispatch</h4>
      <p>Fuzzy match uses a work-estimate thread dispatch instead of collapsing 4-thread searches to 1, so 4T runs actually fan out. Combined with the parallel scoring loop this is where 4T CPU/wall separation comes from.</p>
      <div class="commits">4cd55af &middot; b854946 &middot; c26c04f</div>
    </div>
  </div>
</section>

<section>
  <h2>Independent-run traces</h2>
  <p class="sub">Wall us per counterbalanced block (log scale) &middot; one panel per workload &middot; 6 process runs per subject</p>
  <div class="grid4">
    {"".join(f'<div class="panel">{s}</div>' for s in trace_svgs)}
  </div>
</section>

<section>
  <h2>Resource tradeoffs</h2>
  <table>
    <thead>
      <tr><th>Metric</th><th>Workload</th><th class="num">upstream</th><th class="num">optimized</th><th class="num">ratio</th><th>unit</th></tr>
    </thead>
    <tbody>
      <tr><td rowspan="4"><strong>Read B/op</strong></td>{tr_html(read_cells[:1], "B")}</tr>
      {tr_html(read_cells[1:], "B")}
      <tr><td rowspan="4"><strong>Faults / 100k</strong></td>{tr_html(fault_cells[:1], "")}</tr>
      {tr_html(fault_cells[1:], "")}
      <tr><td rowspan="4"><strong>Peak RSS</strong></td>{tr_html(rss_cells[:1], "MB")}</tr>
      {tr_html(rss_cells[1:], "MB")}
    </tbody>
  </table>
</section>

<section>
  <h2>What the raw capture can&mdash;and cannot&mdash;prove</h2>
  <table class="sigtable">
    <thead>
      <tr><th>Signal</th><th>Status</th><th>What it supports</th></tr>
    </thead>
    <tbody>
      <tr><td>Wall us / op</td><td>measured</td><td>End-to-end latency per search op; QPS = 1e6 / us_per_op.</td></tr>
      <tr><td>CPU us / op</td><td>measured</td><td>Separates busy-time wins from scheduling / IO wait.</td></tr>
      <tr><td>CPU cycles / op</td><td>measured</td><td>Work done independent of frequency scaling.</td></tr>
      <tr><td>Peak RSS</td><td>measured</td><td>Working-set cost of each subject on the same corpus.</td></tr>
      <tr><td>Faults / 100k</td><td>measured</td><td>Major/minor fault pressure during the run.</td></tr>
      <tr><td>IO Read B / op</td><td>measured</td><td>Bytes read per op (binary-sniff / re-sort effects).</td></tr>
      <tr><td>Checksum</td><td>measured</td><td>Result parity gate between subjects per (query, threads, block).</td></tr>
      <tr><td>Context switches</td><td>unavailable</td><td>Not in the driver sensor set.</td></tr>
      <tr><td>Energy / power</td><td>unavailable</td><td>No RAPL / battery counters in this capture.</td></tr>
      <tr><td>SSD miss / device latency</td><td>unavailable</td><td>Warm index only; no device-level miss attribution.</td></tr>
    </tbody>
  </table>
  <p class="sub" style="margin-top:10px">Checksum gate: <span class="pill {ck_class}">{ck_text}</span> &middot; hard requirement: every (query, threads, block) upstream checksum equals optimized checksum.</p>
  {problems_html}
</section>

<div class="meta">
  source: archive/paired-sensors-c227k.jsonl &middot; runners: archive/run_paired_sensors_c227k.ps1 &middot;
  drivers: archive/bench-binaries/fff-profile-driver-{{upstream,optimized}}.exe &middot;
  corpus: C:/Users/kanal/fff-corpus/sumo-worktree (227440 files, no .git) &middot;
  branch: perf/keepers (subject label reported as optimized)
</div>

</div>
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding="utf-8")
    return {
        "wall": wall_rows,
        "cpu": cpu_rows,
        "cyc": cyc_rows,
        "qps": qps_rows,
        "read": read_cells,
        "fault": fault_cells,
        "rss": rss_cells,
        "ck_ok": ck_ok,
        "ck_problems": ck_problems,
    }


# ---------------- Pillow multi-panel PNG ----------------

def load_font(n):
    for p in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ):
        try:
            return ImageFont.truetype(p, n)
        except Exception:
            pass
    return ImageFont.load_default()


def render_png(G, info):
    W, H = 1440, 2100
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ft = load_font(28)
    fs = load_font(15)
    fb = load_font(12)
    fn = load_font(11)
    fx = load_font(10)

    d.text((40, 28), "Bundle-level before -> after", fill=TEXT, font=ft)
    d.text(
        (40, 66),
        "Six counterbalanced process runs per workload · sumo 227k worktree · warm index · lower is better except QPS",
        fill=MUTED,
        font=fs,
    )
    d.rectangle((40, 96, 56, 110), fill=UP)
    d.text((64, 97), "upstream", fill=MUTED, font=fb)
    d.rectangle((150, 96, 166, 110), fill=OPT)
    d.text((174, 97), "optimized", fill=MUTED, font=fb)

    # four bar panels 2x2
    panels = [
        ("Wall latency / op", "us (lower better)", info["wall"], False),
        ("CPU time / op", "us (lower better)", info["cpu"], False),
        ("CPU cycles / op", "cycles (lower better)", info["cyc"], False),
        ("Sustained single-client QPS", "qps = 1e6/us_per_op (higher better)", info["qps"], True),
    ]
    origins = [(40, 130), (740, 130), (40, 560), (740, 560)]
    for (title, unit, rows, qps), (x0, y0) in zip(panels, origins):
        d.rectangle((x0, y0, x0 + 660, y0 + 390), fill=CARD, outline="#243044")
        d.text((x0 + 18, y0 + 14), title, fill=TEXT, font=fs)
        d.text((x0 + 18, y0 + 38), unit, fill=DIM, font=fn)
        vmax = max(max(b, a) for _, b, a in rows) * 1.15 or 1
        bar_x = x0 + 160
        bar_max = 380
        yy = y0 + 70
        for label, before, after in rows:
            d.text((x0 + 18, yy + 8), label, fill=MUTED, font=fn)
            bw = int(bar_max * before / vmax)
            aw = int(bar_max * after / vmax)
            d.rectangle((bar_x, yy, bar_x + bw, yy + 16), fill=UP)
            d.rectangle((bar_x, yy + 20, bar_x + aw, yy + 36), fill=OPT)
            btxt = f"{before:,.0f}" if before >= 100 else f"{before:,.1f}"
            atxt = f"{after:,.0f}" if after >= 100 else f"{after:,.1f}"
            d.text((bar_x + bw + 8, yy + 2), btxt, fill=UP2, font=fx)
            d.text((bar_x + aw + 8, yy + 22), atxt, fill=OPT, font=fx)
            r = ratio_txt(before, after, lower_is_better=not qps)
            good = (after < before) if not qps else (after > before)
            d.text((x0 + 620, yy + 12), r, fill=(OPT if good else WARN), font=fn)
            yy += 72

    # mechanism cards
    y = 990
    d.text((40, y), "What the bundle changed", fill=TEXT, font=fs)
    cards = [
        (
            "Cheaper matcher / scoring loop",
            "Zero-alloc filename checks, byte walks for distance",
            "penalty, max_path_len once/pass, no memset-before-read,",
            "parallel per-match scoring.",
            "e85b9c1 b85c731 6e367ec fba573a c26c04f",
        ),
        (
            "Less IO",
            "Binary sniff stops at EOF early (no full text read),",
            "8M-String re-sort removed. Lower Read B/op and faults",
            "on the same warm index.",
            "040e1bb",
        ),
        (
            "Parallel work-estimate dispatch",
            "Fuzzy match work-estimate thread dispatch; 4T no longer",
            "collapsed to 1 thread. 4T wall/CPU separation comes",
            "from real fan-out + parallel scoring.",
            "4cd55af b854946 c26c04f",
        ),
    ]
    cx = 40
    for title, l1, l2, l3, commits in cards:
        d.rectangle((cx, y + 30, cx + 440, y + 200), fill=CARD, outline="#243044")
        d.text((cx + 16, y + 44), title, fill=OPT2, font=fs)
        d.text((cx + 16, y + 78), l1, fill=MUTED, font=fn)
        d.text((cx + 16, y + 98), l2, fill=MUTED, font=fn)
        d.text((cx + 16, y + 118), l3, fill=MUTED, font=fn)
        d.text((cx + 16, y + 160), commits, fill=DIM, font=fx)
        cx += 460

    # traces row
    y = 1220
    d.text((40, y), "Independent-run traces (log wall us per block)", fill=TEXT, font=fs)
    tx = 40
    for q, t, label in WORKLOADS:
        k = (q, t)
        ups = list(G[k]["upstream.us_per_op"])[:6]
        opts = list(G[k]["optimized.us_per_op"])[:6]
        d.rectangle((tx, y + 30, tx + 320, y + 220), fill=CARD, outline="#243044")
        d.text((tx + 12, y + 42), label, fill=TEXT, font=fn)
        vals = [v for v in ups + opts if v > 0] or [1]
        lo = math.log10(min(vals) * 0.85)
        hi = math.log10(max(vals) * 1.15)
        if hi - lo < 0.1:
            hi = lo + 0.2
        px0, py0, pw, ph = tx + 36, y + 70, 260, 120

        def ypos(v):
            frac = (math.log10(max(v, 1e-6)) - lo) / (hi - lo)
            return py0 + ph * (1 - frac)

        def xpos(i, n=6):
            return px0 + (pw * i / (n - 1) if n > 1 else pw / 2)

        d.rectangle((px0, py0, px0 + pw, py0 + ph), outline="#243044")
        for series, col in ((ups, UP), (opts, OPT2)):
            pts = [(xpos(i), ypos(v)) for i, v in enumerate(series)]
            if len(pts) >= 2:
                d.line(pts, fill=col, width=2)
            for x, yy2 in pts:
                d.ellipse((x - 3, yy2 - 3, x + 3, yy2 + 3), fill=col)
        d.text((tx + 12, y + 198), "pink=upstream  blue=optimized", fill=DIM, font=fx)
        tx += 340

    # resource table
    y = 1480
    d.text((40, y), "Resource tradeoffs (medians, before -> after)", fill=TEXT, font=fs)
    headers = ["metric", "workload", "upstream", "optimized", "ratio"]
    col_x = [40, 220, 480, 700, 940]
    yy = y + 36
    for h, x in zip(headers, col_x):
        d.text((x, yy), h, fill=DIM, font=fn)
    yy += 22
    for metric, cells in (
        ("Read B/op", info["read"]),
        ("Faults/100k", info["fault"]),
        ("Peak RSS", info["rss"]),
    ):
        for i, (lab, b, a, r) in enumerate(cells):
            if i == 0:
                d.text((col_x[0], yy), metric, fill=TEXT, font=fn)
            d.text((col_x[1], yy), lab, fill=MUTED, font=fn)
            d.text((col_x[2], yy), fmt(b, "us"), fill=UP, font=fn)
            d.text((col_x[3], yy), fmt(a, "us"), fill=OPT, font=fn)
            d.text((col_x[4], yy), r, fill=TEXT, font=fn)
            yy += 20
        yy += 6

    # signals
    y = 1760
    d.text((40, y), "What the raw capture can -- and cannot -- prove", fill=TEXT, font=fs)
    signals = [
        ("Wall / CPU / cycles", "measured"),
        ("RSS / faults / IO read", "measured"),
        ("Checksum parity", "measured"),
        ("Context switches", "unavailable"),
        ("Energy / power", "unavailable"),
        ("SSD misses / device latency", "unavailable"),
    ]
    yy = y + 36
    for name, st in signals:
        d.text((40, yy), name, fill=MUTED, font=fn)
        col = OPT if st == "measured" else WARN
        d.text((360, yy), st, fill=col, font=fn)
        yy += 20

    ck = "OK" if info["ck_ok"] == 24 and not info["ck_problems"] else "PROBLEMS"
    d.text(
        (40, H - 56),
        f"Checksum gate: {ck} ({info['ck_ok']} matching pairs) · subject label: optimized · n_rows={sum(len(G[(q,t)]['upstream.us_per_op'])+len(G[(q,t)]['optimized.us_per_op']) for q,t,_ in WORKLOADS)}",
        fill=MUTED,
        font=fn,
    )
    d.text(
        (40, H - 34),
        "source: archive/paired-sensors-c227k.jsonl · corpus sumo-worktree 227k · warm index · no .git walk",
        fill=DIM,
        font=fx,
    )

    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(PNG_OUT)
    return PNG_OUT


# ---------------- Markdown summary ----------------

def render_md(G, info, n_rows, subjects, ck_ok, ck_problems):
    lines = []
    lines.append("# optimized vs upstream - sumo 227k paired sensors")
    lines.append("")
    lines.append("Six counterbalanced process runs per workload. Warm index. Lower is better except QPS.")
    lines.append("")
    lines.append(f"- Capture: `archive/paired-sensors-c227k.jsonl` ({n_rows} rows)")
    lines.append(f"- Subjects: {', '.join(sorted(subjects))}")
    lines.append(f"- Checksum gate: **{ck_ok}/24 pairs match**" + ("" if not ck_problems else f" - {len(ck_problems)} problems"))
    if ck_problems:
        lines.append("")
        lines.append("## Checksum problems")
        for p in ck_problems:
            lines.append(f"- {p}")
    lines.append("")
    lines.append("## Medians")
    lines.append("")
    lines.append(
        "| workload | metric | upstream | optimized | ratio (up/opt) |"
    )
    lines.append("|---|---|---:|---:|---:|")

    metrics = [
        ("us_per_op", "wall us", "us", True),
        ("cpu_us_per_op", "cpu us", "us", True),
        ("cycles_per_op", "cycles", "int", True),
        ("read_b_op", "read B/op", "us", True),
        ("faults_per_100k", "faults/100k", "us", True),
        ("peak_rss_mb", "peak RSS MB", "mb", True),
        ("qps", "QPS", "num", False),
    ]
    for q, t, label in WORKLOADS:
        k = (q, t)
        for field, mlabel, kind, lower in metrics:
            b = med_of(G, k, "upstream", field)
            a = med_of(G, k, "optimized", field)
            r = ratio_txt(b, a, lower_is_better=lower)
            lines.append(
                f"| {label} | {mlabel} | {fmt(b, kind)} | {fmt(a, kind)} | {r} |"
            )
    lines.append("")
    lines.append("## QPS")
    lines.append("")
    lines.append("QPS = 1e6 / us_per_op (sustained single-client). Higher is better; ratio is optimized/upstream.")
    lines.append("")
    lines.append("| workload | upstream QPS | optimized QPS | ratio |")
    lines.append("|---|---:|---:|---:|")
    for q, t, label in WORKLOADS:
        k = (q, t)
        b = med_of(G, k, "upstream", "qps")
        a = med_of(G, k, "optimized", "qps")
        r = ratio_txt(b, a, lower_is_better=False)
        lines.append(f"| {label} | {b:.2f} | {a:.2f} | {r} |")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `archive/optimized-vs-upstream-atlas.html`")
    lines.append("- `archive/charts/optimized-vs-upstream-atlas.png`")
    lines.append("- `archive/optimized-vs-upstream-summary.md`")
    lines.append("")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MD_OUT


def main():
    rows = load_rows()
    G, ck_by_block, subjects = build_aggregates(rows)
    ck_ok, ck_problems = checksum_status(ck_by_block)

    # hard gate on subject labels
    if "keepers" in subjects:
        raise SystemExit("FAIL: subject label 'keepers' found; expected 'optimized'")

    n_rows = len(rows)
    print(f"rows={n_rows} subjects={sorted(subjects)} checksum_pairs_ok={ck_ok} problems={len(ck_problems)}")
    if n_rows != 48:
        print(f"WARN: expected 48 rows, got {n_rows}")

    info = render_html(G, ck_ok, ck_problems, n_rows, subjects)
    render_png(G, info)
    render_md(G, info, n_rows, subjects, ck_ok, ck_problems)
    print(f"wrote {HTML_OUT}")
    print(f"wrote {PNG_OUT}")
    print(f"wrote {MD_OUT}")

    # console median table
    print()
    print(f"{'workload':<18} {'metric':<12} {'upstream':>12} {'optimized':>12} {'ratio':>8}")
    for q, t, label in WORKLOADS:
        k = (q, t)
        for field, mlabel, kind, lower in (
            ("us_per_op", "wall us", "us", True),
            ("cpu_us_per_op", "cpu us", "us", True),
            ("cycles_per_op", "cycles", "int", True),
            ("read_b_op", "read B", "us", True),
            ("faults_per_100k", "faults", "us", True),
            ("peak_rss_mb", "RSS mb", "mb", True),
        ):
            b = med_of(G, k, "upstream", field)
            a = med_of(G, k, "optimized", field)
            r = ratio_txt(b, a, lower_is_better=lower)
            print(f"{label:<18} {mlabel:<12} {fmt(b, kind):>12} {fmt(a, kind):>12} {r:>8}")

    if ck_problems:
        print("CHECKSUM PROBLEMS:")
        for p in ck_problems:
            print(" ", p)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
