#!/usr/bin/env python3
"""FFF file-search A/B: perf/keepers vs upstream origin/main.

Measured p50 (us unless noted), paired single-process warm runs:
synthetic corpora 200/1k/5k/20k at 1T and 4T (median of 3 queries),
static-repo 477 files, mega-repo 170,818 files, typing curves, and
20 labeled ranking queries.

"before" = branch HEAD as found (50e8c75, did not compile for bindings).
"after"  = + build fixes, tombstone filter, honoured thread budget,
           bit-parallel gate.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UP, BEFORE, AFTER = "#7d8794", "#e08a1e", "#1f6feb"
LABEL = {"upstream": "upstream origin/main", "before": "branch (as found)",
         "after": "branch + fixes"}
SIZES = [200, 1000, 5000, 20000]
SWEEP = {
    ("upstream", 1): [62, 246, 1279, 4912],
    ("before", 1): [216, 796, 1514, 2277],
    ("after", 1): [202, 764, 1306, 2258],
    ("upstream", 4): [68, 287, 1075, 2536],
    ("before", 4): [212, 792, 1793, 2771],
    ("after", 4): [212, 766, 1519, 2338],
}
MEGA_US = {"score": (30592, 26061, 18259),
           "file_picker.rs": (16443, 7418, 3066),
           "test": (44734, 11802, 9073)}
SMALL_US = {"AGENTS.md": (19, 262, 95), "readme": (112, 572, 266),
            "score": (128, 600, 292)}
KEYS = ["s", "sc", "sco", "scor", "score", "file_p", "file_pi", "file_pic",
        "file_pick", "file_picker"]
TYPING_MS = {"upstream": [3.0, 20.6, 23.0, 24.0, 44.3, 29.7, 16.3, 13.1, 12.5, 13.9],
             "keepers": [10.8, 10.1, 10.3, 9.2, 19.4, 10.7, 5.1, 3.1, 3.7, 9.1]}
QUALITY = {"clean fff (1.8k)": {"upstream": (16, 0.838), "keepers": (18, 0.927)},
           "noisy tree (477+)": {"upstream": (13, 0.724), "keepers": (17, 0.875)}}


def sweep_panel(ax, threads):
    for (who, th), vals in SWEEP.items():
        if th != threads:
            continue
        c = {"upstream": UP, "before": BEFORE, "after": AFTER}[who]
        ax.plot(SIZES, vals, "--" if who == "before" else "-", color=c,
                lw=2.2, marker="o", ms=5, label=LABEL[who])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(SIZES); ax.set_xticklabels([f"{s:,}" for s in SIZES])
    ax.set_xlabel("files in corpus"); ax.set_ylabel("median p50 search wall (us)")
    ax.set_title(f"Corpus-size sweep, {threads}T")
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper left")


def ratio_panel(ax):
    for who, c in (("before", BEFORE), ("after", AFTER)):
        ratio = [u / k for u, k in zip(SWEEP[("upstream", 1)], SWEEP[(who, 1)])]
        ax.plot(SIZES, ratio, "--" if who == "before" else "-", color=c,
                lw=2.2, marker="o", ms=5, label=f"branch {who} fixes")
        for xi, v in zip(SIZES, ratio):
            ax.annotate(f"{v:.2f}x", (xi, v), textcoords="offset points",
                        xytext=(0, 8 if who == "after" else -14), ha="center",
                        fontsize=7.5, color=c)
    ax.axhline(1.0, color="#a02020", lw=1.2, ls=":")
    ax.text(SIZES[0], 1.08, "parity with upstream", color="#a02020", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.2, 5.0)
    ax.set_xticks(SIZES); ax.set_xticklabels([f"{s:,}" for s in SIZES])
    ax.set_xlabel("files in corpus"); ax.set_ylabel("speedup vs upstream (x)")
    ax.set_title("Where the branch wins: crossover moves in")
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="lower right")


def bars_panel(ax):
    labels, up, bef, aft = [], [], [], []
    for k, v in SMALL_US.items():
        labels.append(f"{k}\n(477 files)"); up.append(v[0]); bef.append(v[1]); aft.append(v[2])
    for k, v in MEGA_US.items():
        labels.append(f"{k}\n(170k files)"); up.append(v[0]); bef.append(v[1]); aft.append(v[2])
    x, w = np.arange(len(labels)), 0.26
    ax.bar(x - w, up, w, color=UP, label=LABEL["upstream"])
    ax.bar(x, bef, w, color=BEFORE, label=LABEL["before"])
    ax.bar(x + w, aft, w, color=AFTER, label=LABEL["after"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylabel("p50 search wall (us)")
    ax.set_title("Per-query: small corpus and mega corpus")
    ax.grid(True, axis="y", which="both", alpha=0.25, lw=0.6)
    ax.legend(fontsize=8)


def typing_panel(ax):
    x = np.arange(len(KEYS))
    ax.plot(x, TYPING_MS["upstream"], "-o", color=UP, lw=2, ms=4, label="upstream")
    ax.plot(x, TYPING_MS["keepers"], "-o", color=AFTER, lw=2, ms=4, label="branch + fixes")
    ax.set_xticks(x); ax.set_xticklabels(KEYS, rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel("typed prefix (one search per keystroke)")
    ax.set_ylabel("p50 per keystroke (ms)")
    ax.set_title("Typing on 170k files, 1T")
    ax.grid(True, alpha=0.25, lw=0.6); ax.legend(fontsize=8)


def quality_panel(ax):
    corpora = list(QUALITY)
    x, w = np.arange(len(corpora)), 0.34
    hit = {who: [QUALITY[c][who][0] for c in corpora] for who in QUALITY[corpora[0]]}
    mrr = {who: [QUALITY[c][who][1] for c in corpora] for who in QUALITY[corpora[0]]}
    for j, who in enumerate(("upstream", "keepers")):
        off = (j - 0.5) * w
        bars = ax.bar(x + off, hit[who], w * 0.9,
                      color=UP if j == 0 else AFTER, alpha=0.55 if j == 0 else 1.0,
                      label=f"{who} hit@1 of 20")
        for b, v in zip(bars, hit[who]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, str(v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(corpora, fontsize=8)
    ax.set_ylabel("hit@1 (of 20 labeled queries)", color="#222")
    ax.set_ylim(0, 21)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper left")

    ax2 = ax.twinx()
    for j, who in enumerate(("upstream", "keepers")):
        off = (j - 0.5) * w
        line, = ax2.plot(x + off, mrr[who], "D--", color="#b8860b" if j == 0 else "#0b3d91",
                         lw=1.8, ms=6, label=f"{who} MRR")
        for xi, v in zip(x + off, mrr[who]):
            ax2.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=7.5, color="#553")
    ax2.set_ylabel("MRR (0-1)", color="#553")
    ax2.set_ylim(0, 1.15)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", ncol=2)


def main():
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 14.5))
    sweep_panel(axes[0][0], 1); sweep_panel(axes[0][1], 4)
    bars_panel(axes[1][0]); typing_panel(axes[1][1])
    ratio_panel(axes[2][0]); quality_panel(axes[2][1])
    fig.suptitle("FFF file search - perf/keepers vs upstream origin/main\n"
                 "measured p50, paired single-process warm runs; "
                 "'branch + fixes' = build restoration, tombstone filter, "
                 "honoured thread budget, bit-parallel streaming gate", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = "archive/search-improvements-2026-10-01.png"
    fig.savefig(out, dpi=160); fig.savefig(out.replace(".png", ".svg"))
    print("wrote", out)


if __name__ == "__main__":
    main()