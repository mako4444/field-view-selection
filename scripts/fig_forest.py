"""Figure of the paired differences of both experiments (sparse and dense maps).

python scripts/fig_forest.py --results outputs/results --out outputs/figures/forest.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SPARSE, DENSE = "#2a78d6", "#eb6834"
INK, MUTED = "#222222", "#888888"


def get(res, name, pair, block=None):
    d = json.load(open(Path(res) / f'{name}.json'))
    m, lo, hi = (d[block] if block else d)['diffs'][pair]
    star = lo > 0 or hi < 0
    return round(m, 1), round(lo, 1), round(hi, 1), star


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', default='outputs/results')
    ap.add_argument('--out', default='outputs/figures/forest.pdf')
    a = ap.parse_args()
    r = a.results
    groups = [("Experiment 1: objects",
               [("field-sum $-$ count", get(r, 'objects_sparse', 'field_sum - cnt', 'field/pooled'),
                 get(r, 'objects_dense', 'field_sum - cnt', 'field/pooled')),
                ("field-sum $-$ Harris-sum", get(r, 'objects_sparse', 'field_sum - harris_sum', 'field/pooled'),
                 get(r, 'objects_dense', 'field_sum - harris_sum', 'field/pooled'))]),
              ("Experiment 2: rooms",
               [("H1: G+F $-$ G", get(r, 'test_sparse_field', 'u_G+F - u_G'), get(r, 'test_dense', 'u_G+F - u_G')),
                ("H2: G+F $-$ G+H", get(r, 'test_sparse_field', 'u_G+F - u_G+H'), get(r, 'test_dense', 'u_G+F - u_G+H')),
                ("H3: G+F $-$ G+C9", get(r, 'test_sparse_field', 'u_G+F - u_G+C9'),
                 get(r, 'test_dense', 'u_G+F - u_G+C9'))])]
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    off = 0.19
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    fig.subplots_adjust(left=0.37, right=0.975, top=0.86, bottom=0.15)
    y, ticks, labs, heads = 0.0, [], [], []
    for gname, rows in groups:
        heads.append((y, gname))
        y -= 0.85
        for lab, sp, de in rows:
            for (e, lo, hi, star), dy, col, mk in ((sp, off, SPARSE, "o"), (de, -off, DENSE, "s")):
                ax.plot([lo, hi], [y + dy] * 2, color=col, lw=1.0, solid_capstyle="round", zorder=2)
                ax.plot([e], [y + dy], marker=mk, ms=4.0, mec=col, mew=1.0, mfc=col if star else "white", zorder=3)
            ticks.append(y)
            labs.append(lab)
            y -= 1.0
        y -= 0.15
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs)
    ax.tick_params(axis="y", length=0, labelsize=7, colors=INK)
    ax.set_ylim(y + 0.55, 0.35)
    for yy, gname in heads:
        ax.text(-0.56, yy - 0.25, gname, transform=ax.get_yaxis_transform(), fontsize=7, weight="bold", color=INK,
                va="center", ha="left")
    for sp_ in ("top", "right", "left"):
        ax.spines[sp_].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=INK, labelsize=6.5, width=0.6, length=2)
    ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Paired difference (percentage points)", fontsize=6.8, color=INK, labelpad=2)
    h = [Line2D([], [], color=SPARSE, marker="o", ms=4, lw=1.0, label="sparse map"),
         Line2D([], [], color=DENSE, marker="s", ms=4, lw=1.0, label="dense map"),
         Line2D([], [], color=MUTED, marker="o", ms=4, mfc="white", lw=0, label="interval includes 0")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
