"""Two panels of F1 boxplots: one per decoder, five boxes each.

Hue carries the restraint (DMS / 2A3 / none); fill saturation separates the
per-row profiles from the per-structure averaged profiles. Both are also named
on the x axis, so identity is never colour-alone.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common import OUT, SCORES_CSV

DMS, A2A3, NONE = "#1baf7a", "#2a78d6", "#eb6834"

# (kind, experiment, label, hue, filled)
BOXES = [
    ("row", "DMS_MaP", "DMS\nper-row", DMS, False),
    ("row", "2A3_MaP", "2A3\nper-row", A2A3, False),
    ("mean", "DMS_MaP", "DMS\nmean", DMS, True),
    ("mean", "2A3_MaP", "2A3\nmean", A2A3, True),
    ("noshape", "", "noSHAPE", NONE, True),
]
PANELS = [("MEA", "MEA  (-M)"), ("PK", "ThreshKnot  (-T)")]

INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"


def main():
    scores = pd.read_csv(SCORES_CSV).fillna({"experiment": ""})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)

    for ax, (decoder, title) in zip(axes, PANELS):
        sub = scores[scores.decoder == decoder]
        data, labels, colors, fills = [], [], [], []
        for kind, exp, label, hue, filled in BOXES:
            f1 = sub[(sub.kind == kind) & (sub.experiment == exp)].f1
            data.append(f1.values)
            labels.append(f"{label}\nn={len(f1)}")
            colors.append(hue)
            fills.append(filled)

        bp = ax.boxplot(
            data, patch_artist=True, widths=0.55, showfliers=True,
            medianprops=dict(color=INK, linewidth=2),
            whiskerprops=dict(color=MUTED, linewidth=1),
            capprops=dict(color=MUTED, linewidth=1),
            flierprops=dict(marker="o", markersize=2.5, markerfacecolor=MUTED,
                            markeredgecolor="none", alpha=0.35),
        )
        for patch, hue, filled in zip(bp["boxes"], colors, fills):
            patch.set_facecolor(hue if filled else "white")
            patch.set_alpha(1.0 if filled else 1.0)
            patch.set_edgecolor(hue)
            patch.set_linewidth(2)

        # selective direct labels: the median only, not a number per point
        for i, values in enumerate(data, start=1):
            ax.text(i, 1.04, f"{pd.Series(values).median():.2f}", ha="center",
                    va="bottom", fontsize=9, color=MUTED)

        ax.set_xticklabels(labels, fontsize=9, color=INK)
        ax.set_title(title, fontsize=11, color=INK, pad=22, loc="left")
        ax.set_ylim(-0.03, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0, colors=MUTED)

    axes[0].set_ylabel("F1 vs PDB reference", fontsize=10, color=INK)
    axes[1].legend(
        handles=[
            Patch(facecolor="white", edgecolor=MUTED, linewidth=2, label="per-row profile"),
            Patch(facecolor=MUTED, edgecolor=MUTED, linewidth=2, label="averaged profile"),
        ],
        loc="lower right", frameon=False, fontsize=9, labelcolor=INK,
    )
    fig.suptitle(
        "SHAPE-restrained LinearPartition folding, DMS vs 2A3",
        fontsize=13, color=INK, x=0.075, ha="left", y=0.985,
    )
    fig.text(0.075, 0.925, "DMS restricted to A/C; G and U masked as no-data",
             fontsize=10, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    path = f"{OUT}/f1_boxplots.png"
    fig.savefig(path, dpi=200, facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
