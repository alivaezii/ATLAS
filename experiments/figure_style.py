from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito palette (colorblind-safe) with grayscale-distinguishable ordering.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#666666",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def apply_journal_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.2,
            "lines.markersize": 4.0,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def cycle_colors(names: Iterable[str]) -> list[str]:
    base = [
        PALETTE["blue"],
        PALETTE["orange"],
        PALETTE["green"],
        PALETTE["red"],
        PALETTE["purple"],
        PALETTE["cyan"],
        PALETTE["black"],
    ]
    out = []
    for i, _ in enumerate(names):
        out.append(base[i % len(base)])
    return out


def export_figure(fig: plt.Figure, out_base: Path, png_fallback: bool = True) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf")
    if png_fallback:
        fig.savefig(out_base.with_suffix(".png"), format="png", dpi=600)
