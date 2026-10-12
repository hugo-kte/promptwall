"""Report figures (matplotlib, Agg). Same visual spec as the rest of the portfolio:
colour follows the series identity, hairline solid grids, rounded bar ends, text in ink."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABELS = {
    "spml": "SPML\n(unseen chatbots)",
    "deepset": "deepset",
    "jackhhao": "jackhhao\n(jailbreaks)",
}

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _figure(width: float, height: float) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return fig, ax


def _titles(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=28)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def _bars(ax: Axes, xs: np.ndarray, values: np.ndarray, width: float, color: str) -> None:
    """Vertical bars with a 3 pt rounded top and a square baseline (one path each)."""
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    r = 3 * ax.figure.dpi / 72
    rx, ry_unit = r / (bbox.width / (x1 - x0)), r / (bbox.height / (y1 - y0))
    k = 0.5523
    for x, v in zip(xs, values, strict=True):
        if not v > 0:
            continue
        ry, rxx = min(ry_unit, v / 2), min(rx, width / 2)
        lo, hi = x - width / 2, x + width / 2
        verts = [
            (lo, 0),
            (lo, v - ry),
            (lo, v - ry + k * ry),
            (lo + rxx - k * rxx, v),
            (lo + rxx, v),
            (hi - rxx, v),
            (hi - rxx + k * rxx, v),
            (hi, v - ry + k * ry),
            (hi, v - ry),
            (hi, 0),
            (lo, 0),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            *[MplPath.CURVE4] * 3,
            MplPath.LINETO,
            *[MplPath.CURVE4] * 3,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, lw=0, snap=False))


def cross_source(experiments: dict[str, Any], path: Path) -> Path:
    series = [
        ("single-source", "Trained on SPML only"),
        ("multi-source", "Trained on all three sources"),
    ]
    sources = ["spml", "deepset", "jackhhao"]
    fig, ax = _figure(7.2, 4.2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(sources) - 0.4)
    ax.set_ylim(0, 1.1)  # headroom for the value labels
    fig.canvas.draw()
    width, step = 0.3, 0.33
    for k, (key, _) in enumerate(series):
        vals = np.array([experiments[key]["results"][s]["detection_rate"] for s in sources])
        xs = np.arange(len(sources)) + (k - 0.5) * step
        _bars(ax, xs, vals, width, SERIES[k])
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(np.arange(len(sources)), [LABELS[s] for s in sources])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Injections detected")
    ax.legend(
        handles=[Patch(color=SERIES[k], label=lab) for k, (_, lab) in enumerate(series)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=9,
        labelcolor=INK2,
    )
    _titles(
        ax,
        "Detection collapses on attacks from another source",
        "Test data; threshold calibrated for a 1% false-positive budget on every benign source",
    )
    return _save(fig, path)


def roc(curves: dict[str, Any], threshold: float, path: Path) -> Path:
    fig, ax = _figure(7.2, 4.4)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    lo = 1e-3
    for k, source in enumerate(["spml", "deepset", "jackhhao"]):
        y, scores = curves[source]
        fpr, tpr, _ = roc_curve(y, scores)
        keep = fpr >= lo
        ax.plot(
            np.r_[lo, fpr[keep]],
            np.r_[np.interp(lo, fpr, tpr), tpr[keep]],
            color=SERIES[k],
            lw=2,
            solid_capstyle="round",
            label=LABELS[source].replace("\n", " "),
        )
        # operating point of the shipped threshold, computed exactly from the scores
        op_fpr = float(np.mean(scores[y == 0] >= threshold))
        op_tpr = float(np.mean(scores[y == 1] >= threshold))
        ax.scatter(
            [max(op_fpr, lo)],
            [op_tpr],
            s=64,
            color=SERIES[k],
            edgecolors=SURFACE,
            linewidths=2,
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_xlim(lo, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-positive rate (log scale)")
    ax.set_ylabel("Detection rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK2)
    _titles(ax, "Shipped classifier, per source", "Dots mark the shipped threshold")
    return _save(fig, path)
