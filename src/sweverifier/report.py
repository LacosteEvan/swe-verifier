"""README charts, light and dark PNGs."""
from __future__ import annotations

from pathlib import Path

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "neutral": "#8a8983"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"], "neutral": "#8a8983"},
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax, t, grid_axis="both"):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8, zorder=0)


def _save(fig, out_stem: Path, mode: str, t) -> Path:
    path = out_stem.with_name(f"{out_stem.name}-{mode}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=t["surface"])
    return path


def selection_bars(rows: list[tuple[str, float, float, float, str]], reference: tuple[str, float],
                   out_stem: Path) -> list[Path]:
    """Issues resolved per selection method, with the 95% interval of its gain over a random pick. `rows`: (label,
    resolved, low, high, kind) with kind 'baseline' (neutral), 'method' or 'best'; `reference`: a dashed line."""
    plt = _plt()
    paths = []
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(rows) + 1.0), dpi=200)
        fig.patch.set_facecolor(t["surface"])
        _style(ax, t, grid_axis="x")
        ax.spines["left"].set_visible(False)
        ys = list(range(len(rows)))[::-1]
        for y, (_, value, lo, hi, kind) in zip(ys, rows, strict=True):
            color = {"baseline": t["neutral"], "method": t["series"][0], "best": t["series"][1]}[kind]
            ax.barh(y, 100 * value, 0.62, color=color, zorder=3)
            if lo == lo:  # an interval, not NaN
                ax.plot([100 * lo, 100 * hi], [y, y], color=t["ink"], linewidth=1, zorder=4)
            ax.annotate(f"{100 * value:.1f}%", (100 * max(value, hi if hi == hi else value), y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=t["ink"])
        ax.axvline(100 * reference[1], color=t["muted"], linestyle="--", linewidth=1.1, zorder=2)
        ax.annotate(reference[0], (100 * reference[1], ys[-1] - 0.55), xytext=(-4, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.5, color=t["muted"])
        ax.set_yticks(ys, [r[0] for r in rows], color=t["ink"], fontsize=8)
        ax.set_xlim(0, 75)
        ax.set_ylim(ys[-1] - 0.9, ys[0] + 0.5)
        ax.set_xlabel("Issues resolved by the selected patch (%)", color=t["muted"], fontsize=8.5)
        fig.tight_layout()
        paths.append(_save(fig, out_stem, mode, t))
        plt.close(fig)
    return paths


def cost_scatter(points: list[tuple[str, float, float, str, tuple[int, int]]], out_stem: Path) -> list[Path]:
    """Cost per issue (log scale) against issues resolved. `points`: (label, cost, resolved, kind, label offset);
    kind 'qwen' (filled), 'ceiling' (hollow) or 'reference' (second color)."""
    plt = _plt()
    paths = []
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=200)
        fig.patch.set_facecolor(t["surface"])
        _style(ax, t)
        for label, cost, value, kind, offset in points:
            color = t["series"][1] if kind == "reference" else t["series"][0]
            face = t["surface"] if kind == "ceiling" else color
            ax.scatter([cost], [100 * value], s=58, color=face, edgecolor=color if kind == "ceiling" else t["surface"],
                       linewidth=1.6 if kind == "ceiling" else 1.2, zorder=4)
            ax.annotate(label, (cost, 100 * value), xytext=offset, textcoords="offset points", fontsize=7.5,
                        color=t["ink"], ha="right" if offset[0] < 0 else "left", va="center")
        ax.set_xscale("log")
        ax.set_xlim(0.004, 0.5)
        ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        ax.set_xticks(ticks, [f"${x:g}" for x in ticks])
        ax.minorticks_off()
        ax.set_ylim(35, 75)
        ax.set_xlabel("Cost per issue ($, log scale)", color=t["muted"], fontsize=8.5)
        ax.set_ylabel("Issues resolved (%)", color=t["muted"], fontsize=8.5)
        fig.tight_layout()
        paths.append(_save(fig, out_stem, mode, t))
        plt.close(fig)
    return paths
