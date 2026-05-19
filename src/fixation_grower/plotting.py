"""Shared plotting helpers and figure-saving utility."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from fixation_grower import config
from fixation_grower.paths import figures_dir


def save_figure(fig: plt.Figure, name: str, fmt: str = "png") -> Path:
    """Save *fig* to ``figures/<name>.<fmt>`` at 300 dpi.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    name:
        Base filename without extension.
    fmt:
        ``"png"`` (default) or ``"svg"``.
    """
    if fmt not in {"png", "svg"}:
        raise ValueError("fmt must be 'png' or 'svg'")
    out_dir = figures_dir()
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def box_strip_legacy_vs_fixgrower(
    data,
    x: str,
    order,
    y: str,
    ax: plt.Axes,
    hue_type: str = "experiment",
    alpha: float = 0.75,
    ylabel: str | None = None,
    xlabel: str | None = None,
    dodge="auto",
    whis: float = 0,
    s: float = 5,
    width: float = 0.5,
    **kwargs,
) -> None:
    """Box + strip plot comparing Legacy and FixGrower groups.

    Parameters
    ----------
    hue_type:
        ``"experiment"`` colors by curriculum arm; ``"animal"`` colors by
        individual animal ID.
    """
    sns.despine()

    sns.boxplot(
        x=x, y=y, data=data, order=order, ax=ax,
        hue="fix_experiment", hue_order=config.HUE_ORDER_EXP,
        palette=config.EXP_PALETTE,
        fill=False, showfliers=False,
        dodge=dodge, whis=whis, width=width,
        **kwargs,
    )

    if hue_type == "animal":
        hue = "animal_id"
        hue_order = config.HUE_ORDER_ANIMALS
        palette = config.ANIMAL_PALETTE
    else:
        hue = "fix_experiment"
        hue_order = config.HUE_ORDER_EXP
        palette = config.EXP_PALETTE

    sns.stripplot(
        x=x, y=y, data=data, order=order, ax=ax,
        hue=hue, hue_order=hue_order, palette=palette,
        dodge=dodge, legend=False, alpha=alpha, s=s,
        **kwargs,
    )

    ax.legend(title=None, frameon=False)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
