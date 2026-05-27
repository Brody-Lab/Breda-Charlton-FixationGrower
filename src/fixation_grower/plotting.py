"""Shared plotting helpers and figure-saving utility."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fixation_grower import config
from fixation_grower.paths import figures_dir
from fixation_grower.transforms import compute_days_relative_to_stage


def save_figure(fig: plt.Figure, name: str, fmt: str = "png", verbose: bool = False):
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

    if verbose:
        print(f"Saved figure to {path}")


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

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels, title=None, frameon=False)
    elif ax.get_legend():
        ax.get_legend().remove()

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def plot_stage_progression(
    trials_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 3),
    dpi: int = 300,
    title: str = "",
    x_var: str = "date",
    relative_to_stage: int | None = None,
    hue: str | None = None,
    xlabel: str = "",
    legend_title: str | None = None,
    rotate_x_labels: bool = False,
    ylim: tuple[int, int] | None = None,
    palette=None,
    alpha: float = 1.0,
    markersize: float | None = None,
    linewidth: float | None = None,
    marker: str = "o",
    hide_legend: bool = False,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Mean training ``stage`` vs time (date or days relative to a reference stage).

    Uses ``sns.lineplot`` with ``drawstyle=\"steps-post\"`` (trial means aggregated
    by x-axis bin and hue). When *relative_to_stage* is set and *x_var* is ``date``,
    *x_var* is replaced by ``days_relative_to_stage_<stage>`` via
    :func:`~fixation_grower.transforms.compute_days_relative_to_stage`.

    Requires columns: ``date``, ``stage``, and optionally *hue*.
    """
    df_work = trials_df.copy()
    fig_own = ax is None
    if fig_own:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if relative_to_stage is not None and x_var == "date":
        df_work = compute_days_relative_to_stage(
            df_work, stage=relative_to_stage
        ).reset_index(drop=True)
        x_var = f"days_relative_to_stage_{relative_to_stage}"

    group_cols = [x_var] if hue is None else [x_var, hue]
    plot_df = df_work.groupby(group_cols, observed=True)["stage"].mean().reset_index()

    line_kw: dict = {"drawstyle": "steps-post", "ax": ax, "marker": marker, **kwargs}
    if palette is not None:
        line_kw["palette"] = palette
    if alpha is not None:
        line_kw["alpha"] = alpha
    if markersize is not None:
        line_kw["markersize"] = markersize
    if linewidth is not None:
        line_kw["linewidth"] = linewidth

    sns.lineplot(data=plot_df, x=x_var, y="stage", hue=hue, **line_kw)

    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    if ylim is not None:
        y_lo, y_hi = ylim
        yticks_list = list(range(y_lo, y_hi + 1))
        use_ylim = ylim
    else:
        max_stage = int(df_work["stage"].max())
        use_ylim = (0, max_stage + 1)
        yticks_list = list(range(max_stage + 1))

    ax.set(
        ylabel="Stage #",
        xlabel=xlabel,
        title=title,
        ylim=use_ylim,
        yticks=yticks_list,
    )
    ax.grid(True)
    leg = ax.get_legend()
    if hide_legend and leg is not None:
        leg.remove()
    elif legend_title is not None and leg is not None:
        leg.set_title(legend_title)

    return fig, ax


def plot_fixation_and_violations(
    ax: plt.Axes,
    df: pd.DataFrame,
    experiment: str,
) -> None:
    """Per-animal fixation growth panel: max fixation, rig switches, violation rate.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    df:
        Trial-level subset for one animal in fixation growth (stages 5–7).
        Must contain ``days_relative_to_stage_5``, ``fixation_dur``, ``rigid``,
        and for FixGrower also ``n_settling_ins``.
    experiment:
        ``"Legacy"`` or ``"FixGrower"`` (controls fixation color and violation
        metric: Legacy uses ``violations`` column; FixGrower computes rate from
        trials vs settling-ins).
    """
    fix_color = (
        config.LEGACY_COLOR if experiment == "Legacy" else config.FIXGROWER_COLOR
    )

    sns.lineplot(
        data=df,
        x="days_relative_to_stage_5",
        y="fixation_dur",
        estimator="max",
        ax=ax,
        marker="o",
        color=fix_color,
    )
    ax.axhline(y=2, color="k", linewidth=1)
    ax.set(ylim=(-0.1, 2.5), xlabel="Days in Fixation Growth Curriculum")
    ax.set_ylabel("Max Fix Dur [s]", color=fix_color, weight="semibold")

    df_sorted = df.sort_values("days_relative_to_stage_5")
    changes = df_sorted["rigid"].ne(df_sorted["rigid"].shift())
    for day in df_sorted.loc[changes, "days_relative_to_stage_5"].unique():
        if day != 0:
            ax.axvline(x=day, color="darkgray", linestyle="--", alpha=0.7)

    ax2 = ax.twinx()
    if experiment == "Legacy":
        sns.lineplot(
            data=df,
            x="days_relative_to_stage_5",
            y="violations",
            color=config.VIOLATION_COLOR,
            ax=ax2,
            marker="o",
            linestyle="--",
            errorbar=None,
        )
    else:
        grouped = (
            df.groupby("days_relative_to_stage_5")
            .agg(
                n_trials=("animal_id", "size"),
                n_settling=("n_settling_ins", "sum"),
            )
            .reset_index()
        )
        grouped["violation_rate"] = 1 - grouped["n_trials"] / grouped["n_settling"]
        sns.lineplot(
            data=grouped,
            x="days_relative_to_stage_5",
            y="violation_rate",
            color=config.VIOLATION_COLOR,
            ax=ax2,
            marker="o",
            linestyle="--",
            errorbar=None,
        )

    ax2.set_ylabel(
        "Violation Rate", color=config.VIOLATION_COLOR, weight="semibold"
    )
    ax2.set_ylim(0, 1)


def plot_fixation_simulation_example(
    ax: plt.Axes,
    trials_true: pd.DataFrame,
    trials_recovery: pd.DataFrame,
    trials_perfect: pd.DataFrame,
) -> None:
    """Panel S2A: true, recovery, and perfect fixation trajectories for one animal."""
    sns.lineplot(
        data=trials_true,
        x="session",
        y="fixation_dur",
        ax=ax,
        estimator="max",
        marker="o",
        label="True",
        color=config.LEGACY_COLOR,
    )
    sns.lineplot(
        data=trials_recovery,
        x="session",
        y="fixation_dur",
        ax=ax,
        estimator="max",
        marker="s",
        label="Recovered",
        color=config.LEGACY_COLOR,
        linestyle=":",
    )
    sns.lineplot(
        data=trials_perfect,
        x="session",
        y="fixation_dur",
        ax=ax,
        estimator="max",
        marker="^",
        label="Perfect",
        color="salmon",
    )
    ax.grid()
    ax.axhline(y=2, color="k")
    ax.set(
        xlabel="Days in Fixation Growth Curriculum",
        ylabel="Fixation Duration [s]",
        ylim=(0, None),
    )


def plot_days_to_target_comparison(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    order: list[str],
    x_labels: list[str],
    marker_colors: list[str],
    tick_colors: list[str],
    *,
    markers: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
    n_animals: int = 8,
) -> None:
    """Paired/group days-to-target pointplot with per-animal gray lines."""
    sns.pointplot(
        data=summary_df,
        x="mode",
        y="days_to_target",
        ax=ax,
        hue="animal_id",
        palette=["gray"] * n_animals,
        alpha=0.25,
        order=order,
    )

    point_kw: dict = {
        "data": summary_df.query("mode in @order"),
        "x": "mode",
        "y": "days_to_target",
        "ax": ax,
        "errorbar": "se",
        "hue": "mode",
        "palette": marker_colors,
        "order": order,
        "linestyle": None,
    }
    if markers is not None:
        point_kw["markers"] = markers
    sns.pointplot(**point_kw)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0)
    for label, color in zip(ax.get_xticklabels(), tick_colors):
        label.set_color(color)

    sns.despine(ax=ax)
    ax.set_ylabel("N Days to Target")
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, None)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
