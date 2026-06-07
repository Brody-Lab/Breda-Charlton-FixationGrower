"""Data transformation and feature-engineering functions."""

import numpy as np
import pandas as pd

from fixation_grower import config

# ---------------------------------------------------------------------------
# Curriculum phase constants (shared by supplement notebooks)
# ---------------------------------------------------------------------------

#: Internal stage_type values, in display order.
STAGE_TYPE_ORDER = ["spoke", "grow_fix", "probe"]
#: Human-readable x-axis labels matching STAGE_TYPE_ORDER.
STAGE_CURRICULUM_LABELS = ["Side Poke", "Fixation Growth", "Probe"]


# ---------------------------------------------------------------------------
# Stage-type helpers (Figure S1)
# ---------------------------------------------------------------------------

def add_stage_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stage_type`` column: ``spoke``, ``grow_fix``, or ``probe``.

    Bins stages 1–4 → ``spoke``, 5–8 → ``grow_fix``, 9–10 → ``probe``.
    Raises ``ValueError`` if any ``stage`` value falls outside 1–10.
    """
    out = df.copy()
    conditions = [
        (out["stage"] >= 1) & (out["stage"] <= 4),
        (out["stage"] >= 5) & (out["stage"] <= 8),
        (out["stage"] >= 9) & (out["stage"] <= 10),
    ]
    out["stage_type"] = np.select(conditions, STAGE_TYPE_ORDER, default=pd.NA)
    if out["stage_type"].isna().any():
        invalid = out.loc[out["stage_type"].isna(), "stage"].unique()
        raise ValueError(f"Unexpected stage values encountered: {invalid}")
    return out


def make_supp_s1_trials_df(tdf: pd.DataFrame) -> pd.DataFrame:
    """Session-level trial count for Fig. S1 panel A.

    Side Poke uses unique ``trial`` count; Fixation Growth and Probe use
    ``n_settling_ins`` sum (center pokes attempted). Sessions with
    ``n_trials >= 900`` are dropped (Bpod/port error days).
    """
    spoke = (
        tdf.query("stage_type == 'spoke'")
        .groupby(
            ["animal_id", "stage_type", "fix_experiment", "date"],
            observed=True,
        )["trial"]
        .nunique()
        .reset_index(name="n_trials")
    )
    non_spoke = (
        tdf.query("stage_type != 'spoke'")
        .groupby(
            ["animal_id", "stage_type", "fix_experiment", "date"],
            observed=True,
        )["n_settling_ins"]
        .sum()
        .reset_index(name="n_trials")
    )
    trials_df = pd.concat([spoke, non_spoke], ignore_index=True)
    return trials_df.loc[trials_df["n_trials"] < 900].copy()


def make_hit_rate_df(tdf: pd.DataFrame) -> pd.DataFrame:
    """Session-level mean hit rate per (animal, stage_type, fix_experiment, date)."""
    return (
        tdf.groupby(
            ["animal_id", "fix_experiment", "stage_type", "date"],
            observed=True,
        )["hits"]
        .mean()
        .reset_index(name="hit_rate")
    )


def make_no_answer_rate_df(tdf: pd.DataFrame) -> pd.DataFrame:
    """Session-level no-answer rate (``result == 6``) per session."""
    return (
        tdf.groupby(
            ["animal_id", "date", "stage_type", "fix_experiment"],
            observed=True,
        )["result"]
        .apply(lambda x: (x == 6).mean())
        .reset_index(name="no_answer_rate")
    )


def make_n_days_stage_type_df(tdf: pd.DataFrame) -> pd.DataFrame:
    """Unique calendar days per (animal, fix_experiment, stage_type)."""
    return (
        tdf.groupby(
            ["animal_id", "fix_experiment", "stage_type"],
            observed=True,
        )["date"]
        .nunique()
        .reset_index(name="n_days_in_stage_type")
    )


def _count_rig_switches(rig_series: pd.Series) -> int:
    """Count rig-ID changes within a pre-sorted (animal, stage_type) group."""
    return int((rig_series != rig_series.shift(1)).sum() - 1)


def make_rig_switch_df(ddf: pd.DataFrame) -> pd.DataFrame:
    """Rig switches and distinct rig count per (animal, stage_type).

    Requires ``ddf`` to have ``rigid`` (rig identifier), ``stage_type``,
    ``fix_experiment``, and ``date``. Sorting by date is done internally.
    ``n_rigs`` = rig switches + 1.
    """
    ddf_sorted = ddf.sort_values(["animal_id", "date"])
    rig_df = (
        ddf_sorted.groupby(["animal_id", "stage_type"], observed=True)
        .agg(
            n_rig_switches=("rigid", _count_rig_switches),
            fix_experiment=("fix_experiment", "first"),
        )
        .reset_index()
    )
    rig_df["n_rigs"] = rig_df["n_rig_switches"] + 1
    return rig_df


def compute_days_relative_to_stage(
    df: pd.DataFrame,
    stage: int,
    date_col_name: str = "date",
    dense_rank: bool = True,
) -> pd.DataFrame:
    """Add a ``days_relative_to_stage_<stage>`` column to *df*.

    Parameters
    ----------
    df:
        DataFrame with columns ``animal_id``, ``stage``, and *date_col_name*.
    stage:
        Reference stage whose first day maps to day 0.
    date_col_name:
        Name of the date column.
    dense_rank:
        When ``True`` (default), use a dense integer rank so calendar gaps
        between sessions don't inflate the day count.
    """
    df["datetime_col"] = pd.to_datetime(df[date_col_name])

    if dense_rank:
        return (
            df.groupby("animal_id")[df.columns]
            .apply(_compute_relative_dense_dates, stage=stage)
            .reset_index(drop=True)
        )

    # Calendar-day calculation
    min_date_stage = (
        df.query("stage == @stage")
        .groupby("animal_id")["datetime_col"]
        .min()
        .reset_index()
        .rename(columns={"datetime_col": f"min_date_stage_{stage}"})
    )
    df = df.merge(min_date_stage, on="animal_id", how="left")
    df[f"days_relative_to_stage_{stage}"] = (
        df["datetime_col"] - df[f"min_date_stage_{stage}"]
    ).dt.days
    df.drop(columns=["datetime_col", f"min_date_stage_{stage}"], inplace=True)
    return df.copy().reset_index()


def _compute_relative_dense_dates(df: pd.DataFrame, stage: int) -> pd.DataFrame:
    """Per-animal helper: assign dense-rank days relative to *stage*."""
    df["dense_rank"] = df["datetime_col"].rank(method="dense").astype(int)
    min_stage_rank = df.query("stage == @stage")["dense_rank"].min()
    df[f"days_relative_to_stage_{stage}"] = df["dense_rank"] - min_stage_rank
    df.drop(columns=["dense_rank"], inplace=True)
    return df


def determine_fixation_dur(row: pd.Series) -> float:
    """Compute fixation duration for DMS2 protocol rows.

    The fixation_dur column was not computed directly in DMS2; this function
    reconstructs it from ``settling_in_dur`` and ``pre_go_dur``.
    """
    if row["stage"] in (5, 6, 7, 8):
        if row["session"] == 1 and row["cumulative_trial"] == 1:
            return row.settling_in_dur
        return row.settling_in_dur + row.pre_go_dur
    elif row["stage"] >= 9:
        return row.pre_go_dur
    raise KeyError(f"Unexpected stage: {row['stage']}")


def make_fixation_growth_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per (animal, date, days_relative_to_stage_5).

    Returns a dataframe with ``max_fixation_dur``, ``trials``,
    ``n_settling_ins``, ``valid_trials``, and ``fixation_growth``
    (day-over-day change in max fixation duration).
    """
    max_fixation_df = (
        df.query("stage >= 5")
        .groupby(
            ["date", "animal_id", "days_relative_to_stage_5", "fix_experiment"],
            observed=True,
        )
        .agg(
            max_fixation_dur=("fixation_dur", "max"),
            trials=("trial", "nunique"),
            n_violations=("violations", "sum"),
            n_settling_ins=("n_settling_ins", "sum"),
        )
        .reset_index()
    )
    max_fixation_df["valid_trials"] = (
        max_fixation_df["trials"] - max_fixation_df["n_violations"]
    )
    max_fixation_df.drop(columns=["n_violations"], inplace=True)
    max_fixation_df["fixation_growth"] = max_fixation_df.groupby(
        "animal_id"
    ).max_fixation_dur.diff()
    return max_fixation_df


def make_days_to_target_df(tdf: pd.DataFrame) -> pd.DataFrame:
    """First day (relative to stage 5) each animal reached the fixation target."""
    return (
        tdf.query("has_reached_target_fixation == True")
        .groupby(["animal_id", "fix_experiment"], observed=True)[
            "days_relative_to_stage_5"
        ]
        .min()
        .reset_index(name="days_to_target")
    )


def make_probe_violation_session_df(tdf: pd.DataFrame) -> pd.DataFrame:
    """Per-session violation rate (%) in probe stages 9–10."""
    return (
        tdf.query("stage in @config.PROBE_STAGES")
        .groupby(
            [
                "date",
                "fix_experiment",
                "animal_id",
                "stage",
                "days_relative_to_stage_10",
            ],
            observed=True,
        )
        .agg(violation_rate=("violations", "mean"))
        .reset_index()
        .assign(violation_rate=lambda d: d["violation_rate"] * 100)
    )


def make_growing_stage_session_summary(
    tdf: pd.DataFrame,
    animal_id: str,
) -> pd.DataFrame:
    """Per-session trial count and violation rate in fixation growth (stages 5–7)."""
    return (
        tdf.query(
            "animal_id == @animal_id and stage in @config.GROWING_STAGES"
        )
        .groupby(["animal_id", "date", "days_relative_to_stage_5"], observed=True)
        .agg(
            n_trials=("trial", "nunique"),
            violation_rate=("violations", "mean"),
        )
        .reset_index()
    )
