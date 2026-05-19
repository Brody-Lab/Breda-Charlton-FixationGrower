"""Data transformation and feature-engineering functions."""

import pandas as pd


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
