"""Data loading functions.

All loaders read from the resolved data directory (see paths.py) and
accept an optional ``root_dir`` override for flexibility.
"""

from pathlib import Path

import pandas as pd

from fixation_grower.paths import data_dir


def load_trials_df(stages: str = "all", root_dir: Path | None = None) -> pd.DataFrame:
    """Load the trial-level dataframe.

    Parameters
    ----------
    stages:
        ``"all"`` | ``"spoke"`` | ``"fix_growth"`` | ``"probe"``
    root_dir:
        Directory containing ``trials_df.parquet``. Defaults to ``data/``.
    """
    root = Path(root_dir) if root_dir is not None else data_dir()
    tdf = pd.read_parquet(root / "trials_df.parquet")

    if stages == "spoke":
        tdf = tdf.query("stage < 5").copy()
    elif stages == "fix_growth":
        tdf = tdf.query("stage > 4 and stage < 9").copy()
    elif stages == "probe":
        tdf = tdf.query("stage == 9 or stage == 10").copy()

    return tdf


def load_days_df(stages: str = "all", root_dir: Path | None = None) -> pd.DataFrame:
    """Load the day-level summary dataframe.

    Parameters
    ----------
    stages:
        ``"all"`` | ``"spoke"`` | ``"fix_growth"`` | ``"probe"``
    root_dir:
        Directory containing ``days_df.parquet``. Defaults to ``data/``.
    """
    root = Path(root_dir) if root_dir is not None else data_dir()
    ddf = pd.read_parquet(root / "days_df.parquet")

    if stages == "spoke":
        ddf = ddf.query("stage < 5").copy()
    elif stages == "fix_growth":
        ddf = ddf.query("stage > 4 and stage < 9").copy()
    elif stages == "probe":
        ddf = ddf.query("stage == 9 or stage == 10").copy()

    return ddf


def load_poke_df(stages: str = "all", root_dir: Path | None = None) -> pd.DataFrame:
    """Load the poke-level dataframe.

    Parameters
    ----------
    stages:
        ``"all"`` | ``"fix_growth"`` | ``"probe"``
    root_dir:
        Directory containing ``poke_df.parquet``. Defaults to ``data/``.
    """
    root = Path(root_dir) if root_dir is not None else data_dir()
    pdf = pd.read_parquet(root / "poke_df.parquet")

    if stages == "fix_growth":
        pdf = pdf.query("stage > 4 and stage < 9").copy()
    elif stages == "probe":
        pdf = pdf.query("stage == 9 or stage == 10").copy()

    return pdf


def load_fixgrower_continued_df(root_dir: Path | None = None) -> pd.DataFrame:
    """Load stage-11 continued assessment trials (FixGrower only)."""
    root = Path(root_dir) if root_dir is not None else data_dir()
    return pd.read_parquet(root / "fig04_fixgrower_continued_stage11.parquet")
