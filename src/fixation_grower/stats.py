"""Statistical comparison functions for Legacy vs FixGrower groups."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests


def check_normality(data, alpha: float = 0.05) -> bool:
    """Return ``True`` if *data* passes Shapiro-Wilk at significance *alpha*."""
    _, p = shapiro(data)
    return p >= alpha


def compare_two_groups(
    legacy_values,
    fixgrower_values,
    alpha: float = 0.05,
) -> dict:
    """Compare two arrays with Welch's t-test (normal) or Mann-Whitney U.

    Returns a dict with keys:
        ``normality_Legacy``, ``normality_FixGrower``,
        ``n_Legacy``, ``n_FixGrower``,
        ``test_type``, ``test_statistic``, ``p_val_raw``.
    """
    normal_legacy = check_normality(legacy_values, alpha=alpha)
    normal_fixgrower = check_normality(fixgrower_values, alpha=alpha)

    result = {
        "normality_Legacy": normal_legacy,
        "normality_FixGrower": normal_fixgrower,
        "n_Legacy": len(legacy_values),
        "n_FixGrower": len(fixgrower_values),
    }

    if normal_legacy and normal_fixgrower:
        stat, p_val = ttest_ind(
            legacy_values, fixgrower_values, equal_var=False)
        result.update(test_type="welch_t",
                      test_statistic=stat, p_val_raw=p_val)
    else:
        stat, p_val = mannwhitneyu(
            legacy_values, fixgrower_values, alternative="two-sided"
        )
        result.update(test_type="mannwhitney",
                      test_statistic=stat, p_val_raw=p_val)

    return result


def compare_legacy_fixgrower(
    df: pd.DataFrame,
    metric_col: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare Legacy vs FixGrower for a single stage (one-row result).

    Parameters
    ----------
    df:
        Must contain ``fix_experiment`` (values ``"Legacy"`` / ``"FixGrower"``)
        and *metric_col*.
    """
    legacy_data = df.loc[df["fix_experiment"]
                         == "Legacy", metric_col].to_numpy()
    fixgrower_data = df.loc[
        df["fix_experiment"] == "FixGrower", metric_col
    ].to_numpy()
    result = compare_two_groups(legacy_data, fixgrower_data, alpha=alpha)
    return pd.DataFrame([result])


def compare_fixation_growth_by_day(
    df: pd.DataFrame,
    max_days: int,
    metric_col: str = "fixation_growth",
    *,
    verbose: bool = False,
) -> dict[int, dict]:
    """Per-day Mann-Whitney U with Bonferroni multiplicity on *p*-values.

    For each curriculum day ``1 .. max_days`` (matching
    ``days_relative_to_stage_5``), compares *metric_col* between Legacy and
    FixGrower. ``p_adj`` multiplies the raw *p*-value by *max_days*.
    Original notebook annotated stars using raw ``p_value`` (not ``p_adj``).

    Parameters
    ----------
    df:
        Must contain ``days_relative_to_stage_5``, ``fix_experiment``, and
        *metric_col*.
    max_days:
        Maximum day index tested and Bonferroni factor.
    metric_col:
        Column to compare (default ``fixation_growth``).
    verbose:
        If ``True``, print per-day summaries to stdout.

    Returns
    -------
    dict[int, dict]
        Keys ``1 .. max_days`` with ``test_type``, ``stat``, ``p_value``,
        ``p_adj``, ``normality_Legacy``, ``normality_FixGrower``.
    """
    results: dict[int, dict] = {}

    for day in range(1, max_days + 1):
        day_data = df.loc[df["days_relative_to_stage_5"] == day]

        group_legacy = (
            day_data.loc[day_data["fix_experiment"] == "Legacy", metric_col]
            .dropna()
        )
        group_fixgrower = (
            day_data.loc[
                day_data["fix_experiment"] == "FixGrower",
                metric_col,
            ].dropna()
        )

        normality_legacy = (
            shapiro(group_legacy).pvalue > 0.05 if len(
                group_legacy) >= 3 else False
        )
        normality_fixgrower = (
            shapiro(group_fixgrower).pvalue > 0.05
            if len(group_fixgrower) >= 3
            else False
        )

        if len(group_legacy) == 0 or len(group_fixgrower) == 0:
            test_type = "insufficient data"
            stat = np.nan
            p_val = np.nan
        else:
            test_type = "mann-whitney"
            stat, p_val = mannwhitneyu(
                group_legacy, group_fixgrower, alternative="two-sided"
            )

        p_adj = (
            min(float(p_val) * max_days, 1.0) if not np.isnan(p_val) else np.nan
        )

        results[day] = {
            "test_type": test_type,
            "stat": stat,
            "p_value": p_val,
            "p_adj": p_adj,
            "normality_Legacy": normality_legacy,
            "normality_FixGrower": normality_fixgrower,
        }

        if verbose:
            sig = "**SIGNIFICANT**" if (not np.isnan(p_adj)
                                        and p_adj < 0.05) else ""
            print(f"Day {day} [{test_type}]: {sig}")
            print(
                f"  Normality Legacy: {normality_legacy}, "
                f"Normality FixGrower: {normality_fixgrower}"
            )
            print(
                f"  Stat = {stat:.3f}, p = {p_val:.3e}, Bonferroni adj p = {p_adj:.3e}\n"
            )

    return results


def compare_legacy_fixgrower_mixedlm(
    df: pd.DataFrame,
    metric_col: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mixed-effects model comparing Legacy vs FixGrower with random animal intercept.

    Parameters
    ----------
    df:
        Must contain ``animal_id``, ``fix_experiment``, and *metric_col*.
    """
    required = ["animal_id", "fix_experiment", metric_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"DataFrame is missing required column: '{col}'")

    if df[df["fix_experiment"] == "Legacy"].empty or df[
        df["fix_experiment"] == "FixGrower"
    ].empty:
        raise ValueError("Both groups (Legacy and FixGrower) must be present.")

    # Legacy as reference baseline (manuscript-aligned). Alphabetical ordering would
    # put FixGrower first as baseline, renaming the contrast T.Legacy vs T.FixGrower.
    formula = (
        f'{metric_col} ~ C(fix_experiment, Treatment(reference="Legacy"))'
    )
    model = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df["animal_id"],
    )
    try:
        fit = model.fit(method="lbfgs", disp=False)
    except Exception as e:
        raise RuntimeError(f"Model fitting failed: {e}") from e

    coef_keys = [k for k in fit.params.index if "FixGrower" in str(k)]
    if len(coef_keys) != 1:
        raise ValueError(
            "Could not find FixGrower contrast in model params: "
            f"{list(fit.params.index)}"
        )
    coef_key = coef_keys[0]

    residuals_normal = check_normality(fit.resid, alpha=alpha)
    result = {
        "test_type": "mixedlm_random_intercept",
        "test_statistic": fit.tvalues[coef_key],
        "p_val_raw": fit.pvalues[coef_key],
        "residuals_normal": residuals_normal,
    }
    return pd.DataFrame([result])


# ---------------------------------------------------------------------------
# Supplement S1 statistical helpers
# ---------------------------------------------------------------------------

def _mixedlm_stage_row(
    df: pd.DataFrame,
    metric_col: str,
    stage_type: str,
    alpha: float,
) -> dict:
    """Fit one mixed-LM row for a single stage_type subset.

    Legacy is the reference group. Random intercept by ``animal_id``.
    """
    sub = df.loc[df["stage_type"] == stage_type].dropna(subset=[metric_col])
    row: dict = {"stage_type": stage_type}

    if sub.empty or sub["fix_experiment"].nunique() < 2:
        row.update(
            test_type="skipped",
            test_statistic=np.nan,
            p_val_raw=np.nan,
            residuals_normal=np.nan,
        )
        return row

    formula = f'{metric_col} ~ C(fix_experiment, Treatment(reference="Legacy"))'
    model = smf.mixedlm(formula=formula, data=sub, groups=sub["animal_id"])
    try:
        fit = model.fit(method="lbfgs", disp=False)
    except Exception:
        row.update(
            test_type="failed",
            test_statistic=np.nan,
            p_val_raw=np.nan,
            residuals_normal=np.nan,
        )
        return row

    coef_keys = [k for k in fit.params.index if "FixGrower" in str(k)]
    if len(coef_keys) != 1:
        row.update(
            test_type="contrast_not_found",
            test_statistic=np.nan,
            p_val_raw=np.nan,
            residuals_normal=np.nan,
        )
        return row

    coef_key = coef_keys[0]
    row.update(
        test_type="mixedlm_random_intercept",
        test_statistic=float(fit.tvalues[coef_key]),
        p_val_raw=float(fit.pvalues[coef_key]),
        residuals_normal=check_normality(fit.resid, alpha=alpha),
    )
    return row


def compare_supp_s1_session_metrics_mixedlm_holm(
    df: pd.DataFrame,
    metric_col: str,
    stage_order: tuple[str, ...] = ("spoke", "grow_fix", "probe"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fig. S1 panels A–C: mixed LM per ``stage_type``, Holm correction across phases.

    Model: ``metric ~ C(fix_experiment, Treatment(reference="Legacy"))``
    with random intercept for ``animal_id``. Holm correction is applied
    across the three stage-type comparisons.

    Returns a tidy DataFrame with one row per stage_type and columns:
    ``stage_type``, ``test_type``, ``test_statistic``, ``p_val_raw``,
    ``residuals_normal``, ``p_val_holm``, ``reject_h0_holm``.
    """
    rows = [_mixedlm_stage_row(df, metric_col, st, alpha) for st in stage_order]
    out = pd.DataFrame(rows)

    raw = out["p_val_raw"].to_numpy(dtype=float)
    valid = ~np.isnan(raw)
    p_adj = np.full(len(raw), np.nan)
    reject = np.full(len(raw), np.nan)
    if valid.any():
        rej, pv_adj, _, _ = multipletests(raw[valid], alpha=alpha, method="holm")
        p_adj[valid] = pv_adj
        reject[valid] = rej.astype(float)

    out["p_val_holm"] = p_adj
    out["reject_h0_holm"] = reject
    return out


def compare_supp_s1_animal_metrics_by_stage_holm(
    df: pd.DataFrame,
    metric_col: str,
    stage_order: tuple[str, ...] = ("spoke", "grow_fix", "probe"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fig. S1 panels D–E: Welch / Mann–Whitney per phase, Holm across phases.

    One row per ``stage_type``. Normality tested via Shapiro-Wilk; Welch's
    t-test used when both groups are normal, otherwise Mann–Whitney U.
    Holm correction is applied across the three stage-type comparisons.
    """
    rows: list[dict] = []
    raw_p: list[float] = []

    for st in stage_order:
        sub = df.loc[df["stage_type"] == st]
        legacy = sub.loc[sub["fix_experiment"] == "Legacy", metric_col].dropna().to_numpy()
        fg = sub.loc[sub["fix_experiment"] == "FixGrower", metric_col].dropna().to_numpy()

        row: dict = {"stage_type": st}
        if len(legacy) == 0 or len(fg) == 0:
            row.update(
                normality_Legacy=np.nan,
                normality_FixGrower=np.nan,
                n_Legacy=len(legacy),
                n_FixGrower=len(fg),
                test_type="insufficient_data",
                test_statistic=np.nan,
                p_val_raw=np.nan,
            )
            raw_p.append(np.nan)
        else:
            cmp = compare_two_groups(legacy, fg, alpha=alpha)
            row.update(
                normality_Legacy=cmp["normality_Legacy"],
                normality_FixGrower=cmp["normality_FixGrower"],
                n_Legacy=cmp["n_Legacy"],
                n_FixGrower=cmp["n_FixGrower"],
                test_type=cmp["test_type"],
                test_statistic=cmp["test_statistic"],
                p_val_raw=cmp["p_val_raw"],
            )
            raw_p.append(cmp["p_val_raw"])
        rows.append(row)

    out = pd.DataFrame(rows)
    raw = np.array(raw_p, dtype=float)
    valid = ~np.isnan(raw)
    p_adj = np.full(len(raw), np.nan)
    reject = np.full(len(raw), np.nan)
    if valid.any():
        rej, pv_adj, _, _ = multipletests(raw[valid], alpha=alpha, method="holm")
        p_adj[valid] = pv_adj
        reject[valid] = rej.astype(float)

    out["p_val_holm"] = p_adj
    out["reject_h0_holm"] = reject
    return out


def compare_paired_wilcoxon(
    df: pd.DataFrame,
    metric_col: str,
    group_a: str,
    group_b: str,
    mode_col: str = "mode",
) -> pd.DataFrame:
    """Wilcoxon signed-rank on paired values from two *mode* levels."""
    wide = (
        df.loc[df[mode_col].isin([group_a, group_b])]
        .pivot(index="animal_id", columns=mode_col, values=metric_col)[
            [group_a, group_b]
        ]
        .dropna()
    )
    stat, p_val = wilcoxon(wide[group_b], wide[group_a])
    return pd.DataFrame(
        [
            {
                "test_type": "wilcoxon_signed_rank",
                "group_a": group_a,
                "group_b": group_b,
                "n_pairs": len(wide),
                "test_statistic": stat,
                "p_val_raw": p_val,
            }
        ]
    )
