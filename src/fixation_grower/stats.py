"""Statistical comparison functions for Legacy vs FixGrower groups."""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
from scipy.stats import mannwhitneyu, shapiro, ttest_ind


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
        stat, p_val = ttest_ind(legacy_values, fixgrower_values, equal_var=False)
        result.update(test_type="welch_t", test_statistic=stat, p_val_raw=p_val)
    else:
        stat, p_val = mannwhitneyu(
            legacy_values, fixgrower_values, alternative="two-sided"
        )
        result.update(test_type="mannwhitney", test_statistic=stat, p_val_raw=p_val)

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
    legacy_data = df.loc[df["fix_experiment"] == "Legacy", metric_col].to_numpy()
    fixgrower_data = df.loc[
        df["fix_experiment"] == "FixGrower", metric_col
    ].to_numpy()
    result = compare_two_groups(legacy_data, fixgrower_data, alpha=alpha)
    return pd.DataFrame([result])


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

    model = smf.mixedlm(
        formula=f"{metric_col} ~ fix_experiment",
        data=df,
        groups=df["animal_id"],
    )
    try:
        fit = model.fit(method="lbfgs", disp=False)
    except Exception as e:
        raise RuntimeError(f"Model fitting failed: {e}") from e

    coef_key = "fix_experiment[T.FixGrower]"
    if coef_key not in fit.params.index:
        raise ValueError(f"Coefficient '{coef_key}' not found in model.")

    residuals_normal = check_normality(fit.resid, alpha=alpha)
    result = {
        "test_type": "mixedlm_random_intercept",
        "test_statistic": fit.tvalues[coef_key],
        "p_val_raw": fit.pvalues[coef_key],
        "residuals_normal": residuals_normal,
    }
    return pd.DataFrame([result])
