"""Legacy fixation-growth simulation (Figure S2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng

from fixation_grower import config
from fixation_grower.paths import data_dir
from fixation_grower.transforms import compute_days_relative_to_stage

# Legacy animals included in growth-ceiling simulation (R044 excluded per Methods).
LEGACY_SIM_ANIMALS = [
    "R040",
    "R042",
    "R046",
    "R048",
    "R050",
    "R052",
    "R054",
    "R056",
]

SIM_MODES = ("recovery", "perfect")
N_SIM_REPLICATES = 30
SIM_SEEDS = range(N_SIM_REPLICATES)
PANEL_A_SEED = 30
SIM_CACHE_SEEDS = (*SIM_SEEDS, PANEL_A_SEED)
PANEL_A_ANIMAL = "R050"

_MODE_PARAMS: dict[str, tuple[float | None, float | None]] = {
    "recovery": (None, None),
    "perfect": (0.0, 0.0),
}


def simulation_results_path(seed: int) -> Path:
    """Path to cached trial-level results for one RNG seed."""
    return data_dir() / f"figS2_simulations/simulation_results_seed_{seed}.parquet"


class LegacyFixationGrowthSimulator:
    """Post-hoc simulate Legacy fixation growth for one animal."""

    def __init__(
        self,
        tdf: pd.DataFrame,
        animal_id: str,
        wu_vr: float | None = None,
        pwu_vr: float | None = None,
        max_n_sessions: int = 90,
        experiment_name: str = "experiment",
        seed: int | None = None,
    ):
        if "days_relative_to_stage_5" not in tdf.columns:
            tdf = compute_days_relative_to_stage(tdf.copy(), stage=5)

        self.tdf = tdf
        self.animal_id = animal_id
        self.max_n_sessions = max_n_sessions
        self.experiment_name = experiment_name
        self.simulated_days_to_target = 0
        self.prev_session_dur = 0.010
        self.rng = default_rng(seed)

        (
            self.emperical_wu_vr_mean,
            self.emperical_wu_vr_var,
            self.emperical_pwu_vr_mean,
            self.emperical_pwu_vr_var,
            self.emperical_n_trial_mean,
            self.emperical_n_trial_var,
            self.emperical_n_days_to_target,
        ) = self._get_animal_parameters()

        self.wu_vr = self.emperical_wu_vr_mean if wu_vr is None else wu_vr
        self.pwu_vr = self.emperical_pwu_vr_mean if pwu_vr is None else pwu_vr

    def _get_animal_parameters(self) -> tuple[float, ...]:
        animal_df = self.tdf.query(
            "animal_id == @self.animal_id and stage in @config.GROWING_STAGES"
        ).copy()

        trial_summary = (
            animal_df.groupby(
                ["animal_id", "date", "days_relative_to_stage_5"], observed=True
            )
            .agg(n_trials=("trial", "nunique"))
            .reset_index()
        )

        warm_up_summary = (
            animal_df.query("warm_up_imp == True")
            .groupby(
                ["animal_id", "date", "days_relative_to_stage_5"], observed=True
            )
            .agg(warm_up_violation_rate=("violations", "mean"))
            .reset_index()
        )

        non_warm_up_summary = (
            animal_df.query("warm_up_imp == False")
            .groupby(
                ["animal_id", "date", "days_relative_to_stage_5"], observed=True
            )
            .agg(non_warm_up_violation_rate=("violations", "mean"))
            .reset_index()
        )

        session_summary = trial_summary.merge(
            warm_up_summary,
            on=["animal_id", "date", "days_relative_to_stage_5"],
            how="left",
        ).merge(
            non_warm_up_summary,
            on=["animal_id", "date", "days_relative_to_stage_5"],
            how="left",
        )

        pwu_vr_mean = session_summary["non_warm_up_violation_rate"].mean()
        pwu_vr_var = session_summary["non_warm_up_violation_rate"].var(ddof=1)
        if pd.isna(pwu_vr_var):
            pwu_vr_var = 0.0

        wu_vr_mean = session_summary["warm_up_violation_rate"].mean()
        wu_vr_var = session_summary["warm_up_violation_rate"].var(ddof=1)
        if pd.isna(wu_vr_var):
            wu_vr_var = 0.0

        n_trial_mean = session_summary["n_trials"].mean()
        n_trial_var = session_summary["n_trials"].var(ddof=1)
        if pd.isna(n_trial_var):
            n_trial_var = 0.0

        n_days_to_target = len(session_summary)

        return (
            wu_vr_mean,
            wu_vr_var,
            pwu_vr_mean,
            pwu_vr_var,
            n_trial_mean,
            n_trial_var,
            n_days_to_target,
        )

    @staticmethod
    def _mean_var_to_alpha_beta(mean: float, var: float) -> tuple[float, float]:
        if mean == 0:
            return 0.0, 0.0
        if mean >= 1.0:
            mean = 0.9999
        if mean <= 0.0:
            mean = 0.0001
        if var < 1e-6:
            var = 1e-6

        alpha = (mean**2) * ((1.0 - mean) / var - 1.0 / mean)
        beta = alpha * (1.0 / mean - 1.0)
        if alpha <= 0 or beta <= 0:
            alpha = max(mean * 100, 1e-3)
            beta = max((1 - mean) * 100, 1e-3)
        return alpha, beta

    def _sample_session_violation_rate(self, in_warm_up: bool) -> float:
        if in_warm_up:
            mean_vr = self.wu_vr
            var_vr = self.emperical_wu_vr_var
        else:
            mean_vr = self.pwu_vr
            var_vr = self.emperical_pwu_vr_var

        if mean_vr == 0:
            return 0.0

        alpha, beta = self._mean_var_to_alpha_beta(mean_vr, var_vr)
        if alpha == 0 and beta == 0:
            return 0.0
        return float(self.rng.beta(alpha, beta))

    def _sample_n_trials(self) -> int:
        sd = np.sqrt(self.emperical_n_trial_var)
        raw = self.rng.normal(self.emperical_n_trial_mean, sd)
        return max(int(round(raw)), 1)

    def _compute_warm_up_step(self) -> float:
        return (self.prev_session_dur - 0.01) / 20.0

    def _simulate_session(self) -> pd.DataFrame:
        n_trials = self._sample_n_trials()
        warm_up_on = self.prev_session_dur > 0.01
        warm_up_step = self._compute_warm_up_step() if warm_up_on else 0.0

        wuvr = self._sample_session_violation_rate(in_warm_up=True)
        pwuvr = self._sample_session_violation_rate(in_warm_up=False)

        fixation_dur = 0.01
        trial_data: dict[str, list] = {
            "session": [],
            "trial": [],
            "fixation_dur": [],
            "violation": [],
            "warm_up_on": [],
        }

        for t_idx in range(1, n_trials + 1):
            if warm_up_on and t_idx <= 20:
                p_violate = wuvr
                did_violate = self.rng.random() < p_violate
                if not did_violate:
                    fixation_dur += warm_up_step
            else:
                p_violate = pwuvr
                did_violate = self.rng.random() < p_violate
                if not did_violate:
                    grow_amount = max(0.001, 0.001 * fixation_dur)
                    fixation_dur += grow_amount

            trial_data["session"].append(self.simulated_days_to_target + 1)
            trial_data["trial"].append(t_idx)
            trial_data["fixation_dur"].append(fixation_dur)
            trial_data["violation"].append(did_violate)
            trial_data["warm_up_on"].append(warm_up_on and t_idx <= 20)

        self.simulated_days_to_target += 1
        self.prev_session_dur = fixation_dur
        return pd.DataFrame(trial_data)

    def run_simulation(self) -> tuple[pd.DataFrame, dict]:
        """Run until fixation >= 2 s or ``max_n_sessions`` reached."""
        all_sessions: list[pd.DataFrame] = []
        for _ in range(self.max_n_sessions):
            session_df = self._simulate_session()
            all_sessions.append(session_df)
            if self.prev_session_dur >= 2.0:
                break

        sim_data = pd.concat(all_sessions, ignore_index=True)
        summary = {
            "animal_id": self.animal_id,
            "wu_vr": self.wu_vr,
            "pwu_vr": self.pwu_vr,
            "emperical_pwu_vr_var": self.emperical_pwu_vr_var,
            "emperical_pwu_vr_mean": self.emperical_pwu_vr_mean,
            "emperical_wu_vr_var": self.emperical_wu_vr_var,
            "emperical_wu_vr_mean": self.emperical_wu_vr_mean,
            "emperical_days_to_target": self.emperical_n_days_to_target,
            "simulated_days_to_target": self.simulated_days_to_target,
            "experiment_name": self.experiment_name,
        }
        return sim_data, summary


def run_simulations_for_seed(
    tdf: pd.DataFrame,
    seed: int,
    animals: tuple[str, ...] | list[str] = LEGACY_SIM_ANIMALS,
) -> pd.DataFrame:
    """Run recovery + perfect simulations for all *animals* at one *seed*."""
    frames: list[pd.DataFrame] = []
    for animal_id in animals:
        for mode, (wu_vr, pwu_vr) in _MODE_PARAMS.items():
            sim = LegacyFixationGrowthSimulator(
                tdf,
                animal_id=animal_id,
                wu_vr=wu_vr,
                pwu_vr=pwu_vr,
                experiment_name=mode,
                seed=seed,
            )
            sim_data, summary = sim.run_simulation()
            sim_data = sim_data.assign(
                animal_id=animal_id,
                mode=mode,
                seed=seed,
                simulated_days_to_target=summary["simulated_days_to_target"],
            )
            frames.append(sim_data)
    return pd.concat(frames, ignore_index=True)


def load_or_run_simulation_seed(
    tdf: pd.DataFrame,
    seed: int,
    *,
    animals: tuple[str, ...] | list[str] = LEGACY_SIM_ANIMALS,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Load cached trial-level results for *seed*, or run and save."""
    path = simulation_results_path(seed)
    if path.exists() and not force_rerun:
        return pd.read_parquet(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    result = run_simulations_for_seed(tdf, seed, animals=animals)
    result.to_parquet(path, index=False)
    return result


def load_all_simulation_results(
    tdf: pd.DataFrame,
    seeds: range | list[int] = SIM_CACHE_SEEDS,
    *,
    animals: tuple[str, ...] | list[str] = LEGACY_SIM_ANIMALS,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Concat trial-level simulation results for all *seeds*."""
    parts = [
        load_or_run_simulation_seed(
            tdf, seed, animals=animals, force_rerun=force_rerun
        )
        for seed in seeds
    ]
    return pd.concat(parts, ignore_index=True)


def summarize_simulation_replicates(sim_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (animal_id, mode, seed) with days_to_target."""
    return (
        sim_df.groupby(["animal_id", "mode", "seed"], observed=True)[
            "simulated_days_to_target"
        ]
        .first()
        .reset_index(name="days_to_target")
    )


def mean_simulated_days_to_target(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Mean days_to_target across replicate seeds (0–29), per (animal_id, mode)."""
    replicates = summarize_simulation_replicates(sim_df)
    replicates = replicates[replicates["seed"].isin(SIM_SEEDS)]
    return (
        replicates.groupby(["animal_id", "mode"], observed=True)[
            "days_to_target"]
        .mean()
        .reset_index()
    )
