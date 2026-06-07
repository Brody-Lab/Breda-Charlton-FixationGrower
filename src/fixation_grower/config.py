"""Experiment metadata, cohort definitions, and aesthetic parameters.

Terminology:
- Legacy   : the standard fixation curriculum used previously (formerly "V1")
- FixGrower: the novel adaptive curriculum introduced in this work (formerly "V2")
"""

from datetime import datetime

import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Animal cohorts
# ---------------------------------------------------------------------------

ANIMALS = [
    "R040", "R041", "R042", "R043", "R044", "R045", "R046", "R047",
    "R048", "R049", "R050", "R051", "R052", "R053", "R054", "R055",
    "R056", "R057",
]

GROUP_1_ANIMALS = ["R040", "R041", "R042", "R043", "R044", "R045", "R046", "R047"]
GROUP_2_ANIMALS = ["R048", "R049", "R050", "R051", "R052", "R053", "R054", "R055", "R056", "R057"]

LEGACY_ANIMALS = ["R040", "R042", "R044", "R046", "R048", "R050", "R052", "R054", "R056"]
FIXGROWER_ANIMALS = ["R041", "R043", "R045", "R047", "R049", "R051", "R053", "R055", "R057"]

# Representative example animals for figure panels
LEGACY_DEMO_ANIMAL = "R042"
FIXGROWER_DEMO_ANIMAL = "R043"

MICE = ["R512", "R513", "R621", "R622", "R623"]

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

START_DATES = {"group_1": "2024-07-20", "group_2": "2024-07-27"}
DATE_MIN = START_DATES["group_1"]
DATE_MAX = "2024-10-17"


def get_start_date(group: str, type: str = "datetime"):
    """Return start date of a cohort subgroup as datetime or string."""
    start_date = START_DATES[group]
    if type == "datetime":
        return datetime.strptime(start_date, "%Y-%m-%d")
    return start_date


DATE_DROPS = {
    "R042": pd.Timestamp("2024-07-29").date(),
    "R043": [
        pd.Timestamp("2024-07-26").date(),
        pd.Timestamp("2024-07-27").date(),
        pd.Timestamp("2024-07-28").date(),
        pd.Timestamp("2024-07-29").date(),
    ],
    "R046": pd.Timestamp("2024-08-12").date(),
    "R052": pd.Timestamp("2024-08-14").date(),
    "R051": [
        pd.Timestamp("2024-08-20").date(),
        pd.Timestamp("2024-08-21").date(),
        pd.Timestamp("2024-08-22").date(),
    ],
}

DROP_COLUMNS = [
    "n_incorr_spokes_during_give_del", "sa", "sb", "stimuli_on",
    "pre_dur", "adj_pre_dur", "stimulus_dur", "post_dur", "sb_extra_dur",
    "give_delay_dur", "give_xtra_light_delay_dur", "give_use",
    "give_del_growth_type", "give_del_adagrow_trial_subset",
    "give_del_adagrow_alpha_plus", "give_del_adagrow_alpha_minus",
    "give_del_adagrow_threshold", "give_del_adagrow_subset_prev_perf",
    "give_del_adagrow_step_size", "give_del_adagrow_window_size",
    "stim_set", "block_switch_type", "give_delay_strict",
    "volume_multiplier", "sound_pair",
]

# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------

HUE_ORDER_ANIMALS = LEGACY_ANIMALS + FIXGROWER_ANIMALS
HUE_ORDER_EXP = ["Legacy", "FixGrower"]

LEGACY_COLOR = "#752F8F"
FIXGROWER_COLOR = "#56B4E9"
EXP_PALETTE = [LEGACY_COLOR, FIXGROWER_COLOR]

LEGACY_PALETTE = sns.color_palette("RdPu", len(LEGACY_ANIMALS))
FIXGROWER_PALETTE = sns.color_palette("YlGnBu", len(FIXGROWER_ANIMALS))

ANIMAL_PALETTE = {
    **{animal: color for animal, color in zip(LEGACY_ANIMALS, LEGACY_PALETTE)},
    **{animal: color for animal, color in zip(FIXGROWER_ANIMALS, FIXGROWER_PALETTE)},
}

MOUSE_COLOR = "darkblue"
SB_COLOR = "teal"

VALID_COLOR = "#079d4b"
VIOLATION_COLOR = "#e1c63c"
HIT_COLOR = "lime"
N_TRIAL_COLOR = "purple"

LEFT_COLOR = "salmon"
RIGHT_COLOR = "cornflowerblue"
MULTI_CHOICE_PAL = [LEFT_COLOR, RIGHT_COLOR, VIOLATION_COLOR]
VALID_PALETTE = [VIOLATION_COLOR, VALID_COLOR]

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

PROBE_STAGES = [9, 10]
GROWING_STAGES = [5, 6, 7]
SPOKE_STAGES = [1, 2, 3, 4]

# Excluded from Fig. 3 Panel C mixed-effects model (Methods: Excluded Animals)
PROBE_OUTLIER = "R047"
