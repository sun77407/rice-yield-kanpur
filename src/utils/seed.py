"""Global seed pinning for reproducibility (see CLAUDE_CODE_PROMPT.md rule 5)."""

from __future__ import annotations

import os
import random

import numpy as np

GLOBAL_SEED: int = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> int:
    """Pin every RNG used in the project and return the seed applied.

    Covers Python ``random``, NumPy, and ``PYTHONHASHSEED``. Libraries that take
    an explicit seed (XGBoost, LightGBM, Optuna, PyMC) must be passed
    ``GLOBAL_SEED`` by the caller.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
