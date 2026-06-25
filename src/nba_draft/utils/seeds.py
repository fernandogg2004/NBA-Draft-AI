"""Reproducibility: a single place to seed all sources of randomness."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and the hash seed for reproducible runs.

    Model libraries (XGBoost, PyMC, etc.) take their own seeds at construction time;
    those are passed explicitly from config rather than relied upon globally.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
