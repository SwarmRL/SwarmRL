"""Dataclass container for off-policy transitions."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """Single transition tuple for off-policy RL."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
