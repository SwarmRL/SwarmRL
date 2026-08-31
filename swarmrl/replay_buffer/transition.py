"""Dataclass container for off-policy transitions."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Transition:
    """
    Single transition tuple for off-policy RL.

    `terminated` marks true terminal states where bootstrapped target values
    should be masked. `truncated` marks externally cut episodes such as
    time limits, which end rollouts but still allow bootstrapping.
    """

    observation: np.ndarray  # s_t
    action: np.ndarray  # a_t
    reward: float  # r_t
    next_observation: np.ndarray  # s_{t+1}
    terminated: float  # 1.0 if real end (killed, goal reached), else 0.0
    truncated: float = 0.0  # 1.0 if episode ended by truncation, else 0.0
