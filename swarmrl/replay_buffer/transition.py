"""Dataclass container for off-policy transitions."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Transition:
    """
    Single transition tuple for off-policy RL.
    Distinguishes between terminated and truncated endings for
    correct state value computation.
    # Target = r + gamma * (1 - terminated) * max_a Q(s', a)

    """

    observation: np.ndarray  # s_t
    action: np.ndarray  # a_t
    reward: float  # r_t
    next_observation: np.ndarray  # s_{t+1}
    terminated: float  # 1.0 if real end (killed, goal reached), else 0.0
