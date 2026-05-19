"""Dataclass container for off-policy transitions."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Transition:
    """Single transition tuple for off-policy RL."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool


# TODO: split done in terminated and truncated for correct state value computation.
# terminated -> value = 0
# truncated -> value = as is...
# $$\text{Target} = r + \gamma \cdot (1 - \text{terminated}) \cdot \max_a Q(s', a)$$
