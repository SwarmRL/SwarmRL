"""
Main module for actions.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass
class Action:
    """
    Holds the 3 quantities that are applied to the colloid plus an identifier
    """

    id = 0
    force: float = 0.0
    torque: np.ndarray = None
    new_direction: np.ndarray = None
