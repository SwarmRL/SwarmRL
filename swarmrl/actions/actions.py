"""
Main module for actions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Action:
    """
    Holds the 3 quantities that are applied to the colloid plus an identifier
    """

    id = 0
    force: float = 0.0
    torque: np.ndarray = None
    new_direction: np.ndarray = None
