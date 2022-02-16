"""
Model to compute external forces in an espresso simulation.
"""
import dataclasses
from typing import Union

import numpy as np
import torch


@dataclasses.dataclass
class Action:
    """
    Holds the 3 quantities that are applied to the colloid plus and identifier
    """

    id = 0
    force: float = 0.0
    torque: np.ndarray = np.zeros((3,))
    new_direction: np.ndarray = None


class InteractionModel(torch.nn.Module):
    """
    Parent class to compute external forces on colloids in a simulation.
    Inherits from the module class of Torch.
    """

    def calc_action(self, colloid, other_colloids) -> Action:
        """
        Compute the next action on colloid.

        Parameters
        ----------
        colloid : object
                Colloid for which an action is being computed
        other_colloids
                Other colloids in the system.

        Returns
        -------
        The action
        """
        raise NotImplementedError("Interaction models must define a calc_action method")
