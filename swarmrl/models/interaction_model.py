"""
Model to compute external forces in an espresso simulation.
"""

import dataclasses
import typing

import numpy as np

from swarmrl.components.colloid import Colloid


@dataclasses.dataclass
class Action:
    """
    Holds the 3 quantities that are applied to the colloid plus an identifier
    """

    id = 0
    force: float = 0.0
    torque: np.ndarray = np.zeros((3,))
    new_direction: np.ndarray = None


class InteractionModel:
    """
    Parent class to compute external forces on colloids in a simulation.
    Inherits from the module class of Torch.
    """

    _kill_switch: bool = False

    @property
    def kill_switch(self):
        """
        If true, kill the simulation.
        """
        return self._kill_switch

    @kill_switch.setter
    def kill_switch(self, value):
        """
        Set the kill switch.
        """
        self._kill_switch = value

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Compute the next action on colloid.

        Parameters
        ----------
        colloids : list
                List of all Colloids for which an action is being computed

        Returns
        -------
        action : list
                List of Actions for all Colloids in the same order as input colloids
        """
        raise NotImplementedError("Interaction models must define a calc_action method")
