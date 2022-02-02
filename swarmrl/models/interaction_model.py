"""
Model to compute external forces in an espresso simulation.
"""
import torch
import numpy as np
from typing import Union
import dataclasses


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

    def compute_state(self, colloid, other_colloids) -> Union[None, np.ndarray]:
        """
        Compute the state of the active learning algorithm.

        If the model is not an active learner this method is ignored.
        """

    def forward(self, colloids: torch.Tensor):
        """
        Perform the forward pass over the model.

        In this method, all other stages of the model should be called. In the case of
        a simple algebraic model, this should just be a call to compute_force.

        Parameters
        ----------
        colloids : torch.Tensor
                Tensor of colloids on which to operate. shape=(n_colloids, n_properties)
                where properties can very between test_models.

        Returns
        -------
        forces : np.ndarray
                Numpy array of forces to apply to the colloids. shape=(n_colloids, 3)
        """
        raise NotImplementedError("Implemented in the child classes.")
