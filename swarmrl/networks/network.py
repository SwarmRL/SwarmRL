"""
Parent class for the networks.
"""

from typing import List

import jax.numpy as np
from flax.core.frozen_dict import FrozenDict

from swarmrl.components.colloid import Colloid


class Network:
    """
    A parent class for the networks that will be used.
    """

    def compute_action(self, observables: List[Colloid], explore_mode: bool = False):
        """
        Compute and action from the action space.

        This method computes an action on all colloids of the relevent type.

        Parameters
        ----------
        observables : List[Colloid]
                Colloids in the system for which the action should be computed.
        explore_mode : bool
                If true, an exploration vs exploitation function is called.

        Returns
        -------
        action : int
                An integer bounded between 0 and the number of output neurons
                corresponding to the action chosen by the agent.
        """
        raise NotImplementedError("Implemented in child class.")

    def __call__(self, params: FrozenDict, feature_vector: np.ndarray):
        """
        Perform the forward pass on the model. This method is
        used in the update. It uses a vmapped version of the
        model.apply function.

        Parameters
        ----------
        params : FrozenDict
                Parameters of the model.
        feature_vector : np.ndarray
                Current state of the agent on which actions should be made.

        Returns
        -------

        """
        raise NotImplementedError("Implemented in child class.")

    def export_model(self, filename: str = "models", directory: str = "Models"):
        """
        Export the model state to a directory.

        Parameters
        ----------
        filename : str (default=models)
                Name of the file the models are saved in.
        directory : str (default=Models)
                Directory in which to save the models. If the directory is not
                in the currently directory, it will be created.

        """
        raise NotImplementedError("Implemented in child class")

    def restore_model_state(self, filename, directory):
        """
        Restore the model state from a file.

        Parameters
        ----------
        filename : str
                Name of the model state file.
        directory : str
                Path to the model state file.

        Returns
        -------
        Updates the model state.
        """
        raise NotImplementedError("Implemented in child class")

    def update_model(
        self,
        grads,
    ):
        """
        Train the model.

        For jax model grads are used to update a model state directly. This method
        takes the grads and updates the params dict corresponding to the relevant
        model.

        Parameters
        ----------
        grads : dict
                Dict of grads from a jax value_and_grad call.
        """
        raise NotImplementedError("Implemented in child class.")
