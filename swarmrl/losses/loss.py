"""
Module for the loss parent class.
"""
import jax.numpy as np

from swarmrl.networks.network import Network


class Loss:
    """
    Parent class for a SwarmRL loss model.
    """

    def compute_loss(
        self,
        network: Network,
        episode_data: np.ndarray,
    ):
        """
        Compute loss on models.

        Parameters
        ----------
        network : Network
                Actor-critic network.
        episode_data : dict
                A dictionary of episode data.
        """
        raise NotImplementedError("Implemented in child class.")
