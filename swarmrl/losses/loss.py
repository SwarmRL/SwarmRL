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
        actor: Network,
        critic: Network,
        episode_data: np.ndarray,
    ):
        """
        Compute loss on models.

        Parameters
        ----------
        actor : Network
                Actor network to train
        critic : Network
                Critic network to train.
        episode_data : dict
                A dictionary of episode data.
        """
        raise NotImplementedError("Implemented in child class.")
