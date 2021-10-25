"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
from swarmrl.interaction_model import InteractionModel
import torch
import numpy as np


class MLPRL(InteractionModel):
    """
    Class for the simple MLP RL implementation.

    The multi-layer perceptron learner is a simple global network model wherein all
    particles are passed into the network and forces computed for each individually.
    """
    def __init__(self):
        """
        Constructor for the MLP RL.
        """
        pass

    def compute_force(self, colloids: torch.Tensor) -> np.ndarray:
        """
        Compute the force on all of the particles with the newest model.

        Parameters
        ----------
        colloids : tf.Tensor
                Tensor of colloids on which to operate. shape=(n_colloids, n_properties)
                where properties can very between test_models.

        Returns
        -------
        forces : np.ndarray
                Numpy array of forces to apply to the colloids. shape=(n_colloids, 3)
        """
        pass

    def update_network(self):
        """
        Perform an update on the network.

        Returns
        -------

        """