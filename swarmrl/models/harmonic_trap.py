"""
Example force computation.
"""

from abc import ABC

import jax.numpy as np

from swarmrl.models.interaction_model import InteractionModel


class HarmonicTrap(InteractionModel, ABC):
    """
    Class for the harmonic trap potential.
    """

    def __init__(
        self, stiffness: float, center: np.ndarray = np.array([0.0, 0.0, 0.0])
    ):
        """
        Constructor for the Harmonic trap interaction rule.

        Parameters
        ----------
        stiffness : float
                Stiffness of the interaction potential.
        center : np.ndarray
                Center of the potential. The force is computed based on this distance.
        """
        super(HarmonicTrap, self).__init__()
        self.stiffness = stiffness
        self.center = center

    def compute_force(self, colloids: np.ndarray) -> np.ndarray:
        """
        Compute the forces on the colloids.

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
        return -self.stiffness * (colloids - self.center)

    def __call__(self, colloids: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        """
        Perform the forward pass over the model.

        Simply call compute forces and return the values.

        Parameters
        ----------
        colloids : tf.Tensor
                Tensor of colloids on which to operate. shape=(n_colloids, n_properties)
                where properties can very between test_models.
        state : torch.Tensor
                State of the system on which a reward may be computed. Defaults to None
                to allow for non-NN models.

        Returns
        -------
        forces : np.ndarray
                Numpy array of forces to apply to the colloids. shape=(n_colloids, 3)
        """
        return self.compute_force(colloids)
