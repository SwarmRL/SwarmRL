"""
Example force computation.
"""

import jax.numpy as np

from swarmrl.agents.classical_agent import ClassicalAgent


class HarmonicTrap(ClassicalAgent):
    """
    Class for the harmonic trap potential agent.
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

    def compute_agent_action(self, colloids: np.ndarray) -> np.ndarray:
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
