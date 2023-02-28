"""
Parent class for the observable.
"""
from typing import List

from swarmrl.models.interaction_model import Colloid


class Observable:
    """
    Parent class for observables.

    Observables act as inputs to the neural networks.
    """

    _observable_shape: tuple

    def initialize(self, colloids: List[Colloid]):
        """
        Initialize the observable with starting positions of the colloids.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids with which to initialize the observable.

        Returns
        -------
        Updates the class state.
        """
        raise NotImplementedError("Implemented in child class.")

    def compute_observable(self, colloids: List[Colloid]):
        """
        Compute the current state observable.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of all colloids in the system.

        Returns
        -------
        observables : List[np.ndarray] (n_colloids, dimension)
                List of observables, one for each colloid.

        """
        raise NotImplementedError("Implemented in child class.")

    @property
    def observable_shape(self):
        """
        Unchangeable shape of the observable.
        Returns
        -------

        """
        return self._observable_shape
