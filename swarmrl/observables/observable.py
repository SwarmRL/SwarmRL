"""
Parent class for the observable.
"""
from typing import List
import sys
sys.path.append('swarmrl/observables')
sys.path.append('swarmrl/models')
from interaction_model import Colloid


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

    def compute_observable(self, colloid: object, other_colloids: list):
        """
        Compute the current state observable.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.
        other_colloids
                Other colloids in the system.

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
