"""
Class for an observable which computes several observables.
"""
from abc import ABC
from typing import List

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable


class MultiSensing(Observable, ABC):
    """
    Takes several observables and returns them as a list of observables.
    """

    def __init__(
        self,
        observables: List[Observable],
    ):
        """
        Constructor for the observable.

        In this observables, the order with which the observables are
        passed to the constructor is the order in which they are
        concatenated.

        Parameters
        ----------
        Observables : List[Observable]
                List of observables.
        """
        self.observables = observables

    def initialize(self, colloids: List[Colloid]):
        """
        Initialize the observables as needed.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids with which to initialize the observable.

        Returns
        -------
        Some of the observables passed to the constructor might need to be
        initialized with the positions of the colloids. This method does
        that.
        """
        for item in self.observables:
            item.initialize(colloids)

    def compute_observable(self, colloids: List[Colloid]) -> List:
        """
        Computes all observables and returns them in a concatenated list.

        Parameters
        ----------
        colloids : list of all colloids.

        Returns
        -------
        List of observables, computed in the order that they were given
        at initialization.
        """
        # Get the observables for each colloid.
        unshaped_observable = []  # shape (n_obs, n_colloids, ...)
        for item in self.observables:
            unshaped_observable.append(item.compute_observable(colloids))

        n_colloids = len(unshaped_observable[0])

        # Reshape the observables to be (n_colloids, n_observables, )
        observable = [[] for _ in range(n_colloids)]
        for i, item in enumerate(unshaped_observable):
            for j, colloid in enumerate(item):
                observable[j].append(colloid)

        return observable
