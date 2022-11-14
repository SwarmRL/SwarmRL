"""
Class for an observable which computes several observables.
"""
from abc import ABC
from typing import List

import jax.numpy as np

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
        IMPORTANT: ALWAYS KEEP ORDER.
        1. concentration_field
        2. vision_cone

        Parameters
        ----------
        Observables : List[Observable]
                List of observables. Their order is extremely important!
        """
        self.observables = observables

    def initialize(self, colloids: List[Colloid]):
        """
        Initialize the observables with starting positions of the colloids.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids with which to initialize the observable.

        Returns
        -------
        Initialisation step used to generate historic positions for the concentration
        field observable to calculate change in gradient.
        """
        for item in self.observables:
            item.initialize(colloids)

    def compute_observable(
        self, colloid: Colloid, other_colloids: List[Colloid]
    ) -> List[Observable]:
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
        observable = []
        for item in self.observables:
            observable.append(
                item.compute_observable(colloid=colloid, other_colloids=other_colloids)
            )

        return np.concatenate(observable, axis=-1)
