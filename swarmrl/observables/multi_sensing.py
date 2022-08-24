"""
Class for an observable which computes several observables.
"""
from abc import ABC
from typing import List

import numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable


class MultiSensing(Observable, ABC):
    """
    Takes several observables and returns them as a list of observables.
    """

    def __init__(
        self,
        observables: List[Observable],
        source: np.ndarray,
        decay_fn: callable,
        box_length: np.ndarray,
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
        source : np.ndarray
                Source of the field.
        decay_fn : callable
                Decay function of the field.
        box_length : np.ndarray
                Array for scaling of the distances.
        """
        self.observables = observables
        self.source = (source,)
        self.decay_fn = (decay_fn,)
        self.box_length = (box_length,)
        self.historic_positions = {}

    def init_task(self) -> Observable:
        """
        Prepare the task for running.

        Returns
        -------
        observable :
                Returns the observable required for the task.
        """

        observable = []
        for item in self.observables:
            observable.append(item.init_task())
        return observable

    def initialize(self, colloids: List[Colloid]):
        """
        Initialize the observables with starting positions of the colloids.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids with which to initialize the observable.

        Returns
        -------
        Updates the class state.
        """
        for item in self.observables:
            item.initialize(colloids)

    def compute_observable(
        self, colloid: Colloid, other_colloids: List[Colloid]
    ) -> List[Observable]:
        """
        Computes vision cone for all colloids

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
