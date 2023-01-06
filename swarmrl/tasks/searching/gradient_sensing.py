"""
Run and tumble task

This task uses the change in the gradient to determine whether a move was good or not.

Notes
-----
Requires a warm up step.
"""
from abc import ABC

import jax.numpy as np

from swarmrl.observables.concentration_field import ConcentrationField
from swarmrl.tasks.task import Task


class GradientSensing(Task, ABC):
    """
    Find a location in a box using distances.
    """

    def __init__(
        self,
        source: np.ndarray = np.array([0, 0, 0]),
        decay_function: callable = None,
        box_size: np.ndarray = np.array([1.0, 1.0, 0.0]),
        reward_scale_factor: int = 10,
    ):
        """
        Constructor for the find origin task.

        Parameters
        ----------
        source : np.ndarray (default = (0, 0 0))
                Source of the gradient.
        decay_function : callable (required=True)
                A function that describes the decay of the field along one dimension.
                This cannot be left None. The function should take a distance from the
                source and return the magnitude of the field at this point.
        box_size : np.ndarray
                Side length of the box.
        reward_scale_factor : int (default=10)
                The amount the field is scaled by to get the reward.

        """
        super(GradientSensing, self).__init__()
        self.source = source
        self.decay_fn = decay_function
        self.reward_scale_factor = reward_scale_factor
        self.box_size = box_size

    def init_task(self):
        """
        Prepare the task for running.

        Returns
        -------
        observable :
                Returns the observable required for the task.
        """
        return ConcentrationField(self.source, self.decay_fn, self.box_size)

    def change_source(self, new_source: np.ndarray):
        """
        Changes the concentration field source.

        Parameters
        ----------
        new_source : np.ndarray
                Coordinates of the new source.

        Returns
        -------
        observable :
                Returns the observable required for the task.
        """
        self.source = new_source
        return ConcentrationField(self.source, self.decay_fn, self.box_size)

    def __call__(
        self,
        observable: np.ndarray,
        colloid: object,
        colloids: list,
        other_colloids: list,
    ) -> float:
        """
        Compute the reward.

        In this case of this task, the observable itself is the gradient of the field
        that the colloid is swimming in. Therefore, the change is simply scaled and
        returned.

        Parameters
        ----------
        observable : np.ndarray (dimension, )
                Observable of a single colloid.
        colloid : object
                The colloid, the reward should be computed for.
        colloids : list
                A list of all colloids in the system.
        other_colloids : list
                A list of all other colloids in the system.
        Returns
        ----------
        reward : float
                Reward for the colloid at the current state.
        """
        return self.reward_scale_factor * np.clip(observable[0], 0, None)
