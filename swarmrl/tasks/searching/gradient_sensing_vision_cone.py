"""
Run and tumble task

This task uses the change in the gradient to determine whether a move as good or not.
It also rewards other particles in the frontal vision cone to incentivise constructing a
circle.

Notes
-----
Requires a warm up step.
"""
from abc import ABC

import jax.numpy as np

from swarmrl.observables.concentration_field import ConcentrationField
from swarmrl.observables.multi_sensing import MultiSensing
from swarmrl.observables.vision_cone import VisionCone
from swarmrl.tasks.task import Task


class GradientSensingVisionCone(Task, ABC):
    """
    Find a location in a box using the gradient and the vision cone.
    """

    def __init__(
        self,
        source: np.ndarray = np.array([0, 0, 0]),
        decay_function: callable = None,
        box_size: np.ndarray = np.array([1.0, 1.0, 0.0]),
        grad_reward_scale_factor: int = 10,
        cone_reward_scale_factor: float = 0.5,
        vision_angle: int = 60,
        detect_source: bool = True,
        return_cone: bool = False,
        vision_direction: complex = complex(0, 1),
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
        grad_reward_scale_factor : int (default=10)
                The amount the field is scaled by to get the reward.
        cone_reward_scale_factor : float (default=0.5)
                The cone reward is cone_reward_scale_factor / (mean distance to other
                particles in vision cone).
        vision_angle : int (default = 60)
                Measurement how big the vision cone is.
        detect_source : bool (default = True)
                Whether the source should be detected in the vision cone. If true and
                the source is in the vision cone,the vision cone returns a number alpha
                close to 0 such that the reward = cone_reward_scale_factor / alpha >> 1.
                This incentivises the particle to move towards the source.
        return_cone : bool (default = False)
                Whether the entire cone should be returned (True) or only the mean
                distance of particles within the cone.
        vision_direction : complex (default = complex(0,1))
                Direction which the vision cone points to. Default is front,
                complex(1,0) would be right, ...
        """

        self.source = source
        self.decay_fn = decay_function
        self.box_size = box_size
        self.grad_reward_scale_factor = grad_reward_scale_factor
        self.cone_reward_scale_factor = cone_reward_scale_factor
        self.return_cone = return_cone
        self.detect_source = detect_source
        self.vision_angle = vision_angle
        self.vision_range = np.inf  # TODO: np.linalg.norm(box_size) / 4
        self.vision_direction = vision_direction

    def init_task(self):
        """
        Prepare the task for running.

        Returns
        -------
        Multisensing : MultiSensing
                Returns a collection of multiple observables.
        """
        concentration = ConcentrationField(self.source, self.decay_fn, self.box_size)
        vision_cone = VisionCone(
            vision_direction=self.vision_direction,
            vision_angle=self.vision_angle,
            vision_range=self.vision_range,
            return_cone=self.return_cone,
            source=self.source,
            detect_source=self.detect_source,
            box_size=self.box_size,
        )

        return MultiSensing(observables=[concentration, vision_cone])

    def change_source(self, new_source: np.ndarray):
        """
        Changes the concentration field source.

        Parameters
        ----------
        new_source : np.ndarray
                Coordinates of the new source.

        Returns
        -------
        Multisensing : MultiSensing
                Returns a collection of multiple observables. Similar output as
                init_task.
        """
        self.source = new_source
        concentration = ConcentrationField(self.source, self.decay_fn, self.box_size)
        vision_cone = VisionCone(
            vision_direction=self.vision_direction,
            vision_angle=self.vision_angle,
            vision_range=self.vision_range,
            return_cone=self.return_cone,
            source=self.source,
            detect_source=self.detect_source,
            box_size=self.box_size,
        )

        return MultiSensing(observables=[concentration, vision_cone])

    def __call__(
        self,
        observable: np.ndarray,
        colloid: object,
        colloids: list,
        other_colloids: list,
    ) -> float:
        """
        Compute the reward.

        For the concentration gradient reward, the observable itself is the gradient of
        the field that the colloid is swimming in. Therefore, the change is simply
        scaled and returned.

        For the vision cone, the reward is a scale factor over the mean distance to all
        particles in the vision cone.

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
        -------
        total_reward : jnp.array
                Reward for the colloid at the current state.
        """
        gradient_sensing_reward = self.grad_reward_scale_factor * np.clip(
            observable[0], 0, None
        )

        if observable[1] == 0.0:
            cone_reward = 0.0
        else:
            cone_reward = np.clip(
                -self.cone_reward_scale_factor * np.log(observable[1]), 0, None
            )

        total_reward = gradient_sensing_reward + cone_reward

        return total_reward
