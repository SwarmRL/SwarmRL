"""
Class for Resobee's swarm task.
"""

from typing import List

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.components.colloid import Colloid
from swarmrl.tasks.task import Task



class ResoBeeSwarm(Task):
    """
    Collective Motion.
    """

    def __init__(
        self,
        box_length: np.ndarray,
        particle_type: int = 0,
        repulsion_radius: float=1,
        alignment_radius: float=1,
        desired_speed:float =1,
        
        

    ):
        """
        Constructor for the ResoBee swarm task.

        Parameters
        ----------
        partition : bool (default=True)
                Whether to partition the reward by particle contribution.
        particle_type : int (default=0)
                Type of particle receiving the reward.

        """
        self.particle_type=particle_type
        self.repulsion_radius = repulsion_radius
        self.alignment_radius = alignment_radius
        self.desired_speed = desired_speed
        self.box_length = box_length
        super().__init__(particle_type=particle_type)
        

    def initialize(self, colloids: List[Colloid]):
        """
        Prepare the task for running.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids to be used in the task.

        Returns
        -------
        Updates the class state.
        """
        pass


    #only used for repulstion, rename, remove obs
    def eval_function_distance_threshold(self,
                                        func: callable,
                                        position: np.ndarray,
                                        observable: np.ndarray,
                                        threshold: int = 1) -> np.ndarray:
        pairwise_euclidean = lambda x: np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
        @jax.jit
        def distance_and_func(x):
            distances = pairwise_euclidean(x)
            return np.where((distances < threshold) & (distances > 0), func(distances), 0)
        result = distance_and_func(position)
        chunk_size = np.shape(position)[0]
        return np.sum(result.reshape((chunk_size, chunk_size)), axis=1)


    def __call__(self, colloids: List[Colloid]):
        """
        Compute the reward.

        In this case of this task, the observable itself is the gradient of the field
        that the colloid is swimming in. Therefore, the change is simply scaled and
        returned.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of colloids to be used in the task.

        Returns
        -------
        rewards : List[float] (n_colloids, )
                Rewards for each colloid.
        """
        # Collect the important data.

        chosen_colloids = [
            colloid for colloid in colloids if colloid.type == self.particle_type
        ]
        colloid_positions = np.array([colloid.pos for colloid in chosen_colloids])
        colloid_velocities = np.array([colloid.velocity for colloid in chosen_colloids])
        alignment = lambda v_i, v_j: np.dot(v_i,v_j)
        repulsion = lambda dist: dist/24.0-1/24.0
        reward_homing = -np.linalg.norm(colloid_positions/self.box_length-0.5,axis = -1)
        # reward_alignment = 0
        # reward_friction = -np.power(np.linalg.norm(colloid_velocities,axis=1)-self.desired_speed,2)/self.desired_speed
        # reward_repulsion = self.eval_function_distance_threshold(repulsion,colloid_positions,colloid_positions, self.repulsion_radius)
        

        return reward_homing#+reward_alignment+reward_friction+reward_repulsion
