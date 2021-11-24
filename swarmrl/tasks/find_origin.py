"""
Task of identifying the origin of a box.

The reward in this case is computed based on the center of mass of the particles along
with their radius of gyration. The idea being that they should form a ring around the
(0, 0, 0) coordinate of the box and that ring should be tightly formed. The radius of
gyration reward may be adjusted such that the colloids form a tighter or wider ring.
"""
from abc import ABC
from swarmrl.tasks.task import Task
import numpy as np
from swarmrl.engine.engine import Engine


class FindOrigin(Task, ABC):
    """
    Find the origin of a box.
    """

    com: list
    r_g: list

    def __init__(
        self,
        engine: Engine,
        origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        r_g_goal: float = 1.0,
        memory: bool = False,
    ):
        """
        Constructor for the find origin task.

        loss = alpha * L_single + beta * L_com + gamma * L_rg

        Parameters
        ----------
        engine : Engine
                SwarmRL engine used to generate environment data. Could be, e.g. a
                simulation or an experiment.
        origin : np.ndarray
                The desired origin.
        r_g_goal : float
                The goal for the radius of gyration.
        alpha
        beta
        gamma
        memory : bool (default = False)
                If true, include an improvement factor in the reward.
        """
        super(FindOrigin, self).__init__(engine)
        self.origin = origin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.memory = memory
        self.r_g_goal = r_g_goal

    def compute_center_of_mass_reward(self, positions: np.ndarray):
        """
        Compute the center of mass reward for the colloids.

        Parameters
        ----------
        positions : np.ndarray
                A numpy array of positions of the shape (n_particles, 3)
        Returns
        -------
        com_vector: np.ndarray
                A center of mass vector.

        Notes
        -----
        Assume here a mass of 1.

        TODO: Remove assumption
        """
        total_mass = np.shape(positions)[0]
        summed_position = np.sum(positions, axis=0)

        com = summed_position / total_mass

        reward = 1 / np.linalg.norm(com - self.origin)

        return com, self.beta * reward

    def compute_radius_of_gyration_reward(self, positions: np.ndarray, com: np.ndarray):
        """
        Compute the radius of gyration reward for the colloids.

        Returns
        -------

        """
        averaging_factor = np.shape(positions)[0]
        shifted_positions = np.sum(np.linalg.norm(positions - com, axis=1) ** 2, axis=0)

        r_g = shifted_positions / averaging_factor

        reward = 1 / abs(r_g - self.r_g_goal)

        return r_g, self.gamma * reward

    def compute_particle_reward(self, colloid: object):
        """
        Compute the reward for the individual particle.

        Parameters
        ----------
        colloid : object
                Colloid for which the reward is being computed.

        Returns
        -------
        reciprocal_distance : np.ndarray
                reciprocal distance from (0, 0, 0) acting as a reward for reducing this
                value.
        """
        distance = np.linalg.norm(colloid.pos - self.origin)

        return self.alpha * 1 / distance

    def compute_reward(self, colloid: object):
        """
        Compute the reward on the whole group of particles.

        Parameters
        ----------
        colloid : object
                Colloid for which the reward is being computed.


        Returns
        -------

        """
        # {"Unwrapped_Positions": (n, 3), "Velocities": (n, 3), "Directors": (n, 3)}
        colloid_data = self.engine.get_particle_data()

        com, com_reward = self.compute_center_of_mass_reward(
            colloid_data["Unwrapped_Positions"]
        )
        r_g, r_g_reward = self.compute_radius_of_gyration_reward(
            colloid_data["Unwrapped_Positions"], com
        )
        single_reward = self.compute_particle_reward(colloid)

        self.com.append(com)
        self.r_g.append(r_g)

        reward = single_reward + com_reward + r_g_reward

        return reward
