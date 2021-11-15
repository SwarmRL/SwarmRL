"""
Task of identifying the origin of a box.

The reward in this case is computed based on the center of mass of the particles along
with their radius of gyration. The idea being that they should form a ring around the
(0, 0, 0) coordinate of the box and that ring should be tightly formed. The radius of
gyration reward may be adjusted such that the colloids form a tighter or wider ring.
"""
from abc import ABC
from swarmrl.tasks.task import Task


class FindOrigin(Task, ABC):
    """
    Find the origin of a box.
    """
    def compute_radius_of_gyration_reward(self):
        """
        Compute the radius of gyration PENALTY/Reward on the colloids.
        Returns
        -------

        """
        pass

    def compute_center_of_mass_reward(self):
        """
        Compute the center of mass reward for the colloids.

        Returns
        -------

        """
    def compute_reward(self):
        """
        Compute the reward on the whole group of particles.

        Returns
        -------

        """


