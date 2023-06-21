"""
Class for rod rotation task.
"""
from typing import List

import jax
import jax.numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.task import Task


class DrugDealing(Task):
    """
    Rotate a rod.
    """

    def __init__(
        self,
        destination=np.array([800, 800, 0]),
        box_size=np.array([1000, 1000, 1000]),
        particle_type: int = 0,
        drug_type: int = 1,
    ):
        super().__init__(particle_type=particle_type)
        self.destination = destination / box_size
        self.drug_type = drug_type
        self.decay_fn = jax.jit(lambda x: 1 - x)
        self.hist_drug_pos = None
        self.historic_position = None
        self.box_size = box_size
        self.alpha = [1000, 18, 100]
        self.beta = [1000, 10, 100]

    def initialize(self, colloids: List[Colloid]):
        for colloid in colloids:
            if colloid.type == self.drug_type:
                self.hist_drug_pos = colloid.pos / self.box_size
            else:
                self.historic_position = (
                    np.array(
                        [
                            colloid.pos
                            for colloid in colloids
                            if colloid.type is self.particle_type
                        ]
                    )
                    / self.box_size
                )
        pass

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
        # new drug - destination distance
        drug = [colloid for colloid in colloids if colloid.type == self.drug_type]
        drug_position = np.array([colloid.pos for colloid in drug]) / self.box_size
        new_drug_dest_distance = np.linalg.norm(
            drug_position - self.destination, axis=1
        )

        # old drug - destination distance
        old_drug_dest_distance = np.linalg.norm(self.hist_drug_pos - self.destination)

        # delta drug - destination distance
        delta_drug_dest_distance = new_drug_dest_distance - old_drug_dest_distance

        drug_reward = -self.alpha[0] * delta_drug_dest_distance

        # compute single reward vmaped
        new_positions = (
            np.array(
                [
                    colloid.pos
                    for colloid in colloids
                    if colloid.type is self.particle_type
                ]
            )
            / self.box_size
        )
        new_part_drug_dist = np.linalg.norm(new_positions - drug_position, axis=1)

        old_part_drug_distance = np.linalg.norm(
            self.historic_position - self.hist_drug_pos, axis=1
        )

        delta_part_drug_dist = new_part_drug_dist - old_part_drug_distance
        part_reward = -self.alpha[1] * delta_part_drug_dist
        drug_reward *= np.ones_like(part_reward)

        reward = drug_reward + part_reward
        # #
        # # print("new drug - particle distance: ", new_part_drug_dist)
        # # print("old drug - particle distance: ", old_part_drug_distance)
        # #
        # # print("new drug - destination distance: ", new_drug_dest_distance)
        # # print("old drug - destination distance: ", old_drug_dest_distance)
        #
        # reward_be_at_drug = self.alpha[0] * np.exp(-self.alpha[2]*new_
        # part_drug_dist)
        # print("reward be at drug: ", reward_be_at_drug)
        # reward_get_to_drug = - self.alpha[2] * delta_part_drug_dist *
        # (1 - 1/6 * reward_be_at_drug)
        # print("reward get to drug: ", reward_get_to_drug)
        #
        #
        # # update historic position
        self.hist_drug_pos = drug_position
        self.historic_position = new_positions
        #
        #
        # delta_reward = self.beta[0] * (old_drug_dest_distance - new_d
        # rug_dest_distance)
        # dist_reward = self.beta[2] * np.exp(-new_drug_dest_distance * self.beta[1])
        #
        #
        # delta_reward *= np.ones_like(reward_be_at_drug)
        # dist_reward *= np.ones_like(reward_be_at_drug)
        #
        #
        # # print("move to drug reward: ", r1)
        # # print("move to destination reward: ", delta_reward)
        # # print("be at destination reward: ", dist_reward)
        # # reward
        # reward = delta_reward + dist_reward + reward_be_at_drug +
        # reward_get_to_drug
        reward = 100 * np.clip(reward, 0, None)
        return reward
