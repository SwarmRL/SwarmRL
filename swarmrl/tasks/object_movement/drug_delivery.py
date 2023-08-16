"""
Class for the species search task.
"""
import logging
from typing import List

import jax.numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class DrugDelivery(Task):
    """
    Class for the species search task.
    """

    def __init__(
        self,
        decay_fn: callable,
        box_length: np.ndarray,
        drug_type: int = 1,
        destination: np.ndarray = np.array([800, 800, 0]),
        scale_factor: int = 1000,
        particle_type: int = 0,
    ):
        """
        Constructor for the observable.

        Parameters
        ----------
        decay_fn : callable
                Decay function of the field.
        box_size : np.ndarray
                Array for scaling of the distances.
        drug_type : int (default=0)
                Type of drug to transport.
        scale_factor : int (default=100)
                Scaling factor for the observable.
        destination : np.ndarray (default=np.array([800, 800, 0]))
                Destination of the drug.
        particle_type : int (default=0)
                Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)

        self.decay_fn = decay_fn
        self.box_length = box_length
        self.drug_type = drug_type
        self.scale_factor = scale_factor
        self.destination = destination / box_length

        self.historical_positions = dict()

        self.colloid_indices = dict()

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
        # be careful when you set up and create the colloids in the simulation file
        # the reset function might swap the indices of the destination and other
        # colloids. create drug and dealer before the destination colloid

        # get the relevant particle indices
        drug_indices = [
            i for i, colloid in enumerate(colloids) if colloid.type == self.drug_type
        ]
        transporter_indices = [
            i
            for i, colloid in enumerate(colloids)
            if colloid.type == self.particle_type
        ]

        # store the indices
        self.colloid_indices["drug"] = drug_indices
        self.colloid_indices["transporter"] = transporter_indices

        if len(self.colloid_indices["drug"]) == 0:
            raise ValueError

        # store the historical positions
        self.historical_positions["drug"] = np.array(
            [colloids[drug_index].pos / self.box_length for drug_index in drug_indices]
        )

        self.historical_positions["transporter"] = np.array(
            [
                colloids[transporter_index].pos / self.box_length
                for transporter_index in transporter_indices
            ]
        )

    def __call__(self, colloids: List[Colloid]):
        """
        Compute the reward on the colloids.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of all colloids in the system.

        Returns
        -------
        rewards : List[float] (n_colloids, dimension)
                List of rewards, one for each colloid.
        """

        if self.historical_positions == {}:
            msg = (
                f"{type(self).__name__} requires initialization. Please set the "
                "initialize attribute of the gym to true and try again."
            )
            raise ValueError(msg)

        # compute the new distance between the transporter and the drug
        new_drug_position = np.array(
            [
                colloids[drug_index].pos / self.box_length
                for drug_index in self.colloid_indices["drug"]
            ]
        )
        new_transporter_position = np.array(
            [
                colloids[transporter_index].pos / self.box_length
                for transporter_index in self.colloid_indices["transporter"]
            ]
        )

        # compute the distance between the transporter and the drug

        new_distance = np.linalg.norm(
            new_drug_position - new_transporter_position, axis=1
        )

        old_drug_positions = self.historical_positions["drug"]

        old_transporter_positions = self.historical_positions["transporter"]

        old_distance = np.linalg.norm(
            old_drug_positions - old_transporter_positions, axis=1
        )

        delta_dist = old_distance - new_distance
        # first the colloids get a reward for getting to the drug
        reward1 = np.where(delta_dist > 0, delta_dist, 0)

        old_drug_dest_dist = np.linalg.norm(old_drug_positions - self.destination)
        new_drug_dest_dist = np.linalg.norm(new_drug_position - self.destination)
        # second reward is for moving the drug closer to the destination
        delta_delivery_dist = old_drug_dest_dist - new_drug_dest_dist
        reward2 = np.where(delta_delivery_dist > 0, 10 * delta_delivery_dist, 0)
        self.historical_positions["drug"] = new_drug_position
        self.historical_positions["transporter"] = new_transporter_position

        return self.scale_factor * (reward1 + reward2)


class DrugTransport(Task):
    """
    Class for the species search task.
    """

    def __init__(
        self,
        decay_fn: callable,
        box_length: np.ndarray,
        drug_type: int = 1,
        destination: np.ndarray = np.array([800, 800, 0]),
        scale_factor: int = 1000,
        particle_type: int = 0,
    ):
        """
        Constructor for the observable.

        Parameters
        ----------
        decay_fn : callable
                Decay function of the field.
        box_size : np.ndarray
                Array for scaling of the distances.
        drug_type : int (default=0)
                Type of drug to transport.
        scale_factor : int (default=100)
                Scaling factor for the observable.
        destination : np.ndarray (default=np.array([800, 800, 0]))
                Destination of the drug.
        particle_type : int (default=0)
                Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)

        self.decay_fn = decay_fn
        self.box_length = box_length
        self.drug_type = drug_type
        self.scale_factor = scale_factor
        self.destination = destination / box_length

        self.historical_positions = dict()

        self.colloid_indices = dict()

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
        # be careful when you set up and create the colloids in the simulation file
        # the reset function might swap the indices of the destination and other
        # colloids. create drug and dealer before the destination colloid

        # get the relevant particle indices
        drug_indices = [
            i for i, colloid in enumerate(colloids) if colloid.type == self.drug_type
        ]
        transporter_indices = [
            i
            for i, colloid in enumerate(colloids)
            if colloid.type == self.particle_type
        ]

        # store the indices
        self.colloid_indices["drug"] = drug_indices
        self.colloid_indices["transporter"] = transporter_indices

        if len(self.colloid_indices["drug"]) == 0:
            raise ValueError

        # store the historical positions
        self.historical_positions["drug"] = np.array(
            [colloids[drug_index].pos / self.box_length for drug_index in drug_indices]
        )

        self.historical_positions["transporter"] = np.array(
            [
                colloids[transporter_index].pos / self.box_length
                for transporter_index in transporter_indices
            ]
        )

    def _compute_r1(self, old_part_drug_distance, new_part_drug_distance, r1_factor=1):
        delta_dist = old_part_drug_distance - new_part_drug_distance
        reward1 = np.where(delta_dist > 0, delta_dist, 0)
        return r1_factor * reward1

    def _compute_r2_factor(self, new_part_drug_dist, a=200, b=1.35, c=0.04):
        reward = b / (1 + np.exp(a * (new_part_drug_dist - c)))
        return reward

    def _compute_r2(self, old_drug_dest_dist, new_drug_dest_dist, r3_factor=10):
        delta_delivery_dist = old_drug_dest_dist - new_drug_dest_dist
        reward3 = np.where(delta_delivery_dist > 0, delta_delivery_dist, 0)
        delivered = np.where(new_drug_dest_dist < 0.01, 0.2, 0)
        print(delivered)
        return r3_factor * reward3 + delivered

    def __call__(self, colloids: List[Colloid]):
        """
        Compute the reward on the colloids.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of all colloids in the system.

        Returns
        -------
        rewards : List[float] (n_colloids, dimension)
                List of rewards, one for each colloid.
        """

        if self.historical_positions == {}:
            msg = (
                f"{type(self).__name__} requires initialization. Please set the "
                "initialize attribute of the gym to true and try again."
            )
            raise ValueError(msg)

        # compute the new distance between the transporter and the drug
        new_drug_position = np.array(
            [
                colloids[drug_index].pos / self.box_length
                for drug_index in self.colloid_indices["drug"]
            ]
        )
        new_transporter_position = np.array(
            [
                colloids[transporter_index].pos / self.box_length
                for transporter_index in self.colloid_indices["transporter"]
            ]
        )

        # compute the distance between the transporter and the drug

        new_part_drug_dist = np.linalg.norm(
            new_drug_position - new_transporter_position, axis=1
        )

        old_drug_positions = self.historical_positions["drug"]

        old_transporter_positions = self.historical_positions["transporter"]

        old_part_drug_dist = np.linalg.norm(
            old_drug_positions - old_transporter_positions, axis=1
        )

        old_drug_dest_dist = np.linalg.norm(
            old_drug_positions - self.destination, axis=1
        )

        new_drug_dest_dist = np.linalg.norm(
            new_drug_position - self.destination, axis=1
        )

        reward1 = self._compute_r1(old_part_drug_dist, new_part_drug_dist)
        reward2 = self._compute_r2(old_drug_dest_dist, new_drug_dest_dist)
        r2_factor = self._compute_r2_factor(new_part_drug_dist)

        self.historical_positions["drug"] = new_drug_position
        self.historical_positions["transporter"] = new_transporter_position

        return self.scale_factor * (reward1 + r2_factor * reward2)
