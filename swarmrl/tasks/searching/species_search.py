"""
Class for the species search task.
"""
from swarmrl.tasks.task import Task
import logging
from typing import List

import jax.numpy as np
import jax
import numpy as onp

from swarmrl.models.interaction_model import Colloid

logger = logging.getLogger(__name__)


class SpeciesSearch(Task):
    """
    Class for the species search task.
    """

    def __init__(
            self, 
            decay_fn: callable, 
            box_length: np.ndarray,
            sensing_type: int = 0,
            avoid: bool = False,
            scale_factor: int = 100,
            particle_type: int = 0
        ):
        """
        Constructor for the observable.

        Parameters
        ----------
        decay_fn : callable
                Decay function of the field.
        box_size : np.ndarray
                Array for scaling of the distances.
        sensing_type : int (default=0)
                Type of particle to sense.
        scale_factor : int (default=100)
                Scaling factor for the observable.
        avoid : bool (default=False)
                Whether to avoid or move to the sensing type.
        particle_type : int (default=0)
                Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)

        self.decay_fn = decay_fn
        self.box_length = box_length
        self.sensing_type = sensing_type
        self.scale_factor = scale_factor
        self.avoid = avoid

        self.historical_field = {}

        self.observable_fn = jax.vmap(
            self.compute_single_observable, in_axes=(0, 0, None, 0)
            )


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
        reference_ids = self.get_colloid_indices(colloids)
        historic_values = np.zeros(len(reference_ids))

        positions = []
        indices = []
        for index in reference_ids:
            indices.append(colloids[index].id)
            positions.append(colloids[index].pos)
       
        test_points = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.sensing_type]
        )

        out_indices, field_values = self.observable_fn(
            np.array(indices), np.array(positions), test_points, historic_values
            )
        
        for index, value in zip(out_indices, onp.array(field_values)):
            self.historical_field[str(index)] = value

    def compute_single_observable(
            self, 
            index: int,
            reference_position: np.ndarray, 
            test_positions: np.ndarray, 
            historic_value: float
        ) -> tuple:
        """
        Compute the observable for a single colloid.

        Parameters
        ----------
        index : int
                Index of the colloid to compute the observable for.
        reference_position : np.ndarray (3,)
                Position of the reference colloid.
        test_positions : np.ndarray (n_colloids, 3)
                Positions of the test colloids.
        historic_value : float
                Historic value of the observable.

        Returns
        -------
        tuple (index, observable_value)
        index : int
                Index of the colloid to compute the observable for.
        observable_value : float
                Value of the observable.
        """
        distances = np.linalg.norm(
            (test_positions - reference_position) / self.box_length, axis=-1
            )
        field_value = self.decay_fn(distances).sum()

        return index, field_value - historic_value
    
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
        if self.historical_field == {}:
            msg = (
                f"{type(self).__name__} requires initialization. Please set the "
                "initialize attribute of the gym to true and try again."
            )
            raise ValueError(msg)
        
        reference_ids = self.get_colloid_indices(colloids)
        positions = []
        indices = []
        historic_values = []
        for index in reference_ids:
            indices.append(colloids[index].id)
            positions.append(colloids[index].pos)
            historic_values.append(self.historical_field[str(colloids[index].id)])
       
        test_points = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.sensing_type]
        )

        out_indices, field_values = self.observable_fn(
            np.array(indices), np.array(positions), test_points, np.array(historic_values)
            )
       
        for index, value in zip(out_indices, onp.array(field_values)):
            self.historical_field[str(index)] = value

        if self.avoid:
            rewards = np.clip(field_values, None, 0)
        else:
            rewards = np.clip(field_values, 0, None)

        return self.scale_factor * rewards
