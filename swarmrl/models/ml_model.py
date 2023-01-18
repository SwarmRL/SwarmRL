"""
Espresso interaction model capable of handling a neural network as a function.
"""
import os
import typing
from typing import Dict

import numpy as np

from swarmrl.models.interaction_model import Action, Colloid, InteractionModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task
from swarmrl.utils.utils import record_trajectory, record_rewards



class MLModel(InteractionModel):
    """
    Class for a NN based espresso interaction model.
    """

    def __init__(
        self,
        models: Dict[str, Network],
        observables: Dict[str, Observable],
        tasks: Dict[str, Task],
        record_traj: bool = False,
        actions: dict = None,
    ):
        """
        Constructor for the NNModel.

        Parameters
        ----------
        models : dict
                SwarmRl networks to use in the action computation. This is a dict so as
                to allow for multiple models to be passed.
        observables : dict[Observables]
                A method to compute an observable given a current system state.This is a
                dict so as to allow for multiple models to be passed.
        tasks : dict[Task]
                Tasks for each particle type to perform.
        record_traj : bool (default=False)
                If true, store trajectory data to disk for training.
        """
        super().__init__()
        self.models = models
        self.observables = observables
        self.tasks = tasks
        self.record_traj = record_traj
        self.eps = np.finfo(np.float32).eps.item()

        self.actions = actions
        # Used in the data saving.
        self.particle_types = [item for item in self.models]
        for item in self.particle_types:
            try:
                os.remove(f".traj_data_{item}.npy")
            except FileNotFoundError:
                pass

    def calc_action(
        self, colloids: typing.List[Colloid], explore_mode: bool = False
    ) -> typing.List[Action]:
        """
        Compute the state of the system based on the current colloid position.

        In the case of the ML models, this method undertakes the following steps:

        1. Compute observable
        2. Compute action probabilities
        3. Compute action

        Returns
        -------
        action: Action
                Return the action the colloid should take.
        """
        actions = []
        action_indices = {item: [] for item in self.particle_types}
        logits = {item: [] for item in self.particle_types}
        rewards = {item: [] for item in self.particle_types}
        feature_vectors = {item: [] for item in self.particle_types}

        for colloid in colloids:
            other_colloids = [c for c in colloids if c is not colloid]

            # Compute the action for a specific colloid type.
            try:
                feature_vector = self.observables[str(colloid.type)].compute_observable(
                    colloid, other_colloids
                )
                reward = self.tasks[str(colloid.type)](
                    feature_vector, colloid, colloids, other_colloids
                )
                action_index, logit = self.models[str(colloid.type)].compute_action(
                    feature_vector=feature_vector, explore_mode=explore_mode
                )
                actions.append(
                    self.actions[str(colloid.type)][
                        list(self.actions[str(colloid.type)])[int(action_index)]
                    ]
                )

                action_indices[str(colloid.type)].append(action_index)
                feature_vectors[str(colloid.type)].append(feature_vector)
                logits[str(colloid.type)].append(logit)
                rewards[str(colloid.type)].append(reward)
            except KeyError:
                actions.append(Action())

        # Record the trajectory if required.
        if self.record_traj:
            for item in self.particle_types:
                record_trajectory(
                    particle_type=item,
                    features=np.array(feature_vectors[item]),
                    actions=np.array(action_indices[item]),
                    logits=np.array(logits[item]),
                    rewards=np.array(rewards[item]),
                )
        for item in self.particle_types:
            record_rewards(particle_type=item, new_rewards=np.array(rewards[item]))

        return actions
