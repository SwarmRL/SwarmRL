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
from swarmrl.utils.utils import record_trajectory


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
        actions = {colloid.id: Action() for colloid in colloids}
        action_indices = {item: [] for item in self.particle_types}
        logits = {item: [] for item in self.particle_types}
        rewards = {item: [] for item in self.particle_types}
        observables = {item: [] for item in self.particle_types}

        for item in self.particle_types:
            observables[item] = self.observables[item].compute_observable(colloids)
            rewards[item] = self.tasks[item](colloids)
            action_indices[item], logits[item] = self.models[item].compute_action(
                observables=observables[item], explore_mode=explore_mode
            )
            chosen_actions = np.take(
                list(self.actions[item].values()), action_indices[item], axis=-1
            )

            count = 0  # Count the colloids of a specific species.
            for colloid in colloids:
                if str(colloid.type) == item:
                    actions[colloid.id] = chosen_actions[count]
                    count += 1
        actions = list(actions.values())  # convert to a list.

        # Record the trajectory if required.
        if self.record_traj:
            for item in self.particle_types:
                record_trajectory(
                    particle_type=item,
                    features=np.array(observables[item]),
                    actions=np.array(action_indices[item]),
                    logits=np.array(logits[item]),
                    rewards=np.array(rewards[item]),
                )

        return actions
