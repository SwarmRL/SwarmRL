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
from swarmrl.utils.utils import record_graph_trajectory


class SharedModel(InteractionModel):
    """
    Class for a NN based espresso interaction model.
    """

    def __init__(
        self,
        force_models: Dict[str, Network],
        observables: Dict[str, Observable or None],
        tasks: Dict[str, Task or None],
        record_traj: bool = False,
        actions: dict = None,
        global_task=None,
    ):
        """
        Constructor for the NNModel.

        Parameters
        ----------
        force_models : dict(
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
        self.force_models = force_models
        self.observables = observables
        self.tasks = tasks
        self.global_task = global_task
        self.record_traj = record_traj
        self.eps = np.finfo(np.float32).eps.item()

        self.actions = actions
        # Used in the data saving.
        self.particle_types = [type_ for type_ in self.force_models]
        for type_ in self.particle_types:
            try:
                os.remove(f".traj_data_{type_}.npy")
            except FileNotFoundError:
                pass
        self.particle_dict = {type_: 0 for type_ in self.particle_types}

    def initialize_model(self, colloids: typing.List[Colloid]):
        """
        Initialize the model. This function checks how many particles of each type
        are in the simulation and initializes the model accordingly. This method
        does not take into account the number of passive particles in the simulation.
        Only particles for which models are defined are taken into account.

        Parameters
        ----------
        colloids : list[Colloid]
                Colloids to initialize the model with.
        """
        for colloid in colloids:
            if colloid.type in self.particle_types:
                self.particle_dict[colloid.type] += 1

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
        actions = {int(np.copy(colloid.id)): Action() for colloid in colloids}
        action_indices = {type_: [] for type_ in self.particle_types}
        log_probs = {type_: [] for type_ in self.particle_types}

        observables = {type_: [] for type_ in self.particle_types}

        if self.global_task is not None:
            rewards = self.global_task(colloids)
        else:
            rewards = {type_: [] for type_ in self.particle_types}

        for type_ in self.particle_types:
            if self.force_models[type_].kind == "network":
                observables[type_] = self.observables[type_].compute_observable(
                    colloids
                )
                if self.tasks[type_] is not None:
                    rewards[type_] = self.tasks[type_](colloids)
                print("we came here")
                action_indices[type_], log_probs[type_] = self.force_models[
                    type_
                ].compute_action(
                    observables=observables[type_], explore_mode=explore_mode
                )
                chosen_actions = np.take(
                    list(self.actions[type_].values()), action_indices[type_], axis=-1
                )
                count = 0  # Count the colloids of a specific species.
                for colloid in colloids:
                    if str(colloid.type) == type_:
                        actions[colloid.id] = chosen_actions[count]
                        count += 1
            elif self.force_models[type_].kind == "classical":
                classical_actions = self.force_models[type_].compute_action(colloids)
                for index, action in classical_actions.items():
                    actions[index] = action

        actions = list(actions.values())

        # Record the trajectory if required.
        if self.record_traj:
            for type_ in self.particle_types:
                record_graph_trajectory(
                    particle_type=type_,
                    features=observables[type_],
                    actions=np.array(action_indices[type_]),
                    log_probs=np.array(log_probs[type_]),
                    rewards=np.array(rewards[type_]),
                )
        return actions
