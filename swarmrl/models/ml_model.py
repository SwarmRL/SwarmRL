"""
Espresso interaction model capable of handling a neural network as a function.
"""
import typing
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from swarmrl.agents import Colloid, Swarm
from swarmrl.models.interaction_model import Action, InteractionModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task


@dataclass
class TrajectoryInformation:
    """
    Helper dataclass for training RL models.
    """

    particle_type: int
    features: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)


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
        self.particle_types = [type_ for type_ in self.models]

        # Trajectory data to be filled in after each action.
        self.trajectory_data = {
            type_: TrajectoryInformation(particle_type=type_)
            for type_ in self.particle_types
        }

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
        swarm = Swarm.create_swarm(colloids)

        actions = {int(np.copy(colloid.id)): Action() for colloid in colloids}
        action_indices = {_type: [] for _type in self.particle_types}
        log_probs = {_type: [] for _type in self.particle_types}
        rewards = {_type: [] for _type in self.particle_types}
        observables = {_type: [] for _type in self.particle_types}
        for _type in self.particle_types:
            observables[_type] = self.observables[_type].compute_observable(swarm)
            rewards[_type] = self.tasks[_type](colloids)
            action_indices[_type], log_probs[_type] = self.models[_type].compute_action(
                observables=observables[_type], explore_mode=explore_mode
            )
            chosen_actions = np.take(
                list(self.actions[_type].values()), action_indices[_type], axis=-1
            )

            count = 0  # Count the colloids of a specific species.
            for colloid in colloids:
                if str(colloid.type) == _type:
                    actions[colloid.id] = chosen_actions[count]
                    count += 1

        # Record the trajectory if required.
        # if self.record_traj:
        for type_ in self.particle_types:
            self.trajectory_data[type_].features.append(observables[type_])
            self.trajectory_data[type_].actions.append(action_indices[type_])
            self.trajectory_data[type_].log_probs.append(log_probs[type_])
            self.trajectory_data[type_].rewards.append(rewards[type_])

        return list(actions.values())
