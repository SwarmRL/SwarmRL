"""
Espresso interaction model capable of handling a neural network as a function.
"""
import typing
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from swarmrl.models.interaction_model import Action, Colloid, InteractionModel
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

from swarmrl.rl_protocols import RLProtocol


class MLModel(InteractionModel):
    """
    Class for a NN based espresso interaction model.
    """

    def __init__(
            self,
            protocols: Dict[str, RLProtocol],
            models: Dict[str, Network],
            observables: Dict[str, Observable],
            tasks: Dict[str, Task],
            record_traj: bool = False,):
            #actions: dict = None,
    #):
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
        self.protocols = protocols
        self.models = models
        self.observables = observables
        self.tasks = tasks
        self.record_traj = record_traj
        self.eps = np.finfo(np.float32).eps.item()
        #self.actions = actions
        # Used in the data saving.
        self.particle_types = [item for item in self.models]

        # Trajectory data to be filled in after each action.
        self.trajectory_data = {
            item: TrajectoryInformation(particle_type=item)
            for item in self.particle_types
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
        calc_actions = {int(np.copy(colloid.id)): Action() for colloid in colloids}
        for item in self.particle_types:
            observables, rewards, calc_actions = self.protocols[item].compute_episode_step(
                item,
                colloids,
                calc_actions,
            )

        actions = list(calc_actions.values())  # convert to a list.
        # Record the trajectory if required.
        # if self.record_traj:
        #for _type in self.particle_types:
        #    self.trajectory_data[_type].features.append(observables[_type])
        #    self.trajectory_data[_type].actions.append(action_indices[_type])
        #    self.trajectory_data[_type].log_probs.append(log_probs[_type])
        #    self.trajectory_data[_type].rewards.append(rewards[_type])

        return actions
