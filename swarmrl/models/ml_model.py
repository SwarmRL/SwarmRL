# noinspection PyInterpreter
"""
Espresso interaction model capable of handling a neural network as a function.
"""
import os
import typing

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
        model: Network,
        observable: Observable,
        task: Task,
        record_traj: bool = False,
    ):
        """
        Constructor for the NNModel.

        Parameters
        ----------
        model : Network
                A SwarmRl network to use in the action computation.
        observable : Observable
                A method to compute an observable given a current system state.
        record_traj : bool
                If true, store trajectory data to disk for training.

        Notes
        -----
        TODO: Move the action definitions to user input.
        """
        super().__init__()
        self.model = model
        self.observable = observable
        self.task = task
        self.record_traj = record_traj

        translate = Action(force=10.0)
        rotate_clockwise = Action(torque=np.array([0.0, 0.0, 15.0]))
        rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -15.0]))
        do_nothing = Action()

        self.actions = {
            "RotateClockwise": rotate_clockwise,
            "Translate": translate,
            "RotateCounterClockwise": rotate_counter_clockwise,
            "DoNothing": do_nothing,
        }

        try:
            os.remove(".traj_data.npy")
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
        action_indices = []
        log_probs = []
        rewards = []

        # (n_particles, n_dims) array of features
        feature_vectors = []

        for colloid in colloids:
            other_colloids = [c for c in colloids if c is not colloid]
            feature_vector = self.observable.compute_observable(colloid, other_colloids)
            reward = self.task(feature_vector)
            action_index, log_prob = self.model.compute_action(
                feature_vector=feature_vector, explore_mode=explore_mode
            )
            actions.append(self.actions[list(self.actions)[int(action_index)]])

            action_indices.append(action_index)
            feature_vectors.append(feature_vector)
            log_probs.append(log_prob)
            rewards.append(reward)

        # Record the trajectory if required.
        if self.record_traj:
            record_trajectory(
                features=np.array(feature_vectors),
                actions=np.array(action_indices),
                log_probs=np.array(log_probs),
                rewards=np.array(rewards),
            )

        return actions
