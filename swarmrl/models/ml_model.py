"""
Espresso interaction model capable of handling a neural network as a function.
"""

from swarmrl.models.interaction_model import InteractionModel
from swarmrl.observables.observable import Observable
from swarmrl.models.interaction_model import Action
from swarmrl.tasks.task import Task

import numpy as np
import torch
import torch.nn.functional
from torch.distributions import Categorical


class MLModel(InteractionModel):
    """
    Class for a NN based espresso interaction model.
    """
    def __init__(
            self,
            gym,
            observable: Observable,
            reward_cls: Task = None,
            record: bool = False
    ):
        """
        Constructor for the NNModel.

        Parameters
        ----------
        actor : torch.nn.Sequential
                A torch model to use in the action computation. In principle this need
                not be a torch model and could simply be any callable.
        observable : Observable
                A method to compute an observable given a current system state.
        critic : torch.nn.Sequential
                A critic model to collect information from.
        reward_cls : Task
                A callable with which to compute a reward.
        record : bool
                If true, record the outputs of the actor, critic, and reward function.
        """
        super().__init__()
        self.gym = gym
        self.reward_cls = reward_cls
        self.observable = observable
        self.record = record
        self.recorded_values = []

        translate = Action(force=10.0)
        rotate_clockwise = Action(torque=np.array([0.0, 0.0, 1]))
        rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -1]))
        do_nothing = Action()

        self.actions = {
            "RotateClockwise": rotate_clockwise,
            "Translate": translate,
            "RotateCounterClockwise": rotate_counter_clockwise,
            "DoNothing": do_nothing
        }

    def _record_parameters(
            self,
            action_log_prob: float,
            action_dist_entropy: float,
            feature_vector: torch.Tensor
    ):
        """
        Record the outputs of the model.

        Parameters
        ----------
        action_log_prob : float
                The log prob of the action. Returned separately here as this can always
                be recorded.
        action_dist_entropy : float
                Distribution entropy of the RL.
        feature_vector : np.array
                Feature vector used by the models to make predictions.

        Returns
        -------
        Updates the class state.
        """
        try:
            value = self.gym.critic.model(feature_vector)
        except TypeError:
            value = None
        try:
            reward = self.reward_cls.compute_reward(feature_vector)
        except AttributeError:
            reward = None

        self.recorded_values.append([action_log_prob, value, reward, action_dist_entropy])

    def calc_action(self, colloid, other_colloids) -> Action:
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
        feature_vector = self.observable.compute_observable(colloid, other_colloids)

        action_probabilities = torch.nn.functional.softmax(
            self.gym.actor.model(feature_vector), dim=-1
        )
        action_distribution = Categorical(action_probabilities)
        action_idx = action_distribution.sample()

        if self.record:
            action_log_prob = action_distribution.log_prob(action_idx)

            distribution_entropy = action_distribution.entropy()
            self._record_parameters(
                action_log_prob, distribution_entropy, feature_vector
            )

        return self.actions[list(self.actions)[action_idx.item()]]
