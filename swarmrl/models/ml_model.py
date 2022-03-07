"""
Espresso interaction model capable of handling a neural network as a function.
"""
import os
import typing

import numpy as np
import torch
import torch.nn.functional
from torch.distributions import Categorical

from swarmrl.models.interaction_model import Action, Colloid, InteractionModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable


def _record_trajectory(colloids: typing.List[Colloid]):
    """
    Record trajectory if required.

    Returns
    -------

    """
    try:
        data = torch.load(".traj_data.pt")
        os.remove(".traj_data.pt")
    except FileNotFoundError:
        data = []

    data.append(colloids)
    torch.save(data, ".traj_data.pt")


class MLModel(InteractionModel):
    """
    Class for a NN based espresso interaction model.
    """

    def __init__(
        self, model: Network, observable: Observable, record_traj: bool = False
    ):
        """
        Constructor for the NNModel.

        Parameters
        ----------
        model : Network
                A SwarmRl model to use in the action computation.
        observable : Observable
                A method to compute an observable given a current system state.
        record_traj : bool
                If true, store trajectory data to disk for training.
        """
        super().__init__()
        self.model = model
        self.observable = observable
        self.record_traj = record_traj

        translate = Action(force=10.0)
        rotate_clockwise = Action(torque=np.array([0.0, 0.0, 0.1]))
        rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -0.1]))
        do_nothing = Action()

        self.actions = {
            "RotateClockwise": rotate_clockwise,
            "Translate": translate,
            "RotateCounterClockwise": rotate_counter_clockwise,
            "DoNothing": do_nothing,
        }

        try:
            os.remove(".traj_data.pt")
        except FileNotFoundError:
            pass

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
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
        # Record the trajectory if required.
        if self.record_traj:
            _record_trajectory(colloids)

        for colloid in colloids:
            other_colloids = [c for c in colloids if c is not colloid]
            feature_vector = self.observable.compute_observable(colloid, other_colloids)

            # action_probabilities = torch.nn.functional.softmax(
            #     self.model(feature_vector), dim=-1
            # )

            initial_prob = self.model(feature_vector)
            initial_prob = initial_prob / torch.max(initial_prob)
            action_probabilities = torch.nn.functional.softmax(initial_prob, dim=-1)

            action_distribution = Categorical(action_probabilities)

            j = np.random.random()
            if j >= 0.8:
                action_idx = np.random.randint(0, len(self.actions))
                actions.append(self.actions[list(self.actions)[action_idx]])

            else:
                action_idx = action_distribution.sample()
                actions.append(self.actions[list(self.actions)[action_idx.item()]])

            # action_log_prob = action_distribution.log_prob(action_idx)



        return actions
