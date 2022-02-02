"""
Espresso interaction model capable of handling a neural network as a function.
"""

from swarmrl.models.interaction_model import InteractionModel
from swarmrl.observables.observable import Observable
from swarmrl.models.interaction_model import Action
import numpy as np
import torch
from torch.distributions import Categorical


class MLModel(InteractionModel):
    """
    Class for a NN based espresso interaction model.
    """
    def __init__(self, model: torch.nn.Sequential, observable: Observable):
        """
        Constructor for the NNModel.

        Parameters
        ----------
        model : Network
                A torch model to use in the action computation. In principle this need
                not be a torch model and could simply be any callable.
        observable : Observable
                A method to compute an observable given a current system state.
        """
        super().__init__()
        self.model = model
        self.observable = observable

        translate = Action(force=10.0)
        rotate_clockwise = Action(torque=np.array([0.0, 0.0, 0.1]))
        rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -0.1]))
        do_nothing = Action()

        self.actions = {
            "RotateClockwise": rotate_clockwise,
            "Translate": translate,
            "RotateCounterClockwise": rotate_counter_clockwise,
            "DoNothing": do_nothing
        }

    def calc_action(self, colloid, other_colloids) -> Action:
        """
        Compute the state of the system based on the current colloid position.

        In the case of the ML models, this method undertakes the following steps:

        1. Compute observable
        2. Compute action probabilities
        3. Compute action
        """
        scaling = torch.nn.Softmax()
        feature_vector = self.observable.compute_observable(colloid, other_colloids)
        action_probabilities = scaling(self.model(feature_vector))
        action_distribution = Categorical(action_probabilities)
        action_idx = action_distribution.sample()

        return self.actions[list(self.actions)[action_idx]]




