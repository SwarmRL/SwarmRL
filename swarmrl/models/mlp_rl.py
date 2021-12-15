"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
from swarmrl.models.interaction_model import InteractionModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task
from swarmrl.loss_models.loss import Loss
from swarmrl.models.interaction_model import Action
import torch
import numpy as np
from typing import Union


class MLPRL(InteractionModel):
    """
    Class for the simple MLP RL implementation.

    The multi-layer perceptron learner is a simple global network model wherein all
    particles are passed into the network and forces computed for each individually.

    Attributes
    ----------
    actor : Network
                A sequential torch model to use as an actor.
    critic : Network
                A sequential torch model to use as a critic.
    task : callable
                Callable function from which a reward can be computed.
    """

    def __init__(
        self,
        actor: Network,
        critic: Network,
        task: Task,
        loss: Loss,
        observable: Observable,
    ):
        """
        Constructor for the MLP RL.

        Parameters
        ----------
        actor : torch.nn.Sequential
                A sequential torch model to use as an actor.
        critic : torch.nn.Sequential
                A sequential torch model to use as a critic.
        task : Task
                Callable from which a reward can be computed.
        loss : Loss
                A loss model to use in the A-C loss computation.
        actions : dict
                A dictionary of possible actions. Key names should describe the action
                and the value should be a data class. See the actions module for
                more information.
        observable : Observable
                Observable class with which to compute the input vector.
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.task = task
        self.loss = loss
        self.observable = observable

        # Properties stored during the episode.
        self.true_rewards = []
        self.predicted_rewards = []
        self.action_probabilities = []

        # Actions on particles.
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.new_direction = None

        translate = Action(force=10.0)
        rotate_clockwise = Action(torque=np.array([0.0, 0.0, 1.0]))
        rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -1.0]))
        do_nothing = Action()

        self.actions = {
            "RotateClockwise": rotate_clockwise,
            "Translate": translate,
            "RotateCounterClockwise": rotate_counter_clockwise,
            "DoNothing": do_nothing
        }

    def calc_action(self, colloid, other_colloids) -> Action:
        """
        Return the selected action on the particles.

        Parameters
        ----------
        colloid
        other_colloids

        Returns
        -------

        """
        action = self.compute_state(colloid, other_colloids)
        print(f"Action: {action}")

        return self.actions[list(self.actions)[action]]

    def compute_feature_vector(self, colloid: object, other_colloids: list):
        """
        Compute the feature vector.

        Parameters
        ----------
        colloid : object
                Colloid of interest
        other_colloids : list
                List of all other colloids.

        Returns
        -------

        """
        return self.observable.compute_observable(colloid, other_colloids)

    def compute_state(self, colloid, other_colloids) -> int:
        """
        Compute the state of the active learning algorithm.

        If the model is not an active learner this method is ignored.

        Notes
        -----
        1.) Compute current state
        2.) Store necessary properties
        3.) Compute and set new forces / torques.
        """
        # Collect current state information.
        scaling = torch.nn.Softmax()
        state = self.compute_feature_vector(colloid, other_colloids)
        action_logits = self.actor(state)
        selector = torch.distributions.Categorical(action_logits)
        action = selector.sample()
        action_probabilities = scaling(action_logits)
        predicted_reward = self.critic(state)
        reward = self.task.compute_reward(colloid)

        # Update the stored data.
        self.action_probabilities.append(
            list(action_probabilities.detach().numpy())[action]
        )
        self.true_rewards.append(reward)
        self.predicted_rewards.append(float(predicted_reward.detach().numpy()))

        return action

    def update_critic(self, loss: torch.Tensor):
        """
        Perform an update on the critic network.

        Returns
        -------
        Runs back-propagation on the critic model.
        """
        self.critic.update_model(loss)

    def update_actor(self, loss: torch.Tensor):
        """
        Perform an update on the actor network.

        This method undertakes two tasks. First the actor reward must be predicted by
        mixing the true and predicted tasks stored during the run. Second, the final
        reward is passed to the actor for weight updates.

        Returns
        -------
        Runs back-propagation on the actor model.
        """
        self.actor.update_model(loss)

    def update_rl(self):
        """
        Update the RL algorithm.
        """
        actor_loss, critic_loss = self.loss.compute_loss(
            torch.tensor(self.action_probabilities),
            torch.tensor(self.predicted_rewards),
            torch.tensor(self.true_rewards),
        )
        self.update_actor(actor_loss)
        self.update_critic(critic_loss)

        self.true_rewards = []
        self.predicted_rewards = []
        self.action_probabilities = []
