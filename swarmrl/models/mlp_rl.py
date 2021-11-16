"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
from swarmrl.models.interaction_model import InteractionModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task
from swarmrl.loss_models.loss import Loss
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
            actions: dict,
            observable: Observable
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
        self.actions = actions
        self.observable = observable

        # Properties stored during the episode.
        self.true_rewards = []
        self.predicted_rewards = []
        self.action_probabilities = []

        # Actions on particles.
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.new_direction = None

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

    def compute_state(self, colloid, other_colloids) -> None:
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

        # Handle the action.
        self.handle_action(action, colloid)

        # Update the stored data.
        self.action_probabilities.append(list(action_probabilities.detach().numpy()))
        self.true_rewards.append(reward)
        self.predicted_rewards.append(float(predicted_reward.detach().numpy()))

        return None

    def handle_action(self, action: torch.Tensor, colloid):
        """
        Handle the action chosen.

        Parameters
        ----------
        colloid : object
                A colloid object on which an action is taken.
        action : torch.Tensor
                The action chosen for the next step.

        Returns
        -------

        """
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.new_direction = None

        chosen_key = list(self.actions)[action.numpy()]

        update = self.actions[chosen_key]

        if update.property == "force":
            self.force = update.action(colloid)
        elif update.property == "torque":
            self.torque = update.action(colloid)
        elif update.property == "new_direction":
            self.new_direction = update.action(colloid)

    def calc_new_direction(self, colloid, other_colloids) -> Union[None, np.ndarray]:
        """
        Compute the new direction of the colloid.

        Parameters
        ----------
        colloid
        other_colloids

        Returns
        -------

        """
        return self.new_direction

    def calc_force(self, colloid, other_colloids) -> np.ndarray:
        """
        Compute the force on all of the particles with the newest model.

        The model may simply be called. Upon calling it will generate several options
        for the next step. One of these options will be selected based upon a defined
        probability distribution.

        Returns
        -------
        forces : np.ndarray
                Numpy array of forces to apply to the colloids. shape=(n_colloids, 3)
        """
        return self.force

    def calc_torque(self, colloid, other_colloids) -> np.ndarray:
        """
        Compute the torque acting on the particle an return it.
        Returns
        -------

        """
        return self.torque

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
