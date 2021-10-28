"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
from swarmrl.models.interaction_model import InteractionModel
import torch
import numpy as np
from typing import Callable


class MLPRL(InteractionModel):
    """
    Class for the simple MLP RL implementation.

    The multi-layer perceptron learner is a simple global network model wherein all
    particles are passed into the network and forces computed for each individually.

    Attributes
    ----------
    actor : torch.nn.Sequential
                A sequential torch model to use as an actor.
    critic : torch.nn.Sequential
                A sequential torch model to use as a critic.
    reward_function : callable
                Callable function from which a reward can be computed.
    actor_loss : torch.nn.Module
                Callable to compute the loss on the actor.
    critic_loss : torch.nn.Module
                Callable to compute the loss on the critic.
    actor_optimizer : torch.nn.Module
                Optimizer for the actor model.
    critic_optimizer : torch.nn.Module
                Optimizer for the critic model.
    """
    def __init__(
            self,
            actor: torch.nn.Sequential,
            critic: torch.nn.Sequential,
            reward_function: Callable,
            actor_loss: torch.nn.Module,
            critic_loss: torch.nn.Module,
            actor_optimizer: torch.nn.Module,
            critic_optimizer: torch.nn.Module
    ):
        """
        Constructor for the MLP RL.

        Parameters
        ----------
        actor : torch.nn.Sequential
                A sequential torch model to use as an actor.
        critic : torch.nn.Sequential
                A sequential torch model to use as a critic.
        reward_function : callable
                Callable function from which a reward can be computed.
        actor_loss : torch.nn.Module
                Callable to compute the loss on the actor.
        critic_loss : torch.nn.Module
                Callable to compute the loss on the critic.
        actor_optimizer : torch.nn.Module
                Optimizer for the actor model.
        critic_optimizer : torch.nn.Module
                Optimizer for the critic model.
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_function = reward_function
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def compute_force(self, colloids: torch.Tensor) -> np.ndarray:
        """
        Compute the force on all of the particles with the newest model.

        The model may simply be called. Upon calling it will generate several options
        for the next step. One of these options will be selected based upon a defined
        probability distribution.

        Parameters
        ----------
        colloids : tf.Tensor
                Tensor of colloids on which to operate. shape=(n_colloids, n_properties)
                where properties can very between test_models.

        Returns
        -------
        forces : np.ndarray
                Numpy array of forces to apply to the colloids. shape=(n_colloids, 3)
        """
        return self.actor(colloids)

    def compute_reward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the reward based on the state of the system.

        Parameters
        ----------
        state : torch.Tensor
                State of the system on which the reward should be computed.

        Returns
        -------
        reward : torch.Tensor
                torch tensor containing the reward
        """
        return self.reward_function(state)

    def update_critic(self, reward: torch.Tensor):
        """
        Perform an update on the critic network.

        Parameters
        ----------
        reward : torch.Tensor
                Reward on which to update the critic network.

        Returns
        -------
        Runs back-propagation on the critic model.
        """
        loss = self.critic_loss(reward)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, reward: torch.Tensor, predicted_reward: torch.Tensor):
        """
        Perform an update on the actor network.

        Parameters
        ----------
        reward : torch.Tensor
                Real reward computed by the system.
        predicted_reward : torch.Tensor
                Reward predicted by the critic.

        Returns
        -------
        Runs back-propagation on the actor model.
        """
        loss = self.actor_loss(reward, predicted_reward)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def predict_reward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Use the current critic network to compute a predicted award.

        Parameters
        ----------
        state : torch.Tensor
                Current state of the system on which to predict the reward.

        Returns
        -------
        reward : torch.Tensor
                predicted reward from the critic network.
        """
        pass

    def forward(
            self, colloids: torch.Tensor, state: torch.Tensor = None
    ) -> np.ndarray:
        """
        Perform the forward pass over the model.

        Perform the following steps:

        0.) Compute the reward
        1.) Compute new action
        2.) Update weights of critic
        3.) Compute predicted reward
        4.) Update weights of actor with old reward + predicted reward.

        Parameters
        ----------
        colloids : torch.Tensor
                Tensor of colloids on which to operate. shape=(n_colloids, n_properties)
                where properties can very between test_models.
        state : torch.Tensor
                State of the system on which to compute the reward.

        Returns
        -------
        forces : np.ndarray
                Numpy array of forces to apply to the colloids. shape=(n_colloids, 3)
        """
        if state is None:
            raise ValueError("State cannot be None")
        reward = self.compute_reward(state)
        updated_forces = self.compute_force(colloids)
        self.update_critic(reward)
        predicted_reward = self.predict_reward(state)
        self.update_actor(reward, predicted_reward)

        return updated_forces
