"""
Module for the implementation of policy gradient loss.

Policy gradient is the most simplistic loss function where critic loss drives the entire
policy learning.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""
import logging

import jax
import jax.numpy as np
import optax
from flax.core.frozen_dict import FrozenDict

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.expected_returns import ExpectedReturns

logger = logging.getLogger(__name__)


class PolicyGradientLoss(Loss):
    """
    Parent class for the reinforcement learning tasks.

    Notes
    -----
    """

    def __init__(self, value_function: ExpectedReturns):
        """
        Constructor for the reward class.

        Parameters
        ----------
        value_function : ExpectedReturns
        """
        super(Loss, self).__init__()
        self.value_function = value_function
        self.n_particles = None
        self.n_time_steps = None

    def _compute_actor_loss(
        self,
        actor_params: FrozenDict,
        feature_data: np.ndarray,
        rewards: np.ndarray,
        action_indices: np.ndarray,
        actor: Network,
        critic: Network,
    ):
        """
        Compute the actor loss.

        Parameters
        ----------
        actor_params : FrozenDict
                Parameters of the actor model used.
        feature_data : np.ndarray (n_timesteps, n_particles, feature_dimension)
                Observable data for each time step and particle within the episode.
        rewards : np.ndarray (n_timesteps, n_particles, reward_dimension)
                Reward data for each time step and particle within the episode.
        action_indices : np.ndarray (n_timesteps, n_particles)
                Indices of the chosen actions at each time step so that exploration
                is preserved in the model training.
        actor : Network
                Actor model to use in the analysis.
        critic : Network
                Critic model to use in the analysis.

        Returns
        -------
        loss : float
                The loss for the episode.
        """
        actor_apply_fn = jax.vmap(actor.apply_fn, in_axes=(None, 0))
        # (n_timesteps, n_particles, n_possibilities)
        logits = actor_apply_fn({"params": actor_params}, feature_data)
        probabilities = jax.nn.softmax(logits)  # get probabilities
        chosen_probabilities = gather_n_dim_indices(probabilities, action_indices)
        log_probs = np.log(chosen_probabilities)
        logger.debug(f"{log_probs.shape=}")

        value_function_values = self.value_function(rewards)
        logger.debug(f"{value_function_values.shape}")

        critic_apply_fn = jax.vmap(critic.__call__, in_axes=(0))
        critic_values = critic_apply_fn(feature_data)[
            :, :, 0
        ]  # zero for trivial dimension
        logger.debug(f"{critic_values.shape=}")

        # (n_timesteps, n_particles)
        advantage = value_function_values - critic_values
        logger.debug(f"{advantage=}")

        loss = -1 * ((log_probs * advantage).sum(axis=0)).mean()
        logger.debug(f"{loss=}")

        return loss

    def _compute_critic_loss(
        self,
        critic_params: FrozenDict,
        feature_data: np.ndarray,
        rewards: np.ndarray,
        critic: Network,
    ):
        """
        Callable to be wrapped in grad for the critic loss.

        Parameters
        ----------
        critic_params : FrozenDict
                Parameters of the critic model used.
        feature_data : np.ndarray (n_timesteps, n_particles, feature_dimension)
                Observable data for each time step and particle within the episode.
        rewards : np.ndarray (n_timesteps, n_particles, reward_dimension)
                Reward data for each time step and particle within the episode.
        critic : Network
                Critic model to use in the analysis.

        Returns
        -------
        loss : float
                Critic loss for the episode.
        """
        critic_apply_fn = jax.vmap(critic.apply_fn, in_axes=(None, 0))
        critic_values = critic_apply_fn({"params": critic_params}, feature_data)[
            :, :, 0
        ]
        logger.debug(f"{critic_values.shape=}")
        value_function_values = self.value_function(rewards)
        logger.debug(f"{value_function_values.shape=}")

        loss = np.sum(optax.huber_loss(critic_values, value_function_values), axis=0)

        loss = np.mean(loss)

        return loss

    def compute_loss(
        self,
        actor: Network,
        critic: Network,
        episode_data: np.ndarray,
    ):
        """
        Compute the loss functions for the actor and critic based on the reward.

        Returns
        -------
        loss_tuple : tuple
                (actor_loss, critic_loss)
        """
        feature_data = episode_data.item().get("features")
        action_data = episode_data.item().get("actions")
        reward_data = episode_data.item().get("rewards")

        self.n_particles = np.shape(feature_data)[1]
        self.n_time_steps = np.shape(feature_data)[0]

        actor_grad_fn = jax.value_and_grad(self._compute_actor_loss)
        actor_loss, actor_grads = actor_grad_fn(
            actor.model_state.params,
            feature_data,
            reward_data,
            action_data,
            actor,
            critic,
        )

        critic_grad_fn = jax.value_and_grad(self._compute_critic_loss)
        critic_loss, critic_grads = critic_grad_fn(
            critic.model_state.params, feature_data, reward_data, critic
        )

        return actor_grads, critic_grads
