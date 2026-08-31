"""
Module for the implementation of policy gradient loss.

Policy gradient is the most simplistic loss function where critic loss drives the entire
policy learning.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.utils.logging_utils import log_jax_runtime_value
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.expected_returns import ExpectedReturns


class PolicyGradientLoss(Loss):
    """
    Parent class for the reinforcement learning tasks.

    Notes
    -----
    """

    def __init__(self, value_function: ExpectedReturns = ExpectedReturns()):
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

    def _calculate_loss(
        self,
        network_params: FrozenDict,
        network: Network,
        feature_data: jax.Array,
        action_indices: jax.Array,
        rewards: jax.Array,
        final_observation: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
    ) -> jax.Array:
        """
        Compute the loss of the shared actor-critic network.

        Parameters
        ----------
        network : FlaxModel
            The actor-critic network that approximates the policy.
        network_params : FrozenDict
            Parameters of the actor-critic model used.
        feature_data : jax.Array (n_time_steps, n_particles, feature_dimension)
            Observable data for each time step and particle within the episode.
        action_indices : jax.Array (n_time_steps, n_particles)
            The actions taken by the policy for all time steps and particles during one
            episode.
        rewards : jax.Array (n_time_steps, n_particles)
            The rewards received for all time steps and particles during one episode.
        final_observation : jax.Array (n_particles, feature_dimension)
            Observation reached after the final transition, used only to bootstrap
            the critic and not associated with an action or reward.
        terminated : jax.Array (n_time_steps,)
            Per-transition task termination flags.
        truncated : jax.Array (n_time_steps,)
            Per-transition environment-reset or time-limit flags.


        Returns
        -------
        loss : float
            The loss of the actor-critic network for the last episode.
        """

        # Include the resulting final state for critic bootstrapping.
        # Shape: (n_timesteps + 1, n_particles, feature_dimension).
        all_feature_data = jnp.concatenate(
            (feature_data, final_observation[jnp.newaxis, ...]), axis=0
        )
        all_logits, all_values = network(network_params, all_feature_data)
        logits = all_logits[:-1]
        all_values = jnp.squeeze(all_values, axis=-1)
        predicted_values = all_values[:-1]
        probabilities = jax.nn.softmax(logits, axis=-1)  # get probabilities
        chosen_probabilities = gather_n_dim_indices(probabilities, action_indices)
        log_probs = jnp.log(chosen_probabilities + 1e-8)
        log_jax_runtime_value("log_probs", log_probs)

        returns = self.value_function(rewards, all_values, terminated, truncated)
        # Necessary because bootstrapped returns contain critic values; the targets
        # must not receive gradients during the critic update.
        returns = jax.lax.stop_gradient(returns)
        log_jax_runtime_value("returns", returns)

        log_jax_runtime_value("predicted_values", predicted_values)

        # (n_timesteps, n_particles)
        advantage = returns - predicted_values
        log_jax_runtime_value("advantage", advantage)

        # Sum over time steps and average over agents.
        critic_loss = optax.huber_loss(predicted_values, returns).sum(axis=0).sum()

        advantage = jax.lax.stop_gradient(advantage)
        actor_loss = -1 * ((log_probs * advantage).sum(axis=0)).sum()
        log_jax_runtime_value("actor_loss", actor_loss)

        return actor_loss + critic_loss

    def compute_loss(self, network: Network, episode_data):
        """
        Compute the loss and update the shared actor-critic network.

        Parameters
        ----------
        network : Network
                actor-critic model to use in the analysis.
        episode_data : np.ndarray (n_timesteps, n_particles, feature_dimension)
                Observable data for each action state. The final observation is stored
                separately in ``episode_data.final_observation``.

        Returns
        -------

        """
        # Rewards are computed after the action is taken (in calc_reward, called
        # after the integrator step), so index i already holds the matching
        # (state_i, action_i, reward_i+1) tuple. No shifting is needed here.
        feature_data = jnp.array(episode_data.features)
        action_data = jnp.array(episode_data.actions)
        reward_data = jnp.array(episode_data.rewards)
        final_observation = jnp.array(episode_data.final_observation)
        terminated = jnp.array(episode_data.terminated, dtype=bool)
        truncated = jnp.array(episode_data.truncated, dtype=bool)

        self.n_particles = jnp.shape(feature_data)[1]
        self.n_time_steps = jnp.shape(feature_data)[0]

        network_grad_fn = jax.value_and_grad(self._calculate_loss)
        _, network_grads = network_grad_fn(
            network.model_state.params,
            network=network,
            feature_data=feature_data,
            action_indices=action_data,
            rewards=reward_data,
            final_observation=final_observation,
            terminated=terminated,
            truncated=truncated,
        )

        network.update_model(network_grads)
