"""
Loss functions based on Proximal policy optimization.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

from abc import ABC
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from jax import jit

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy
from swarmrl.utils.logging_utils import log_jax_runtime_value
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.generalized_advantage_estimate import GAE


class ProximalPolicyLoss(Loss, ABC):
    """
    Class to implement the proximal policy loss.
    """

    def __init__(
        self,
        value_function: GAE = GAE(),
        sampling_strategy: SamplingStrategy = GumbelDistribution(),
        n_epochs: int = 20,
        epsilon: float = 0.2,
        entropy_coefficient: float = 0.01,
    ):
        """
        Constructor for the PPO class.

        Parameters
        ----------
        value_function : Callable
            A the state value function that computes the value of a series of states for
            using the reward of the trajectory visiting these states
        n_epochs : int
            number of PPO updates
        epsilon : float
            the maximum of the relative distance between old and updated policy.
        entropy_coefficient : float
            Entropy coefficient for the PPO update. # TODO Add more here.

        """
        self.value_function = value_function
        self.sampling_strategy = sampling_strategy
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.entropy_coefficient = entropy_coefficient
        self.eps = 1e-8

    @partial(jit, static_argnums=(0, 2))
    def _calculate_loss(
        self,
        network_params: FrozenDict,
        network: Network,
        feature_data: jax.Array,
        action_indices: jax.Array,
        rewards: jax.Array,
        old_log_probs: jax.Array,
        final_observation: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
    ) -> jax.Array:
        """
        A function that computes the actor loss.

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
        old_log_probs : jax.Array (n_time_steps, n_particles)
            The log probabilities of the actions taken by the policy for all time steps
            and particles during one episode.
        final_observation : jax.Array (n_particles, feature_dimension)
            Observation reached after the final transition, used only to bootstrap
            the critic and not associated with an action or reward.
        terminated : jax.Array (n_time_steps,)
            Per-transition task termination flags.
        truncated : jax.Array (n_time_steps,)
            Per-transition environment-reset or time-limit flags.

        Returns
        -------
        loss: float
            The loss of the actor-critic network for the last episode.
        """

        # Include the resulting final state for critic bootstrapping. It has no action.
        all_feature_data = jnp.concatenate(
            (feature_data, final_observation[jnp.newaxis, ...]), axis=0
        )
        # compute the probabilities of the old actions under the new policy
        all_logits, all_values = network(network_params, all_feature_data)
        new_logits = all_logits[:-1]
        all_values = jnp.squeeze(all_values, axis=-1)
        predicted_values = all_values[:-1]

        log_jax_runtime_value("predicted_values", predicted_values)

        # compute the advantages and returns
        advantages, returns = self.value_function(
            rewards=rewards,
            values=all_values,
            terminated=terminated,
            truncated=truncated,
        )
        returns = jax.lax.stop_gradient(returns)
        log_jax_runtime_value("advantages", advantages)
        log_jax_runtime_value("returns", returns)

        # compute the probabilities of the old actions under the new policy
        new_probabilities = jax.nn.softmax(new_logits, axis=-1)

        # compute the entropy of the whole distribution
        entropy = self.sampling_strategy.compute_entropy(new_probabilities).sum()
        log_jax_runtime_value("entropy", entropy)
        chosen_log_probs = jnp.log(
            gather_n_dim_indices(new_probabilities, action_indices) + self.eps
        )

        # compute the ratio between old and new probs
        ratio = jnp.exp(chosen_log_probs - old_log_probs)

        # Compute critic loss
        total_critic_loss = (
            optax.huber_loss(predicted_values, returns).sum(axis=0).sum()
        )

        # Compute the actor loss

        # compute the clipped loss
        advantages = jax.lax.stop_gradient(advantages)
        clipped_loss = -1 * jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages,
        )
        particle_actor_loss = jnp.sum(clipped_loss, axis=0)
        actor_loss = jnp.sum(particle_actor_loss)
        # Compute combined loss
        loss = actor_loss - self.entropy_coefficient * entropy + 0.5 * total_critic_loss
        log_jax_runtime_value("actor_loss", actor_loss)
        log_jax_runtime_value("total_critic_loss", total_critic_loss)
        log_jax_runtime_value("loss", loss)

        return loss

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
        old_log_probs_data = jnp.array(episode_data.log_probs)
        feature_data = jnp.array(episode_data.features)
        action_data = jnp.array(episode_data.actions)
        reward_data = jnp.array(episode_data.rewards)
        final_observation = jnp.array(episode_data.final_observation)
        terminated = jnp.array(episode_data.terminated, dtype=bool)
        truncated = jnp.array(episode_data.truncated, dtype=bool)

        for _ in range(self.n_epochs):
            network_grad_fn = jax.value_and_grad(self._calculate_loss)
            _, network_grad = network_grad_fn(
                network.model_state.params,
                network=network,
                feature_data=feature_data,
                action_indices=action_data,
                rewards=reward_data,
                old_log_probs=old_log_probs_data,
                final_observation=final_observation,
                terminated=terminated,
                truncated=truncated,
            )

            network.update_model(network_grad)
        log_jax_runtime_value("network_grad", network_grad)
