"""
Loss functions based on Proximal policy optimization.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""
from abc import ABC

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict

from swarmrl.losses.loss import Loss
from swarmrl.networks.flax_network import FlaxModel
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy
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

    def compute_critic_loss(
        self, critic_params: FrozenDict, critic: FlaxModel, features, true_values
    ) -> jnp.array:
        """
        A function that computes the critic loss.

        Parameters
        ----------
        critic : FlaxModel
            The critic network that approximates the state value function.
        critic_params : FrozenDict
            Parameters of the critic model used.
        features : np.ndarray (n_time_steps, n_particles, feature_dimension)
            Observable data for each time step and particle within the episode.
        true_values : np.ndarray (n_time_steps, n_particles)
            The state value computed for all time steps and particles during one episode
            using the value_function given to the class.

        Returns
        -------
        critic_loss: jnp.array ()
            Critic loss of an episode, summed over all time steps and meaned over
            all particles.
        """
        predicted_values = critic.apply_fn({"params": critic_params}, features)
        predicted_values = jnp.squeeze(predicted_values)

        value_loss = optax.huber_loss(predicted_values, true_values)

        particle_loss = jnp.sum(value_loss, 1)
        critic_loss = jnp.sum(particle_loss)

        if self.record_training:
            self.memory["critic_loss"].append(critic_loss.primal)

        return critic_loss

    def _calculate_loss(
        self,
        network_params: FrozenDict,
        network: FlaxModel,
        feature_data,
        action_indices,
        rewards,
        old_log_probs,
    ) -> jnp.array:
        """
        A function that computes the actor loss.

        Parameters
        ----------
        actor : FlaxModel
            The actor network that computes the log probs of the possible actions for a
            given observable vector
        actor_params : FrozenDict
            Parameters of the actor model used.
        critic : FlaxModel
            The critic network that approximates the state value function.
        features : np.ndarray (n_time_steps, n_particles, feature_dimension)
            Observable data for each time step and particle within the episode.
        actions : np.ndarray (n_time_steps, n_particles)
            The actions taken during the episode at each time steps and by each agent.
        old_log_probs : np.ndarray (n_time_steps, n_particles)
            The log_probs of the taken action during the episode at each time steps and
            by each agent.
        true_values : np.ndarray (n_time_steps, n_particles)
            The state value computed using the rewards received during the episode. To
            compute them one uses the value function given to the class.
        entropies : np.ndarray (n_time_steps, n_particles)
            The Shannon entropy of the distribution

        Returns
        -------
        actor_loss: float
            The actor loss of an episode, summed over all time steps and meaned over
            all particles.
        """

        # compute the probabilities of the old actions under the new policy
        new_logits, predicted_values = network.apply_fn(
            {"params": network_params}, feature_data
        )

        # compute the advantages and returns
        advantages, returns = self.value_function(
            rewards=rewards, values=predicted_values
        )

        # compute the probabilities of the old actions under the new policy
        new_probabilities = jax.nn.softmax(new_logits, axis=-1)

        # compute the entropy of the whole distribution
        entropy = jnp.sum(self.sampling_strategy.compute_entropy(new_probabilities))
        chosen_log_probs = jnp.log(
            gather_n_dim_indices(new_probabilities, action_indices) + self.eps
        )

        # compute the ratio between old and new probs
        ratio = jnp.exp(chosen_log_probs - old_log_probs)

        # compute the clipped loss
        clipped_loss = -1 * jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages,
        )

        # mean over the time steps
        particle_actor_loss = jnp.sum(clipped_loss, axis=0)

        # mean over the particle losses
        actor_loss = jnp.sum(particle_actor_loss)
        critic_loss = optax.huber_loss(predicted_values, returns)

        particle_critic_loss = jnp.sum(critic_loss, 1)
        total_critic_loss = jnp.sum(particle_critic_loss)

        loss = actor_loss + self.entropy_coefficient * entropy + 0.5 * total_critic_loss

        return loss

    def compute_loss(self, network: FlaxModel, episode_data):
        """
        Compute the loss and update the shared actor-critic network.

        Parameters
        ----------
        network : Network
                actor-critic model to use in the analysis.
        episode_data : np.ndarray (n_timesteps, n_particles, feature_dimension)
                Observable data for each time step and particle within the episode.

        Returns
        -------

        """
        feature_data = episode_data.item().get("features")
        old_log_probs_data = episode_data.item().get("log_probs")
        action_data = episode_data.item().get("actions")
        reward_data = episode_data.item().get("rewards")

        for _ in range(self.n_epochs):
            network_grad_fn = jax.value_and_grad(self._calculate_loss)
            network_loss, network_grad = network_grad_fn(
                network.model_state.params,
                network=network,
                feature_data=feature_data,
                action_indices=action_data,
                rewards=reward_data,
                old_log_probs=old_log_probs_data,
            )

            network.update_model(network_grad)
