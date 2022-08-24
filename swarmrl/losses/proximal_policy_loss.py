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
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.expected_returns import ExpectedReturns


class ProximalPolicyLoss(Loss, ABC):
    """
    Class to implement the proximal policy loss.
    """

    def __init__(
        self,
        value_function: ExpectedReturns,
        n_epochs: int = 10,
        epsilon: float = 0.2,
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
        """
        self.value_function = value_function
        self.n_epochs = n_epochs
        self.epsilon = epsilon

    def compute_critic_loss(
        self, critic: FlaxModel, critic_params: FrozenDict, features, true_values
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

        particle_loss = jnp.mean(value_loss, 0)

        critic_loss = jnp.mean(particle_loss)
        return critic_loss

    def compute_actor_loss(
        self,
        actor: FlaxModel,
        actor_params: FrozenDict,
        critic: FlaxModel,
        features,
        actions,
        old_log_probs,
        true_values,
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
            The log probs of the taken action during the episode at each time steps and
            by each agent.
        true_values : np.ndarray (n_time_steps, n_particles)
            The state value computed using the rewards received during the episode. To
            compute them one uses the value function given to the class.

        Returns
        -------
        actor_loss: float
            The actor loss of an episode, summed over all time steps and meaned over
            all particles.
        """

        # compute the probabilities of the old actions under the new policy
        new_log_props = actor.apply_fn({"params": actor_params}, features)
        new_log_props = gather_n_dim_indices(new_log_props, actions)

        # compute the ratio between old and new probs
        ratio = jnp.exp(new_log_props - old_log_probs)

        # compute the predicted values and to get the advantage
        predicted_values = critic(features)
        advantage = true_values - jnp.squeeze(predicted_values)
        # advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-10)

        # compute the clipped loss
        clipped_loss = -1 * jnp.minimum(
            ratio * advantage,
            jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage,
        )

        # sum over the time steps
        particle_loss = jnp.mean(clipped_loss, 0)

        # mean over the particle losses
        actor_loss = jnp.mean(particle_loss)

        return actor_loss

    def compute_loss(self, actor: FlaxModel, critic: FlaxModel, episode_data) -> tuple:
        """
        Compute the loss and update the actor and critic.
        Parameters
        ----------
        actor : FlaxModel
            The actor network that computes the log probs of the possible actions for a
            given observable vector
        critic : FlaxModel
            The critic network that approximates the state value function.
        episode_data : dict
            A dictionary containing the features, log_probs, actions and rewards of the
            previous episode at each time step for each colloid.
        Returns
        -------
        model_tuple : tuple ( FlaxModel, FlaxModel
            The updated actor and critic network.
        """
        feature_data = episode_data.item().get("features")
        old_log_probs_data = episode_data.item().get("log_probs")
        action_data = episode_data.item().get("actions")
        reward_data = episode_data.item().get("rewards")

        for _ in range(self.n_epochs):
            actor_grad_fn = jax.value_and_grad(self.compute_actor_loss, 1)
            actor_loss, actor_grad = actor_grad_fn(
                actor,
                actor.model_state.params,
                critic,
                feature_data,
                action_data,
                old_log_probs_data,
                self.value_function(reward_data),
            )

            critic_grad_fn = jax.value_and_grad(self.compute_critic_loss, 1)
            critic_loss, critic_grad = critic_grad_fn(
                critic,
                critic.model_state.params,
                feature_data,
                self.value_function(reward_data),
            )

            actor.update_model(actor_grad)
            critic.update_model(critic_grad)
        return actor.model_state, critic.model_state
