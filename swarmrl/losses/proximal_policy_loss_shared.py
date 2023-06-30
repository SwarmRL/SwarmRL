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
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.generalized_advantage_estimate import GAE


class SharedProximalPolicyLoss(Loss, ABC):
    """
    Class to implement the proximal policy loss.
    """

    def __init__(
        self,
        value_function: GAE = GAE(),
        sampling_strategy: GumbelDistribution = GumbelDistribution(),
        n_epochs: int = 10,
        epsilon: float = 0.2,
        entropy_coefficient: float = 0.01,
        record_training=False,
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
        record_training : bool
            If true, training data is stored.
        """
        self.value_function = value_function
        self.sampling_strategy = sampling_strategy
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.entropy_coefficient = entropy_coefficient
        self.eps = 1e-8

    def calculate_loss(
        self,
        network_params: FrozenDict,
        network: FlaxModel,
        features,
        actions,
        old_log_probs,
        rewards,
    ) -> jnp.array:
        """
        A function that computes the critic loss.

        Parameters
        ----------
        network_params : FrozenDict
            Parameters of the network used.
        network : FlaxModel
            The network used. Output of the network is a tuple of logits and values.
        features : np.ndarray (n_time_steps, n_particles, feature_dimension)
            Observable data for each time step and particle within the episode.
        actions : np.ndarray (n_time_steps, n_particles)
            The actions taken by each particle at each time step.
        old_log_probs : np.ndarray (n_time_steps, n_particles)
            The log probabilities of the actions taken by each particle at each time
            step.
        rewards : np.ndarray (n_time_steps, n_particles)
            The rewards received by each particle at each time step.

        Returns
        -------
        critic_loss: jnp.array ()
            Critic loss of an episode, summed over all time steps and meaned over
            all particles.
        """
        new_logits = []
        predicted_values = []
        for i in range(len(features)):
            logits, values = network.apply_fn({"params": network_params}, features[i])
            new_logits.append(logits)
            predicted_values.append(values)

        new_logits = jnp.array(new_logits)
        predicted_values = jnp.array(predicted_values)

        predicted_values = jnp.squeeze(predicted_values)
        # compute the advantages and returns
        advantages, returns = self.value_function(
            rewards=rewards, values=predicted_values
        )

        # compute the probabilities of the old actions under the new policy
        new_probabilities = jax.nn.softmax(new_logits)

        # compute the entropy of the whole distribution
        entropy = jnp.sum(self.sampling_strategy.compute_entropy(new_probabilities))
        chosen_log_probs = jnp.log(
            gather_n_dim_indices(new_probabilities, actions) + self.eps
        )

        # compute the ratio between old and new probs
        ratio = jnp.exp(chosen_log_probs - old_log_probs)
        # compute the clipped loss
        clipped_loss = -1 * jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages,
        )

        # sum over the time steps
        particle_actor_loss = jnp.mean(clipped_loss, axis=0)

        # sum over the particle losses
        actor_loss = jnp.sum(particle_actor_loss)
        value_loss = optax.huber_loss(predicted_values, returns)

        particle_critic_loss = jnp.sum(value_loss, 1)
        critic_loss = jnp.mean(particle_critic_loss)
        print(
            "loss: ",
            (
                actor_loss + self.entropy_coefficient * entropy + 0.5 * critic_loss
            ).primal,
        )
        return actor_loss + self.entropy_coefficient * entropy + 0.5 * critic_loss

    def compute_loss(self, network: FlaxModel, episode_data):
        """
        Compute the loss and update the actor and critic.
        Parameters
        ----------
        network : FlaxModel
            The actor and critic network. The actor is used to compute the policy
            gradient and the critic is used to compute the value function.
            The output of the network is a tuple of the actor and critic.
        episode_data : dict
            A dictionary containing the features, log_probs, actions and rewards of the
            previous episode at each time step for each colloid.
        Returns
        -------
        model_tuple : tuple  FlaxModel, FlaxModel
            The updated actor and critic network.
        """
        feature_data = episode_data.item().get("features")
        old_log_probs_data = episode_data.item().get("log_probs")
        action_data = episode_data.item().get("actions")
        # will return the reward per particle.
        reward_data = episode_data.item().get("rewards")
        for _ in range(self.n_epochs):
            network_grad_fn = jax.value_and_grad(self.calculate_loss)
            network_loss, network_grad = network_grad_fn(
                network.model_state.params,
                network=network,
                features=feature_data,
                actions=action_data,
                old_log_probs=old_log_probs_data,
                rewards=reward_data,
            )

            network.update_model(network_grad)
