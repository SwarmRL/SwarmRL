"""
Loss functions based on Proximal policy optimization.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

import logging
import sys
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
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.generalized_advantage_estimate import GAE


class LoggerWriter:
    """
    Class that redirects stdout to a logger.

    Parameters
    ----------
    logger: Logger
        The logger object to redirect stdout to.
    level: int
        The logging level to use.
    linebuf: str
        Buffer to store the current line being written.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        """
            Writes the given buffer to the logger.

        Parameters
        ----------
            buf: str
                The buffer to write.

        Returns
        -------
            None
        """
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())


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

        # Store the original stdout, create a logger and the respective logger_stdout
        self.original_stdout = sys.stdout
        self.logger = logging.getLogger(__name__)
        self.logger_stdout = LoggerWriter(self.logger, logging.DEBUG)

    @partial(jit, static_argnums=(0, 2))
    def _calculate_loss(
        self,
        network_params: FrozenDict,
        network: Network,
        feature_data,
        action_indices,
        rewards,
        old_log_probs,
    ) -> jnp.array:
        """
        A function that computes the actor loss.

        Parameters
        ----------
        network : FlaxModel
            The actor-critic network that approximates the policy.
        network_params : FrozenDict
            Parameters of the actor-critic model used.
        feature_data : np.ndarray (n_time_steps, n_particles, feature_dimension)
            Observable data for each time step and particle within the episode.
        action_indices : np.ndarray (n_time_steps, n_particles)
            The actions taken by the policy for all time steps and particles during one
            episode.
        rewards : np.ndarray (n_time_steps, n_particles)
            The rewards received for all time steps and particles during one episode.
        old_log_probs : np.ndarray (n_time_steps, n_particles)
            The log probabilities of the actions taken by the policy for all time steps
            and particles during one episode.

        Returns
        -------
        loss: float
            The loss of the actor-critic network for the last episode.
        """

        # compute the probabilities of the old actions under the new policy
        new_logits, predicted_values = network(network_params, feature_data)
        predicted_values = predicted_values.squeeze()

        jax.debug.print("predicted_values = {}", predicted_values, ordered=True)

        # compute the advantages and returns
        advantages, returns = self.value_function(
            rewards=rewards, values=predicted_values
        )
        jax.debug.print("advantages = {}", advantages, ordered=True)
        jax.debug.print("returns = {}", returns, ordered=True)

        # compute the probabilities of the old actions under the new policy
        new_probabilities = jax.nn.softmax(new_logits, axis=-1)

        # compute the entropy of the whole distribution
        entropy = self.sampling_strategy.compute_entropy(new_probabilities).sum()
        jax.debug.print("entropy = {}", entropy, ordered=True)
        chosen_log_probs = jnp.log(
            gather_n_dim_indices(new_probabilities, action_indices) + self.eps
        )

        # compute the ratio between old and new probs
        ratio = jnp.exp(chosen_log_probs - old_log_probs)

        # Compute the actor loss

        # compute the clipped loss
        clipped_loss = -1 * jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages,
        )
        particle_actor_loss = jnp.sum(clipped_loss, axis=0)
        actor_loss = jnp.sum(particle_actor_loss)

        # Compute critic loss
        total_critic_loss = (
            optax.huber_loss(predicted_values, returns).sum(axis=0).sum()
        )

        # Compute combined loss
        loss = actor_loss - self.entropy_coefficient * entropy + 0.5 * total_critic_loss
        jax.debug.print("actor_loss = {}", actor_loss, ordered=True)
        jax.debug.print("total_critic_loss = {}", total_critic_loss, ordered=True)
        jax.debug.print("loss = {}", loss, ordered=True)

        return loss

    def compute_loss(self, network: Network, episode_data):
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
        old_log_probs_data = jnp.array(episode_data.log_probs)
        feature_data = jnp.array(episode_data.features)
        action_data = jnp.array(episode_data.actions)
        reward_data = jnp.array(episode_data.rewards)

        sys.stdout = self.logger_stdout
        jax.debug.print("\n---------PROXIMAL POLICY LOSS LOG---------\n", ordered=True)
        for _ in range(self.n_epochs):
            network_grad_fn = jax.value_and_grad(self._calculate_loss)
            _, network_grad = network_grad_fn(
                network.model_state.params,
                network=network,
                feature_data=feature_data,
                action_indices=action_data,
                rewards=reward_data,
                old_log_probs=old_log_probs_data,
            )

            network.update_model(network_grad)
        jax.debug.print("network_grad = {}", network_grad, ordered=True)
        jax.debug.print(
            "\n---------END PROXIMAL POLICY LOSS LOG---------\n", ordered=True
        )
        sys.stdout = self.original_stdout
