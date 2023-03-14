"""
Loss functions based on Proximal policy optimization.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""
from abc import ABC

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict

from swarmrl.losses.loss import Loss
from swarmrl.networks.flax_network import FlaxModel
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.generalized_advantage_estimate import GAE


class ProximalPolicyLoss(Loss, ABC):
    """
    Class to implement the proximal policy loss.
    """

    def __init__(
        self,
        value_function: GAE,
        sampling_strategy: GumbelDistribution,
        n_epochs: int = 20,
        epsilon: float = 0.2,
        entropy_coefficient: float = 0.01,
        record_training=True,
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
        self.record_training = record_training
        self.memory = {"feature_data": [],
                        "rewards": [],
                        "action_indices:": [],
                        "old_probs": [],
                        "advantages": [],
                        "returns": [],
                        "critic_vals": [],
                        "new_logits": [],
                        "entropy": [],
                        "chosen_log_probs": [],
                        "ratio": [],
                        "actor_loss": [],
                        "critic_loss": []
                        }

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

        particle_loss = jnp.mean(value_loss, 1)
        critic_loss = jnp.sum(particle_loss)

        if self.record_training:
            self.memory["critic_loss"].append(critic_loss.primal)

        return critic_loss

    def compute_actor_loss(
        self,
        actor_params: FrozenDict,
        actor: FlaxModel,
        features,
        actions,
        old_probs,
        advantages,
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
        old_probs : np.ndarray (n_time_steps, n_particles)
            The probs of the taken action during the episode at each time steps and
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
        new_logits = actor.apply_fn({"params": actor_params}, features)
        new_probabilities = jax.nn.softmax(new_logits)

        # compute the entropy of the whole distribution
        entropy = jnp.sum(self.sampling_strategy.compute_entropy(new_probabilities))
        chosen_log_probs = jnp.log(gather_n_dim_indices(new_probabilities, actions)+self.eps)

        # compute the ratio between old and new probs
        ratio = jnp.exp(chosen_log_probs - jnp.log(old_probs+self.eps))

        # compute the clipped loss
        clipped_loss = -1 * jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages,
        )

        # mean over the time steps
        particle_loss = jnp.sum(clipped_loss, axis=0)

        # mean over the particle losses
        actor_loss = jnp.sum(particle_loss)

        if self.record_training:
            self.memory["new_logits"].append(new_logits.primal)
            self.memory["entropy"].append(entropy.primal)
            self.memory["chosen_probs"].append(chosen_log_probs.primal)
            self.memory["ratio"].append(ratio.primal)
            self.memory["actor_loss"].append(clipped_loss.primal)

        return actor_loss + self.entropy_coefficient * entropy

    def compute_loss(self, actor: FlaxModel, critic: FlaxModel, episode_data):
        """
        Compute the loss and update the actor and critic.
        Parameters
        ----------
        actor : FlaxModel
            The actor network that computes the log probs of the possible actions for a
            given observable vector.
        critic : FlaxModel
            The critic network that approximates the state value function.
        episode_data : dict
            A dictionary containing the features, log_probs, actions and rewards of the
            previous episode at each time step for each colloid.
        Returns
        -------
        model_tuple : tuple  FlaxModel, FlaxModel
            The updated actor and critic network.
        """
        feature_data = episode_data.item().get("features")
        old_probs_data = episode_data.item().get("logits")
        action_data = episode_data.item().get("actions")
        # will return the reward per particle.
        reward_data = episode_data.item().get("rewards")

        # in case of partial rewards. They are summed up here to give a total reward.
        try:
            reward_data = np.sum(reward_data, axis=2)
        except:
            pass

        for _ in range(self.n_epochs):

            # compute the advantages and returns (true_values) for that epoch
            predicted_values = np.squeeze(critic(feature_data))
            advantages = self.value_function(rewards=reward_data,
                                              values=predicted_values
                                              )
            returns = self.value_function.returns(advantages=advantages,
                                                  values=predicted_values)

            actor_grad_fn = jax.value_and_grad(self.compute_actor_loss)
            actor_loss, actor_grad = actor_grad_fn(
                actor.model_state.params,
                actor=actor,
                features=feature_data,
                actions=action_data,
                old_probs=old_probs_data,
                advantages=advantages
            )
            critic_grad_fn = jax.grad(self.compute_critic_loss)
            critic_grad = critic_grad_fn(
                critic.model_state.params,
                critic=critic,
                features=feature_data,
                true_values=returns,
            )

            actor.update_model(actor_grad)
            critic.update_model(critic_grad)
            if self.record_training:
                self.memory["returns"].append(returns)
                self.memory["advantages"].append(advantages)
                self.memory["critic_vals"].append(predicted_values)
            # write training specs to disc

        if self.record_training:
            self.memory["feature_data"] = feature_data
            self.memory["old_probs"] = old_probs_data
            self.memory["action_indices"] = action_data
            self.memory["rewards"] = reward_data
            self.memory = data_saver(self.memory)

def data_saver(data: dict):
    empty_memory = {"feature_data": [],
                    "rewards": [],
                    "action_indices:": [],
                    "old_probs": [],
                    "advantages": [],
                    "returns": [],
                    "critic_vals": [],
                    "new_logits": [],
                    "entropy": [],
                    "chosen_log_probs": [],
                    "ratio": [],
                    "actor_loss": [],
                    "critic_loss": []
                    }

    try:
        reloaded_dict = np.load("dummy_data.npy", allow_pickle=True).item()
        for key, item in reloaded_dict.items():
            reloaded_dict[key].append(data[key])
        np.save("dummy_data.npy", reloaded_dict, allow_pickle=True)
    except FileNotFoundError:
        print('new_one')
        for key, item in empty_memory.items():
            empty_memory[key].append(data[key])
        np.save("dummy_data.npy", empty_memory, allow_pickle=True)

    return empty_memory
