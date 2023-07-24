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

# from swarmrl.observables.col_graph_V0 import GraphObservable
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.generalized_advantage_estimate_shared import GAE


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

        # Make vmap function
        print(jnp.shape(features))
        actor_apply_fn = jax.vmap(network.apply_fn, in_axes=(None, 0))
        new_logits, predicted_values = actor_apply_fn(
            {"params": network_params}, features
        )
        predicted_values = jnp.squeeze(predicted_values)
        # compute the advantages and returns
        advantages, returns = self.value_function(
            rewards=rewards, values=predicted_values
        )
        advantages = advantages

        # compute the probabilities of the old actions under the new policy
        new_probabilities = jax.nn.softmax(new_logits, axis=-1)

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
        loss = actor_loss + self.entropy_coefficient * entropy + 0.5 * critic_loss
        # print("loss", loss.primal)
        return loss

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
        features = episode_data.item().get("features")
        # new_nodes = jnp.array([graph.nodes for graph in features])
        # new_edges = jnp.array([graph.edges for graph in features])
        # new_destinations = jnp.array([graph.destinations for graph in features])
        # new_receivers = jnp.array([graph.receivers for graph in features])
        # new_senders = jnp.array([graph.senders for graph in features])
        # new_globals = jnp.array([graph.globals_ for graph in features])
        # new_n_node = jnp.array([graph.n_node for graph in features])
        # new_n_edge = jnp.array([graph.n_edge for graph in features])
        # new_graph = GraphObservable(
        #     nodes=new_nodes,
        #     edges=new_edges,
        #     destinations=new_destinations.astype(int),
        #     receivers=new_receivers.astype(int),
        #     senders=new_senders.astype(int),
        #     globals_=new_globals,
        #     n_node=new_n_node.astype(int)[0],
        #     n_edge=new_n_edge.astype(int),
        # )
        old_log_probs_data = episode_data.item().get("log_probs")
        action_data = episode_data.item().get("actions")

        # will return the reward per particle.
        reward_data = episode_data.item().get("rewards")
        for _ in range(self.n_epochs):
            network_grad_fn = jax.value_and_grad(self.calculate_loss)
            network_loss, network_grad = network_grad_fn(
                network.model_state.params,
                network=network,
                features=features,
                actions=action_data,
                old_log_probs=old_log_probs_data,
                rewards=reward_data,
            )
            network.update_model(network_grad)
