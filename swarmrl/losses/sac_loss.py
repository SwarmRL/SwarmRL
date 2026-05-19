"""
Module for computing Soft Actor-Critic (SAC) losses for the critic,
actor, and adaptive temperature parameter.

This implementation is based on "Soft Actor-Critic Algorithms and
Applications" by Haarnoja et al. (arXiv:1812.05905, 2018/2019).
"""

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as np

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.value_functions.td_return_sac import TDReturnsSAC


class SoftActorCriticLoss(Loss):
    """
    Implements the loss function for the Soft Actor-Critic (SAC) algorithm.
    Based on the paper "Soft Actor-Critic Algorithms and Applications"
    by Haarnoja et al. (2018).
    """

    def __init__(
        self,
        value_function: TDReturnsSAC = TDReturnsSAC(),
        target_entropy: float | None = None,
    ):
        """
        Constructor for the SAC loss class.

        Parameters
        ----------
        value_function: TDReturnsSAC
            Value function to use.
        target_entropy : float, optional
            The target minimum entropy threshold (typically -dim(A)).
        """
        super().__init__()
        self.value_function = value_function or TDReturnsSAC()
        self.target_entropy = target_entropy

    @partial(jax.jit, static_argnums=(0, 1, 3))
    def compute_losses(
        self,
        actor_network: Network,
        actor_network_params: Any,
        critic_network: Network,
        critic_network_params: Any,
        target_critic_network_params: Any,
        rewards: jax.Array,
        dones: jax.Array,
        log_alpha: jax.Array,
        actions: jax.Array,
        state_inputs: Dict[str, Any],
        next_state_inputs: Dict[str, Any],
    ) -> Tuple[Dict[str, jax.Array], Dict[str, Any]]:
        """
        Computes critic, actor, and temperature losses using a unified dictionary
        interface. This decouples the loss calculations from specific neural network
        architectures.

        Parameters
        ----------
        actor_network : Network
            The policy/actor network instance.
        actor_network_params : Any
            Parameters (weights) of the actor network.
        critic_network : Network
            The Q-network/critic network instance.
        critic_network_params : Any
            Parameters (weights) of the critic network.
        target_critic_network_params : Any
            Parameters (weights) of the target critic network.
        rewards : jax.Array
            Immediate rewards. Shape: (batch_size, 1) or (batch_size,)
        dones : jax.Array
            Terminal flags (1.0 for terminal transition, 0.0 otherwise).
        log_alpha : jax.Array
            The log of the temperature parameter (scalar).
        actions : jax.Array
            The historical actions chosen by the policy (from replay buffer).
        state_inputs : dict
            Unified input dictionary containing any inputs needed for evaluating
            the network at state s_t (e.g. feature_data, carry, node_features, masks).
        next_state_inputs : dict
            Unified input dictionary containing any inputs needed for evaluating
            the network at state s_{t+1}.

        Returns
        -------
        losses : dict
            Contains the calculated losses: 'critic_loss', 'actor_loss', and
            optionally 'temperature_loss'.
        aux : dict
            Contains auxiliary statistics and outputs for state updates and logging.
        """
        alpha = np.exp(log_alpha)

        # 1. Target Q-Value Computation: evaluate target actor on s_{t+1} to
        # sample next action candidates
        next_actions, next_log_probs, updated_stats_target_actor, _ = (
            actor_network.compute_action_training(
                actor_network_params, **next_state_inputs
            )
        )
        next_log_probs = np.expand_dims(next_log_probs, axis=-1)

        # Evaluate target critics on next action candidates using target parameters
        q1_next, q2_next, updated_stats_target_critic = (
            critic_network.compute_q_values_target(
                target_critic_network_params, actions=next_actions, **next_state_inputs
            )
        )
        q_next_min = np.minimum(q1_next, q2_next)

        # Calculate Soft TD Target
        target_q = self.value_function(
            rewards=rewards,
            q_next_min=q_next_min,
            temperature=alpha,
            next_log_probs=next_log_probs,
            dones=dones,
        )
        target_q = jax.lax.stop_gradient(target_q)

        # 2. Critic Predictions & Loss (using s_t inputs and replay buffer actions)
        q1_pred, q2_pred, updated_stats_critic = critic_network.compute_q_values_critic(
            critic_network_params, actions=actions, **state_inputs
        )
        critic_loss = self._calculate_critic_loss(q1_pred, q2_pred, target_q)

        # 3. Actor Predictions & Loss (using s_t inputs and newly sampled actions)
        new_actions, log_probs, updated_stats_actor, _ = (
            actor_network.compute_action_training(actor_network_params, **state_inputs)
        )
        log_probs = np.expand_dims(log_probs, axis=-1)

        q1_action, q2_action, _ = critic_network.compute_q_values_critic(
            critic_network_params, actions=new_actions, **state_inputs
        )
        actor_loss = self._calculate_actor_loss(q1_action, q2_action, log_probs, alpha)

        # 4. Temperature/Alpha Loss (if target_entropy is provided)
        losses = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

        if self.target_entropy is not None:
            temperature_loss = self._calculate_temperature_loss(log_alpha, log_probs)
            losses["temperature_loss"] = temperature_loss

        aux = {
            "updated_stats_actor": updated_stats_actor,
            "updated_stats_critic": updated_stats_critic,
            "updated_stats_target_critic": updated_stats_target_critic,
            "target_q": target_q,
            "log_probs": log_probs,
            "alpha": alpha,
        }

        return losses, aux

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_critic_loss(
        self, q1_pred: jax.Array, q2_pred: jax.Array, target_q: jax.Array
    ) -> jax.Array:
        """
        Computes the Mean Squared Bellman Error (MSBE) for the twin Q-networks.
        Eq 5 in SAC-paper.

        Parameters
        ----------
        q1_pred : jax.Array
            Predicted Q-values from the first Q-network.
            Expected shape: (batch_size, 1) or (batch_size,)
        q2_pred : jax.Array
            Predicted Q-values from the second Q-network.
            Expected shape: Matches `q1_pred`.
        target_q : jax.Array
            The computed 1-step soft TD targets (Bellman targets).
            Expected shape: Matches `q1_pred`.

        Returns
        -------
        critic_loss : jax.Array
            A scalar tensor representing the combined loss for both Q-networks.
        """
        critic_loss = 0.5 * np.mean(
            (q1_pred - target_q) ** 2 + (q2_pred - target_q) ** 2
        )
        return critic_loss

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_actor_loss(
        self,
        q1_action: jax.Array,
        q2_action: jax.Array,
        log_probs: jax.Array,
        alpha: jax.Array,
    ) -> jax.Array:
        """
        Computes the policy loss for the actor using the reparameterization trick.
        Maximizes expected return while maximizing entropy (exploration).
        Eq 7 in SAC-paper.

        Parameters
        ----------
        q1_action : jax.Array
            Q-value predictions from the first online Q-network evaluated on
            newly sampled actions.
        q2_action : jax.Array
            Q-value predictions from the second online Q-network evaluated on
            newly sampled actions.
        log_probs : jax.Array
            Log-probabilities of actions sampled from the current policy.
        alpha : jax.Array
            The current entropy temperature parameter (alpha).

        Returns
        -------
        actor_loss : jax.Array
            A scalar tensor representing the actor loss to be minimized.
        """
        actor_loss = np.mean(alpha * log_probs - np.minimum(q1_action, q2_action))
        return actor_loss

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_temperature_loss(
        self,
        log_alpha: jax.Array,
        log_probs: jax.Array,
    ) -> jax.Array:
        """
        Computes the loss for the self-tuning entropy temperature parameter (alpha).
        Eq 18 in SAC-paper.

        Parameters
        ----------
        log_alpha : jax.Array
            The log-temperature parameter to optimize (scalar).
        log_probs : jax.Array
            Log-probabilities of actions. Automatically detached using stop_gradient
            to isolate updates strictly to alpha.

        Returns
        -------
        temperature_loss : jax.Array
            A scalar tensor representing the temperature loss to be minimized.
        """
        alpha = np.exp(log_alpha)
        log_probs_detached = jax.lax.stop_gradient(log_probs)

        temperature_loss = -alpha * np.mean(log_probs_detached + self.target_entropy)
        return temperature_loss

    def compute_loss(self, network: Network, episode_data: dict):
        raise NotImplementedError(
            "Use compute_losses() from SAC training loop with explicit tensors."
        )
