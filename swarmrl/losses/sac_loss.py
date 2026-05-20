"""
Module for computing Soft Actor-Critic (SAC) losses for the critic,
actor, and adaptive temperature parameter.

This implementation is based on "Soft Actor-Critic Algorithms and
Applications" by Haarnoja et al. (arXiv:1812.05905, 2018/2019).
"""

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from swarmrl.losses.loss import Loss
from swarmrl.networks.multi_flax_networks import MultiFlaxModel
from swarmrl.value_functions.td_return_sac import TDReturnsSAC


class SoftActorCriticLoss(Loss):
    """
    Implements the loss function for the Soft Actor-Critic (SAC) algorithm.
    Based on the paper "Soft Actor-Critic Algorithms and Applications"
    by Haarnoja et al. (2018).

    Concept:
    Math Blocks: _calculate_X_loss
    Forward_pipeline: compute_losses
    Core kernel: _train_step (jax.jit)
    parameter_update: compute_loss()
    """

    def __init__(
        self,
        value_function: TDReturnsSAC = TDReturnsSAC(),
        target_entropy: float | None = None,
        polyak_tau: float = 0.005,
    ):
        """
        Constructor for the SAC loss class.

        Parameters
        ----------
        value_function: TDReturnsSAC
            Value function to use.
        target_entropy : float, optional
            The target minimum entropy threshold (typically -dim(A)).
        polyak_tau: float, optional
            An SAC hyperparameter for setting the poylak averaging.
        """
        super().__init__()
        self.value_function = value_function or TDReturnsSAC()
        self.target_entropy = target_entropy
        self.polyak_tau = polyak_tau

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
        critic_loss = 0.5 * jnp.mean(
            (q1_pred - target_q) ** 2 + (q2_pred - target_q) ** 2
        )
        return critic_loss

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
        actor_loss = jnp.mean(alpha * log_probs - jnp.minimum(q1_action, q2_action))
        return actor_loss

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
        alpha = jnp.exp(log_alpha)
        log_probs_detached = jax.lax.stop_gradient(log_probs)

        temperature_loss = -alpha * jnp.mean(log_probs_detached + self.target_entropy)
        return temperature_loss

    def compute_losses(
        self,
        actor_module: Any,
        actor_network_params: Any,
        critic_module: Any,
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
         interface. Fully compatible with MultiFlaxModel registries.

         .. warning::
         JAX TRACER COMPILATION REQUIREMENT:
            Both `state_inputs` and `next_state_inputs` MUST strictly contain only
            valid JAX array types (e.g., jax.Array). Passing raw Python primitives
            such as strings, integers, or booleans inside these dictionaries will
            break JAX tracing and trigger compilation errors. Static metadata or
            flags must be handled at network initialization.

         Parameters
         ----------
         actor_module : flax.linen.Module
             The raw structural blueprint/module of the policy/actor network.
         actor_network_params : Any
             Parameters (weights) of the actor network.
        critic_module : flax.linen.Module
             The raw structural blueprint/module of the Q-network/critic network.
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
             The historical actions sampled from the replay buffer.
             Shape: (batch_size, action_dim)
         state_inputs : dict
             Unified input dictionary containing any inputs needed for evaluating
             the network at state s_t (e.g. feature_data, carry, node_features, masks).
         state_inputs : Dict[str, jax.Array]
             Unified input PyTree dictionary passed directly to the modules for
             state s_t evaluation (e.g., {"feature_data": matrix}, node_features,
             sequence masks).
             MUST strictly contain only valid dynamic JAX array types.
         next_state_inputs : Dict[str, jax.Array]
             Unified input PyTree dictionary passed directly to the modules for state
             s_{t+1} evaluation. Follows the same strict array-only typing rules
             as `state_inputs`.

         Returns
         -------
         losses : Dict[str, jax.Array]
             Contains the scalar loss tensors: 'critic_loss', 'actor_loss', and
             optionally 'temperature_loss'.
         aux : Dict[str, Any]
             Contains auxiliary execution footprints, tracking statistics, and hidden
             model carries used for monitoring and evaluation.
        """
        alpha = jnp.exp(log_alpha)

        # 1. Target Q-Value Computation: evaluate target actor on s_{t+1} to
        # sample next action candidates
        next_actions, next_log_probs, updated_stats_target_actor, _ = (
            actor_module.apply(
                {"params": actor_network_params},
                **next_state_inputs,
            )
        )
        next_log_probs = next_log_probs[..., None]

        # Evaluate target critics on next action candidates using target parameters
        q1_next, q2_next, updated_stats_target_critic = critic_module.apply(
            {"params": target_critic_network_params},
            actions=next_actions,
            # TODO/NOTE: method = "method_name, if needed!"
            **next_state_inputs,
        )
        q_next_min = jnp.minimum(q1_next, q2_next)

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
        q1_pred, q2_pred, updated_stats_critic = critic_module.apply(
            {"params": critic_network_params},
            actions=actions,
            **state_inputs,
        )
        critic_loss = self._calculate_critic_loss(q1_pred, q2_pred, target_q)

        # 3. Actor Predictions & Loss (using s_t inputs and newly sampled actions)
        new_actions, log_probs, updated_stats_actor, _ = actor_module.apply(
            {"params": actor_network_params}, **state_inputs
        )
        log_probs = log_probs[..., None]

        q1_action, q2_action, _ = critic_module.apply(
            {"params": critic_network_params}, actions=new_actions, **state_inputs
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

    @partial(jax.jit, static_argnums=(0, 1, 3))
    def _train_step(
        self,
        actor_module,  # raw network class
        actor_state,  # TrainState tracking actor weights + optimizer
        critic_module,  # raw network class
        critic_state,  # TrainState tracking critic weights + optimizer
        target_critic_params,  # Frozen parameters of the target network
        log_alpha_state,  # TrainState tracking the scalar log_alpha parameter
        rewards: jax.Array,
        dones: jax.Array,
        actions: jax.Array,
        state_inputs: Dict[str, Any],
        next_state_inputs: Dict[str, Any],
    ):
        critic_loss, critic_grads = jax.value_and_grad(
            lambda p: self.compute_losses(
                actor_module,
                actor_state.params,
                critic_module,
                p,
                target_critic_params,
                rewards,
                dones,
                log_alpha_state.params,
                actions,
                state_inputs,
                next_state_inputs,
            )[0]["critic_loss"]
        )(critic_state.params)

        actor_loss, actor_grads = jax.value_and_grad(
            lambda p: self.compute_losses(
                actor_module,
                p,
                critic_module,
                critic_state.params,
                target_critic_params,
                rewards,
                dones,
                log_alpha_state.params,
                actions,
                state_inputs,
                next_state_inputs,
            )[0]["actor_loss"]
        )(actor_state.params)

        temp_loss, alpha_grads = jax.value_and_grad(
            lambda p: self.compute_losses(
                actor_module,
                actor_state.params,
                critic_module,
                critic_state.params,
                target_critic_params,
                rewards,
                dones,
                p,
                actions,
                state_inputs,
                next_state_inputs,
            )[0]["temperature_loss"]
        )(log_alpha_state.params)

        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_log_alpha_state = log_alpha_state.apply_gradients(grads=alpha_grads)

        # Polyak Averaging
        new_target_critic_params = jax.tree.map(
            lambda target, online: (1.0 - self.polyak_tau) * target
            + self.polyak_tau * online,
            target_critic_params,
            new_critic_state.params,
        )

        return (
            new_actor_state,
            new_critic_state,
            new_target_critic_params,
            new_log_alpha_state,
            {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "temp_loss": temp_loss,
            },
        )

    def compute_loss(
        self, network: MultiFlaxModel, episode_data: dict[str, Any]
    ) -> Dict[str, jax.Array]:
        """
        The unified entrypoint invoked by the sac_agent. Takes the flat dictionary
        output from ReplayBuffer.sample(), packages observations for broad PyTree
        compatibility, and executes the optimized JIT training step kernel.

        Parameters
        ----------
        network : Network
            A container holding your multi-component states
            (actor_state, critic_state, etc.).
        episode_data : dict
            The dictionary returned by ReplayBuffer.sample().
        """
        # 1. Cast flat arrays safely to JAX tensors
        rewards_data = jnp.array(episode_data["reward"])
        dones_data = jnp.array(episode_data["done"])
        actions_data = jnp.array(episode_data["action"])

        # 2. Package observations into input dictionaries.
        state_inputs = {"feature_data": jnp.array(episode_data["observation"])}
        next_state_inputs = {
            "feature_data": jnp.array(episode_data["next_observation"])
        }

        # 3. Run the JIT-compiled kernel
        (
            new_actor_state,
            new_critic_state,
            new_target_critic_params,
            new_log_alpha_state,
            loss_metrics,
        ) = self._train_step(
            actor_network=network.networks["actor"],
            actor_state=network.states["actor"],
            critic_network=network.networks["critic"],
            critic_state=network.states["critic"],
            target_critic_params=network.target_params["critic"],
            log_alpha_state=network.states["alpha"],
            rewards=rewards_data,
            dones=dones_data,
            actions=actions_data,
            state_inputs=state_inputs,
            next_state_inputs=next_state_inputs,
        )

        # 4. Synchronize the newly updated pure JAX states back into the containers
        network.states["actor"] = new_actor_state
        network.states["critic"] = new_critic_state
        network.target_params["critic"] = new_target_critic_params
        network.states["alpha"] = new_log_alpha_state

        # Track training progress cleanly
        network.epoch_count += 1

        # 5. Return loss data back to agent.py for metrics tracking.
        # Optional, proximal_policy.compute_loss() does not return anything at all.
        return loss_metrics
