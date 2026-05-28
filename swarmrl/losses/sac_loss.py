"""
Module for computing Soft Actor-Critic (SAC) losses for the critic,
actor, and adaptive temperature parameter.

This implementation is based on "Soft Actor-Critic Algorithms and
Applications" by Haarnoja et al. (arXiv:1812.05905, 2018/2019).
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from swarmrl.losses.loss import Loss
from swarmrl.networks.multi_flax_networks import MultiFlaxModel
from swarmrl.sampling_strategies.sampling_strategy import ContinuousSamplingStrategy
from swarmrl.value_functions.td_return_sac import TDReturnsSAC


def _extract_log_alpha(log_alpha_params: Any) -> jax.Array:
    """Return the scalar temperature leaf from scalar or dict-style params."""
    if isinstance(log_alpha_params, dict):
        if "params" in log_alpha_params and len(log_alpha_params) == 1:
            return log_alpha_params["params"]
        if "log_alpha" in log_alpha_params and len(log_alpha_params) == 1:
            return log_alpha_params["log_alpha"]
    return log_alpha_params


def calculate_critic_loss(
    q1_pred: jax.Array, q2_pred: jax.Array, target_q: jax.Array
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
        Expected shape: strictly matches `q1_pred` (e.g., (batch_size, 1)).
    target_q : jax.Array
        The computed 1-step soft TD targets (Bellman targets).
        Expected shape: Matches `q1_pred`.
        WARNING: Mixing (batch_size, 1) and (batch_size,) will cause silent
        and fatal broadcasting errors in JAX to (batch_size, batch_size).

    Returns
    -------
    critic_loss : jax.Array
        A scalar tensor representing the combined loss for both Q-networks.
    """
    return 0.5 * jnp.mean((q1_pred - target_q) ** 2 + (q2_pred - target_q) ** 2)


def calculate_actor_loss(
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
        The current entropy temperature parameter (alpha) (should be detached).

    Returns
    -------
    actor_loss : jax.Array
        A scalar tensor representing the actor loss to be minimized.
    """
    return jnp.mean(alpha * log_probs - jnp.minimum(q1_action, q2_action))


def calculate_temperature_loss(
    alpha_val: jax.Array,
    log_probs: jax.Array,
    target_entropy: float,
) -> jax.Array:
    """
    Computes the loss for the self-tuning entropy temperature parameter (alpha).
    Eq 18 in SAC-paper.

    Parameters
    ----------
    alpha_val : jax.Array
        The exponentiated temperature parameter (not detached).
    log_probs : jax.Array
        Log-probabilities of actions. Automatically detached using stop_gradient
        to isolate updates strictly to alpha.
    target_entropy : float
        The target minimum entropy threshold.

    Returns
    -------
    temperature_loss : jax.Array
        A scalar tensor representing the temperature loss to be minimized.
    """
    log_probs_detached = jax.lax.stop_gradient(log_probs)
    return -alpha_val * jnp.mean(log_probs_detached + target_entropy)


def sac_loss_fn(
    trainable_params: dict[str, jax.Array],
    actor_module: Any,
    critic_module: Any,
    target_critic_params: Any,
    value_function_call: Any,
    sampling_strategy: Any,
    target_entropy: float | None,
    episode_data: dict[str, Any],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """
    Pure JAX math function.
    Computes critic, actor, and temperature losses using a unified dictionary
    interface. Fully compatible with MultiFlaxModel registries.

    Parameters
    ----------
    trainable_params: dict[str, jax.Array],
        network weights dictionary
    actor_module : Any,
        Actor network template
    critic_module : Any,
        Critic network template
    target_critic_params : Any,
        The frozen weights of the slow-moving target critic network.
        Used to calculate stable 1-step TD targets (the Bellman backup)
        without the moving-target instability of the live critic.
    value_function_call : Any,
        __call__ method of the value function
    target_entropy : float | None,

    episode_data : dict[str, Any],
        Sampled transition batch returned by ``ReplayBuffer.sample()``.
        Expected keys are ``observation``, ``next_observation``, ``action``,
        ``reward``, ``terminated``, ``actor_rng`` and ``next_actor_rng``.

    Returns
    -------
    total_loss : Any
        The total loss.
    loss_dict : dict[str, Any]
        Contains the scalar loss tensors: 'critic_loss', 'actor_loss',
        'temperature_loss', 'alpha' and 'q1_mean'.
    """
    batch_data = episode_data
    state_inputs = {"feature_data": jnp.array(batch_data["observation"])}
    next_state_inputs = {"feature_data": jnp.array(batch_data["next_observation"])}
    actions = jnp.array(batch_data["action"])
    rewards = jnp.array(batch_data["reward"]).reshape(-1, 1)
    terminated = jnp.array(batch_data["terminated"]).reshape(-1, 1)

    actor_rng = batch_data["actor_rng"]
    next_actor_rng = batch_data["next_actor_rng"]

    next_network_key, next_sample_key = jax.random.split(next_actor_rng)
    live_network_key, live_sample_key = jax.random.split(actor_rng)

    batch_size = actions.shape[0]

    actor_p = trainable_params["actor"]
    critic_p = trainable_params["critic"]
    log_alpha_p = _extract_log_alpha(trainable_params["log_alpha"])

    alpha_val = jnp.exp(log_alpha_p)
    alpha_detached = jax.lax.stop_gradient(alpha_val)

    # Split Network Pass and Sampling

    # 1. Target Actor: Pure Forward Pass -> Sample
    next_logits = actor_module.apply(
        {"params": actor_p}, rng_key=next_network_key, **next_state_inputs
    )
    next_actions, next_log_probs = sampling_strategy(
        logits=next_logits,
        rng_key=next_sample_key,
        calculate_log_probs=True,
        deployment_mode=False,
    )
    next_log_probs = next_log_probs[..., None]

    # 2. Live Actor: Pure Forward Pass -> Sample
    new_logits = actor_module.apply(
        {"params": actor_p}, rng_key=live_network_key, **state_inputs
    )
    new_actions, log_probs = sampling_strategy(
        logits=new_logits,
        rng_key=live_sample_key,
        calculate_log_probs=True,
        deployment_mode=False,
    )
    log_probs = log_probs[..., None]

    # Optimized Batch Critic Evaluation
    concatenated_actions = jnp.concatenate([actions, new_actions], axis=0)
    concatenated_states = jax.tree_util.tree_map(
        lambda x: jnp.concatenate([x, x], axis=0), state_inputs
    )
    q1_concat, q2_concat = critic_module.apply(
        {"params": critic_p}, actions=concatenated_actions, **concatenated_states
    )
    q1_pred, q1_action = q1_concat[:batch_size], q1_concat[batch_size:]
    q2_pred, q2_action = q2_concat[:batch_size], q2_concat[batch_size:]

    # Target Critic Evaluation
    q1_next, q2_next = critic_module.apply(
        {"params": target_critic_params}, actions=next_actions, **next_state_inputs
    )
    q_next_min = jnp.minimum(q1_next, q2_next)

    # Calculate Soft TD Target
    target_q = value_function_call(
        rewards=rewards,
        q_next_min=q_next_min,
        temperature=alpha_detached,
        next_log_probs=next_log_probs,
        terminated=terminated,
    )
    target_q = jax.lax.stop_gradient(target_q)

    # MSBE Critic Loss
    critic_loss = calculate_critic_loss(q1_pred, q2_pred, target_q)

    # Actor Policy Loss
    actor_loss = calculate_actor_loss(q1_action, q2_action, log_probs, alpha_detached)

    # Total sum for JAX to differentiate
    total_loss = critic_loss + actor_loss

    temperature_loss = 0.0
    if target_entropy is not None:
        temperature_loss = calculate_temperature_loss(
            alpha_val, log_probs, target_entropy
        )
        total_loss += temperature_loss

    return total_loss, {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "temperature_loss": temperature_loss,
        "alpha": alpha_val,
        "q1_mean": jnp.mean(q1_pred),
    }


@partial(jax.jit, static_argnums=(1, 2, 4, 6))
def get_sac_grads(
    trainable_params: dict[str, jax.Array],
    actor_module: Any,
    critic_module: Any,
    target_critic_params: Any,
    value_function_call: Any,
    sampling_strategy: Any,
    target_entropy: float | None,
    episode_data: dict[str, Any],
) -> tuple[tuple[jax.Array, dict[str, jax.Array]], dict[str, Any]]:
    """JIT-compiled wrapper that calculates the gradients"""
    loss_grad_fn = jax.value_and_grad(sac_loss_fn, argnums=0, has_aux=True)
    return loss_grad_fn(
        trainable_params,
        actor_module,
        critic_module,
        target_critic_params,
        value_function_call,
        sampling_strategy,
        target_entropy,
        episode_data,
    )


@partial(jax.jit, static_argnums=(2,))
def _apply_polyak_update(live_params, target_params, tau: float):
    """JIT-compiled Polyak averaging for maximum speed."""
    return jax.tree_util.tree_map(
        lambda live, target: tau * live + (1.0 - tau) * target,
        live_params,
        target_params,
    )


class SoftActorCriticLoss(Loss):
    """
    Implements the loss function for the Soft Actor-Critic (SAC) algorithm.
    Based on the paper "Soft Actor-Critic Algorithms and Applications"
    by Haarnoja et al. (2018).

    Concept:
    Math Blocks: calculate_X_loss
    Pure Math Engine: sac_loss_fn
    Compiled Gradients: get_sac_grads
    Parameter Update: compute_loss()
    """

    def __init__(
        self,
        sampling_strategy: ContinuousSamplingStrategy,
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
        self.sampling_strategy = sampling_strategy
        self.value_function = value_function or TDReturnsSAC()
        self.target_entropy = target_entropy
        self.polyak_tau = float(polyak_tau)

    def compute_loss(
        self, network: MultiFlaxModel, episode_data: dict[str, Any]
    ) -> dict[str, jax.Array]:
        """
        Calculates gradients and directly applies updates to the network registries.
        Returns the logging metrics.
        Parameters
        ----------
        network : Network
            A container holding multi-component states
            (actor_state, critic_state, etc.).
        episode_data : dict
            Sampled transition batch returned by ReplayBuffer.sample().
        """
        # Package active weights
        trainable_params = {
            "actor": network.states["actor"].params,
            "critic": network.states["critic"].params,
            "log_alpha": network.states["log_alpha"].params,
        }

        # Get gradients
        (total_loss, metrics), grads = get_sac_grads(
            trainable_params,
            network.networks["actor"],
            network.networks["critic"],
            network.target_params["critic"],
            self.value_function.__call__,
            self.sampling_strategy,
            self.target_entropy,
            episode_data,
        )

        # Apply gradients
        network.states["critic"] = network.states["critic"].apply_gradients(
            grads=grads["critic"]
        )
        network.states["actor"] = network.states["actor"].apply_gradients(
            grads=grads["actor"]
        )
        if self.target_entropy is not None:
            network.states["log_alpha"] = network.states["log_alpha"].apply_gradients(
                grads=grads["log_alpha"]
            )

        # Apply polyak target update
        network.target_params["critic"] = _apply_polyak_update(
            network.states["critic"].params,
            network.target_params["critic"],
            self.polyak_tau,
        )

        return metrics
