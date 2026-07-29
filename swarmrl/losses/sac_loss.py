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
from swarmrl.networks.flax_network import FlaxModel
from swarmrl.sampling_strategies.sampling_strategy import ContinuousSamplingStrategy
from swarmrl.value_functions.td_return_sac import TDReturnsSAC


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
    if q1_pred.shape != q2_pred.shape or q1_pred.shape != target_q.shape:
        raise ValueError(
            "SAC critic predictions and targets must have matching shapes; "
            f"got q1={q1_pred.shape}, q2={q2_pred.shape}, target={target_q.shape}."
        )
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
    trainable_params: Any,
    model: Any,
    target_critic_params: Any,
    value_function_call: Any,
    sampling_strategy: Any,
    target_entropy: float | None,
    episode_data: dict[str, Any],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """
    Computes critic, actor, and temperature losses for a single Flax module.

    Parameters
    ----------
    trainable_params : Any
        Trainable parameter pytree of the live SAC module.
    model : Any
        Flax module exposing ``actor(...)``, ``critic(...)``, and ``alpha()``.
    target_critic_params : Any
        Frozen target-critic parameter pytree used for stable TD targets.
    value_function_call : Any
        ``__call__`` method of the SAC value function.
    sampling_strategy : Any
        Continuous action sampling strategy used for policy evaluation.
    target_entropy : float | None
        Target entropy used for adaptive temperature updates.
    episode_data : dict[str, Any]
        Sampled transition batch returned by ``ReplayBuffer.sample()``.
        Expected keys are ``observation``, ``next_observation``, ``action``,
        ``reward``, ``terminated``, ``truncated``, ``actor_rng`` and
        ``next_actor_rng``. Observations are expected to be array batches.

    Returns
    -------
    total_loss : Any
        The summed SAC loss.
    loss_dict : dict[str, Any]
        Scalar metrics including critic, actor, and temperature losses.
    """
    batch_data = episode_data
    state_inputs = {"feature_data": jnp.array(batch_data["observation"])}
    next_state_inputs = {"feature_data": jnp.array(batch_data["next_observation"])}

    actions = jnp.array(batch_data["action"])
    rewards = jnp.array(batch_data["reward"]).reshape(-1, 1)
    terminated = jnp.array(batch_data["terminated"]).reshape(-1, 1)
    _truncated = jnp.array(batch_data["truncated"]).reshape(-1, 1)

    actor_rng = batch_data["actor_rng"]
    next_actor_rng = batch_data["next_actor_rng"]

    next_network_key, next_sample_key = jax.random.split(next_actor_rng)
    live_network_key, live_sample_key = jax.random.split(actor_rng)

    alpha_val = model.apply({"params": trainable_params}, method=model.alpha)
    alpha_detached = jax.lax.stop_gradient(alpha_val)

    detached_params = jax.tree_util.tree_map(jax.lax.stop_gradient, trainable_params)

    next_logits = model.apply(
        {"params": detached_params},
        rng_key=next_network_key,
        method=model.actor,
        **next_state_inputs,
    )
    next_actions, next_log_probs = sampling_strategy(
        logits=next_logits,
        rng_key=next_sample_key,
        calculate_log_probs=True,
        deployment_mode=False,
    )
    next_log_probs = next_log_probs[..., None]

    new_logits = model.apply(
        {"params": trainable_params},
        rng_key=live_network_key,
        method=model.actor,
        **state_inputs,
    )
    new_actions, log_probs = sampling_strategy(
        logits=new_logits,
        rng_key=live_sample_key,
        calculate_log_probs=True,
        deployment_mode=False,
    )
    log_probs = log_probs[..., None]

    q1_pred, q2_pred = model.apply(
        {"params": trainable_params},
        actions=actions,
        method=model.critic,
        **state_inputs,
    )

    q1_next, q2_next = model.apply(
        {"params": target_critic_params},
        actions=next_actions,
        method=model.critic,
        **next_state_inputs,
    )
    q_next_min = jnp.minimum(q1_next, q2_next)

    target_q = value_function_call(
        rewards=rewards,
        q_next_min=q_next_min,
        temperature=alpha_detached,
        next_log_probs=next_log_probs,
        terminated=terminated,
    )
    target_q = jax.lax.stop_gradient(target_q)

    critic_loss = calculate_critic_loss(q1_pred, q2_pred, target_q)

    q1_action, q2_action = model.apply(
        {"params": detached_params},
        actions=new_actions,
        method=model.critic,
        **state_inputs,
    )
    actor_loss = calculate_actor_loss(q1_action, q2_action, log_probs, alpha_detached)

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


@partial(jax.jit, static_argnums=(1, 3, 5))
def get_sac_grads(
    trainable_params: Any,
    model: Any,
    target_critic_params: Any,
    value_function_call: Any,
    sampling_strategy: Any,
    target_entropy: float | None,
    episode_data: dict[str, Any],
) -> tuple[tuple[jax.Array, dict[str, jax.Array]], Any]:
    """JIT-compiled wrapper that calculates gradients for a single module."""
    loss_grad_fn = jax.value_and_grad(sac_loss_fn, argnums=0, has_aux=True)
    return loss_grad_fn(
        trainable_params,
        model,
        target_critic_params,
        value_function_call,
        sampling_strategy,
        target_entropy,
        episode_data,
    )


@partial(jax.jit, static_argnums=(2,))
def _apply_polyak_update(live_params, target_params, tau: float):
    """JIT-compiled Polyak averaging."""
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
        self, network: FlaxModel, episode_data: dict[str, Any]
    ) -> dict[str, jax.Array]:
        """
        Calculate gradients and apply updates to the FlaxModel state.
        Returns logging metrics.
        Parameters
        ----------
        network : Network
            FlaxModel Network holding actor, critic and alpha
        episode_data : dict
            Sampled transition batch returned by ReplayBuffer.sample().
        """
        (total_loss, metrics), grads = get_sac_grads(
            network.model_state.params,
            network.model,
            network.target_params["critic"],
            self.value_function.__call__,
            self.sampling_strategy,
            self.target_entropy,
            episode_data,
        )

        network.model_state = network.model_state.apply_gradients(grads=grads)
        network.target_params["critic"] = _apply_polyak_update(
            network.model_state.params,
            network.target_params["critic"],
            self.polyak_tau,
        )

        return metrics
