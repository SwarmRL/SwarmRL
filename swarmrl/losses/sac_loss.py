"""Loss primitives for Soft Actor-Critic updates."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from swarmrl.losses.loss import Loss
from swarmrl.value_functions.td_return_sac import TDReturnsSAC


class SoftActorCriticLoss(Loss):
    """Compute SAC losses from actor/critic forward-call outputs."""

    def __init__(
        self,
        value_function: TDReturnsSAC | None = None,
        target_entropy: float | None = None,
    ):
        self.value_function = value_function or TDReturnsSAC()
        self.target_entropy = target_entropy

    def compute_losses(
        self,
        rewards: jnp.ndarray,
        dones: jnp.ndarray,
        q1_pred: jnp.ndarray,
        q2_pred: jnp.ndarray,
        q1_next: jnp.ndarray,
        q2_next: jnp.ndarray,
        next_log_probs: jnp.ndarray,
        log_probs: jnp.ndarray,
        alpha: float,
    ) -> dict[str, jnp.ndarray]:
        q_next_min = jnp.minimum(q1_next, q2_next)
        target_q = self.value_function(
            rewards=rewards,
            q_next_min=q_next_min,
            temperature=alpha,
            next_log_probs=next_log_probs,
            dones=dones,
        )
        target_q = jax.lax.stop_gradient(target_q)

        critic_loss = 0.5 * (
            jnp.mean((q1_pred - target_q) ** 2) + jnp.mean((q2_pred - target_q) ** 2)
        )

        q_min_pred = jnp.minimum(q1_pred, q2_pred)
        actor_loss = jnp.mean(alpha * log_probs - q_min_pred)

        losses = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "target_q": target_q,
        }
        if self.target_entropy is not None:
            alpha_loss = -jnp.mean(jnp.log(alpha) * (log_probs + self.target_entropy))
            losses["alpha_loss"] = alpha_loss

        return losses

    def compute_loss(self, network, episode_data):
        raise NotImplementedError(
            "Use compute_losses() from SAC training loop with explicit tensors."
        )
