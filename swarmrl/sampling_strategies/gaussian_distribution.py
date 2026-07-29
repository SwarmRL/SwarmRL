"""Continuous Gaussian sampling strategy."""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct

from swarmrl.sampling_strategies.sampling_strategy import ContinuousSamplingStrategy


def action_limits_from_bounds(
    action_dimension: int,
    low: float,
    high: float,
    float_precision: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Build per-dimension action limits from scalar bounds."""
    if action_dimension < 1:
        raise ValueError("action_dimension must be at least 1")
    if high < low:
        raise ValueError("high must be greater than or equal to low")
    limits = jnp.array([[low, high]] * action_dimension, dtype=float_precision)
    return limits


# Register the class as a JAX PyTree so it can cross JIT boundaries safely
@struct.dataclass
class ContinuousGaussianDistribution(ContinuousSamplingStrategy):
    """
    Sample continuous actions from a Gaussian policy parameterization.

    Expected logits trailing dimension is ``2 * action_dimension`` where the
    first half encodes mean and the second half encodes log-std. Sampled actions
    are optionally tanh-squashed to ``action_limits``, if provided.
    In deployment mode, actions are deterministic (mean action) and
    no log-probabilities are produced.
    """

    # Static metadata fields are excluded from PyTree leaves.
    action_dimension: int = struct.field(pytree_node=False)
    action_limits: Optional[jnp.ndarray] = struct.field(pytree_node=True, default=None)
    log_scale: Optional[jnp.ndarray] = struct.field(pytree_node=True, default=None)
    log_std_min: float = struct.field(pytree_node=False, default=-20.0)
    log_std_max: float = struct.field(pytree_node=False, default=1.0)
    float_precision: jnp.dtype = struct.field(pytree_node=False, default=jnp.float32)

    @classmethod
    def create(
        cls,
        action_dimension: int,
        action_limits: Optional[jnp.ndarray] = None,
        log_std_min: float = -20.0,
        log_std_max: float = 1.0,
        float_precision: jnp.dtype = jnp.float32,
    ) -> "ContinuousGaussianDistribution":
        """Factory method to handle the initialization validation safely."""
        if action_limits is None:
            action_limits = action_limits_from_bounds(
                action_dimension, -1.0, 1.0, float_precision
            )
        else:
            if action_limits.shape != (action_dimension, 2):
                raise ValueError(
                    f"action_limits shape is {action_limits.shape} "
                    f"but should be {(action_dimension, 2)}"
                )
            action_limits = jnp.asarray(action_limits, dtype=float_precision)

        scale = (action_limits[:, 1] - action_limits[:, 0]) / 2.0
        log_scale = jnp.log(scale)

        return cls(
            action_dimension=int(action_dimension),
            action_limits=action_limits,
            log_scale=log_scale,
            log_std_min=float(log_std_min),
            log_std_max=float(log_std_max),
            float_precision=float_precision,
        )

    def squash_action(self, action: jnp.ndarray) -> jnp.ndarray:
        """Squash actions to configured limits via tanh-affine transform."""
        low = self.action_limits[:, 0]
        high = self.action_limits[:, 1]
        scale = (high - low) / 2.0
        mid = (high + low) / 2.0
        return jnp.tanh(action) * scale + mid

    def __call__(
        self,
        logits: jnp.ndarray,
        rng_key: Optional[jax.Array] = None,
        calculate_log_probs: bool = True,
        deployment_mode: bool = False,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Return sampled continuous actions and optional per-sample log-probs.

        Parameters
        ----------
        logits : jnp.ndarray
            Tensor with shape ``(batch_size, 2 * action_dimension)``.
        rng_key : Optional[jax.Array]
            PRNG key for sampling.
        calculate_log_probs : bool
            If true, compute tanh-corrected Gaussian log-probabilities.
        deployment_mode : bool
            If true, use deterministic actions (mean) and do not sample.

        Returns
        -------
        tuple[jnp.ndarray, Optional[jnp.ndarray]]
            ``(actions, log_probs)`` where ``log_probs`` is ``None`` when
            ``calculate_log_probs`` is false or ``deployment_mode`` is true.
        """

        logits = jnp.asarray(logits, dtype=self.float_precision)

        # Flexibly check trailing dimension to support arbitrary batching/vmap
        if logits.shape[-1] != 2 * self.action_dimension:
            raise ValueError(
                f"Logits trailing dimension must be 2 * {self.action_dimension}. "
                f"Got shape {logits.shape}."
            )

        mean = logits[..., : self.action_dimension]

        if deployment_mode:
            pre_squash_action = mean
            log_probs = None
        else:
            if rng_key is None:
                raise ValueError("A valid JAX PRNGKey is required during training.")

            log_std = jnp.clip(
                logits[..., self.action_dimension :],
                self.log_std_min,
                self.log_std_max,
            )
            std = jnp.exp(log_std)

            noise = jax.random.normal(
                rng_key, shape=mean.shape, dtype=self.float_precision
            )
            pre_squash_action = noise * std + mean

            if calculate_log_probs:
                log_probs = -0.5 * (
                    ((pre_squash_action - mean) / std) ** 2
                    + 2.0 * log_std
                    + jnp.log(2.0 * jnp.pi)
                )
                log_probs = log_probs.sum(axis=-1)

                log_scale_term = self.log_scale if self.log_scale is not None else 0.0

                correction = (
                    2.0
                    * (
                        jnp.log(2.0)
                        - pre_squash_action
                        - jax.nn.softplus(-2.0 * pre_squash_action)
                    )
                    + log_scale_term
                ).sum(axis=-1)

                log_probs = log_probs - correction
            else:
                log_probs = None

        actions = (
            self.squash_action(pre_squash_action)
            if self.action_limits is not None
            else pre_squash_action
        )

        return actions, log_probs
