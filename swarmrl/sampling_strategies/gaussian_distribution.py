"""Continuous Gaussian sampling strategy."""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as onp

from swarmrl.sampling_strategies.sampling_strategy import ContinuousSamplingStrategy

logger = logging.getLogger(__name__)


class ContinuousGaussianDistribution(ContinuousSamplingStrategy):
    """
    Sample continuous actions from a Gaussian policy parameterization.

    Expected logits shape is ``(batch_size, 2 * action_dimension)`` where the
    first half encodes mean and the second half encodes log-std. Sampled actions
    are optionally tanh-squashed to ``action_limits``, if provided. In deployment mode, actions
    are deterministic (mean action) and no log-probabilities are produced.
    """

    def __init__(
        self, action_dimension: int, action_limits: Optional[jnp.ndarray] = None
    ):
        self.action_dimension = int(action_dimension)
        if action_limits is not None:
            if action_limits.shape != (action_dimension, 2):
                raise ValueError(f"action_limits shape is {action_limits.shape} but should be {(action_dimension, 2)}")
        self.action_limits = (
            None
            if action_limits is None
            else jnp.asarray(action_limits, dtype=jnp.float32)
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
        subkey: Optional[jax.Array] = None,
        rng_key: Optional[jax.Array] = None,
        calculate_log_probs: bool = True,
        deployment_mode: bool = False,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Return sampled continuous actions and optional per-sample log-probs.

        Parameters
        ----------
        logits : jnp.ndarray
            Tensor with shape ``(batch_size, 2 * action_dimension)``.
        subkey : Optional[jax.Array]
            PRNG subkey for sampling. Preferred key argument.
        rng_key : Optional[jax.Array]
            Alias for ``subkey`` for caller compatibility. Used only when
            ``subkey`` is ``None``.
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
        logits = jnp.asarray(logits, dtype=jnp.float32)
        if logits.shape[1] != 2 * self.action_dimension:
            raise ValueError(
                "Logits must have shape (batch_size, 2 * action_dimension). "
                f"Got {logits.shape} for action_dimension={self.action_dimension}."
            )

        mean = logits[:, : self.action_dimension]
        if deployment_mode:
            pre_squash_action = mean
            log_probs = None
        else:
            if subkey is None:
                subkey = rng_key
            if subkey is None:
                subkey = jax.random.PRNGKey(onp.random.randint(0, 1236534623))
            log_std = jnp.clip(logits[:, self.action_dimension :], -20.0, 1.0)
            std = jnp.exp(log_std)
            pre_squash_action = jax.random.normal(subkey, shape=mean.shape) * std + mean

            if calculate_log_probs:
                log_probs = -0.5 * (
                    ((pre_squash_action - mean) / std) ** 2
                    + 2.0 * jnp.log(std)
                    + jnp.log(2.0 * jnp.pi)
                )
                log_probs = log_probs.sum(axis=-1)

                correction = (
                    2.0
                    * (
                        jnp.log(2.0)
                        - pre_squash_action
                        - jax.nn.softplus(-2.0 * pre_squash_action)
                    )
                ).sum(axis=-1)
                log_probs = log_probs - correction
            else:
                log_probs = None

        actions = (
            self.squash_action(pre_squash_action)
            if self.action_limits is not None
            else pre_squash_action
        )
        logger.debug(f"{actions=}, {log_probs=}, shape={actions.shape}")

        return actions, log_probs
