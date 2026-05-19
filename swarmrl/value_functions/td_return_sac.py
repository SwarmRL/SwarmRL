"""Temporal-difference target helper for Soft Actor-Critic."""

from functools import partial

import jax
import jax.numpy as np


class TDReturnsSAC:
    """Compute one-step entropy-regularized TD targets."""

    def __init__(self, gamma: float = 0.99, standardize: bool = False):
        self.gamma = gamma
        self.standardize = standardize
        self.eps = np.finfo(np.float32).eps.item()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        rewards: np.ndarray,
        q_next_min: np.ndarray,
        temperature: float,
        next_log_probs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        target = rewards + (1.0 - dones) * self.gamma * (
            q_next_min - temperature * next_log_probs
        )
        if self.standardize:
            mean_vector = np.mean(target)
            std_vector = np.std(target) + self.eps
            target = (target - mean_vector) / std_vector
        return target
