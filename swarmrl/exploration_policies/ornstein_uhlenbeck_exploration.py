"""
Global Ornstein-Uhlenbeck (OU) exploration for continuous actions.
"""

from typing import Any

import jax
import jax.numpy as np

from swarmrl.exploration_policies.exploration_policy import ContinuousExplorationPolicy


class GlobalOUExploration(ContinuousExplorationPolicy):
    """
    Adds temporally-correlated OU noise to continuous model actions,
    as defined here https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process#Definition:
    dx_t = theta * (mu - x_t) * dt + sigma * dW_t. For simplicity dt=1 is assumed.

    Parameters/Symbols:
    - theta: Mean-reversion speed.
    - mu: Long-term mean (the value the noise settles around).
    - sigma: Volatility / scaling factor.
    - dW_t: Wiener process increment (Standard Brownian Motion).
        In discrete time with dt=1, dW_t is sampled from N(0, 1).

    Epsilon gating is applied per action dimension.
    For an action vector like ``[a, b, c, ...]``, each component is
    perturbed independently with probability ``epsilon`` at every step.
    The OU internal state is still updated every step, even when a component
    is not applied to the output action in that step.

    """

    def __init__(
        self,
        action_limits: Any,
        drift: float = 0.1,
        volatility: float = 0.1,
        long_term_mean: float = 0.0,
        action_dimension: int = 3,
        epsilon: float = 1.0,
    ) -> None:
        """
        Initialize the OU exploration process.

        Parameters
        ----------
        action_limits : array-like, shape (action_dimension, 2)
            Per-action lower/upper bounds used for clipping and noise scaling.
        drift : float
            Mean-reversion strength (theta). Must be > 0.
        volatility : float
            Diffusion scale (sigma).
        long_term_mean : float
            Long-term mean (mu) of the OU process.
        action_dimension : int
            Number of continuous action dimensions.
        epsilon : float
            Probability of applying OU noise in a step. Must be > 0.
        """
        self.drift: float = float(drift)
        self.volatility: float = float(volatility)
        self.long_term_mean: np.ndarray = np.asarray(long_term_mean, dtype=np.float32)
        self.action_dimension: int = int(action_dimension)
        self.action_limits: np.ndarray = np.asarray(action_limits, dtype=np.float32)
        self.noise: np.ndarray = np.zeros(self.action_dimension, dtype=np.float32)
        self.epsilon: np.ndarray = np.asarray(epsilon, dtype=np.float32)

        if self.drift <= 0:
            raise ValueError("drift needs to be greater than 0")
        if float(self.epsilon) <= 0:
            raise ValueError("epsilon needs to be greater than 0")

        if self.action_limits.shape != (self.action_dimension, 2):
            raise ValueError(
                f"action_limits shape is {self.action_limits.shape}, should be"
                f" {(self.action_dimension, 2)}."
            )

    def reduce_randomness(self, decay: float = 0.95) -> None:
        """
        Reduce OU state magnitude and epsilon (called every 10 episodes by trainer).
        """
        decay = np.asarray(decay, dtype=np.float32)
        self.noise = self.noise * decay
        self.epsilon = np.maximum(self.epsilon * decay, np.float32(0.01))

    def __call__(
        self, model_actions: np.ndarray, rng_key: jax.random.PRNGKey
    ) -> np.ndarray:
        """
        Add OU noise to model actions.

        OU state is always updated; epsilon only gates whether noise is applied
        to the current action output. Gating is dimension-wise (independent per
        action component), not one shared gate for the whole action vector.
        """
        model_actions = np.asarray(model_actions, dtype=np.float32)
        key_normal, key_uniform = jax.random.split(rng_key)

        value_range = self.action_limits[:, 1] - self.action_limits[:, 0]

        # Update OU state.
        long_term_noise_shift = self.drift * (self.long_term_mean - self.noise)
        random_noise = (
            self.volatility
            * value_range
            * jax.random.normal(key_normal, shape=self.noise.shape, dtype=np.float32)
        )

        self.noise = self.noise + long_term_noise_shift + random_noise

        # Decide whether to apply exploration in this step (per action dimension).
        should_explore = (
            jax.random.uniform(key_uniform, shape=self.noise.shape, dtype=np.float32)
            < self.epsilon
        ).astype(np.float32)

        noise_to_apply = self.noise * should_explore
        if model_actions.ndim > 1:
            noise_to_apply = noise_to_apply.reshape(1, -1)

        actions = model_actions + noise_to_apply
        actions = np.clip(actions, self.action_limits[:, 0], self.action_limits[:, 1])
        return actions
