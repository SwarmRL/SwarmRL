"""
Module for the TD returns value function.
"""

from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as np


class TDReturns:
    """
    Class for the TD (Temporal Difference) returns.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        standardize: bool = True,
        standardize_axis: Optional[Union[int, Tuple[int, ...]]] = 0,
    ):
        """
        Constructor for the TD returns class

        Parameters
        ----------
        gamma : float
            A decay factor for the values of the task each time step.
        standardize : bool
            If True, standardize the results of the calculation.
        standardize_axis : int, tuple of ints, or None
            The axis or axes along which to compute the mean and standard deviation
            during standardization. Defaults to 0 (typically the time dimension).
            Pass None or a tuple like (0, 1, 2) to standardize over all dimensions.

        """
        self.gamma = gamma
        self.standardize = standardize
        self.standardize_axis = standardize_axis

        # Set by us to stabilize division operations.
        self.eps = np.finfo(np.float32).eps.item()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        rewards: jax.Array,
        expected_future_rewards: jax.Array,
    ) -> jax.Array:
        """
        Call function for the expected TD returns.

        Parameters
        ----------
        rewards : jax.Array
            np.ndarray (n_time_steps, n_particles, dimension)
            A JAX array of immediate rewards.
        expected_future_rewards : jax.Array
            Estimated future rewards. Can match the shape of `rewards`,
            or be missing the final step on the time axis (axis 0).

        Returns
        -------
        expected_returns : np.ndarray (n_time_steps, n_particles, dimension)
            Expected TD returns for the rewards.
        """

        if rewards.shape != expected_future_rewards.shape:
            missing_steps = rewards.shape[0] - expected_future_rewards.shape[0]

            padding_shape = (missing_steps,) + rewards.shape[1:]
            padding = np.zeros(padding_shape, dtype=expected_future_rewards.dtype)

            expected_future_rewards = np.concatenate(
                [expected_future_rewards, padding], axis=0
            )

        # Compute expected returns via 1-step bootstrapping
        expected_returns = rewards + self.gamma * expected_future_rewards

        if self.standardize:
            # Use keepdims=True to preserve the dimensions of the original array.
            # This ensures robust broadcasting regardless of which axes are selected
            # for standardization.
            mean_vector = np.mean(
                expected_returns, axis=self.standardize_axis, keepdims=True
            )
            std_vector = (
                np.std(expected_returns, axis=self.standardize_axis, keepdims=True)
                + self.eps
            )

            expected_returns = (expected_returns - mean_vector) / std_vector

        return expected_returns
