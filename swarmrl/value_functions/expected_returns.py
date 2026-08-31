"""
Module for the expected returns value function.
"""

from functools import partial

import jax
import jax.numpy as jnp

from swarmrl.utils.logging_utils import log_jax_runtime_value


class ExpectedReturns:
    """
    Class for the expected returns.
    """

    def __init__(self, gamma: float = 0.99, standardize: bool = True):
        """
        Constructor for the Expected returns class

        Parameters
        ----------
        gamma : float
                A decay factor for the values of the task each time step.
        standardize : bool
                If True, standardize the results of the calculation.

        Notes
        -----
        See https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
        for more information.
        """
        self.gamma = gamma
        self.standardize = standardize

        # Set by us to stabilize division operations.
        self.eps = jnp.finfo(jnp.float32).eps.item()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, rewards: jnp.ndarray):
        """
        Call function for the expected returns.
        Parameters
        ----------
        rewards : jnp.ndarray (n_time_steps, n_particles, dimension)
                A numpy array of rewards to use in the calculation.

        Returns
        -------
        expected_returns : jnp.ndarray (n_time_steps, n_particles)
                Expected returns for the rewards.
        """
        log_jax_runtime_value("gamma", self.gamma)

        log_jax_runtime_value("rewards", rewards)

        def return_step(running_return, reward):
            current_return = reward + self.gamma * running_return
            return current_return, current_return

        _, expected_returns = jax.lax.scan(
            return_step,
            jnp.zeros_like(rewards[-1]),
            rewards,
            reverse=True,
        )

        log_jax_runtime_value("expected_returns", expected_returns)

        if self.standardize:
            mean_vector = jnp.mean(expected_returns, axis=0)
            std_vector = jnp.std(expected_returns, axis=0) + self.eps

            expected_returns = (expected_returns - mean_vector) / std_vector

        return expected_returns
