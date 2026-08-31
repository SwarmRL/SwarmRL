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
    def __call__(
        self,
        rewards: jax.Array,
        values: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
    ) -> jax.Array:
        """
        Call function for the expected returns.
        Parameters
        ----------
        rewards : jax.Array (n_time_steps, n_particles, dimension)
                A numpy array of rewards to use in the calculation.
        values : jax.Array (n_time_steps + 1, n_particles)
                Critic predictions for each action state and the final observation.
        terminated : jax.Array (n_time_steps,)
                Whether each transition ended in a terminal task state.
        truncated : jax.Array (n_time_steps,)
                Whether each transition ended because the environment was reset.

        Returns
        -------
        expected_returns : jax.Array (n_time_steps, n_particles)
                Expected returns for the rewards.
        """
        log_jax_runtime_value("gamma", self.gamma)

        log_jax_runtime_value("rewards", rewards)

        def return_step(running_return, transition):
            """
            Boundary-aware return:
            $G_t = r_t + gamma * ((1 - done_t) * G_(t+1) + truncated_t * V_(t+1))$.
            Terminations use zero bootstrap; truncations bootstrap without crossing
            the reset.
            """
            reward, next_value, is_terminated, is_truncated = transition
            continuation = jnp.where(
                is_truncated,
                next_value,
                running_return,
            )
            continuation = jnp.where(
                is_terminated,
                jnp.zeros_like(continuation),
                continuation,
            )
            current_return = reward + self.gamma * continuation
            return current_return, current_return

        # Seed the reverse scan with V(s_T) for rollout (trainer episode) bootstrapping.
        # The scan step masks this value to zero when the final transition terminated.
        _, expected_returns = jax.lax.scan(
            return_step,
            values[-1],
            (rewards, values[1:], terminated, truncated),
            reverse=True,
        )

        log_jax_runtime_value("expected_returns", expected_returns)

        if self.standardize:
            mean_vector = jnp.mean(expected_returns, axis=0)
            std_vector = jnp.std(expected_returns, axis=0) + self.eps

            expected_returns = (expected_returns - mean_vector) / std_vector

        return expected_returns
