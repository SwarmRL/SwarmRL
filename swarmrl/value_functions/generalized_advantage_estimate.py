"""
Module for the expected returns value function.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit


class GAE:
    """
    Class for the expected returns.
    """

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        """
        Constructor for the generalized advantage estimate  class

        Parameters
        ----------
        gamma : float
                A decay factor for the values of the task each time step.
        lambda_ : float
                A decay factor that describes the amount of bias included in the
                advantage calculation.

        Notes
        -----
        See https://arxiv.org/pdf/1506.02438.pdf for more information.
        """
        self.gamma = gamma
        self.lambda_ = lambda_

        # Set by us to stabilize division operations.
        self.eps = jnp.finfo(jnp.float32).eps.item()

    @partial(jit, static_argnums=(0,))
    def __call__(self, rewards: jnp.ndarray, values: jnp.ndarray):
        """
        Call function for the advantage.
        Parameters
        ----------
        rewards : jnp.ndarray (n_time_steps, n_particles)
                A numpy array of rewards to use in the calculation.
        values : jnp.ndarray (n_time_steps, n_particles)
                The prediction of the critic for the episode.
        Returns
        -------
        advantages : jnp.ndarray (n_time_steps, n_particles)
                Expected returns for the rewards.
        """
        gae = 0
        advantages = jnp.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            if t != len(rewards) - 1:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            else:
                delta = rewards[t] - values[t]

            gae = delta + self.gamma * self.lambda_ * gae
            advantages = advantages.at[t].set(gae)

        returns = advantages + values

        advantages = (advantages - jnp.mean(advantages)) / (
            jnp.std(advantages) + self.eps
        )
        return advantages, returns
