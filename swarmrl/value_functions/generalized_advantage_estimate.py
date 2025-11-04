"""
Module for the expected returns value function.
"""

import logging
from functools import partial

import jax.numpy as np
from jax import jit

logger = logging.getLogger(__name__)


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
        self.eps = np.finfo(np.float32).eps.item()

    @partial(jit, static_argnums=(0,))
    def __call__(self, rewards: np.ndarray, values: np.ndarray):
        """
        Call function for the advantage.
        Parameters
        ----------
        rewards : np.ndarray (n_time_steps, n_particles)
                A numpy array of rewards to use in the calculation.
        values : np.ndarray (n_time_steps, n_particles)
                The prediction of the critic for the episode.
        Returns
        -------
        advantages : np.ndarray (n_time_steps, n_particles)
                Expected returns for the rewards.
        """
        gae = 0
        advantages = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]

            gae = delta + self.gamma * self.lambda_ * gae
            advantages = advantages.at[t].set(gae)

        returns = advantages + values[:-1]

        advantages = (advantages - np.mean(advantages)) / (
            np.std(advantages) + self.eps
        )
        return advantages, returns
