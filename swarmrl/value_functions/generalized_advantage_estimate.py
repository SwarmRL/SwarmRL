"""
Module for the expected returns value function.
"""
import logging

import jax.numpy as np
import numpy as onp

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
        expected_returns : np.ndarray (n_time_steps, n_particles)
                Expected returns for the rewards.
        """
        gae = 0
        advantages = onp.zeros_like(rewards)
        for t in reversed(range(len(rewards)-1)):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lambda_ * gae
            advantages[t] = gae
        advantages = ((advantages - np.mean(advantages)) / (np.std(advantages) + self.eps))
        return advantages

    def returns(self, advantages: np.ndarray, values: np.ndarray):
        """
        Function to compute the expected return.
        Parameters
        ----------
        advantages : np.ndarray (n_time_steps, n_particles)
                A numpy array of advantages
        values : np.ndarray (n_time_steps, n_particles)
                The prediction of the critic for the episode.
        Returns
        -------
        expected_returns : np.ndarray (n_time_steps, n_particles)
                Expected returns for the rewards.
        """
        returns = advantages + values
        return returns
