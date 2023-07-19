"""
Module for the expected returns value function.
"""
import logging

import jax.numpy as np

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
        (advantages, returns) :     (np.ndarray (n_time_steps, n_particles),
                                    np.ndarray (n_time_steps, n_particles)
                                    )
                The advantage and the expected return for the episode.
        """

        gae = 0

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            if t != len(rewards) - 1:
                returns.at[t].set(rewards[t] + self.gamma * returns[t + 1])
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            else:
                # print(rewards[t], values[t])
                delta = rewards[t] - values[t]
                returns.at[t].set(rewards[t])

            gae = delta + self.gamma * self.lambda_ * gae
            advantages.at[t].set(gae)

        # returns = advantages + values
        advantages = (advantages - np.mean(advantages)) / (
            np.std(advantages) + self.eps
        )

        # returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)

        return advantages, returns
