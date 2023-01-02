"""
Module for the expected returns value function.
"""
import logging

import jax.numpy as jnp

logger = logging.getLogger(__name__)


class GAE:
    """
    Class for the expected returns.
    """

    def __init__(
            self, gamma: float = 0.99,
            lambda_discount: float = 0.9,
            standardize: bool = True
    ):
        """
        Constructor for the Expected returns class

        Parameters
        ----------
        gamma : float
                Trajectory decay factor for the values of the task each time step.
        lambda_discount : float
                Exponential mean discount.
        standardize : bool
                If True, standardize the results of the calculation.

        Notes
        -----
        See https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
        for more information.
        """
        self.gamma = gamma
        self.lambda_discount = lambda_discount
        self.standardize = standardize

        # Set by us to stabilize division operations.
        self.eps = jnp.finfo(jnp.float32).eps.item()

    def _gae_calculation(self, rewards):
        advantages = []
        gae = 0.
        for t in reversed(range(len(rewards))):
            # Masks used to set next state value to 0 for terminal states.
            value_diff = self.gamma * values[t + 1] * terminal_masks[t] - values[t]
            delta = rewards[t] + value_diff
            # Masks[t] used to ensure that values before and after a terminal state
            # are independent of each other.
            gae = delta + self.gamma * self.lambda_discount * terminal_masks[t] * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        return jnp.array(advantages)

    def __call__(self, rewards: jnp.ndarray):
        """
        Call function for the expected returns.
        Parameters
        ----------
        rewards : np.ndarray (n_time_steps, n_particles, dimension)
                A numpy array of rewards to use in the calculation.

        Returns
        -------
        expected_returns : np.ndarray (n_time_steps, n_particles)
                Expected returns for the rewards.
        """
        logger.debug(f"{self.gamma=}")
        expected_returns = jnp.zeros_like(rewards)
        n_particles = rewards.shape[1]

        final_time = len(rewards) + 1
        logger.debug(rewards)

        for t, reward in enumerate(rewards):
            gamma_array = self.gamma ** jnp.linspace(
                t + 1, final_time, int(final_time - (t + 1)), dtype=int
            )
            gamma_array = jnp.transpose(
                jnp.repeat(gamma_array[None, :], n_particles, axis=0)
            )

            proceeding_rewards = rewards[t:, :]

            returns = proceeding_rewards * gamma_array
            expected_returns = expected_returns.at[t, :].set(returns.sum(axis=0))

        logger.debug(f"{expected_returns=}")

        if self.standardize:
            mean_vector = jnp.mean(expected_returns, axis=0)
            std_vector = jnp.std(expected_returns, axis=0) + self.eps

            expected_returns = (expected_returns - mean_vector) / std_vector

        return expected_returns
