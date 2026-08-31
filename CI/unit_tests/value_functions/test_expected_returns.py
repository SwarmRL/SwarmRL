"""
Test the expected return module.
"""

import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal, assert_array_equal

from swarmrl.value_functions.expected_returns import ExpectedReturns


class TestExpectedReturns:
    """
    Test suite for the expected returns module.
    """

    def test_unstandardized_returns(self):
        """
        Test that unstandardized returns work correctly.
        """
        # Sum over the rewards starting from index i, (0, 0), 1 + 2 + 3 and so on.
        true_values = jnp.array([[6.0, 15.0], [5.0, 11.0], [3.0, 6.0]])

        # Trivial gamma function for analytic simplicity.
        value_function = ExpectedReturns(gamma=1.0, standardize=False)

        # 2 particles, 3 time steps
        rewards = jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        values = jnp.zeros((4, 2))
        terminated = jnp.array([False, False, True])
        truncated = jnp.zeros_like(terminated)

        expected_returns = value_function(rewards, values, terminated, truncated)

        assert_array_equal(expected_returns, true_values)

    def test_unstandardized_returns_discount_consecutive_rewards(self):
        """Discount rewards by their distance from the current transition."""
        rewards = jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        true_values = jnp.array([[2.75, 8.0], [3.5, 8.0], [3.0, 6.0]])
        values = jnp.zeros((4, 2))
        terminated = jnp.array([False, False, True])
        truncated = jnp.zeros_like(terminated)
        value_function = ExpectedReturns(gamma=0.5, standardize=False)

        expected_returns = value_function(rewards, values, terminated, truncated)

        assert_array_equal(expected_returns, true_values)

    def test_standardized_returns(self):
        """
        Test that the standardization of the return is correct.
        """
        value_function = ExpectedReturns(gamma=0.79, standardize=True)

        # True values
        true_mean = jnp.array([0.0, 0.0])
        true_std = jnp.array([1.0, 1.0])

        # 2 particles, 3 time steps
        rewards = jnp.array([
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
            [5.0, 8.0],
            [6.0, 9.0],
            [7.0, 10.0],
        ])
        values = jnp.zeros((8, 2))
        terminated = jnp.array([False] * 6 + [True])
        truncated = jnp.zeros_like(terminated)

        expected_returns = value_function(rewards, values, terminated, truncated)

        mean_vector = jnp.mean(expected_returns, axis=0)
        std_vector = jnp.std(expected_returns, axis=0)

        assert_array_almost_equal(mean_vector, true_mean, decimal=6)
        assert_array_almost_equal(std_vector, true_std, decimal=6)

    def test_rollout_boundary_bootstraps_from_final_value(self):
        value_function = ExpectedReturns(gamma=1.0, standardize=False)
        rewards = jnp.array([[1.0], [2.0]])
        values = jnp.array([[0.0], [0.0], [10.0]])
        terminated = jnp.array([False, False])
        truncated = jnp.array([False, False])

        returns = value_function(rewards, values, terminated, truncated)

        assert_array_equal(returns, jnp.array([[13.0], [12.0]]))

    def test_termination_zeros_bootstrap(self):
        value_function = ExpectedReturns(gamma=1.0, standardize=False)
        rewards = jnp.array([[1.0], [2.0]])
        values = jnp.array([[0.0], [0.0], [10.0]])
        terminated = jnp.array([False, True])
        truncated = jnp.array([False, False])

        returns = value_function(rewards, values, terminated, truncated)

        assert_array_equal(returns, jnp.array([[3.0], [2.0]]))

    def test_truncation_bootstraps_but_stops_return_propagation(self):
        value_function = ExpectedReturns(gamma=1.0, standardize=False)
        rewards = jnp.array([[1.0], [2.0], [3.0]])
        values = jnp.array([[0.0], [10.0], [20.0], [30.0]])
        terminated = jnp.array([False, False, False])
        truncated = jnp.array([False, True, False])

        returns = value_function(rewards, values, terminated, truncated)

        assert_array_equal(returns, jnp.array([[23.0], [22.0], [33.0]]))
