import jax.numpy as jnp
import numpy as onp

from swarmrl.value_functions.generalized_advantage_estimate import GAE


class TestGAE:
    def test_gae(self):
        gae = GAE(gamma=1, lambda_=1)
        rewards = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        values = jnp.array([1, 2, 3, 4, 5, 0])
        terminated = jnp.array([False, False, False, False, True])
        truncated = jnp.zeros_like(terminated)

        expected_advantages = jnp.array([4, 2, 0, -2, -4])

        expected_returns = expected_advantages + values[:-1]

        expected_advantages = (expected_advantages - jnp.mean(expected_advantages)) / (
            jnp.std(expected_advantages) + jnp.finfo(jnp.float32).eps.item()
        )

        advantages, returns = gae(rewards, values, terminated, truncated)

        onp.testing.assert_allclose(
            advantages, expected_advantages, rtol=1e-4, atol=1e-4
        )
        onp.testing.assert_allclose(returns, expected_returns, rtol=1e-4, atol=1e-4)

    def test_rollout_boundary_bootstraps_from_final_value(self):
        gae = GAE(gamma=1, lambda_=1)
        rewards = jnp.array([1.0, 1.0])
        values = jnp.array([10.0, 20.0, 30.0])
        terminated = jnp.array([False, False])
        truncated = jnp.array([False, False])

        _, returns = gae(rewards, values, terminated, truncated)

        onp.testing.assert_allclose(returns, jnp.array([32.0, 31.0]))

    def test_termination_zeros_bootstrap(self):
        gae = GAE(gamma=1, lambda_=1)
        rewards = jnp.array([1.0, 1.0])
        values = jnp.array([10.0, 20.0, 30.0])
        terminated = jnp.array([False, True])
        truncated = jnp.array([False, False])

        _, returns = gae(rewards, values, terminated, truncated)

        onp.testing.assert_allclose(returns, jnp.array([2.0, 1.0]))

    def test_truncation_bootstraps_but_stops_advantage_propagation(self):
        gae = GAE(gamma=1, lambda_=1)
        rewards = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([0.0, 10.0, 20.0, 30.0])
        terminated = jnp.array([False, False, False])
        truncated = jnp.array([False, True, False])

        _, returns = gae(rewards, values, terminated, truncated)

        onp.testing.assert_allclose(returns, jnp.array([23.0, 22.0, 33.0]))
