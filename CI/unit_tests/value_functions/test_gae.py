import jax.numpy as jnp
import numpy as onp

from swarmrl.value_functions.generalized_advantage_estimate import GAE


class TestGAE:
    def test_gae(self):
        gae = GAE(gamma=1, lambda_=1)
        rewards = jnp.array([1, 1, 1, 1, 1])
        values = jnp.array([1, 2, 3, 4, 5])

        expected_advantages = jnp.array([4, 2, 0, -2, -4])

        expected_returns = expected_advantages + values

        expected_advantages = (expected_advantages - jnp.mean(expected_advantages)) / (
            jnp.std(expected_advantages) + jnp.finfo(jnp.float32).eps.item()
        )

        advantages, returns = gae(rewards, values)

        onp.testing.assert_allclose(
            advantages, expected_advantages, rtol=1e-4, atol=1e-4
        )
        onp.testing.assert_allclose(returns, expected_returns, rtol=1e-4, atol=1e-4)
