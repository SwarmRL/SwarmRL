import jax.numpy as np
import numpy as onp

from swarmrl.value_functions.generalized_advantage_estimate import GAE


class TestGAE:
    def test_gae(self):
        gae = GAE(gamma=1, lambda_=1)
        rewards = np.array([1, 1, 1, 1, 1])
        values = np.array([1, 2, 3, 4, 5])

        expected_advantages = np.array([4, 2, 0, -2, -4])
        expected_advantages = (expected_advantages - np.mean(expected_advantages)) / (
            np.std(expected_advantages) + np.finfo(np.float32).eps.item()
        )
        expected_returns = expected_advantages + values

        advantages = gae(rewards, values)
        returns = gae.returns(advantages, values)

        onp.testing.assert_allclose(
            advantages, expected_advantages, rtol=1e-4, atol=1e-4
        )
        onp.testing.assert_allclose(returns, expected_returns, rtol=1e-4, atol=1e-4)
