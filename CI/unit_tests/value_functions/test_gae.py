from swarmrl.value_functions.generalized_advantage_estimate import GAE
import jax.numpy as np
import numpy as onp
import jax


class TestGAE:
    def test_gae(self):
        gae = GAE(gamma=0.95, lambda_=0.95)
        rewards = np.array([1, 2, 3, 4, 5])
        values = np.array([1, 1, 1, 1, 1])

        expected_returns = [9.006775, 7.05697, 5.221145, 3.48829, 1.857385]
        expected_advantages = [7.05697, 4.15004, 2.17116, 0.538415, -0.857385]

        returns, advantages = gae(rewards, values)

        onp.testing.assert_allclose(returns, expected_returns, rtol=1e-4, atol=1e-4)
        onp.testing.assert_allclose(advantages, expected_advantages, rtol=1e-4, atol=1e-4)
