from swarmrl.value_functions.generalized_advantage_estimate import GAE
import numpy as np


def test_gae():
    # Create test data
    rewards = np.array([1, 2, 3, 4, 5])
    values = np.array([5, 4, 3, 2, 1])

    # Create an instance of the GAE class
    gae = GAE()

    # Test the __call__ method
    advantages = gae(rewards, values)
    expected_advantages = np.array([-1, -1, -1, -1, -1])
    assert np.array_equal(advantages,
                          expected_advantages), f"Expected {expected_advantages}, but got {advantages}"

    # Test the returns method
    expected_returns = np.array([4, 3, 2, 1, 0])
    returns = gae.returns(advantages, values)
    assert np.array_equal(returns,
                          expected_returns), f"Expected {expected_returns}, but got {returns}"


test_gae()