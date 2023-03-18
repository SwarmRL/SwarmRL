"""
Test the categorical distribution.
"""
import jax.numpy as np
import numpy as onp
from numpy.testing import assert_array_almost_equal

from swarmrl.sampling_strategies.categorical_distribution import CategoricalDistribution


class TestCategorical:
    """
    Test suite for the categorical distribution.
    """

    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """
        cls.even_logits = np.array([6.0, 6.0, 6.0, 6.0])
        cls.even_probabilities = np.array([0.25, 0.25, 0.25, 0.25])

        cls.definite_logits = np.array([10.0, 0.0])
        cls.definite_probabilities = np.array([1.0, 0.0])

    def test_even_logits(self):
        """
        Test the case for no noise and even logits.
        """
        sampler = CategoricalDistribution(noise="none")

        outcomes = onp.array([0, 0, 0, 0])
        for _ in range(500):
            outcomes[sampler(self.even_logits)] += 1

        assert_array_almost_equal(outcomes / 500, self.even_probabilities, decimal=1)

    def test_gaussian_noise(self):
        """
        Test that the Gaussian noise statistics are correct.
        """
        sampler = CategoricalDistribution(noise="gaussian")
        sampler(self.definite_logits)

        outcomes = onp.array([0, 0])
        for _ in range(500):
            outcomes[sampler(self.definite_probabilities)] += 1

        assert outcomes[1] != 0.0

    def test_uniform_noise(self):
        """
        Test that the Gaussian noise statistics are correct.
        """
        sampler = CategoricalDistribution(noise="uniform")
        sampler(self.definite_logits)

        outcomes = onp.array([0, 0])
        for _ in range(500):
            outcomes[sampler(self.definite_probabilities)] += 1

        assert outcomes[1] != 0.0

    def test_multi_colloid(self):
        """
        Ensure the sampler works for many colloids.
        """
        sampler = CategoricalDistribution()

        logits = np.array([[3.0, 3.0, 3.0, 3.0], [100.0, 0.0, 0.0, 0.0]])
        probabilities = np.array([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])

        # Collect points for different cases.
        single_outcomes_0 = onp.array([0.0, 0.0, 0.0, 0.0])
        single_outcomes_1 = onp.array([0.0, 0.0, 0.0, 0.0])
        full_outcomes = onp.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

        for _ in range(500):
            single_outcomes_0[sampler(logits[0])] += 1
            single_outcomes_1[sampler(logits[1])] += 1
            full_outcome_data = sampler(logits)
            full_outcomes[0][full_outcome_data[0]] += 1
            full_outcomes[1][full_outcome_data[1]] += 1

        assert_array_almost_equal(
            full_outcomes[0] / 500, single_outcomes_0 / 500, decimal=1
        )
        assert_array_almost_equal(
            full_outcomes[1] / 500, single_outcomes_1 / 500, decimal=1
        )
        assert_array_almost_equal(full_outcomes / 500, probabilities, decimal=1)
