"""
Test the categorical distribution.
"""
import jax.numpy as np
import numpy as onp
import pytest
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
        cls.probabilities = np.array([0.25, 0.25, 0.25, 0.25])
        cls.definite_probabilities = np.array([1.0, 0.0])
        cls.logits = np.log(cls.probabilities)

    def test_no_noise_no_entropy(self):
        """
        Test the case for no noise and no entropy.
        """
        sampler = CategoricalDistribution(noise="none")
        outcomes = onp.array([0, 0, 0, 0])
        for _ in range(500):
            outcomes[sampler(self.logits, entropy=False)] += 1

        assert_array_almost_equal(outcomes / 500, self.probabilities, decimal=1)

    def test_no_noise_entropy(self):
        """
        Test the case for no noise but with entropy.
        """
        sampler = CategoricalDistribution(noise="none")

        trivial_entropy = sampler(self.logits, entropy=True)[1]

        assert trivial_entropy == 1.0

        probabilities = np.array([0.7, 0.1, 0.1, 0.1])
        logits = np.log(probabilities)

        entropy = -1 * (probabilities * np.log(probabilities)).sum()
        max_entropy = -1 * np.log(1 / 4)

        print(entropy / max_entropy)
        non_trivial_entropy = sampler(logits, entropy=True)[1]

        assert non_trivial_entropy == pytest.approx(entropy / max_entropy, 0.00001)

    def test_gaussian_noise(self):
        """
        Test that the Gaussian noise statistics are correct.
        """
        sampler = CategoricalDistribution(noise="gaussian")
        sampler(self.logits, entropy=True)

        outcomes = onp.array([0, 0])
        for _ in range(500):
            outcomes[sampler(self.definite_probabilities, entropy=False)] += 1

        assert outcomes[1] != 0.0

    def test_uniform_noise(self):
        """
        Test that the Gaussian noise statistics are correct.
        """
        sampler = CategoricalDistribution(noise="uniform")
        sampler(self.logits, entropy=True)

        outcomes = onp.array([0, 0])
        for _ in range(500):
            outcomes[sampler(self.definite_probabilities, entropy=False)] += 1

        assert outcomes[1] != 0.0
