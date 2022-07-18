"""
Test the Gumbel distribution.
"""
import jax.numpy as np
import numpy as onp
from numpy.testing import assert_array_almost_equal

from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution


class TestCategorical:
    """
    Test suite for the Gumbel distribution.
    """

    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """
        cls.sampler = GumbelDistribution()
        cls.probabilities = np.array([0.25, 0.25, 0.25, 0.25])
        cls.logits = np.log(cls.probabilities)

    def test_simple_probabilities(self):
        """
        Test the uniform probabilities.
        """
        outcomes = onp.array([0, 0, 0, 0])
        for _ in range(500):
            outcomes[self.sampler(self.logits, entropy=False)] += 1

        assert_array_almost_equal(outcomes / 500, self.probabilities, decimal=1)

    def test_non_trivial_predictions(self):
        """
        Test case for non equal probabilities.
        """
        probabilities = np.array([0.85, 0.1, 0.025, 0.025])
        logits = np.log(probabilities)

        outcomes = onp.array([0, 0, 0, 0])
        for _ in range(1000):
            outcomes[self.sampler(logits, entropy=False)] += 1

        assert_array_almost_equal(outcomes / 1000, probabilities, decimal=1)
