"""
Unit tests for the sampling strategy parent.
"""
import numpy as np

from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy


class TestSamplingStrategy:
    """
    Test suite for the sampling strategy parent.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare some initial attributes.
        """
        cls.strategy = SamplingStrategy()

    def test_compute_entropy(self):
        """
        Test the Shannon entropy computation.
        """
        # Test simple shapes
        probabilities = np.array([0.25, 0.25, 0.25, 0.25])
        entropy_should_be = -1 * np.log(1 / 4)
        entropy = self.strategy.compute_entropy(probabilities)
        assert entropy == entropy_should_be

        probabilities = np.array([0.85, 0.1, 0.025, 0.025])
        entropy_should_be = -1 * (probabilities * np.log(probabilities)).sum()
        entropy = self.strategy.compute_entropy(probabilities)
        assert entropy == entropy_should_be

        # Test real world shapes
        probabilities = np.array([[0.25, 0.25, 0.25, 0.25], [0.85, 0.1, 0.025, 0.025]])
        entropy_1 = -1 * np.log(1 / 4)
        entropy_2 = -1 * (probabilities[1] * np.log(probabilities[1])).sum()
        entropy_should_be = entropy_1 + entropy_2
