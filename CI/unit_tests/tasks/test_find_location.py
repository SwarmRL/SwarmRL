"""
Test the find origin task.
"""
import numpy as np
import pytest
import torch

from swarmrl.tasks.searching.find_location import FindLocation


class TestFindLocation:
    """
    Test the find location task.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        cls.task = FindLocation()

    def test_simple_box(self):
        """
        Test the reward for the default unit cube and origin.
        """
        bad_reward = self.task.compute_reward(torch.tensor([1, 1, 0]))
        good_reward = self.task.compute_reward(torch.tensor([0, 0, 0]))
        mixed_reward = self.task.compute_reward(torch.tensor([1.0, 0.0, 0]))

        assert bad_reward == 0.0
        assert good_reward == pytest.approx(1.414, 0.001)
        assert mixed_reward == pytest.approx(0.414, 0.001)

    def test_box_shift(self):
        """
        Test the reward for a shifted box.
        """
        self.task.location = np.array([0.9, 0.9, 0.0])
        good_reward = self.task.compute_reward(torch.tensor([1, 1, 0]))
        bad_reward = self.task.compute_reward(torch.tensor([0, 0, 0]))
        mixed_reward = self.task.compute_reward(torch.tensor([1.0, 0.0, 0]))

        assert good_reward == pytest.approx(1.273, 0.001)
        assert bad_reward == pytest.approx(0.1414, 0.001)
        assert mixed_reward == pytest.approx(0.509, 0.001)

    def test_larger_box(self):
        """
        Test reward_computation in a large box.

        Should return the same results as the first test on a much larger box.
        """
        self.task.location = torch.tensor([0.0, 0.0, 0.0])
        self.task.side_length = torch.tensor([1000.0, 1000.0, 1000.0])
        self.task._compute_max_distance()  # recompute this attribute
        bad_reward = self.task.compute_reward(torch.tensor([1000.0, 1000.0, 0.0]))
        good_reward = self.task.compute_reward(torch.tensor([0, 0, 0.0]))
        mixed_reward = self.task.compute_reward(torch.tensor([1000.0, 0.0, 0.0]))

        assert bad_reward == 0.0
        assert good_reward == pytest.approx(1.414, 0.001)
        assert mixed_reward == pytest.approx(0.414, 0.001)
