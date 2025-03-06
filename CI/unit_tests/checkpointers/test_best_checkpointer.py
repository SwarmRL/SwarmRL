"""
Test the best checkpointer module.
"""

import numpy as np

from swarmrl.checkpointers.best_checkpointer import BestRewardCheckpointer


class TestBestCheckpointer:
    """
    Test suite for the best checkpointer module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the class for testing.
        """
        cls.best_checkpointer = BestRewardCheckpointer(
            min_reward=50, increase_factor=1.05, wait_time=2, window_width=3
        )
        cls.best_checkpointer_2 = BestRewardCheckpointer(
            min_reward=50, increase_factor=1.25, wait_time=0, window_width=3
        )

    def test_initialization(self):
        """
        Test the initialization of the best checkpointer.
        """
        assert self.best_checkpointer.min_reward == 50
        assert self.best_checkpointer.increase_factor == 1.05
        assert self.best_checkpointer.window_width == 3
        assert self.best_checkpointer.wait_time == 2

        assert self.best_checkpointer_2.min_reward == 50
        assert self.best_checkpointer_2.increase_factor == 1.25
        assert self.best_checkpointer_2.window_width == 3
        assert self.best_checkpointer_2.wait_time == 0

    def test_check_for_checkpoint(self):
        """
        Test the check for checkpoint method for the first setup.
        """
        # modify data, such that the reward is increasing
        # and then decreasing for 10 episodes each
        increasing_reward = np.linspace(0, 300, 11)
        decreasing_reward = np.linspace(300, 100, 11)
        rewards = np.concatenate((increasing_reward, decreasing_reward))
        for i in range(len(rewards)):
            if i in [4, 7, 10, 13]:
                assert self.best_checkpointer.check_for_checkpoint(rewards, i)
            else:
                assert not self.best_checkpointer.check_for_checkpoint(rewards, i)

    def test_check_for_checkpointer_2(self):
        """
        Test the check for checkpoint method for the second setup.
        Difference are different values for wait_time and increase factor.
        """
        # Test if this also works for a higher increase factor
        increasing_reward = np.linspace(0, 300, 11)
        decreasing_reward = np.linspace(300, 100, 11)
        rewards = np.concatenate((increasing_reward, decreasing_reward))
        for i in range(0, 23):
            checkpoint_boolean = self.best_checkpointer_2.check_for_checkpoint(
                rewards, i
            )
            if i in [4, 5, 6, 8, 10]:
                assert checkpoint_boolean
            else:
                assert not checkpoint_boolean
