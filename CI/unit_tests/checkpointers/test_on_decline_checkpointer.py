"""
Test the OnDeclineCheckpointer module.
"""

import numpy as np

from swarmrl.checkpointers.on_decline_checkpointer import OnDeclineCheckpointer


class TestOnDeclineCheckpointer:
    """
    Test suite for the on decline checkpointer module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the class for testing
        """
        cls.on_decline_checkpointer = OnDeclineCheckpointer(
            out_path="/dev/null",
            min_reward=200,
            wait_time=2,
            window_width=3,
        )

    def test_initialization(self):
        """
        Test the initialization of the on-decline checkpointer.
        """
        assert self.on_decline_checkpointer.min_reward == 200
        assert self.on_decline_checkpointer.out_path == "/dev/null"

        assert self.on_decline_checkpointer.window_width == 3
        assert self.on_decline_checkpointer.wait_time == 2

    def test_check_for_checkpoint(self):
        """
        Test the check for checkpoint method.
        """
        increasing_reward = np.linspace(0, 300, 11)
        decreasing_reward = np.linspace(300, 100, 11)
        again_increasing_reward = np.linspace(100, 400, 11)
        again_decreasing_reward = np.linspace(400, 200, 11)
        rewards = np.concatenate((
            increasing_reward,
            decreasing_reward,
            again_increasing_reward,
            again_decreasing_reward,
        ))

        # Test checking outside window width
        assert not self.on_decline_checkpointer.check_for_checkpoint(rewards, 1)

        for i in range(len(rewards)):
            if i in [13, 35]:
                assert self.on_decline_checkpointer.check_for_checkpoint(rewards, i)
            else:
                assert not self.on_decline_checkpointer.check_for_checkpoint(rewards, i)
