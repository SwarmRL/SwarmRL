"""
Test the goal checkpointer module
"""

import numpy as np

from swarmrl.checkpointers.goal_checkpointer import GoalCheckpointer


class TestGoalCheckpointer:
    """
    Test suite for the goal checkpointer module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the class for testing.
        """
        cls.goal_checkpointer = GoalCheckpointer(
            out_path="/dev/null/",
            required_reward=100,
            running_out_length=2,
            window_width=3,
            do_goal_break=True,
        )

        cls.goal_checkpointer_2 = GoalCheckpointer(
            out_path="/dev/null/",
            required_reward=100,
            running_out_length=0,
            window_width=3,
            do_goal_break=True,
        )

    def test_initialization(self):
        """
        Test the initialization of the goal checkpointer.
        """
        assert self.goal_checkpointer.required_reward == 100
        assert self.goal_checkpointer.running_out_length == 2
        assert self.goal_checkpointer.window_width == 3
        assert self.goal_checkpointer.DO_GOAL_BREAK
        assert not self.goal_checkpointer.check_for_break()
        assert self.goal_checkpointer.stop_episode == -1

        assert self.goal_checkpointer_2.required_reward == 100
        assert self.goal_checkpointer_2.running_out_length == 0
        assert self.goal_checkpointer_2.window_width == 3
        assert self.goal_checkpointer_2.DO_GOAL_BREAK

    def test_check_for_checkpoint(self):
        """
        Test the check for checkpoint method for the first setup.
        """
        rewards = np.linspace(0, 300, 10)

        for i in range(len(rewards)):
            if i in [4, 5, 6, 7]:
                assert self.goal_checkpointer.check_for_checkpoint(rewards, i)
            else:
                assert not self.goal_checkpointer.check_for_checkpoint(rewards, i)
            if self.goal_checkpointer.BREAK_TRAINING:
                break

    def test_check_for_checkpoint_2(self):
        """
        Test the check for checkpoint method for the second setup.
        """
        rewards = np.linspace(0, 300, 10)

        for i in range(len(rewards)):
            if i in [4]:
                assert self.goal_checkpointer_2.get_stop_episode() == -1
                assert self.goal_checkpointer_2.check_for_checkpoint(rewards, i)
                assert self.goal_checkpointer_2.get_stop_episode() == 4
            else:
                assert not self.goal_checkpointer_2.check_for_checkpoint(rewards, i)

            if self.goal_checkpointer_2.BREAK_TRAINING:
                break
