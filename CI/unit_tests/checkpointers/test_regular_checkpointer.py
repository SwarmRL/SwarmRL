"""
Test the regular checkpointer module.
"""

import random

import numpy as np

from swarmrl.checkpointers.regular_checkpointer import RegularCheckpointer


class TestRegularCheckpointer:
    """
    Test suite for the regular checkpointer module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the class for testing.
        """
        cls.random_integer = random.randint(2, 10)
        cls.regular_checkpointer = RegularCheckpointer(
            out_path="/dev/null/", save_interval=cls.random_integer
        )

    def test_initialization(self):
        """
        Test the initialization of the regular checkpointer.
        """
        assert self.regular_checkpointer.save_interval == self.random_integer
        assert self.regular_checkpointer.out_path == "/dev/null/"

    def test_check_for_checkpoint(self):
        """
        Test the check for checkpoint method.
        """
        rewards = np.random.rand(self.random_integer * 5)

        n_backups = 0
        for episode_index in range(0, len(rewards)):
            if (episode_index + 1) % self.random_integer == 0:
                assert self.regular_checkpointer.check_for_checkpoint(
                    rewards, episode_index
                )
                n_backups += 1
            else:
                assert not self.regular_checkpointer.check_for_checkpoint(
                    rewards, episode_index
                )

        assert n_backups == len(rewards) // self.random_integer
