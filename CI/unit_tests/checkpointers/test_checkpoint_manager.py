"""
Test the checkpoint manager module.
"""

import numpy as np

from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer
from swarmrl.checkpointers.checkpoint_manager import CheckpointManager


class TrueCheckpointer(BaseCheckpointer):
    def check_for_checkpoint(self, *args, **kwargs) -> bool:
        return True


class FalseCheckpointer(BaseCheckpointer):
    def check_for_checkpoint(self, *args, **kwargs) -> bool:
        return False


class TestCheckpointManager:
    """
    Test suite for the checkpoint manager module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the class for testing.
        """
        cls.saved_paths_true = []
        cls.saved_paths_false = []

        cls.true_manager = CheckpointManager(
            checkpointers=[TrueCheckpointer(out_path="/dev/null")],
            checkpoint_path="/tmp/checkpoints",
            save_callback=cls.saved_paths_true.append,
        )

        cls.false_manager = CheckpointManager(
            checkpointers=[FalseCheckpointer(out_path="/dev/null")],
            checkpoint_path="/tmp/checkpoints",
            save_callback=cls.saved_paths_false.append,
        )

    def test_initialization(self):
        """
        Test the initialization of the checkpoint manager.
        """
        assert self.true_manager.checkpoint_path == "/tmp/checkpoints"
        assert len(self.true_manager.checkpointers) == 1
        assert isinstance(self.true_manager.checkpointers[0], TrueCheckpointer)

    def test_check_and_save_triggers_callback(self):
        """
        Save callback should be called once if any checkpointer triggers.
        """
        self.saved_paths_true.clear()

        did_save = self.true_manager.check_and_save(
            rewards=np.array([1.0, 2.0, 3.0]),
            current_episode=2,
            current_reward=3.0,
        )

        assert did_save
        assert len(self.saved_paths_true) == 1
        assert "Model-ep_3" in self.saved_paths_true[0]
        assert "-cur_reward_3.0" in self.saved_paths_true[0]
        assert "-TrueCheckpointer" in self.saved_paths_true[0]

    def test_check_and_save_does_not_trigger_without_export(self):
        """
        Save callback should not be called if no checkpointer triggers.
        """
        self.saved_paths_false.clear()

        did_save = self.false_manager.check_and_save(
            rewards=np.array([1.0, 2.0, 3.0]),
            current_episode=2,
            current_reward=3.0,
        )

        assert not did_save
        assert len(self.saved_paths_false) == 0
