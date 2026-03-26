"""
Module for the CheckpointManager
"""

from collections.abc import Callable

import numpy as np

from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer


class CheckpointManager:
    """
    Evaluate checkpointers and trigger model exports through a callback.
    """

    def __init__(
        self,
        checkpointers: list[BaseCheckpointer],
        checkpoint_path: str,
        save_callback: Callable[[str], None],
    ):
        """
        Parameters
        ----------
        checkpointers : list[BaseCheckpointer]
                Active checkpointers to evaluate.
        checkpoint_path : str
                Base path where checkpoints are stored.
                If individual checkpointers have no explicit out_path
                defined, checkpoints will be stored here.
        save_callback : Callable[[str], None]
                Callback used to persist model states.
        """
        self.checkpointers = checkpointers
        self.checkpoint_path = checkpoint_path
        self.last_saved_paths: list[str] = []

        for checkpointer in self.checkpointers:
            checkpointer.set_save_callback(save_callback)

    def check_and_save(
        self,
        rewards: np.ndarray,
        current_episode: int,
        current_reward: float,
    ) -> bool:
        """
        Evaluate all checkpointers and save a model if any criterion is met.

        Returns
        -------
        bool
            Whether a checkpoint has been written.
        """
        self.last_saved_paths = []

        for checkpointer in self.checkpointers:
            saved_path = checkpointer.evaluate_and_save(
                rewards=rewards,
                current_episode=current_episode,
                current_reward=current_reward,
                default_checkpoint_path=self.checkpoint_path,
            )
            if saved_path is not None:
                self.last_saved_paths.append(saved_path)

        return len(self.last_saved_paths) > 0

    def should_stop_training(self) -> tuple[bool, int]:
        """
        Check whether training should stop and at which episode.
        "First break wins".

        Returns
        -------
        tuple[bool, int]
            `(break_training, stop_after_episode)` where `stop_after_episode` is
            `-1` when no stopping criterion is active.
        """
        for checkpointer in self.checkpointers:
            if checkpointer.check_for_break():
                return True, checkpointer.get_stop_episode()
        return False, -1
