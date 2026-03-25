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
        save_callback : Callable[[str], None]
                Callback used to persist model states.
        """
        self.checkpointers = checkpointers
        self.checkpoint_path = checkpoint_path
        self.save_callback = save_callback

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
        export = []
        save_string = ""

        for checkpointer in self.checkpointers:
            should_export = checkpointer.check_for_checkpoint(rewards, current_episode)
            export.append(should_export)
            if should_export:
                save_string += f"-{checkpointer.__class__.__name__}"

        if not any(export):
            return False

        self.save_callback(
            f"{self.checkpoint_path}/Model-ep_{current_episode + 1}"
            f"-cur_reward_{current_reward:.1f}"
            f"{save_string}/"
        )
        return True
