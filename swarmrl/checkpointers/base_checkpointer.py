"""
Module for the BaseCheckpointer parent.
"""

import logging
import shutil
from collections import deque
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseCheckpointer:
    """
    A base class for all checkpointers.

    Attributes
    ----------

    """

    def __init__(self, out_path: str | None, n_buffer: int | None = None):
        """
        Initializes the checkpointer with the specified output path.
        Args:
            out_path str: Path to the folder where the models should be stored.
            n_buffer int | None: Number of latest checkpoints to keep.
                    If None, all checkpoints are kept.

        """

        if n_buffer is not None and n_buffer <= 0:
            raise ValueError("n_buffer must be greater than 0 when specified")

        self.out_path = out_path
        self.n_buffer = n_buffer
        self._save_callback: Callable[[str], None] | None = None
        self._saved_paths: deque[tuple[str, str]] = deque()

    def check_for_checkpoint(self, *args, **kwargs) -> bool:
        """
        Determine if a checkpoint should be made.

        Returns
        -------
        bool
            Whether or not to save a checkpoint.
        """
        raise NotImplementedError("This method must be implemented in subclass")

    def set_save_callback(self, save_callback: Callable[[str], None]):
        """
        Set the save callback used by this checkpointer.

        Parameters
        ----------
        save_callback : Callable[[str], None]
                Callback that writes a checkpoint to the provided directory.
        """
        self._save_callback = save_callback

    def get_checkpoint_reason(self) -> str:
        """
        Return the string token used in the checkpoint directory name.
        """
        return self.__class__.__name__

    def evaluate_and_save(
        self,
        rewards,
        current_episode: int,
        current_reward: float,
        default_checkpoint_path: str | None = None,
    ) -> str | None:
        """
        Evaluate checkpoint criterion and persist model when criterion is met.

        Returns
        -------
        str | None
            Saved directory path or None if no checkpoint was saved.
        """
        if not self.check_for_checkpoint(rewards, current_episode):
            return None
        if self._save_callback is None:
            raise RuntimeError("Save callback not configured for checkpointer")

        checkpoint_root = (
            self.out_path if self.out_path is not None else default_checkpoint_path
        )
        if checkpoint_root is None:
            raise ValueError("No checkpoint path provided via checkpointer or manager")

        save_directory = self._build_checkpoint_save_directory(
            checkpoint_root=checkpoint_root,
            current_episode=current_episode,
            current_reward=current_reward,
        )
        self._save_callback(save_directory)
        self._track_saved_path_and_cleanup_old_checkpoints(
            save_directory, checkpoint_root
        )
        return save_directory

    def _build_checkpoint_save_directory(
        self,
        checkpoint_root: str,
        current_episode: int,
        current_reward: float,
    ) -> str:
        """
        Build the checkpoint save directory path for one checkpoint event.
        """
        return (
            f"{checkpoint_root}/Model-ep_{current_episode + 1:06d}"
            f"-reward_{float(current_reward):.2f}"
            f"-reason_{self.get_checkpoint_reason()}/"
        )

    def _track_saved_path_and_cleanup_old_checkpoints(
        self,
        save_directory: str,
        checkpoint_root: str,
    ):
        """
        Track a new checkpoint directory and apply retention limits.
        """
        self._saved_paths.append((save_directory, checkpoint_root))
        if self.n_buffer is None:
            return

        while len(self._saved_paths) > self.n_buffer:
            old_directory, old_checkpoint_root = self._saved_paths.popleft()
            old_path = Path(old_directory)
            root_path = Path(old_checkpoint_root)

            if not self._is_safe_checkpoint_deletion_target(old_path, root_path):
                logger.warning(
                    f"Refusing to delete unsafe checkpoint path '{old_directory}'"
                )
                continue

            try:
                shutil.rmtree(old_directory)
            except FileNotFoundError:
                logger.debug(f"Checkpoint path already removed: {old_directory}")
            except OSError as e:
                logger.warning(f"Failed deleting old checkpoint '{old_directory}': {e}")

    @staticmethod
    def _is_safe_checkpoint_deletion_target(path: Path, root: Path) -> bool:
        """
        Check whether a directory is a safe checkpoint deletion target.
        """
        if not path.exists() or not path.is_dir():
            return False
        if not path.name.startswith("Model-ep_"):
            return False
        if path.resolve() == root.resolve():
            return False
        return True

    def check_for_break(self, *args, **kwargs) -> bool:
        """
        Determine if the simulation should be stopped.

        Returns
        -------
        bool
            Whether or not the simulation should be ended.
        """
        return False

    def get_stop_episode(self) -> int:
        """
        Get the episode at which training should stop.

        Returns
        -------
        int
            The episode number at which to stop training, or -1 if not set.
        """
        return -1
