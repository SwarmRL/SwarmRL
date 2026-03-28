"""
Module for the SaveOnDecline Checkpointer
"""

import numpy as np

from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer


class OnDeclineCheckpointer(BaseCheckpointer):
    """
    Checkpointer that saves the current model if the reward starts to decline.
    This model could then be used as a backup in case of forgetting.
    """

    def __init__(
        self,
        out_path: str | None,
        min_reward: float = 250,
        window_width: int = 30,
        wait_time: int = 10,
        n_buffer: int | None = 3,
    ):
        """
        Initializes the OnDeclineCheckpointer.

        Parameters:
        -----------
        out_path: str | None
            Path to the folder where the models should be stored.
            If None, the manager default path is used.
        min_reward: float
            The minimum reward required to trigger a checkpoint.
        window_width: int
            Determines how many episodes should be considered for
            the running reward average.
        wait_time: int
            A minimum number of episodes to wait for the next checkpoint check.
            Can prevent frequent checkpoints for noisy signals.
        n_buffer : int | None
            Number of latest checkpoints to keep for this checkpointer.
        """
        if window_width <= 0:
            raise ValueError("window_width must be greater than 0")

        super().__init__(out_path, n_buffer=n_buffer)
        self.min_reward = min_reward
        self.wait_time = wait_time
        self.window_width = window_width
        self.next_check_episode = -1
        self.last_reward = 0

    def check_for_checkpoint(self, rewards: np.ndarray, current_episode: int) -> bool:
        """
        Check if the average reward in the window exceeds the minimum reward
        and falls below the peak (indicating potential forgetting).

        A checkpoint is triggered when:
        - The average reward is above min_reward (performance threshold)
        - The average reward is below the historical peak (indicates decline)
        - The average reward is below the last recorded average (continuing to decline)

        This creates a dynamic minimum threshold: after each checkpoint, the current
        average becomes the new performance baseline for the next checkpoint check.

        Parameters
        ----------
        rewards : np.ndarray
            Array of rewards.
        current_episode : int
            The current episode number.

        Returns
        -------
        bool
            Whether the checkpoint criteria are met.
        """
        if rewards is None or len(rewards) == 0:
            return False
        if current_episode < 0 or current_episode >= len(rewards):
            return False

        window_end = current_episode + 1
        window_start = max(0, window_end - self.window_width)
        avg_reward = np.mean(rewards[window_start:window_end])

        if (
            current_episode > self.next_check_episode
            and self.min_reward < avg_reward < np.max(rewards[: current_episode + 1])
            and avg_reward < self.last_reward
        ):
            self.min_reward = avg_reward
            self.next_check_episode = current_episode + self.wait_time
            do_checkpoint = True
        else:
            do_checkpoint = False

        # Skip episode 0
        if current_episode != 0:
            self.last_reward = avg_reward

        return do_checkpoint
