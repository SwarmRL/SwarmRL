"""
Module for the Save-at-Goal-Checkpointer
"""

import numpy as np

from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer


class GoalCheckpointer(BaseCheckpointer):
    """
    Checkpointer that saves the model when a reward goal is reached.
    """

    def __init__(
        self,
        out_path: str | None,
        required_reward: float = 200,
        window_width: int = 30,
        do_goal_break: bool = False,
        running_out_length: int = 0,
        save_only_on_first_goal_hit: bool = True,
        n_buffer: int | None = 3,
    ):
        """
        Initializes the GoalCheckpointer.

        Parameters:
        -----------
        out_path: str | None
            Path to the folder where the models should be stored.
            If None, the manager default path is used.
        required_reward: float
            The reward that needs to be achieved to trigger a checkpoint.
        window_width: int
            Determines how many episodes should be considered
            for the running reward average.
        do_goal_break: bool
            Whether to stop training after the goal is reached.
        running_out_length: int
            The number of episodes to run after the goal is reached
            before stopping training.
        save_only_on_first_goal_hit: bool
            If True, trigger checkpoint saving only once on the first
            goal hit. Further goal hits will not trigger additional saves,
            making the n_buffer obsolete for this case.
        n_buffer : int | None
            Number of latest checkpoints to keep for this checkpointer.
        """
        if window_width <= 0:
            raise ValueError("window_width must be greater than 0")
        if running_out_length < 0:
            raise ValueError("running_out_length must not be negative")
        super().__init__(out_path, n_buffer=n_buffer)
        self.required_reward = required_reward
        self.window_width = window_width
        self.do_goal_break = do_goal_break
        self.running_out_length = running_out_length
        self.save_only_on_first_goal_hit = save_only_on_first_goal_hit
        self.break_training = False
        self.stop_episode = -1
        self._did_save_on_goal_hit = False

    def check_for_checkpoint(self, rewards: np.ndarray, current_episode: int) -> bool:
        """
        Check if the average reward in the window is greater than the required reward.

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

        goal_hit = avg_reward >= self.required_reward
        if not goal_hit:
            return False

        if self.do_goal_break:
            if not self.break_training:
                self.break_training = True
                if self.running_out_length > 0:
                    self.stop_episode = current_episode + self.running_out_length
                else:
                    self.stop_episode = current_episode

        if self.save_only_on_first_goal_hit:
            if self._did_save_on_goal_hit:
                return False
            self._did_save_on_goal_hit = True

        return True

    def check_for_break(self):
        return self.break_training

    def get_stop_episode(self):
        return self.stop_episode
