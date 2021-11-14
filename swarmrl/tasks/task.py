"""
Module for the parent class of the tasks.

Notes
-----
The reward classes handle the computation of the reward from an environment and
compute the loss for the models to train on.
"""
import torch


class Task(torch.nn.Module):
    """
    Parent class for the reinforcement learning tasks.
    """
    def __init__(self):
        """
        Constructor for the reward class.
        """
        super(Task, self).__init__()

    @classmethod
    def compute_reward(self):
        """
        Compute the reward for the current state.

        Returns
        -------

        """
        raise NotImplementedError("Implemented in child class.")
