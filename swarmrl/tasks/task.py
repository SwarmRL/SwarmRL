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

        Parameters
        ----------
        engine : Engine
                SwarmRL engine used to generate environment data. Could be, e.g. a
                simulation or an experiment.
        """
        super(Task, self).__init__()

    @classmethod
    def forward(self, observables: torch.Tensor):
        """
        Compute the reward on the whole group of particles.

        Parameters
        ----------
        observables : torch.Tensor
                Observables collected during the episode.


        Returns
        -------

        """
        raise NotImplementedError("Implemented in child class.")
