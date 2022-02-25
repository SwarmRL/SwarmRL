"""
Module for the loss parent class.
"""
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task


class Loss:
    """
    Parent class for a SwarmRL loss model.
    """

    def compute_loss(
        self,
        actor: Network,
        critic: Network,
        observable: Observable,
        episode_data: list,
        task: Task
    ):
        """
        Compute loss on models.

        Parameters
        ----------
        actor : Network
                Actor network to train
        critic : Network
                Critic network to train.
        observable : Observable
                Observable class to compute observables with.
        episode_data : dict
                A dictionary of episode data.
        task : Task
                Task class from which to compute the reward.
        """
        raise NotImplementedError("Implemented in child class.")
