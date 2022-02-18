"""
Module for the loss parent class.
"""
import torch


class Loss:
    """
    Parent class for a SwarmRL loss model.
    """
    def compute_loss(
            self,
            log_probabilities: torch.Tensor,
            values: torch.Tensor,
            rewards: torch.Tensor,
            entropy: torch.Tensor,
            n_particles: int,
            n_time_steps: int
    ):
        """

        Parameters
        ----------
        log_probabilities : torch.Tensor (n_particles, n_steps)
                Log of the actions predicted by the actor.
        values : torch.Tensor (n_particles, n_steps)
                Values predicted by the critic.
        rewards : torch.Tensor (n_particles, n_steps)
                Rewards for each state.
        entropy : torch.Tensor (n_particles, n_steps)
                Entropy computed from the distribution.

        Returns
        -------
        loss_tuple : tuple
                (actor_loss, critic_loss)
        """
        raise NotImplementedError("Implemented in child class.")
