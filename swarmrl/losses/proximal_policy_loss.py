"""
Loss functions based on Proximal policy optimization.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""
from abc import ABC
import torch
from swarmrl.losses.loss import Loss


class ProximalPolicyLoss(Loss, ABC):
    """
    Class to implement the proximal policy loss.
    """
    def _compute_ratios(self):
        """
        Compute the policy ratios.

        Returns
        -------

        """
        pass

    def _compute_surrogates(self):
        """
        Compute the surrogates.
        Returns
        -------

        """
        pass

    def _compute_actor_loss(self):
        """
        Compute the actor loss.

        Returns
        -------

        """
        pass

    def _compute_critic_loss(self):
        """
        Compute the critic loss.

        Returns
        -------

        """
        pass

    def compute_loss(
            self,
            log_probabilities: torch.Tensor,
            values: torch.Tensor,
            rewards: torch.Tensor,
            entropy: torch.Tensor
    ):
        """
        Compute the Proximal policy loss.

        Notes
        -----
        All inputs are of the shape (n_particles, n_episodes)
        """
        # Compute ratios
        # Compute true value
        # Compute advantage
        # Compute surrogates
        # Compute actor/critic loss
