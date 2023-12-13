"""
Module implementing different loss models.
"""

from swarmrl.losses.loss import Loss
from swarmrl.losses.policy_gradient_loss import PolicyGradientLoss
from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss

__all__ = [
    Loss.__name__,
    PolicyGradientLoss.__name__,
    ProximalPolicyLoss.__name__,
]
