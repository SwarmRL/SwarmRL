"""
Package intrinsic_rewards for Agents.
"""
import logging

logger = logging.getLogger(__name__)


from swarmrl.intrinsic_reward.intrinsic_reward import IntrinsicReward

try:
    from swarmrl.intrinsic_reward.random_network_distillation import RNDReward
    from swarmrl.intrinsic_reward.rnd_configs import RNDArchitecture, RNDConfig

    __all__ = [
        IntrinsicReward.__name__,
        RNDArchitecture.__name__,
        RNDConfig.__name__,
        RNDReward.__name__,
    ]
except ImportError:
    logger.info(
        "Could not find optional packages, certain features might be unavailable."
        )
    pass

__all__ = [
        IntrinsicReward.__name__,
    ]
