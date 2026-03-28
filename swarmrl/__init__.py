"""
__init__ file for the swarmrl package.
"""

from loguru import logger as _loguru_logger

from swarmrl import (
    agents,
    checkpointers,
    components,
    exploration_policies,
    intrinsic_reward,
    losses,
    networks,
    observables,
    sampling_strategies,
    tasks,
    trainers,
    training_routines,
    utils,
    value_functions,
)
from swarmrl.engine import espresso

# Setup a swarmrl logger but disable it.
# Use logging_utils.setup_swarmrl_logger() to actually enable/configure the logger.
_loguru_logger.disable("swarmrl")


__all__ = [
    espresso.__name__,
    utils.__name__,
    losses.__name__,
    checkpointers.__name__,
    components.__name__,
    tasks.__name__,
    observables.__name__,
    networks.__name__,
    exploration_policies.__name__,
    trainers.__name__,
    sampling_strategies.__name__,
    value_functions.__name__,
    agents.__name__,
    training_routines.__name__,
    intrinsic_reward.__name__,
]
