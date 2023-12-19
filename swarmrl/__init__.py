"""
__init__ file for the swarmrl package.
"""

import logging

from swarmrl import (
    agents,
    components,
    exploration_policies,
    losses,
    networks,
    observables,
    sampling_strategies,
    tasks,
    trainers,
    training_routines,
    value_functions,
    intrinsic_reward,
)
from swarmrl.engine import espresso
from swarmrl.utils import utils

# Setup a swarmrl logger but disable it.
# Use utils.setup_swarmrl_logger() to actually enable/configure the logger.
_ROOT_NAME = __name__
_logger = logging.getLogger(_ROOT_NAME)
_logger.setLevel(logging.NOTSET)


__all__ = [
    espresso.__name__,
    utils.__name__,
    losses.__name__,
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
