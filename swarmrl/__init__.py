"""
__init__ file for the swarmrl package.
"""

import logging

from swarmrl import (
    agents,
    exploration_policies,
    gyms,
    losses,
    models,
    networks,
    observables,
    rl_protocols,
    sampling_strategies,
    tasks,
    training_routines,
    value_functions,
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
    agents.__name__,
    tasks.__name__,
    observables.__name__,
    networks.__name__,
    models.__name__,
    exploration_policies.__name__,
    gyms.__name__,
    sampling_strategies.__name__,
    value_functions.__name__,
    rl_protocols.__name__,
    training_routines.__name__,
]
