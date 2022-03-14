"""
__init__ file for the swarmrl package.
"""
import logging
from swarmrl import losses, models, networks, observables, tasks
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
    "losses",
    "tasks",
    "observables",
    "networks",
    "models",
]
