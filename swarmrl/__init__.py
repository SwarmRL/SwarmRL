"""
__init__ file for the swarmrl package.
"""
import logging

from swarmrl.engine import espresso
from swarmrl.loss_models import loss
from swarmrl.models import bechinger_models, mlp_rl
from swarmrl.models.interaction_model import InteractionModel
from swarmrl.networks.mlp import MLP
from swarmrl.observables.position import PositionObservable
from swarmrl.tasks.find_origin import FindOrigin
from swarmrl.utils import utils

# Setup a swarmrl logger but disable it.
# Use utils.setup_swarmrl_logger() to actually enable/configure the logger.
_ROOT_NAME = __name__
_logger = logging.getLogger(_ROOT_NAME)
_logger.setLevel(logging.NOTSET)


__all__ = [
    InteractionModel.__name__,
    espresso.__name__,
    bechinger_models.__name__,
    mlp_rl.__name__,
    MLP.__name__,
    loss.__name__,
    FindOrigin.__name__,
    PositionObservable.__name__,
    utils.__name__,
]
