"""
Class to return 0 as task.
"""
import logging
from typing import List

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)

class DummyTask(Task):
    """
    Class to return 0 as task.
    """
    def __init__(self, particle_type: int = 0):
        super().__init__(particle_type=particle_type)

    def initialize(self, colloids: List[Colloid]):
        return None
    def __call__(self, colloids: List[Colloid]):
        return [0 for _ in colloids]
