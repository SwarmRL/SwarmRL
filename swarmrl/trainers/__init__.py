"""
Package to hold all SwarmRL trainers.
"""

from swarmrl.trainers.continuous_trainer import ContinuousTrainer
from swarmrl.trainers.episodic_trainer import EpisodicTrainer
from swarmrl.trainers.trainer import Trainer

__all__ = [ContinuousTrainer.__name__, EpisodicTrainer.__name__, Trainer.__name__]
