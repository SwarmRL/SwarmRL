"""
Package to hold all SwarmRL gyms.
"""

from swarmrl.gyms.episodic_trainer import EpisodicTrainer
from swarmrl.gyms.gym import Gym

__all__ = [Gym.__name__, EpisodicTrainer.__name__]
