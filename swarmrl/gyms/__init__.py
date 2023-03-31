"""
Package to hold all SwarmRL gyms.
"""
from swarmrl.gyms.gym import Gym
from swarmrl.gyms.gym_shared import SharedNetworkGym

__all__ = [Gym.__name__, SharedNetworkGym.__name__]
