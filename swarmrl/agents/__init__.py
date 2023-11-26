"""
Package for RL Protocols
"""

from swarmrl.agents.actor_critic import ActorCriticAgent
from swarmrl.agents.classical_agent import ClassicalAgent

__all__ = [ActorCriticAgent.__name__, ClassicalAgent.__name__]
