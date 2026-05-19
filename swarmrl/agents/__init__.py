"""
Package for Agents.
"""

from swarmrl.agents import bechinger_models, dummy_models, find_point
from swarmrl.agents.actor_critic import ActorCriticAgent
from swarmrl.agents.agent import Agent
from swarmrl.agents.classical_agent import ClassicalAgent
from swarmrl.agents.sac_agent import SACAgent

__all__ = [
    ActorCriticAgent.__name__,
    SACAgent.__name__,
    ClassicalAgent.__name__,
    Agent.__name__,
    bechinger_models.__name__,
    find_point.__name__,
    dummy_models.__name__,
]
