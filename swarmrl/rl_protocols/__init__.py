"""
Package for RL Protocols
"""
from swarmrl.rl_protocols.actor_critic import ActorCritic
from swarmrl.rl_protocols.classical_algorithm import ClassicalAlgorithm
from swarmrl.rl_protocols.rl_protocol import RLProtocol

__all__ = [ActorCritic.__name__, RLProtocol.__name__, ClassicalAlgorithm.__name__]
