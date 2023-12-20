"""
Helper module to instantiate several modules.
"""

from swarmrl.networks.flax_network import FlaxModel
from swarmrl.networks.graph_network import GraphModel

__all__ = [FlaxModel.__name__, GraphModel.__name__]
