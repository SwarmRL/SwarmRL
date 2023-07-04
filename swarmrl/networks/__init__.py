"""
Helper module to instantiate several modules.
"""
from swarmrl.networks.flax_network import FlaxModel
from swarmrl.networks.graph_network2 import GraphModel2
from swarmrl.networks.graph_network_V0 import GraphModel_V0

__all__ = [FlaxModel.__name__, GraphModel2.__name__, GraphModel_V0.__name__]
