"""
Module for the Actor-Critic RL protocol.
"""
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.rl_protocols.rl_protocol import RLProtocol
from swarmrl.tasks.task import Task


class SharedActorCritic(RLProtocol):
    """
    Class to handle the actor-critic RL Protocol for a shared architecture.
    """

    def __init__(
        self,
        particle_type: int,
        network: Network,
        task: Task,
        observable: Observable,
        actions: dict,
    ):
        """
        Constructor for the protocol for a shared actor-critic architecture.

        Parameters
        ----------
        network : Network
                Network for the RL protocol.
        particle_type : int
                Particle ID this RL protocol applies to.
        observable : Observable
                Observable for this particle type and network input
        task : Task
                Task for this particle type to perform.
        actions : dict
                Actions allowed for the particle.
        """

        self.network = network
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.actions = actions
