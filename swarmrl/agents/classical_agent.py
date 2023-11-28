"""
Class for classical agents. These are agents not controlled by
machine learning. They should also not be trainable.
"""

import typing

from swarmrl.components.colloid import Colloid
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task


class ClassicalAgent:
    """
    Class to handle the actor-critic RL Protocol.
    """

    def __init__(
        self,
        particle_type: int,
        actions: dict,
        task: Task = None,
        observable: Observable = None,
    ):
        """
        Constructor for the actor-critic protocol.

        Parameters
        ----------
        network : Network
                Shared Actor-Critic Network for the RL protocol. The apply function
                should return a tuple of (logits, value).
        particle_type : int
                Particle ID this RL protocol applies to.
        observable : Observable
                Observable for this particle type and network input
        task : Task
                Task for this particle type to perform.
        actions : dict
                Actions allowed for the particle.
        """
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.actions = actions

    def compute_agent_action(self, **kwargs):
        """
        Function implemented by specific algorithm to compute the
        action of the agent.
        """
        raise NotImplementedError("Implemented in Child class.")

    def compute_agent_state(self, colloids: typing.List[Colloid]):
        """
        Copmute the new state for the agent.

        Returns the chosen actions to the force function which
        talks to the espresso engine.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids in the system.
        """
        chosen_actions = self.compute_agent_action()

        return chosen_actions, self.task.kill_switch
