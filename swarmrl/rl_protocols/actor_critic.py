"""
Module for the Actor-Critic RL protocol.
"""
import numpy as np
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.rl_protocols.rl_protocol import RLProtocol
from swarmrl.tasks.task import Task

class ActorCritic(RLProtocol):
    """
    Class to handle the actor-critic RL Protocol.
    """

    def __init__(
        self,
        particle_type: int,
        actor: Network,
        critic: Network,
        task: Task,
        observable: Observable,
        actions: dict,
    ):
        """
        Constructor for the actor-critic protocol.

        Parameters
        ----------
        actor : Network
                Actor network for the RL protocol.
        critic : Network
                Critic network for the RL protocol.
        particle_type : int
                Particle ID this RL protocol applies to.
        observable : Observable
                Observable for this particle type and network input
        task : Task
                Task for this particle type to perform.
        actions : dict
                Actions allowed for the particle.
        """
        self.actor = actor
        self.critic = critic
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.actions = actions

    def compute_episode_step(self,
                             item,
                             colloids,
                             actions,
    ):
        """

        """
        observables_computed = self.observable.compute_observable(colloids)
        rewards = self.task(colloids)

        action_indices, log_probs = self.actor.compute_action(
            observables=observables_computed, explore_mode=False)

        chosen_actions = np.take(
            list(self.actions.values()), action_indices, axis=-1
        )

        count = 0  # Count the colloids of a specific species.
        for colloid in colloids:
            if str(colloid.type) == item:
                actions[colloid.id] = chosen_actions[count]
                count += 1

        return observables_computed, rewards, actions
