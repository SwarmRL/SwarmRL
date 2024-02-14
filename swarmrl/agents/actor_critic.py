"""
Module for the Actor-Critic RL protocol.
"""

import typing
from dataclasses import dataclass, field

import numpy as np

from swarmrl.actions.actions import Action
from swarmrl.agents.agent import Agent
from swarmrl.components.colloid import Colloid
from swarmrl.losses import Loss, ProximalPolicyLoss
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task


@dataclass
class TrajectoryInformation:
    """
    Helper dataclass for training RL models.
    """

    particle_type: int
    features: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    killed: bool = False


class ActorCriticAgent(Agent):
    """
    Class to handle the actor-critic RL Protocol.
    """

    def __init__(
        self,
        particle_type: int,
        network: Network,
        task: Task,
        observable: Observable,
        actions: dict,
        loss: Loss = ProximalPolicyLoss(),
        train: bool = True,
    ):
        """
        Constructor for the actor-critic protocol.

        Parameters
        ----------
        particle_type : int
                Particle ID this RL protocol applies to.
        observable : Observable
                Observable for this particle type and network input
        task : Task
                Task for this particle type to perform.
        actions : dict
                Actions allowed for the particle.
        loss : Loss (default=ProximalPolicyLoss)
                Loss function to use to update the networks.
        train : bool (default=True)
                Flag to indicate if the agent is training.
        """
        # Properties of the agent.
        self.network = network
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.actions = actions
        self.train = train
        self.loss = loss

        # Trajectory to be updated.
        self.trajectory = TrajectoryInformation(particle_type=self.particle_type)

    def __name__(self) -> str:
        """
        Give the class a name.

        Return
        ------
        name : str
            Name of the class.
        """
        return "ActorCriticAgent"

    def update_agent(self) -> tuple:
        """
        Update the agents network.

        Returns
        -------
        rewards : float
                Net reward for the agent.
        killed : bool
                Whether or not this agent killed the
                simulation.
        """
        # Collect data for returns.
        rewards = self.trajectory.rewards
        killed = self.trajectory.killed

        # Compute loss for actor and critic.
        self.loss.compute_loss(
            network=self.network,
            episode_data=self.trajectory,
        )

        # Reset the trajectory storage.
        self.reset_trajectory()

        return rewards, killed

    def reset_agent(self, colloids: typing.List[Colloid]):
        """
        Reset several properties of the agent.

        Reset the observables and tasks for the agent.

        Parameters
        ----------
        colloids : typing.List[Colloid]
                Colloids to use in the initialization.
        """
        self.observable.initialize(colloids)
        self.task.initialize(colloids)

    def reset_trajectory(self):
        """
        Set all trajectory data to None.
        """
        self.task.kill_switch = False  # Reset here.
        self.trajectory = TrajectoryInformation(particle_type=self.particle_type)

    def initialize_network(self):
        """
        Initialize all of the models in the gym.
        """
        self.network.reinitialize_network()

    def save_agent(self, directory: str):
        """
        Save the agent network state.

        Parameters
        ----------
        directory : str
                Location to save the models.
        """
        self.network.export_model(
            filename=f"{self.__name__()}_{self.particle_type}", directory=directory
        )

    def restore_agent(self, directory: str):
        """
        Restore the agent state from a directory.
        """
        self.network.restore_model_state(
            filename=f"{self.__name__()}_{self.particle_type}", directory=directory
        )

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Copmute the new state for the agent.

        Returns the chosen actions to the force function which
        talks to the espresso engine.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids in the system.
        """
        state_description = self.observable.compute_observable(colloids)
        action_indices, log_probs = self.network.compute_action(
            observables=state_description
        )
        chosen_actions = np.take(list(self.actions.values()), action_indices, axis=0)#for Resobee "0", for Espresso "-1"

        # Update the trajectory information.
        if self.train:
            self.trajectory.features.append(state_description)
            self.trajectory.actions.append(action_indices)
            self.trajectory.log_probs.append(log_probs)
            self.trajectory.rewards.append(self.task(colloids))
            self.trajectory.killed = self.task.kill_switch

        self.kill_switch = self.task.kill_switch

        return chosen_actions
