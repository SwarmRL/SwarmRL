"""
Module for the Trainer parent.
"""

from typing import List, Tuple

import numpy as np

from swarmrl.agents.actor_critic import ActorCriticAgent
from swarmrl.force_functions.force_fn import ForceFunction
from swarmrl.losses.loss import Loss
from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss


class Trainer:
    """
    Parent class for the RL Trainer.

    Attributes
    ----------
    rl_protocols : list(protocol)
            A list of RL protocols to use in the simulation.
    loss : Loss
            An optimization method to compute the loss and update the model.
    """

    def __init__(
        self,
        agents: List[ActorCriticAgent],
        loss: Loss = ProximalPolicyLoss(),
    ):
        """
        Constructor for the MLP RL.

        Parameters
        ----------
        agents : list
                A list of RL agents
        loss : Loss
                A loss model to use in the A-C loss computation.
        """
        self.loss = loss
        self.agents = {}

        # Add the protocols to an easily accessible internal dict.
        # TODO: Maybe turn into a dataclass? Not sure if it helps yet.
        for agent in agents:
            self.agents[str(agent.particle_type)] = agent

    def initialize_training(self) -> ForceFunction:
        """
        Return an initialized interaction model.

        Returns
        -------
        interaction_model : ForceFunction
                Interaction model to start the simulation with.
        """

        return ForceFunction(
            agents=self.agents,
        )

    def update_rl(self) -> Tuple[ForceFunction, np.ndarray]:
        """
        Update the RL algorithm.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to use in the next episode.
        reward : np.ndarray
                Current mean episode reward. This is returned for nice progress bars.
        killed : bool
                Whether or not the task has ended the training.
        """
        reward = 0.0  # TODO: Separate between species and optimize visualization.
        switches = []

        for agent in self.agents.values():
            episode_data = agent.trajectory

            reward += np.mean(episode_data.rewards)

            # Compute loss for actor and critic.
            self.loss.compute_loss(
                network=agent.network,
                episode_data=episode_data,
            )
            agent.reset_trajectory()
            switches.append(episode_data.killed)

        # Create a new interaction model.
        interaction_model = ForceFunction(agents=self.agents)
        return interaction_model, np.array(reward), any(switches)

    def export_models(self, directory: str = "Models"):
        """
        Export the models to the specified directory.

        Parameters
        ----------
        directory : str (default='Models')
                Directory in which to save the models.

        Returns
        -------
        Saves the actor and the critic to the specific directory.

        Notes
        -----
        This is super lazy. We should add this to the rl protocol. Same with the
        model restoration.
        """
        for type_, val in self.agents.items():
            val.network.export_model(filename=f"Model{type_}", directory=directory)

    def restore_models(self, directory: str = "Models"):
        """
        Export the models to the specified directory.

        Parameters
        ----------
        directory : str (default='Models')
                Directory from which to load the objects.

        Returns
        -------
        Loads the actor and critic from the specific directory.
        """
        for type_, val in self.agents.items():
            val.network.restore_model_state(
                filename=f"Model{type_}", directory=directory
            )

    def initialize_models(self):
        """
        Initialize all of the models in the gym.
        """
        for _, val in self.agents.items():
            val.network.reinitialize_network()

    def perform_rl_training(self, **kwargs):
        """
        Perform the RL training.

        Parameters
        ----------
        **kwargs
            All arguments related to the specific trainer.
        """
        raise NotImplementedError("Implemented in child class")
