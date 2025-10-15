"""
Module for the Trainer parent.
"""

import logging
from typing import List, Tuple

import numpy as np

from swarmrl.agents.actor_critic import ActorCriticAgent
from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer
from swarmrl.force_functions.force_fn import ForceFunction

logger = logging.getLogger(__name__)


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

    _engine = None

    @property
    def engine(self):
        """
        Runner engine property.
        """
        return self._engine

    @engine.setter
    def engine(self, value):
        """
        Set the engine value.
        """
        self._engine = value

    def __init__(
        self,
        agents: List[ActorCriticAgent],
        checkpointers: List[BaseCheckpointer] | None = None,
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
        self.agents = {}
        self.checkpointers = checkpointers if checkpointers is not None else []

        # Add the protocols to an easily accessible internal dict.
        # TODO: Maybe turn into a dataclass? Not sure if it helps yet.
        for agent in agents:
            self.agents[str(agent.particle_type)] = agent

        checkpoint_paths = []
        if len(checkpointers) > 0:
            for checkpointer in checkpointers:
                self.checkpointers.append(checkpointer)
                if checkpointer.out_path is not None:
                    checkpoint_paths.append(checkpointer.out_path)

            if len(checkpoint_paths) == 0:
                print("No checkpointer out_path provided. Storing in './Models/' now.")
                self.checkpoint_path = "./Models/"
            elif len(checkpoint_paths) == 1:
                self.checkpoint_path = checkpoint_paths[0]
            else:
                print(
                    "Found multiple checkpointer paths. Choosing the first entry: "
                    f"{checkpoint_paths[0]}."
                )
                self.checkpoint_path = checkpoint_paths[0]
        else:
            print("No Checkpointer provided.")

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
            if isinstance(agent, ActorCriticAgent):
                ag_reward, ag_killed = agent.update_agent()
                logger.debug(f"{ag_reward=}")
                reward += np.mean(ag_reward)
                switches.append(ag_killed)

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
        """
        for agent in self.agents.values():
            agent.save_agent(directory)

    def restore_models(self, directory: str = "Models"):
        """
        Restore the models from the specified directory.

        Parameters
        ----------
        directory : str (default='Models')
                Directory from which to load the objects.

        Returns
        -------
        Loads the actor and critic from the specific directory.
        """
        for agent in self.agents.values():
            agent.restore_agent(directory)

    def initialize_models(self):
        """
        Initialize all of the models in the gym.
        """
        for agent in self.agents.values():
            agent.initialize_network()

    def perform_rl_training(self, **kwargs):
        """
        Perform the RL training.

        Parameters
        ----------
        **kwargs
            All arguments related to the specific trainer.
        """
        raise NotImplementedError("Implemented in child class")
