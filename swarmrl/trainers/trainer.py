"""
Module for the Trainer parent.
"""

from typing import List, Tuple

import numpy as np
from loguru import logger

from swarmrl.agents.agent import Agent
from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer
from swarmrl.checkpointers.checkpoint_manager import CheckpointManager
from swarmrl.force_functions.force_fn import ForceFunction


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
        agents: List[Agent],
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
        self.checkpointers = list(checkpointers) if checkpointers is not None else []

        # Add the protocols to an easily accessible internal dict.
        # TODO: Maybe turn into a dataclass? Not sure if it helps yet.
        for agent in agents:
            self.agents[str(agent.particle_type)] = agent

        checkpoint_paths = [
            checkpointer.out_path
            for checkpointer in self.checkpointers
            if checkpointer.out_path is not None
        ]
        if len(self.checkpointers) > 0:
            if len(checkpoint_paths) == 0:
                logger.warning(
                    "No checkpointer out_path provided. Storing in './Models/' now."
                )
                self.checkpoint_path = "./Models/"
            elif len(checkpoint_paths) == 1:
                self.checkpoint_path = checkpoint_paths[0]
            else:
                logger.warning(
                    "Found multiple checkpointer paths. Choosing the first entry: "
                    f"{checkpoint_paths[0]}."
                )
                self.checkpoint_path = checkpoint_paths[0]

            self.checkpoint_manager = CheckpointManager(
                checkpointers=self.checkpointers,
                checkpoint_path=self.checkpoint_path,
                save_callback=self.export_models,
            )
            logger.info(f"Activated {len(self.checkpointers)} checkpointers.")
        else:
            self.checkpoint_manager = None
            logger.info("No Checkpointer provided.")

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

    def update_rl(
        self, terminated: bool = False, truncated: bool = False
    ) -> Tuple[ForceFunction, np.ndarray, bool]:
        """
        Update the RL algorithm.

        Parameters
        ----------
        truncated : bool
                Whether the environment will be reset after this rollout (trainer
                episode).
        terminated : bool
                Whether a task ended the shared environment during this rollout.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to use in the next episode.
        reward : np.ndarray
                Current mean episode reward. This is returned for nice progress bars.
        killed : bool
                Whether or not the task has ended the training.

        Notes
        -----
        A rollout is one SwarmRL trainer episode between network updates. A truncated
        rollout ends because the environment will be reset; a terminated rollout ends
        because a task has ended the shared environment. Termination takes precedence
        when both occur at the same rollout boundary. Because all agents share that
        environment, termination is applied to every agent's final transition.
        """
        reward = 0.0  # TODO: Separate between species and optimize visualization.
        truncated = truncated and not terminated
        for agent in self.agents.values():
            agent_rewards = agent.on_rollout_end(
                terminated=terminated,
                truncated=truncated,
            )
            if agent_rewards is not None:
                logger.debug(f"agent rewards={agent_rewards}")
                reward += np.mean(agent_rewards)

        # Create a new interaction model.
        interaction_model = ForceFunction(agents=self.agents)
        return interaction_model, np.array(reward), terminated

    def finalize_agents(self):
        """Finalize agent-side resources after training."""
        for agent in self.agents.values():
            agent.finalize()

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

    def maybe_save_checkpoint(
        self,
        rewards: np.ndarray,
        episode: int,
        current_reward: float,
    ) -> bool:
        """
        Evaluate all checkpointers and save models when a criterion is met.

        Parameters
        ----------
        rewards : np.ndarray
                Reward history.
        episode : int
                Current episode index.
        current_reward : float
                Reward of the current episode.

        Returns
        -------
        bool
            Whether a checkpoint was saved in this episode.
        """
        if self.checkpoint_manager is None:
            return False
        return self.checkpoint_manager.check_and_save(
            rewards=rewards,
            current_episode=episode,
            current_reward=current_reward,
        )

    def check_for_stop_criterion(self) -> tuple[bool, int]:
        """
        Query all checkpointers for a stop criterion.

        Returns
        -------
        tuple[bool, int]
            `(break_training, stop_after_episode)` where `stop_after_episode` is
            `-1` if no stopping criterion is active.
        """
        if self.checkpoint_manager is None:
            return False, -1
        return self.checkpoint_manager.should_stop_training()

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
