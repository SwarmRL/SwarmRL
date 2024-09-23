"""
Module for the Trainer parent.
"""
import logging
from typing import List, Tuple

import numpy as np

from swarmrl.agents.actor_critic import ActorCriticAgent
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
        checkpoint_params: dict = None,
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

        # Add the protocols to an easily accessible internal dict.
        # TODO: Maybe turn into a dataclass? Not sure if it helps yet.
        for agent in agents:
            self.agents[str(agent.particle_type)] = agent

        # Initialize the checkpointer

        self.checkpoint_params = checkpoint_params
        if self.checkpoint_params is None:
            self.DO_CHECKPOINT = False
        else:
            self.DO_CHECKPOINT = True
            self.STOP_TRAINING_NOW = False
            self.initialize_checkpointer(checkpoint_params)

    def initialize_checkpointer(self, checkpoint_params: dict = None):
        """
        Initialize the checkpointer by taking the checkpoint_params key-value pairs.
        It automatically sets the DO_CHECKPOINT key to True if the checkpoint_params is not None.
        All other keys are optional and depend on the specific checkpointing method.
        If not provided, they will be set to default values.
        
        Currently the checkpointer can do the following:
        - reward-goal-checkpointing: Save the model when a certain reward is reached.
            keys:
            - DO_GOAL_MODEL: Boolean to activate the goal-model-checkpointing.
            - required_reward: The reward that should be reached. This is very specific to the task!
            - window_width: The width of the window to calculate the average reward.
            - DO_GOAL_BREAK: Boolean to stop the training after the goal is reached.
            - running_out_length: If the goal is reached, the simulation will continue for another running_out_length episodes.

        - best-reward-checkpointing: Saves the model, when the reward is greater than the previous maximum reward.
            keys:
            - DO_BEST_MODEL: Boolean to activate the best-model-checkpointing.
            - min_reward: The minimum reward that should be reached.
            - window_width: The width of the window to calculate the average reward.
            - increase_factor: The factor the reward should be greater than the previous maximum reward.
            - better_wait_time: The number of episodes to wait before the first checkpoint.
        
        - backup-model-checkpointing: Saves the model, if the reward is sinking suddenly. Could be used for analysing forgetting models.
            keys:
            - DO_BACKUP_MODEL: Boolean to activate the backup-model-checkpointing.
            - backup_wait_time: The number of episodes to wait before the first checkpoint.
            - window_width: The width of the window to calculate the average reward.
            - min_backup_reward: The minimum reward that should be reached to be able to save models that at least learned something.
        
        - Save a model in regular intervals.
            keys:
            - DO_REGULAR: Boolean to activate the regular-checkpointing.
            - save_models_intervall: The intervall size of episodes after which models will be saved.

        Parameters
        ----------
        checkpoint_params : dict
            Parameters to use in the checkpointer.
        """
        self.rewards = []
        self.window_width = checkpoint_params.get('window_width', 30)
        self.STOP_TRAINING_NOW = False

        if self.window_width < 1:
            self.DO_RUNNING_OUT = False
        else:
            self.DO_RUNNING_OUT = True

        # initialize reward-goal-checkpointing
        if 'DO_GOAL_MODEL' not in checkpoint_params.keys():
            self.DO_GOAL_MODEL = False
        else:
            self.DO_GOAL_MODEL = checkpoint_params['DO_GOAL_MODEL']
            self.required_reward = checkpoint_params.get('required_reward', 200)
            self.DO_GOAL_BREAK = checkpoint_params.get('DO_GOAL_BREAK', True)
            self.running_out_length = checkpoint_params.get('running_out_length', 0)
            self.old_max = 0

        # initialize best-model-checkpointing
        if 'DO_BEST_MODEL' not in checkpoint_params.keys():
            self.DO_BEST_MODEL = False
        else:
            self.DO_BEST_MODEL = checkpoint_params['DO_BEST_MODEL']
            self.min_reward = checkpoint_params.get('min_reward', 250)
            self.better_increase_factor = checkpoint_params.get('increase_factor', 1.05)
            self.better_wait_time = checkpoint_params.get('better_wait_time', 20)

        # Initialize backup-model-checkpointing
        if 'DO_BACKUP_MODEL' not in checkpoint_params.keys():
            self.DO_BACKUP_MODEL = False
        else:
            self.DO_BACKUP_MODEL = checkpoint_params['DO_BACKUP_MODEL']
            self.backup_wait_time = checkpoint_params.get('backup_wait_time', 20)
            self.min_backup_reward = checkpoint_params.get('min_backup_reward', 250)

        # Initialize regular-checkpointing
        if 'DO_REGULAR' not in checkpoint_params.keys():
            self.DO_REGULAR = False
        else:
            self.DO_REGULAR = checkpoint_params['DO_REGULAR']
            self.save_models_intervall = checkpoint_params.get('save_models_intervall', 25)
        
        if self.DO_GOAL_BREAK == True:
            self.stop_episode = 0

    def check_for_checkpoint(self, rewards: np.ndarray, n_episodes: int, current_episode: int):
        """
        Check if a model-backup of the current state should be saved.

        Parameters
        ----------
        reward : np.ndarray
                The current reward data.
        n_episodes : int
                The total number of episodes.
        current_episode : int
                The current episode.

        Returns
        -------
        save_string : str
                A string that contains the flags of the checkpointing criterias that were met.
        """
        SAVE_GOAL = False
        SAVE_BEST = False
        SAVE_BACKUP = False
        SAVE_REGULAR = False

        self.rewards = rewards.copy()
        current_reward = rewards[current_episode]

        if self.DO_CHECKPOINT == True:
            if self.DO_REGULAR == True:
                if (current_episode + 1) % self.save_models_intervall == 0:
                    SAVE_REGULAR = True

            if self.DO_GOAL_MODEL == True or self.DO_BEST_MODEL == True or self.DO_BACKUP_MODEL == True:
                if current_episode > self.window_width:
                    average_window_reward = np.mean(self.rewards[current_episode-self.window_width:current_episode+1])
                else:
                    average_window_reward = np.mean(self.rewards[:current_episode + 1])

            # Do goal-model-checkpointing
            if self.DO_GOAL_MODEL == True:
                if (average_window_reward >= self.required_reward and
                    current_reward >= self.required_reward):

                    if self.DO_GOAL_BREAK == True:
                        self.STOP_TRAINING_NOW = True
                        if self.DO_RUNNING_OUT == True:
                            self.stop_episode = current_episode + self.running_out_length + 1
                        else:
                            self.stop_episode = current_episode
                        if self.stop_episode == 0:
                            self.STOP_TRAINING_NOW = False
                    SAVE_GOAL = True
            
            # Do best-model-checkpointing
            if self.DO_BEST_MODEL == True:
                if (current_episode > self.better_wait_time and
                    current_reward > self.min_reward and
                    average_window_reward > self.better_increase_factor * self.old_max and
                    current_reward > self.old_max):

                    self.old_max = current_reward
                    SAVE_BEST = True

            # Do backup-model-checkpointing
            if self.DO_BACKUP_MODEL == True:
                if (current_episode > self.backup_wait_time and
                    self.min_backup_reward < current_reward < np.max(self.rewards)):

                    self.min_backup_reward = current_reward
                    SAVE_BACKUP = True

        checkpoint_flags = {'SAVE_GOAL': SAVE_GOAL, 
                    'SAVE_BEST': SAVE_BEST,
                    'SAVE_BACKUP': SAVE_BACKUP,
                    'SAVE_REGULAR': SAVE_REGULAR,
                    'STOP_TRAINING_NOW': self.STOP_TRAINING_NOW}
        
        save_string = ""
        for flag_name, flag_value in checkpoint_flags.items():
            if flag_value:
                save_string += flag_name + "_"
        save_string = save_string[:-1] if save_string.endswith("_") else save_string
        return save_string

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
