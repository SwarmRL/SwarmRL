"""
Module for the Actor-Critic RL protocol.
"""
import glob
import os

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
        network: Network,
        task: Task,
        observable: Observable,
        actions: dict,
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
        self.network = network
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.actions = actions

    def save_model(self, directory: str = "ckpts", episode: int = 0):
        """
        Save the network parameters to the ckpts directory.

        The naming convention is as follows:
            {model type}_{episode number}_{particle type}
        In this case, the model is of type, ac for actor-critic.

        Parameters
        ----------
        directory : str (default = ckpts)
                Directory in which to save the file.
        episode : int (default = 0)
                Which episode weights to save.

        Notes
        -----
        This will create directories if required, right permissions are
        needed.
        """
        self.network.export_model(
            filename=f"ac_{episode}_{self.particle_type}", directory=directory
        )

    def restore_parameters(self, directory: str = "ckpts", episode: int = None):
        """
        Restore the parameters of the actor and critic networks.

        Parameters
        ----------
        directory : str (default = ckpts)
                Directory from which to load the model.
        episode : int (default = None)
                Which episode weights to load. If None, the latest ones
                will be loaded.
        """
        # Get largest episode if not given.
        if episode is None:
            episodes = glob.glob("directory/*.pkl")

            latest_model_episode = 0
            for item in episodes:
                _, file = os.path.split(item)
                particle_type = file.split("_")[-1]
                episode = file.split("_")[1]

                if particle_type == self.particle_type:
                    if episode > latest_model_episode:
                        latest_model_episode = episode
                else:
                    continue
            episode = latest_model_episode

        self.network.restore_model_state(
            filename=f"ac_{episode}_{self.particle_type}", directory=directory
        )
