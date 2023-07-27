"""
Child class for the ResoBee engine
"""
from swarmrl.engine.engine import Engine
from swarmrl.models.interaction_model import InteractionModel

import numpy as np
import zmq
import time
import yaml
import os


class ResoBee(Engine):
    """
    Child class for the ResoBee Engine.
    """

    def __init__(self, executable_path, config_dir):
        self.executable_path = executable_path
        config_path = os.path.join(config_dir, 'config.yaml')
        self.config = yaml.safe_load(open(config_path, 'r'))

        # initialize the zmq socket for communication with ResoBee
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.config.tcp_address)

    def integrate(
            self,
            n_slices: int,
            force_model: InteractionModel,
    ) -> None:
        """

        Parameters
        ----------
        n_slices: int
            Number of time slices to integrate
        force_model
            A an instance of swarmrl.models.interaction_model.InteractionModel
        """

        print("Computing action...")
        time.sleep(1)

        print("Sending action (forces) to ResoBee client...")
        ret = {
            "f_x": np.random.rand(self.config.n_agents).tolist(),
            "f_y": np.random.rand(self.config.n_agents).tolist()
        }
        self.socket.send_json(ret)

    def get_particle_data(self) -> dict:
        """
        Get type, id, position, velocity and director of the particles
        as a dict of np.array

        The particle data is fetched from the C++ engine and converted to a dict.
        """

        print("Listening for requests from ResoBee client...")
        message = self.socket.recv_json()

        print("Received state (population) from ResoBee client: ", message)
        indices = np.array(message["indexes"])
        types = np.random.rand(self.config.n_agents)  # zeros for now
        unwrapped_positions = np.array(list(zip(message["x"], message["y"])))
        velocities = np.array(list(zip(message["v_x"], message["v_y"])))
        directors = np.random.rand(self.config.n_agents, 2)  # zeros for now

        return {
            "Id": indices,
            "Type": types,
            "Unwrapped_Positions": unwrapped_positions,
            "Velocities": velocities,
            "Directors": directors,
        }
