"""
Child class for the ResoBee engine
"""
from swarmrl.engine.engine import Engine
from swarmrl.models.interaction_model import InteractionModel

import numpy as np
import zmq
import yaml
import os
import subprocess


class Population:
    def __init__(self, n_agents):
        self.indices = np.zeros(n_agents)
        self.types = np.zeros(n_agents)
        self.unwrapped_positions = np.zeros((n_agents, 2))
        self.velocities = np.zeros((n_agents, 2))
        self.directors = np.zeros((n_agents, 2))


class ResoBee(Engine):
    """
    Child class for the ResoBee Engine.
    """

    def __init__(self, resobee_executable, config_dir):
        self.resobee_executable = resobee_executable
        self.config_dir = config_dir
        self.config = yaml.safe_load(open(os.path.join(config_dir, 'config.yaml'), 'r'))
        self.population = Population(self.config["n_agents"])

        # initialize the zmq socket for communication with ResoBee
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        print("Binding to tcp address: ", self.config["tcp_address"])
        self.socket.bind(str(self.config["tcp_address"]))

    def __del__(self):
        self.socket.close()
        self.context.term()

    def receive_state(self):
        print("Listening for requests from ResoBee client...")
        message = self.socket.recv_json()

        print("Received data from ResoBee client.")
        try:
            self.population.indices = np.array(message["indices"])
            self.population.unwrapped_positions = np.array(list(zip(message["x"], message["y"])))
            self.population.velocities = np.array(list(zip(message["v_x"], message["v_y"])))
        except KeyError:
            print("Received an incomplete simulation state from ResoBee client.")

        try:
            simulation_is_finished = message["is_finished"]
        except KeyError:
            print("Received no simulation completion state from ResoBee client.")
            raise KeyError

        return simulation_is_finished

    def compute_action(self, force_model) -> tuple[list, list]:
        # todo: implement this, for now this is random
        forces_x = np.random.rand(self.config["n_agents"]).tolist()
        forces_y = np.random.rand(self.config["n_agents"]).tolist()

        return forces_x, forces_y

    def send_action(self, forces_x: list, forces_y: list):
        print("Sending action (forces) to ResoBee client...")
        ret = {
            "f_x": forces_x,
            "f_y": forces_y
        }
        self.socket.send_json(ret)

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

        # start the ResoBee engine
        try:
            process = subprocess.Popen([self.resobee_executable, self.config_dir])

            # run the ResoBee engine until it finishes
            simulation_is_finished = False

            while simulation_is_finished is False:
                simulation_is_finished = self.receive_state()
                f_x, f_y = self.compute_action(force_model)
                self.send_action(f_x, f_y)

                # check if the ResoBee engine has finished
                if simulation_is_finished:
                    print("ResoBee simulation successfully completed.")
                    break
        finally:
            process.kill()

    def get_particle_data(self) -> dict:
        """
        Get type, id, position, velocity and director of the particles
        as a dict of np.array

        The particle data is fetched from the C++ engine and converted to a dict.
        """

        return {
            "Id": self.population.indices,
            "Type": self.population.types,
            "Unwrapped_Positions": self.population.unwrapped_positions,
            "Velocities": self.population.velocities,
            "Directors": self.population.directors,
        }
