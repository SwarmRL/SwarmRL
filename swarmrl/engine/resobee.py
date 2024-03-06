"""
Child class for the ResoBee engine
"""
from swarmrl.engine.engine import Engine
from swarmrl.force_functions import ForceFunction

import numpy as np
import zmq
import yaml
import os
from dataclasses import dataclass
import subprocess
from swarmrl.components.colloid import Colloid

class Population:
    def __init__(self, n_agents: int):
        self.indices = np.zeros(n_agents)
        self.types = np.zeros(n_agents)
        self.unwrapped_positions = np.zeros((n_agents, 2))
        self.velocities = np.zeros((n_agents, 2))
        self.directors = np.zeros((n_agents, 2))
        self.time_slice = 10


class ResoBee(Engine):
    """
    Child class for the ResoBee Engine.
    """

    def __init__(self, resobee_executable : str, config_dir: str)->None:
        """
        Intializes the ResoBee engine.

        Args:
            resobee_executable (str): path to the executable
            config_dir (str): path to the directory containing "config.yaml"
        """
        self.resobee_executable = resobee_executable
        self.config_dir = config_dir
        self.config = yaml.safe_load(open(os.path.join(config_dir, 'config.yaml'), 'r'))
        self.population = Population(self.config["n_agents"])

        # initialize the zmq socket for communication with ResoBee
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        print("Binding to tcp address: ", self.config["tcp_address"])
        self.socket.bind(str(self.config["tcp_address"]))

    def __del__(self)->None:
        """
        Closes connection to ResoBee
        """
        self.socket.close()
        self.context.term()

    @property
    def colloids(self)->dict:
        """
        Property particle data from ResoBee.

        Returns:
            dict: particle data
        """ 
        self.receive_state()
        return self.get_particle_data()

    def receive_state(self)->bool:
        """
        Fetches simulation state from ResoBee client.

        Raises:
            KeyError: Received no or incomplete simulation state from ResoBee client.

        Returns:
            bool: ResoBee client finished the simulation.
        """
        # print("Listening for requests from ResoBee client...")
        message = self.socket.recv_json()

        # print("Received data from ResoBee client.")
        try:
            self.population.indices = np.array(message["indices"])
            self.population.unwrapped_positions = np.array(list(zip(message["x"], message["y"])))
            self.population.velocities = np.array(list(zip(message["v_x"], message["v_y"])))
        except KeyError:
            # print("Received no or incomplete simulation state from ResoBee client.")
            pass
        try:
            simulation_is_finished = message["is_finished"]
        except KeyError:
            print("Received no simulation completion state from ResoBee client.")
            raise KeyError

        return simulation_is_finished

    def compute_action(self, force_model: ForceFunction) -> tuple[list, list]:
        """
        Determines the force in x and y coordinate for every particle.

        Args:
            force_model (ForceFunction): An instance of swarmrl.force_functions.ForceFunction

        Returns:
            tuple[list, list]: Force in x and y for every particle.
        """
        colloids = self.get_particle_data()
        actions = force_model.calc_action(colloids)
        forces_x = [action.force * np.cos(action.new_direction) for action in actions]
        forces_y = [action.force * np.sin(action.new_direction) for action in actions]
        return forces_x, forces_y

    def send_action(self, forces_x: list, forces_y: list)->None:
        """
        Sends the calculated forces to the ResoBee client.

        Args:
            forces_x (list): Force in x for every particle.
            forces_y (list): Force in y for every particle.
        """
        # print("Sending action (forces) to ResoBee client...")
        ret = {
            "f_x": forces_x,
            "f_y": forces_y
        }
        self.socket.send_json(ret)

    def send_nothing(self)->None:
        """
        Sends nothing to the ResoBee client.
        """
        # print("Sending nothing to ResoBee client...")
        ret = {}
        self.socket.send_json(ret)

    def integrate(
            self,
            force_model: ForceFunction,
    ) -> None:
        """

        Parameters
        ----------
        force_model
            An instance of swarmrl.force_functions.ForceFunction
        """
        # start the ResoBee engine
        try:
            process = subprocess.Popen([self.resobee_executable, self.config_dir])

            # run the ResoBee engine until it finishes
            simulation_is_finished = False
            step = 0

            while simulation_is_finished is False:
                simulation_is_finished = self.receive_state()

                # compute a new action if we enter a new time slice
                if step % self.population.time_slice == 0:
                    f_x, f_y = self.compute_action(force_model)
                    self.send_action(f_x, f_y)
                else:
                    self.send_nothing()
                # check if the ResoBee engine has finished
                if simulation_is_finished:
                    # print("ResoBee simulation successfully completed.")
                    break
                step += 1
        finally:
            process.kill()
           

    def get_particle_data(self) -> dict:
        """
        Get type, id, position, velocity and director of the particles
        as a dict of np.array

        The particle data is fetched from the C++ engine and converted to a dict.
        """
        colloids = []

        for i in range(len(self.population.indices)):
            colloids.append(
                Colloid(
                    pos=np.array(self.population.unwrapped_positions)[i],
                    director=np.array(self.population.directors)[i],
                    id=np.array(self.population.indices)[i],
                    velocity=np.array(self.population.velocities)[i],
                    type=0
                )
            )
        return colloids
