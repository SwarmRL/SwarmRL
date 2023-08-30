"""
    Adding already calculated trajectories from file to Training/Deployment
"""
import logging
import os
import pickle
from abc import ABC
from typing import List

import numpy as np
from swarmrl.networks.network import Network
from swarmrl.models.interaction_model import Action
import h5py as hf

logger = logging.getLogger(__name__)

class TrajectoryFileNetwork(Network):
    def __init__(self,
                 load_directory: str,
                 particle_type: int,
                 particle_gamma_translation: int,
                 particle_gamma_rotation: int,
                 system_runner: object = None,
                 ):
        """
        Parameters
        ----------
        load_directory: str
            Input directory of the position data
        colloid_type: int
            Type of colloid that is controlled
        particle_gamma_translation, particle_gamma_rotation: int, int
            Friction coefficient of the colloid, if None it is taken from the system_runner
            (There was no other easy way of getting the friction coefficient, if set manually to a particle type)
        system_runner: SystemRunner
            espressoMD.SystemRunner object
        """
        self.index_tracker = -1
        self.particle_type = particle_type
        self.gamma = particle_gamma_translation
        self.gamma_rotation = particle_gamma_rotation
        if self.gamma is None and self.gamma_rotation is None:
            if system_runner is None:
                raise ValueError("SystemRunner is None, please provide a SystemRunner object")
            self.gamma, self.gamma_rotation = system_runner.get_friction_coefficients(self.colloid_type)

        db = hf.File(f"{load_directory}/trajectory.hdf5")
        self.colloid_pos = db["Wanted_Positions"][:]
        # there are only positions in the trajectory file no velocities

    def compute_action(self, observables: List, explore_mode: bool = False):
        """
        Calculates the action to get to the next position in the trajectory file

        Parameters
        ----------
        observables : List (n_colloids)

        Returns
        -------
        Action, None
            Needs to return two values for the network to work in protocol
        """
        self.index_tracker += 1
        mass = 1
        velocity = observables[0]
        pos = self.colloid_pos[self.index_tracker]
        pos1 = self.colloid_pos[self.index_tracker + 1]
        force = (pos1 - pos - velocity * 0.01) * 2 * mass / 0.01 ** 2
        force_value = np.linalg.norm(force)
        new_direction = force / force_value
        # if gamma = 0, mass=1, t_step=0.01 then force*0.0004
        # if gamma = 100, mass=1, t_step=0.01 then force*0.0004*gamma*0.25
        #if self.index_tracker < 10:
        #    print("Wanted:", pos, "actual:", colloid.pos, "Force:", force)
        return Action(force=0.0004 * self.gamma * 0.25 * force_value, new_direction=new_direction), None