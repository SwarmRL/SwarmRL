"""
Adding Classical Paths to RL Training
"""
import logging
import os
import pickle
from abc import ABC
from typing import List

import numpy as np
from swarmrl.networks.network import Network
from swarmrl.models.interaction_model import Action

logger = logging.getLogger(__name__)


class ClassicalNetwork(Network):
    def __init__(self,
                 eq_of_motion: callable,
                 params: list,
                 box_length: np.ndarray = np.array([1000.0, 1000.0, 1000.0]),
                 home_pos: np.ndarray = np.array([500.0, 500.0, 0])
                 ):
        """
        Parameters
        ----------
        eq_of_motion: callable
            Function that returns the force vector for the colloid
        params: list
            Parameters for the eq_of_motion function
        box_length: np.ndarray
            Box length of the simulation
        home_pos: np.ndarray
            Home position of the colloid
        """
        self.eq_of_motion = eq_of_motion
        self.t = 0
        self.params = params
        self.home_pos = home_pos
        self.box_length = box_length

    def compute_action(self, observables: List, explore_mode: bool = False):
        """
        observables : List (n_colloids, 2)
        """
        self.t += 0.2 / 5
        pos, director = observables[0][0]*self.box_length, observables[0][1]
        force = self.eq_of_motion(self.t, pos, director, self.home_pos, self.params)
        nd = np.array([force[0], force[1], force[2]])
        new_direction = nd / np.linalg.norm(nd)

        return Action(force=50 * np.linalg.norm(nd), new_direction=new_direction), None  # has to have two return values
