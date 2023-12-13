import typing

import numpy as np

from swarmrl.actions.actions import Action
from swarmrl.agents.classical_agent import ClassicalAgent


class ConstForce(ClassicalAgent):
    def __init__(self, force: float):
        self.action = Action(force=force)

    def calc_action(self, colloids) -> typing.List[Action]:
        return len(colloids) * [self.action]


class ConstTorque(ClassicalAgent):
    def __init__(self, torque: np.ndarray):
        self.action = Action(torque=torque)

    def calc_action(self, colloids) -> typing.List[Action]:
        return len(colloids) * [self.action]


class ConstForceAndTorque(ClassicalAgent):
    def __init__(self, force: float, torque: np.ndarray):
        self.action = Action(force=force, torque=torque)

    def calc_action(self, colloids) -> typing.List[Action]:
        return len(colloids) * [self.action]


class ToConstDirection(ClassicalAgent):
    def __init__(self, direction: np.ndarray):
        self.action = Action(new_direction=direction)

    def calc_action(self, colloids) -> typing.List[Action]:
        return len(colloids) * [self.action]
