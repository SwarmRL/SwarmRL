import typing

import numpy as np

from . import interaction_model


class ConstForce(interaction_model.InteractionModel):
    def __init__(self, force: float):
        self.action = interaction_model.Action(force=force)

    def calc_action(self, colloids) -> typing.List[interaction_model.Action]:
        return len(colloids) * [self.action], False


class ConstTorque(interaction_model.InteractionModel):
    def __init__(self, torque: np.ndarray):
        self.action = interaction_model.Action(torque=torque)

    def calc_action(self, colloids) -> typing.List[interaction_model.Action]:
        return len(colloids) * [self.action], False


class ConstForceAndTorque(interaction_model.InteractionModel):
    def __init__(self, force: float, torque: np.ndarray):
        self.action = interaction_model.Action(force=force, torque=torque)

    def calc_action(self, colloids) -> typing.List[interaction_model.Action]:
        return len(colloids) * [self.action], False


class ToConstDirection(interaction_model.InteractionModel):
    def __init__(self, direction: np.ndarray):
        self.action = interaction_model.Action(new_direction=direction)

    def calc_action(self, colloids) -> typing.List[interaction_model.Action]:
        return len(colloids) * [self.action]
