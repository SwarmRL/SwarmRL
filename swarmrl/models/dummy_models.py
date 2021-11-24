from . import interaction_model
import numpy as np


class ConstForce(interaction_model.InteractionModel):
    def __init__(self, force: float):
        self.action = interaction_model.Action(force=force)

    def calc_action(self, colloid, other_colloids) -> interaction_model.Action:
        return self.action


class ConstTorque(interaction_model.InteractionModel):
    def __init__(self, torque: np.ndarray):
        self.action = interaction_model.Action(torque=torque)

    def calc_action(self, colloid, other_colloids) -> interaction_model.Action:
        return self.action


class ToConstDirection(interaction_model.InteractionModel):
    def __init__(self, direction: np.ndarray):
        self.action = interaction_model.Action(new_direction=direction)

    def calc_action(self, colloid, other_colloids) -> interaction_model.Action:
        return self.action
