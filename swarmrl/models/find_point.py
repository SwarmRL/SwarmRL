import typing

import numpy as np

from swarmrl.models.interaction_model import Action, InteractionModel


class FindPoint(InteractionModel):
    def __init__(
        self,
        act_force,
        act_torque,
        vision_half_angle=np.pi / 4,
        point=np.array([0.0, 0.0, 0.0]),
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.point = point
        self.cos = np.cos(vision_half_angle)

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        for colloid in colloids:
            to_point = self.point - colloid.pos
            if np.dot(to_point, colloid.director) / np.linalg.norm(to_point) > self.cos:
                actions.append(Action(force=self.act_force))
            else:
                actions.append(Action())

        return actions, False
