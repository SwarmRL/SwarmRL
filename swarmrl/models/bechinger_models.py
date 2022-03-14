import numpy as np
import typing

from swarmrl.models.interaction_model import Action, InteractionModel


class Lavergne2019(InteractionModel):
    """
    See doi/10.1126/science.aau5347
    """

    def __init__(
        self, vision_half_angle=np.pi / 2.0, act_force=1, perception_threshold=1
    ):
        self.vision_half_angle = vision_half_angle
        self.act_force = act_force
        self.perception_threshold = perception_threshold

    def calc_action(self, colloids) -> typing.List[Action]:
        # determine perception value
        actions = []
        for colloid in colloids:
            other_colloids = [c for c in colloids if c is not colloid]

            colls_in_vision = get_colloids_in_vision(
                colloid, other_colloids, vision_half_angle=self.vision_half_angle
            )
            perception = 0
            my_pos = np.copy(colloid.pos)
            for coll in colls_in_vision:
                dist = np.linalg.norm(my_pos - coll.pos)
                perception += 1 / (2 * np.pi * dist)

            # set activity on/off
            if perception >= self.perception_threshold:
                actions.append(Action(force=self.act_force))
            else:
                actions.append(Action())

        return actions


class Baeuerle2020(InteractionModel):
    """
    See https://doi.org/10.1038/s41467-020-16161-4
    """

    def __init__(
        self,
        act_force=1.0,
        act_torque=1,
        detection_radius_position=1.0,
        detection_radius_orientation=1.0,
        vision_half_angle=np.pi / 2.0,
        angular_deviation=1,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.detection_radius_position = detection_radius_position
        self.detection_radius_orientation = detection_radius_orientation
        self.vision_half_angle = vision_half_angle
        self.angular_deviation = angular_deviation

    def calc_action(self, colloids) -> typing.List[Action]:
        # get vector to center of mass
        actions = []
        for colloid in colloids:
            other_colloids = [c for c in colloids if c is not colloid]
            colls_in_vision_pos = get_colloids_in_vision(
                colloid,
                other_colloids,
                vision_half_angle=self.vision_half_angle,
                vision_range=self.detection_radius_position,
            )
            if len(colls_in_vision_pos) == 0:
                # not detailed in the paper. take from previous model
                actions.append(Action())
                continue

            com = np.mean(
                np.stack([col.pos for col in colls_in_vision_pos], axis=0), axis=0
            )
            to_com = com - colloid.pos
            to_com_angle = angle_from_vector(to_com)

            # get average orientation of neighbours
            colls_in_vision_orientation = get_colloids_in_vision(
                colloid,
                other_colloids,
                vision_half_angle=self.vision_half_angle,
                vision_range=self.detection_radius_orientation,
            )

            if len(colls_in_vision_orientation) == 0:
                # not detailed in paper
                actions.append(Action())
                continue

            colls_in_vision_orientation.append(colloid)

            mean_orientation_in_vision = np.mean(
                np.stack([col.director for col in colls_in_vision_orientation], axis=0),
                axis=0,
            )
            mean_orientation_in_vision /= np.linalg.norm(mean_orientation_in_vision)

            # choose target orientation based on self.angular_deviation
            target_angle_choices = [
                to_com_angle + self.angular_deviation,
                to_com_angle - self.angular_deviation,
            ]
            target_orientation_choices = [
                vector_from_angle(ang) for ang in target_angle_choices
            ]

            angle_deviations = [
                np.arccos(np.dot(orient, mean_orientation_in_vision))
                for orient in target_orientation_choices
            ]
            target_angle = target_angle_choices[np.argmin(angle_deviations)]
            current_angle = angle_from_vector(colloid.director)
            angle_diff = target_angle - current_angle

            # take care of angle wraparound and bring difference to [-pi, pi]
            if angle_diff >= np.pi:
                angle_diff -= 2 * np.pi
            if angle_diff <= -np.pi:
                angle_diff += 2 * np.pi
            torque_z = np.sin(angle_diff) * self.act_torque

            actions.append(
                Action(force=self.act_force, torque=np.array([0, 0, torque_z]))
            )

        return actions


def get_colloids_in_vision(
    coll, other_coll, vision_half_angle=np.pi, vision_range=np.inf
) -> list:
    my_pos = np.array(coll.pos)
    my_director = coll.director
    colls_in_vision = []
    for other_p in other_coll:
        dist = other_p.pos - my_pos
        dist_norm = np.linalg.norm(dist)
        in_range = dist_norm < vision_range
        if not in_range:
            continue
        in_cone = np.arccos(np.dot(dist / dist_norm, my_director)) < vision_half_angle
        if in_cone and in_range:
            colls_in_vision.append(other_p)
    return colls_in_vision


def angle_from_vector(vec) -> float:
    return np.arctan2(vec[1], vec[0])


def vector_from_angle(angle) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle), 0])
