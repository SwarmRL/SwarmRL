import numpy as np

from swarmrl.actions.actions import Action
from swarmrl.agents.classical_agent import ClassicalAgent


class Lymburn(ClassicalAgent):
    def __init__(
        self,
        force_params: dict,
        detection_radius_position_colls=np.inf,
        detection_radius_position_pred=np.inf,
        home_pos=np.array([500, 500, 0]),
        agent_speed=10,
        predator_type: int = 1,
    ):
        """
        Parameters
        ----------
        force_params : dict
            Dictionary containing the force parameters.
            a = alignment, r = repulsion, h = home, f = friction, p = predator
                e.g.: force_params = {"K_a":K_a, "K_r":K_r,
                  "K_h":K_h, "K_f":K_f, "K_p":K_p}
        detection_radius_position_colls : float
            Radius in which the colloids are detected
        detection_radius_position_pred : float
            Radius in which the predator is detected
        home_pos : np. array
            Position of the home
        agent_speed : float
            Speed of the colloids (used for the friction force)
        predator_type : int
            Type of the predator colloids.
        """
        self.force_params = force_params
        self.detection_radius_position_colls = detection_radius_position_colls
        # implies r_align=r_repulsion
        self.detection_radius_position_pred = detection_radius_position_pred
        self.home_pos = home_pos
        self.predator_type = predator_type
        self.agent_speed = agent_speed

    def update_force_params(self, K_a=None, K_r=None, K_h=None, K_f=None, K_p=None):
        """
        Update the force parameters
        """
        update_params = {"K_a": K_a, "K_r": K_r, "K_h": K_h, "K_f": K_f, "K_p": K_p}
        for key, value in update_params.items():
            if value is not None:
                self.force_params[key] = value

    def calc_action(self, colloids):
        actions = []
        for colloid in colloids:
            if colloid.type == self.predator_type:
                continue

            other_colls = [
                c
                for c in colloids
                if c is not colloid and not c.type == self.predator_type
            ]
            colls_in_vision = get_colloids_in_vision(
                colloid, other_colls, vision_radius=self.detection_radius_position_colls
            )
            predator = [p for p in colloids if p.type == self.predator_type]
            # only one predator in the simulation
            pred_in_vision = get_colloids_in_vision(
                colloid, predator, vision_radius=self.detection_radius_position_pred
            )
            colls_in_vision_position = np.array([c.pos for c in colls_in_vision])
            colls_in_vision_velocity = np.array([c.velocity for c in colls_in_vision])

            pred_in_vision_position = np.array([p.pos for p in pred_in_vision])

            force_a, force_r = np.array([0, 0, 0]), np.array([0, 0, 0])
            if len(colls_in_vision) > 0:
                force_a = np.sum(colls_in_vision_velocity - colloid.velocity, axis=0)

                force_r_notnorm = np.sum(colls_in_vision_position - colloid.pos, axis=0)
                dist_norm = np.linalg.norm(colls_in_vision_position - colloid.pos)
                force_r = force_r_notnorm / dist_norm

            force_h = self.home_pos - colloid.pos

            force_p = np.array([0, 0, 0])
            if len(pred_in_vision) > 0:
                force_p_notnorm = np.sum(colloid.pos - pred_in_vision_position, axis=0)
                dist_norm_pred = np.linalg.norm(colloid.pos - pred_in_vision_position)
                force_p = force_p_notnorm / dist_norm_pred

            force_f = (
                -colloid.velocity
                * (np.abs(colloid.velocity) - self.agent_speed)
                / self.agent_speed
            )

            force = (
                self.force_params["K_a"] * force_a
                + self.force_params["K_r"] * force_r
                + self.force_params["K_h"] * force_h
                + self.force_params["K_p"] * force_p
                + self.force_params["K_f"] * force_f
            )

            force_magnitude = np.linalg.norm(force)
            force_direction = force / force_magnitude

            actions.append(Action(force=force_magnitude, new_direction=force_direction))
        return actions


def get_colloids_in_vision(coll, other_coll, vision_radius):
    my_pos = coll.pos
    colls_in_vision = []
    for other_p in other_coll:
        dist = other_p.pos - my_pos
        dist_norm = np.linalg.norm(dist)
        in_range = dist_norm < vision_radius
        if not in_range:
            continue
        if in_range:
            colls_in_vision.append(other_p)
    return colls_in_vision
