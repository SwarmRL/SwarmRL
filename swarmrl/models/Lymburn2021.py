import numpy as np

from swarmrl.models.interaction_model import Action


class JonnysForceModel:
    def __init__(
        self,
        delay: int = 0,
        col_type: int = 1,
        pred_type: int = 0,
        force_params: dict = None,
        detection_radius_position_colls=np.inf,
        detection_radius_position_pred=np.inf,
        center=np.array([500, 500, 0]),
    ):
        self.kind = "classical"
        if force_params is None:
            a = 1
            force_params = {
                "K_a": 0.01 * a,
                "K_r": -0.01 * a,
                "K_h": -0.05 * a,
                "K_p": 200 * a,
            }
        self.force_params = force_params

        self.delay = delay
        self.counter = 0

        # detection radius
        self.detection_radius_position_colls = detection_radius_position_colls
        self.detection_radius_position_pred = detection_radius_position_pred

        # build a random home_pos
        self.home_pos = center + np.random.random(3) * 1000
        self.home_pos[-1] = 0

        # def types
        self.col_type = col_type
        self.pred_type = pred_type

    def compute_action(self, colloids):
        actions = {}
        for colloid in colloids:
            if colloid.type == self.col_type:
                if self.counter > self.delay:
                    actions[colloid.id] = Action()

                else:
                    other_colloids = [
                        c
                        for c in colloids
                        if c is not colloid and not c.type == self.pred_type
                    ]
                    colls_in_vision = get_colloids_in_vision(
                        colloid,
                        other_colloids,
                        vision_radius=self.detection_radius_position_colls,
                    )

                    predator = [
                        p for p in colloids if p is p.type == self.pred_type
                    ]  # only one predator is taken in account
                    pred_in_vision = get_colloids_in_vision(
                        colloid,
                        predator,
                        vision_radius=self.detection_radius_position_pred,
                    )

                    colls_in_vision_position = np.array(
                        [c.pos for c in colls_in_vision]
                    )
                    colls_in_vision_velocity = np.array(
                        [c.velocity for c in colls_in_vision]
                    )
                    pred_in_vision_position = np.array([p.pos for p in pred_in_vision])

                    force_a, force_r = np.array([0, 0, 0]), np.array([0, 0, 0])
                    if len(colls_in_vision) > 0:
                        force_a = np.sum(
                            colls_in_vision_velocity - colloid.velocity, axis=0
                        )
                        force_r_notnorm = np.sum(
                            colls_in_vision_position - colloid.pos, axis=0
                        )
                        dist_norm = np.linalg.norm(
                            colls_in_vision_position - colloid.pos
                        )
                        alpha = np.where(dist_norm < 20, -10, 10)
                        force_r = alpha * force_r_notnorm / dist_norm

                    force_h = self.home_pos - colloid.pos
                    dist_norm_home = np.linalg.norm(force_h)
                    force_h = 10 * force_h / dist_norm_home

                    force_p = np.array([0, 0, 0])
                    if len(pred_in_vision) > 0:
                        force_p_notnorm = np.sum(
                            colloid.pos - pred_in_vision_position, axis=0
                        )
                        dist_norm_pred = np.linalg.norm(
                            colloid.pos - pred_in_vision_position
                        )
                        force_p = force_p_notnorm / dist_norm_pred**2

                    force = (
                        self.force_params["K_a"] * force_a
                        + self.force_params["K_r"] * force_r
                        + self.force_params["K_h"] * force_h
                        + self.force_params["K_p"] * force_p
                    )

                    force_magnitude = np.linalg.norm(force)
                    force_direction = force / force_magnitude

                    actions[colloid.id] = Action(
                        force=force_magnitude, new_direction=force_direction
                    )
            else:
                pass
        self.counter += 1
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


def rotate_vector_clockwise(v, alpha):
    r = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    return np.matmul(r, v)
