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
                "K_r": 3 * a,
                "K_h": -5.0 * a,
                "K_p": 40000 * a,
            }
        self.force_params = force_params

        self.delay = delay
        self.counter = 0

        # detection radius
        self.detection_radius_position_colls = detection_radius_position_colls
        self.detection_radius_position_pred = detection_radius_position_pred

        # parameter for forces
        self.r_0 = 40

        # build a random home_pos
        self.center = center
        self.home_pos = self.center + (np.random.random(3) - 0.5) * 0
        self.home_pos[-1] = 0

        # def types
        self.col_type = col_type
        self.pred_type = pred_type

    def reset(self):
        self.home_pos = self.center + np.random.random(3) * 0
        self.home_pos[-1] = 0

    def compute_action(self, colloids):
        sheeps = [c for c in colloids if c.type == self.col_type]
        predators = [p for p in colloids if p.type == self.pred_type]
        sheep_pos = np.array([s.pos for s in sheeps])
        pred_pos = np.array([p.pos for p in predators])
        # sheep_vel = np.array([s.velocity for s in sheeps])

        # repulsion force
        dr = sheep_pos[:, None, :] - sheep_pos[None, :, :]
        dr_norm = np.linalg.norm(dr, axis=-1) / 1000
        mask = 1500 * (dr_norm - self.r_0 / 1000) * abs(dr_norm - self.r_0 / 1000)
        dr /= dr_norm[:, :, None] + 1e-10
        force_r = mask[:, :, None] * dr
        force_r = np.sum(-force_r, axis=1)
        froce_r_magnitude = np.linalg.norm(force_r, axis=-1)
        force_r /= froce_r_magnitude[:, None]

        # home force
        force_h_vec = sheep_pos - self.home_pos
        force_h_vec /= np.linalg.norm(force_h_vec, axis=-1)[:, None]
        force_h = force_h_vec

        # preditor force
        dist_vec = pred_pos[None, :, :] - sheep_pos[:, None, :]
        dists = np.linalg.norm(dist_vec, axis=-1)
        mask = np.where(dists < 50, 1 / dists + 1e-10, 0)
        force_p = np.sum(mask[:, :, None] * dist_vec, axis=1)

        # compute the total force
        total_force = 20 * force_r + 20 * force_h + (-50) * force_p
        force_magnitude = np.linalg.norm(total_force, axis=-1)
        total_force /= force_magnitude[:, None]

        # compute the action
        actions = {}
        for i, sheep in enumerate(sheeps):
            actions[sheep.id] = Action(
                force=force_magnitude[i],
                new_direction=total_force[i],
            )
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


k = 1


def preditor_force(position, preditor_position, scale=1, offset=600, temp=10):
    force = position - preditor_position
    dist_norm = np.linalg.norm(force)
    force *= scale * np.exp(-(dist_norm - offset) / temp) / dist_norm
    if k == 1:
        np.save(
            "preditor_force",
            [np.copy(position), preditor_position, force],
            allow_pickle=True,
        )
    return force
