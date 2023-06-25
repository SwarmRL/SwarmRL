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
        self.home_pos = self.center + (np.random.random(3) - 0.5) * 1000
        self.home_pos[-1] = 0

        # def types
        self.col_type = col_type
        self.pred_type = pred_type

    def reset(self):
        self.home_pos = self.center + np.random.random(3) * 1000
        self.home_pos[-1] = 0

    def compute_action(self, colloids):
        sheeps = [c for c in colloids if c.type == self.col_type]
        predators = [p for p in colloids if p.type == self.pred_type]
        sheep_pos = np.array([s.pos for s in sheeps])
        pred_pos = np.array([p.pos for p in predators])
        # sheep_sheep_dist = np.linalg.norm(
        #     sheep_pos[:, None, :] - sheep_pos[None, :, :], axis=-1
        # )
        # sheep_pred_dist = np.linalg.norm(
        #     sheep_pos[:, None, :] - pred_pos[None, :, :], axis=-1
        # )
        sheep_vel = np.array([s.velocity for s in sheeps])

        # compute the force

        # force_a = np.zeros(3)
        # force_r = np.zeros(3)
        # force_h = np.zeros(3)
        # force_p = np.zeros(3)

        k_a = 0.3
        k_r = -0.001
        k_h = -5.0
        k_p = 0.04

        # alignment: difference between the average velocity
        # of all
        # sheep and the velocity of the sheep
        force_a = k_a * np.mean(sheep_vel - sheep_vel[:, None, :], axis=0)
        print("force_a")
        print(force_a)

        # repulsion: repulsion from other sheep. The repulsion should
        # be quadratic in the distance, centered around an r_0
        dr = np.sum(
            (sheep_pos[:, None, :] - sheep_pos[None, :, :]),
            axis=0,
        )
        # force_r_amp = np.linalg.norm(dr, axis=-1)
        force_r = k_r * (dr - self.r_0 * np.array([1, 1, 0])) ** 2
        print("force_r")
        print(force_r)

        # home: repulsion from the home position. The repulsion should
        # be constant in the distance
        force_h_vec = self.home_pos - sheep_pos
        force_h_vec /= np.linalg.norm(force_h_vec, axis=-1)[:, None]
        force_h = k_h * force_h_vec
        print("force_h")
        print(force_h_vec)

        # predator: repulsion from the predator. The repulsion should
        # be cubic in the distance
        force_p = k_p * np.sum(
            (sheep_pos[:, None, :] - pred_pos[None, :, :]),
            axis=0,
        )
        print("force_p")
        print(force_p)

        total_force = force_h + force_a + force_r + force_p
        force_magnitude = np.linalg.norm(total_force, axis=-1)
        total_force /= force_magnitude[:, None]

        # compute the action
        actions = {}
        for i, sheep in enumerate(sheeps):
            # print(force_magnitude[i])
            actions[sheep.id] = Action(
                force=force_magnitude[i],
                new_direction=total_force[i],
            )

        # actions = {}
        # for colloid in colloids:
        #     if colloid.type == self.col_type:
        #         if self.counter < self.delay:
        #             actions[colloid.id] = Action()
        #         else:
        #             other_colloids = [
        #                 c
        #                 for c in colloids
        #                 if c is not colloid and c.type == self.col_type
        #             ]
        #             colls_in_vision = get_colloids_in_vision(
        #                 colloid,
        #                 other_colloids,
        #                 vision_radius=self.detection_radius_
        #                 position_colls,
        #             )
        #
        #             predator = [
        #                 p for p in colloids if p.type == self.pred_type
        #             ]  # only one predator is taken in account
        #             pred_in_vision = get_colloids_in_vision(
        #                 colloid,
        #                 predator,
        #                 vision_radius=self.detection_radius_position_pred,
        #             )
        #
        #             colls_in_vision_position = np.array(
        #                 [c.pos for c in colls_in_vision]
        #             )
        #             colls_in_vision_velocity = np.array(
        #                 [c.velocity for c in colls_in_vision]
        #             )
        #             pred_in_vision_position = np.array([p.pos for p in
        #             pred_in_vision])
        #
        #             force_a, force_r = np.array([0, 0, 0]), np.array([0, 0, 0])
        #             if len(colls_in_vision) > 0:
        #                 force_a = np.sum(
        #                     colls_in_vision_velocity - colloid.velocity, axis=0
        #                 )
        #                 force_r_notnorm = np.sum(
        #                     colls_in_vision_position - colloid.pos, axis=0
        #                 )
        #                 dist_norm = np.linalg.norm(
        #                     colls_in_vision_position - colloid.pos
        #                 )
        #
        #                 force_r = force_r_notnorm / dist_norm
        #
        #             force_h = self.home_pos - colloid.pos
        #             dist_norm_home = np.linalg.norm(force_h)
        #             force_h = force_h / dist_norm_home
        #
        #             force_p = np.array([0, 0, 0])
        #             if len(pred_in_vision) > 0:
        #                 force_p = np.sum(
        #                     preditor_force(colloid.pos,
        #                     pred_in_vision_position), axis=0
        #                 )
        #             # print('preditor force ',
        #             np.linalg.norm(self.force_params["K_p"] * force_p))
        #             # print("home force: ",
        #             np.linalg.norm(self.force_params["K_h"] * force_h))
        #             # print("repel force: ",
        #             np.linalg.norm(self.force_params["K_r"] * force_r))
        #             force = (
        #                 self.force_params["K_a"] * force_a
        #                 + self.force_params["K_r"] * force_r
        #                 + self.force_params["K_h"] * force_h
        #                 + self.force_params["K_p"] * force_p
        #             )
        #
        #             force_magnitude = np.linalg.norm(force)
        #             force_direction = force / force_magnitude
        #
        #             actions[colloid.id] = Action(
        #                 force=force_magnitude, new_direction=force_direction
        #             )
        #     else:
        #         pass
        # self.counter += 1
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
