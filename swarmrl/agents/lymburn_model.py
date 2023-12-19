from swarmrl.actions.actions import Action
from swarmrl.agents.classical_agent import ClassicalAgent
import h5py as hf
import numpy as np


class Lymburn(ClassicalAgent):
    def __init__(self,
                 force_params: dict,
                 pred_movement: callable,
                 pred_params: np.array,
                 from_file: bool = False,
                 detection_radius_position_colls=np.inf,
                 detection_radius_position_pred=np.inf,
                 home_pos=np.array([500, 500, 0]),
                 agent_speed=10,
                 time_slice: float = 0.01
                 ):
        """
        Parameters
        ----------
        force_params : dict
            Dictionary containing the force parameters.
                e.g.: force_params = {"K_a":K_a, "K_r":K_r,
                  "K_h":K_h, "K_f":K_f, "K_p":K_p}
        pred_movement : callable
            Function that describes the movement of the predator
                input: t, pos, velocity, home_pos, pred_params
        pred_params : np. array (3,1)
            Parameters for the predator movement function
                If FromFile=True, force_params[0] is the path to
                    the trajectory file
        from_file: bool
            If True, the predator follows a trajectory from a trajectory file
        detection_radius_position_colls : float
            Radius in which the colloids are detected
        detection_radius_position_pred : float
            Radius in which the predator is detected
        home_pos : np. array
            Position of the home
        agent_speed : float
            Speed of the colloids (used for the friction force)
        time_slice : float
        """
        self.force_params = force_params
        self.pred_movement = pred_movement
        self.pred_params = pred_params
        self.detection_radius_position_colls = detection_radius_position_colls
        # implies r_align=r_repulsion
        self.detection_radius_position_pred = detection_radius_position_pred
        self.home_pos = home_pos
        self.agent_speed = agent_speed

        self.t = 0
        self.time_slice = time_slice

        self.from_file = from_file
        if self.from_file:
            db = hf.File(f"{self.pred_params[0]}/trajectory.hdf5")
            self.wanted_pos = db["Wanted_Positions"][:]
        self.index_tracker = -1

    def update_force_params(self, K_a=None, K_r=None, K_h=None, K_f=None, K_p=None):
        update_params = {"K_a": K_a,
                             "K_r": K_r,
                             "K_h": K_h,
                             "K_f": K_f,
                             "K_p": K_p}
        for key, value in update_params.items():
            if value is not None:
                self.force_params[key] = value

    def update_pred_movement(self, pred_movement):
        self.pred_movement = pred_movement

    def update_pred_params(self, pred_params0=None, pred_params1=None, pred_params2=None):
        update_params = [pred_params0, pred_params1, pred_params2]
        for i, value in enumerate(update_params):
            if value is not None:
                self.pred_params[i] = value

    def calc_action(self, colloids):
        actions = []
        self.t += self.time_slice
        self.index_tracker += 1
        for colloid in colloids:
            if colloid.type == 1:
                if self.from_file:
                    pred_force = traj_from_file(
                        self.wanted_pos[self.index_tracker],
                        self.wanted_pos[self.index_tracker + 1],
                        colloid.velocity)
                else:
                    pred_force = self.pred_movement(
                        self.t,
                        colloid.pos,
                        colloid.velocity,
                        self.home_pos,
                        self.pred_params)

                nd = np.array([pred_force[0], pred_force[1], pred_force[2]])
                new_direction = nd / np.linalg.norm(nd)
                actions.append(Action(force=np.linalg.norm(nd),
                                      new_direction=new_direction))
                continue

            other_colls = [c for c in colloids if c is not colloid and not c.type == 1]
            colls_in_vision = get_colloids_in_vision(
                colloid,
                other_colls,
                vision_radius=self.detection_radius_position_colls)
            predator = [p for p in colloids if p.type == 1]
            # only one predator in the simulation
            pred_in_vision = get_colloids_in_vision(
                colloid,
                predator,
                vision_radius=self.detection_radius_position_pred)
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

            force_f = -colloid.velocity * (np.abs(colloid.velocity) - self.agent_speed) / self.agent_speed

            force = self.force_params["K_a"] * force_a + \
                    self.force_params["K_r"] * force_r + \
                    self.force_params["K_h"] * force_h + \
                    self.force_params["K_p"] * force_p + \
                    self.force_params["K_f"] * force_f

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


def harmonic_1d(t, pos, director, home_pos, params):
    force_x = params[0] * np.cos(params[1] * t)
    force_y = home_pos[1] - pos[1]
    force_z = 0
    return force_x, force_y, force_z


def harmonic_2d(t, pos, director, home_pos, params):
    force_x = params[0] * np.cos(params[1] * t)
    force_y = params[0] * np.sin(params[1] * t)
    force_z = 0
    return force_x, force_y, force_z

def no_force(t, pos, director, home_pos, params):
    return 0, 0, 0


def traj_from_file(pos, pos1, velocity):
    """
    Function to get classical particle on a trajectory from given datafile
    """
    mass = 1
    force = (pos1 - pos - velocity * 0.01) * 2 * mass / 0.01 ** 2
    force = 0.0004 * 1e5 * 0.25 * force/50
    return force[0], force[1], force[2]
