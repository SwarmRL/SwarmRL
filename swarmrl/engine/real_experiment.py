"""
Parent class for the engine.
"""
import swarmrl.models.interaction_model
import numpy as np
import dataclasses


@dataclasses.dataclass
class Colloid:
    pos: np.ndarray
    director: np.ndarray


experiment_actions = {
    "do_nothing": 0,
    "rotate_clockwise": 1,
    "rotate_anticlockwise": 2,
    "be_active": 3,
}


class DummyCommunicator:
    def __init__(self, n_colloids=17):
        self.n_colloids = n_colloids

    def get_particle_state(self) -> np.array:
        pos = 5 * np.random.random((self.n_colloids, 2))
        theta = 2 * np.pi * np.random.random((self.n_colloids,))
        last_action = np.random.randint(1, 5, (self.n_colloids,))
        state = np.column_stack((pos, theta, last_action))
        return state

    def do_experimental_magic(self, orders):
        np.testing.assert_array_equal(orders.shape, [self.n_colloids, 7])


def _handle_one_step(
    state: np.array,
    force_model: swarmrl.models.interaction_model.InteractionModel,
) -> np.array:
    n_colloids = len(state)
    poss = np.column_stack((state[:, 0:2], np.zeros((n_colloids,))))
    directors = np.stack([vector_from_angle(angle) for angle in state[:, 2]])

    colloids = []
    for pos, direc in zip(poss, directors):
        colloids.append(Colloid(pos=pos, director=direc))

    action_ids = []
    for coll in colloids:
        other_colloids = [c for c in colloids if c is not coll]
        # update the state of an active learner, ignored by non ML models.
        force_model.compute_state(coll, other_colloids)
        action = force_model.calc_action(coll, other_colloids)

        if not np.all(action.torque == 0):
            if action.torque[2] > 0:
                action_ids.append(experiment_actions["rotate_anticlockwise"])
            else:
                action_ids.append(experiment_actions["rotate_clockwise"])
            continue
        if not action.force == 0.0:
            action_ids.append(experiment_actions["be_active"])
        else:
            action_ids.append(experiment_actions["do_nothing"])

    ret = np.zeros((n_colloids, 7))
    ret[:, 0] = action_ids
    return ret


def vector_from_angle(angle):
    return np.array([np.sin(angle), np.cos(angle), 0])


class RealExperiment:
    def __init__(self):
        self.communicator = DummyCommunicator()

    def setup_simulation(self) -> None:
        pass

    def integrate(
        self,
        n_slices: int,
        force_model: swarmrl.models.interaction_model.InteractionModel,
    ) -> None:
        for _ in range(n_slices):
            state = self.communicator.get_particle_state()
            actions = _handle_one_step(state, force_model)
            self.communicator.do_experimental_magic(actions)

    def get_particle_data(self) -> dict:
        """
        Get position, velocity and director of the particles as a dict of np.array
        """
        state = self.communicator.get_particle_state()
        n_colloids = len(state)
        poss = np.column_stack(state[:, 0:2], np.zeros((n_colloids,)))
        directors = np.stack([vector_from_angle(angle) for angle in state[:, 2]])
        return {"Unwrapped_Positions": poss, "Directors": directors}

    def finalize(self):
        """
        Optional: to clean up after finishing the simulation (e.g. writing the last chunks of trajectory)
        """
        pass
