"""
Parent class for the engine.
"""
from socket import socket
import swarmrl.models.interaction_model
import numpy as np
import dataclasses
import struct


@dataclasses.dataclass
class Colloid:
    pos: np.ndarray
    director: np.ndarray
    id: int


experiment_actions = {
    "do_nothing": 1,
    "rotate_clockwise": 4,
    "rotate_anticlockwise": 3,
    "be_active": 2,
}


def _handle_one_step(
    state: np.array,
    force_model: swarmrl.models.interaction_model.InteractionModel,
) -> np.array:
    n_colloids = len(state)




def vector_from_angle(angle):
    return np.array([np.sin(angle), np.cos(angle), 0])


class RealExperiment:
    def __init__(self, connection):
        self.connection = connection

    def setup_simulation(self) -> None:
        pass

    def receive_colloids(self, data_size):
        data_size_int = struct.unpack('I', data_size)[0]
        print("Waiting for receiving actual data")
        data = self.connection.recv(8 * data_size_int)
        while data and len(data) < 8 * data_size_int:
            data.extend(self.connection.recv(8 * data_size_int))

        # cast bytestream to double array and reshape to [x y theta id]
        data = np.array(struct.unpack(str(len(data)//8)+"d", data)).reshape((-1, 4))
        print(f"Received data \n {data}")
        colloids = []
        for row in data:
            colloids.append(Colloid(pos=[row[0],row[1],0], director=vector_from_angle(row[2]), id=row[3]))

        return colloids

    def get_actions(self, colloids, force_model):
        n_colloids = len(colloids)
        ret = np.zeros((n_colloids, 2))
        for idx, coll in enumerate(colloids):
            other_colloids = [c for c in colloids if c is not coll]
            # update the state of an active learner, ignored by non ML models.
            force_model.compute_state(coll, other_colloids)
            action = force_model.calc_action(coll, other_colloids)

            if not np.all(action.torque == 0):
                if action.torque[2] > 0:
                    action_id=experiment_actions["rotate_anticlockwise"]
                else:
                    action_id=experiment_actions["rotate_clockwise"]

            if not action.force == 0.0:
                action_id=experiment_actions["be_active"]
            else:
                action_id=experiment_actions["do_nothing"]

            ret[idx,0] = coll.id
            ret[idx,1] = action_id
        return ret

    def send_sctions(self, actions):
        # Flatten data in 'Fortran' style
        data = actions.flatten('F')
        print(f"Data to send \n{data}")
        # transform to bytes
        data_bytes = struct.pack(str(len(data))+"d", *data)

        # and send them (as bytestream)
        self.connection.sendall(data_bytes)

    def integrate(
        self,
        n_slices: int,
        force_model: swarmrl.models.interaction_model.InteractionModel,
    ) -> None:
        for _ in range(n_slices):
            print("Waiting for receiving data_size")
            data_size = self.connection.recv(8)
            print(f"Received data_size = {data_size}")
            if data_size: #received number of particles
                colloids = self.receive_colloids(data_size)
                actions = self.get_actions(colloids, force_model)
                self.send_sctions(actions)
            else:
                force_model.finalize()
                break


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
