"""
Parent class for the engine.
"""
import struct
import typing

import numpy as np

import swarmrl.engine.engine
from swarmrl.models.interaction_model import Colloid


class ConnectionClosedError(Exception):
    """
    Exception to capture when Matlab closes the connection
    """

    pass


experiment_actions = {
    "do_nothing": 1,
    "rotate_clockwise": 4,
    "rotate_anticlockwise": 3,
    "be_active": 2,
}


def vector_from_angle(angle):
    return np.array([np.cos(angle), np.sin(angle), 0])


class RealExperiment(swarmrl.engine.engine.Engine):
    """
    Class for the real experiment interface.
    """

    def __init__(self, connection):
        """
        Constructor for the experiment.

        Parameters
        ----------
        connection : object
                Communication object to be used to talk to the experiment.
        """
        self.connection = connection

    def setup_simulation(self) -> None:
        """
        Method not required in the real experiment.
        """
        pass

    def receive_colloids(self) -> typing.List[Colloid]:
        """
        Collect the colloid data from the experiment.

        Returns
        -------
        colloids : list
                A list of colloids.
        """
        print("Waiting for receiving data_size")
        data_size = self.connection.recv(8)
        # break if connection closed
        if not data_size:
            print("Received connection closed signal")
            raise ConnectionClosedError

        data_size_int = struct.unpack("I", data_size)[0]
        print(f"Received data_size = {data_size_int}")
        print("Waiting for receiving actual data")
        data = self.connection.recv(8 * data_size_int)
        while data and len(data) < 8 * data_size_int:
            data.extend(self.connection.recv(8 * data_size_int))

        # cast bytestream to double array and reshape to [x y theta id]
        data = np.array(struct.unpack(str(len(data) // 8) + "d", data)).reshape((-1, 4))
        print(f"Received data with shape {np.shape(data)} \n")
        colloids = []
        for row in data:
            coll = Colloid(
                pos=np.array([row[0], row[1], 0]),
                director=vector_from_angle(row[2]),
                id=row[3],
            )
            colloids.append(coll)

        return colloids

    def get_actions(
        self,
        colloids: typing.List[Colloid],
        force_model: swarmrl.models.interaction_model.InteractionModel,
    ) -> np.array:
        """
        Collect the actions on the particles.

        This method calls the interaction models and collects all of the actions on the
        colloids.

        Parameters
        ----------
        colloids : list
                List of colloids on which to compute interactions.
        force_model : swarmrl.models.interaction_model.InteractionModel
                Interaction model to use in the action computation.

        Returns
        -------
        ret : np.ndarray
                A numpy array of the actions on each colloid.
        """
        n_colloids = len(colloids)
        ret = np.zeros((n_colloids, 2))
        actions = force_model.calc_action(colloids)
        for idx, coll in enumerate(colloids):
            other_colloids = [c for c in colloids if c is not coll]
            # update the state of an active learner, ignored by non ML models.
            force_model.compute_state(coll, other_colloids)
            action = actions[idx]

            if not action.force == 0.0:
                action_id = experiment_actions["be_active"]
            else:
                action_id = experiment_actions["do_nothing"]

            if not np.all(action.torque == 0):
                if action.torque[2] > 0:
                    action_id = experiment_actions["rotate_anticlockwise"]
                else:
                    action_id = experiment_actions["rotate_clockwise"]

            ret[idx, 0] = coll.id
            ret[idx, 1] = action_id
        return ret

    def send_actions(self, actions: np.ndarray):
        """
        Send the actions to the experiment apparatus.

        Parameters
        ----------
        actions : np.ndarray
                A numpy array of actions to send to the experiment.

        Returns
        -------
        Sends all data to the experiment.
        """
        # Flatten data in 'Fortran' style
        data = actions.flatten("F")
        print(f"Sending data with shape {np.shape(data)} \n")

        data_bytes = struct.pack(str(len(data)) + "d", *data)
        self.connection.sendall(data_bytes)

    def integrate(
        self,
        n_slices: int,
        force_model: swarmrl.models.interaction_model.InteractionModel,
    ) -> None:
        """
        Perform the real-experiment equivalent of an integration step.

        Parameters
        ----------
        n_slices : int
                Number of integration steps to perform.
        force_model : swarmrl.models.interaction_model.InteractionModel
                Model to use in the interaction computations.

        Returns
        -------
        Sends actions to the experiment.

        Notes
        -----
        Can we always refer to real time as discreet? I like it.
        """
        for _ in range(n_slices):
            try:
                colloids = self.receive_colloids()
            except ConnectionClosedError:
                # force_model.finalize()
                self.connection.close()
                break

            actions = self.get_actions(colloids, force_model)
            self.send_actions(actions)
