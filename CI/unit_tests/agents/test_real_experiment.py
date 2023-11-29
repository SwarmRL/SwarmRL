import enum
import struct
import unittest as ut

import numpy as np

import swarmrl.engine.real_experiment
import swarmrl.models.dummy_models

"""
Test the connection between simulation and experiment.
To this end, set up a mock-connection that acts as the experiment does.
The assertions happen inside the mock-connections.

If the experiment changes (i.e. the matlab code), the changes need to be reflected in
MockConnection to ensure this test stays up-to-date.
"""


class MessageType(enum.Enum):
    DATA_SIZE = 0
    ACTUAL_DATA = 1


class MockConnection:
    """
    The experiment sends
    - 1st the data size of the particle property matrix
    - 2nd the actual matrix

    The experiment receives
    - a matrix of actions
    """

    def __init__(self, n_partcl: int, box_l: float):
        self.n_partcl = n_partcl
        self.box_l = box_l

        self.actual_data_size = self.n_partcl * 4  # [x y theta id]
        self.ids = np.array(range(self.n_partcl))

        # the first message of the experiment will always be the data size
        self.next_message = MessageType.DATA_SIZE

    def recv(self, data_size: int) -> bytes:
        """
        Supply the engine with data, alternating between the data size and a matrix of
        random particle properties.
        """
        if self.next_message == MessageType.DATA_SIZE:
            assert data_size == 8
            actual_data_size_bytes = struct.pack("I", self.actual_data_size)
            self.next_message = MessageType.ACTUAL_DATA
            return actual_data_size_bytes
        elif self.next_message == MessageType.ACTUAL_DATA:
            assert data_size == 8 * self.actual_data_size

            xs = self.box_l * np.random.random((self.n_partcl,))
            ys = self.box_l * np.random.random((self.n_partcl,))
            thetas = 2 * np.pi * np.random.random((self.n_partcl,))
            partcl_props = np.column_stack((xs, ys, thetas, self.ids.astype(float)))

            # experiment sends data C-style flattened
            partcl_props = partcl_props.flatten("C")
            partcl_props_bytes = struct.pack(
                str(len(partcl_props)) + "d", *partcl_props
            )

            self.next_message = MessageType.DATA_SIZE
            return partcl_props_bytes

    def sendall(self, data: bytes):
        """
        Receive data from the engine and check that the values are sensible
        """
        # experiment expects data to be F-Style flattened
        data_unpacked = np.array(struct.unpack(str(len(data) // 8) + "d", data))
        data_matrix = data_unpacked.reshape((-1, 2), order="F")
        shape = np.shape(data_matrix)
        assert shape == (self.n_partcl, 2)
        assert np.all(data_matrix[:, 0].astype(int) == self.ids)
        for action_id in data_matrix[:, 1]:
            assert (
                action_id in swarmrl.engine.real_experiment.experiment_actions.values()
            )


class TestRealExperiment(ut.TestCase):
    def test_communication(self):
        connection = MockConnection(n_partcl=17, box_l=8.765)
        runner = swarmrl.engine.real_experiment.RealExperiment(connection)
        runner.setup_simulation()
        f_model = swarmrl.models.dummy_models.ConstForce(123)
        runner.integrate(10, f_model)
        runner.finalize()


if __name__ == "__main__":
    ut.main()
