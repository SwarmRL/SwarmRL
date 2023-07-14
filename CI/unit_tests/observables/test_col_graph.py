"""
Unit test for the position-angle observable.
"""


import numpy as np
import numpy.testing as npt

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.col_graph_V0 import ColGraphV1


def build_circle_cols(n_cols, dist=300):
    cols = []
    pos_0 = 1000 * np.random.random(3)
    pos_0[-1] = 0
    direction_0 = np.random.random(3)
    direction_0[-1] = 0
    for i in range(n_cols):
        theta = np.random.random(1)[0] * 2 * np.pi
        position = pos_0 + dist * np.array([np.cos(theta), np.sin(theta), 0])
        direction = np.random.random(3)
        direction[-1] = 0
        direction = direction / np.linalg.norm(direction)
        cols.append(Colloid(pos=position, director=direction, type=0, id=i))
    return cols


class TestGraphObservable:
    """
    Test suite for the position angle observable.
    """

    def test_initialize(self):
        """
        Tests if for all colloids in the observation radius, the observable returns a
        graph for each colloid with num_cols - 1 nodes.
        """
        # create a list of colloids.
        num_cols = 20
        graph_obs = ColGraphV1(
            num_nodes=20, cutoff=np.sqrt(2), box_size=np.array([1000, 1000, 1000])
        )

        cols = build_circle_cols(num_cols)

        graph_obs.initialize(colloids=cols)

        assert graph_obs.num_nodes == len(cols) - 1

    def test_compute_observable(self):
        num_cols = 20
        graph_obs = ColGraphV1(
            num_nodes=20, cutoff=np.sqrt(2), box_size=np.array([1000, 1000, 1000])
        )

        cols = build_circle_cols(num_cols)

        graph_obs.initialize(colloids=cols)

        graph = graph_obs.compute_observable(colloids=cols)

        nodes = graph.nodes
        assert np.shape(nodes) == (num_cols, num_cols - 1, 3)

    def test_angles(self):
        col1 = Colloid(
            pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]), type=0, id=0
        )
        col2 = Colloid(
            pos=np.array([1, 0, 0]), director=np.array([1, 0, 0]), type=0, id=1
        )
        col3 = Colloid(
            pos=np.array([0, 1, 0]), director=np.array([1, 0, 0]), type=0, id=2
        )

        cols = [col1, col2, col3]

        graph_obs = ColGraphV1(
            num_nodes=2, cutoff=np.sqrt(2), box_size=np.array([1000, 1000, 1000])
        )

        graph_obs.initialize(colloids=cols)

        graph = graph_obs.compute_observable(colloids=cols)

        nodes = graph.nodes
        assert np.shape(nodes) == (3, 2, 3)

        distances = [
            (1 / 1000, 1 / 1000),
            (1 / 1000, np.sqrt(2) / 1000),
            (1 / 1000, np.sqrt(2) / 1000),
        ]
        angles = [(0, np.pi / 2), (np.pi, 3 * np.pi / 4), (-np.pi / 2, -np.pi / 4)]
        for i, node in enumerate(nodes):
            npt.assert_array_almost_equal(node[:, 1], angles[i])
            npt.assert_array_almost_equal(node[:, 0], distances[i])

    def test_edges(self):
        col1 = Colloid(
            pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]), type=0, id=0
        )
        col2 = Colloid(
            pos=np.array([1, 0, 0]), director=np.array([1, 0, 0]), type=0, id=1
        )
        col3 = Colloid(
            pos=np.array([0, 1, 0]), director=np.array([1, 0, 0]), type=0, id=2
        )

        cols = [col1, col2, col3]

        graph_obs = ColGraphV1(
            num_nodes=2, cutoff=np.sqrt(2), box_size=np.array([1000, 1000, 1000])
        )

        graph_obs.initialize(colloids=cols)

        graph = graph_obs.compute_observable(colloids=cols)

        npt.assert_array_almost_equal(graph.n_edge, np.array([[0], [0], [0]]))
