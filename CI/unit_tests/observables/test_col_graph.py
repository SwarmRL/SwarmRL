"""
Unit test for the position-angle observable.
"""


import os
import time

import numpy as np
from numpy.testing import assert_array_equal

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.col_graph import ColGraph


def build_cols(collist):
    """
    Helper function that builds a list of colloids from a list.

    Parameters
    ----------
    collist : list
        List of the number of colloids of each type.

    Returns
    -------
    cols : list

    """
    cols = []
    for type_cols, num_cols in enumerate(collist):
        for i in range(num_cols):
            position = 1000 * np.random.random(3)
            position[-1] = 0
            direction = np.random.random(3)
            direction[-1] = 0
            direction = direction / np.linalg.norm(direction)
            cols.append(Colloid(pos=position, director=direction, type=type_cols, id=i))
    return cols


def build_specific_cols(num_cols, distance):
    """
    Helper function to build a list of colloids that are at a specific distance from a
    central colloid.

    Parameters
    ----------
    num_cols : int
        Number of colloids to build.
    distance : float
        Distance of the colloids from the central colloid.

    Returns
    -------
    cols : list
        The central colloid is the first colloid in the list.
    """
    cols = []
    colloid_0 = Colloid(
        pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]), type=1, id=0
    )
    cols.append(colloid_0)
    for i in range(num_cols):
        random_angle = np.random.random() * 2 * np.pi
        position = np.array(
            [distance * np.cos(random_angle), distance * np.sin(random_angle), 0]
        )
        random_dir = np.random.random(3)
        random_dir[-1] = 0
        random_dir = random_dir / np.linalg.norm(random_dir)
        cols.append(Colloid(pos=position, director=random_dir, type=0, id=i + 1))
    return cols


def build_circle_cols(n_colls, dist=300):
    cols = []
    pos_0 = 1000 * np.random.random(3)
    pos_0[-1] = 0
    direction_0 = np.random.random(3)
    direction_0[-1] = 0
    for i in range(n_colls - 1):
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

    def test_compute_observable_zero_cut(self):
        """
        Tests if for no colloid in the observation radius, the observable returns a
        graph for each colloid with only one node which is all zeros.
        """
        # create a list of colloids.
        cols = build_cols([3, 2, 1])

        # test for r_cut = 0
        graph_obs = ColGraph(cutoff=0.0, box_size=np.array([1000, 1000, 1000]))
        graphs = graph_obs.compute_observable(colloids=cols)
        for graph in graphs:
            assert graph.n_node == 1

    def test_compute_observable_full_box(self):
        """
        Tests if for all colloids in the observation radius, the observable returns a
        graph for each colloid with num_cols - 1 nodes.
        """
        # create a list of colloids.
        num_cols = 20
        graph_obs = ColGraph(cutoff=np.sqrt(2), box_size=np.array([1000, 1000, 1000]))
        for i in range(1):
            cols = build_circle_cols(num_cols)
            start_time = time.time()
            graphs = graph_obs.compute_observable(colloids=cols)
            end_time = time.time()
            print("time for one iteration: ", end_time - start_time)
            # for graph in graphs:
            #     assert graph.n_node == int(num_cols - 1)
        info = {
            "colloids": np.array(cols, dtype=object),
            "graphs": np.array(graphs[0], dtype=object),
        }
        np.save("graph_list.npy", info, allow_pickle=True)

    def test_compute_observable_r500(self):
        """
        This test checks if the first graph has the correct nodes. The first dimension
        of the nodes is the distance of the first colloid to the other colloids. which
        is 500 for all of them.
        The second test checks, if the delta_type is -1 for all nodes (last dimension
        of the node) of the first graph.
        """
        cols_500 = build_specific_cols(10, 500)
        graph_obs = ColGraph(cutoff=0.51, box_size=np.array([1000, 1000, 1000]))
        graphs = graph_obs.compute_observable(colloids=cols_500)
        np.testing.assert_array_almost_equal(graphs[0].nodes[:, 0], 0.5 * np.ones(10))
        np.testing.assert_array_almost_equal(graphs[0].nodes[:, -1], -1 * np.ones(10))

    def test_compute_observable_directions(self):
        """
        Test if the angle between the director and the position of the other colloid is
        correct. For col1 it should be Pi/4 and for col2 it should be 3Pi/4.
        The second test checks if the angel between the directors are correct.
        For col1 this should be Pi/2 and for col2 it should be -Pi/2
        """
        cols_direction = []
        col1 = Colloid(
            pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]), type=0, id=0
        )
        col2 = Colloid(
            pos=np.array([500, 500, 0]), director=np.array([0, 1, 0]), type=1, id=1
        )
        cols_direction.append(col1)
        cols_direction.append(col2)

        graph_obs_angle = ColGraph(cutoff=0.9, box_size=np.array([1000, 1000, 1000]))
        graphs = graph_obs_angle.compute_observable(colloids=cols_direction)
        pos_angle1 = np.pi / 4
        pos_angle2 = 3 * np.pi / 4

        np.testing.assert_almost_equal(graphs[0].nodes[0, 1], pos_angle1)
        np.testing.assert_almost_equal(graphs[1].nodes[0, 1], pos_angle2)

        dir_angle1 = np.pi / 2
        dir_angle2 = -1 * np.pi / 2
        np.testing.assert_almost_equal(graphs[0].nodes[0, 2], dir_angle1)
        np.testing.assert_almost_equal(graphs[1].nodes[0, 2], dir_angle2)
        print(graphs[0])

    def test_save_observable(self):
        """
        Test the saving of an list of colloids in a npy file.
        """
        cols = []
        for i in range(2):
            position = np.random.random(3)
            direction = np.random.random(3)
            cols.append(Colloid(pos=position, director=direction, type=0, id=i))

        np.save("test.npy", cols)
        loaded_cols = np.load("test.npy", allow_pickle=True)
        os.remove("test.npy")

        assert_array_equal(loaded_cols, cols)
