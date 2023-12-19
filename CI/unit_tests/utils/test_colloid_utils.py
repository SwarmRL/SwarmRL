"""
Test the colloid utils.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from swarmrl.components.colloid import Colloid
from swarmrl.utils.colloid_utils import (
    compute_distance_matrix,
    compute_forces,
    compute_torque,
    compute_torque_partition_on_rod,
    get_colloid_indices,
)


class TestColloidUtils:
    """
    Test suite for colloid utils.
    """

    def test_get_colloid_indices(self):
        """
        Test the get_colloid_indices function.
        """
        colloid_list = [
            Colloid(
                pos=np.array([0.0, 0.0, 0.0]),
                director=np.array([0.0, 0.0, 1.0]),
                id=0,
                type=0,
            ),
            Colloid(
                pos=np.array([0.0, 0.0, 0.0]),
                director=np.array([0.0, 0.0, 1.0]),
                id=1,
                type=1,
            ),
            Colloid(
                pos=np.array([0.0, 0.0, 0.0]),
                director=np.array([0.0, 0.0, 1.0]),
                id=2,
                type=0,
            ),
            Colloid(
                pos=np.array([0.0, 0.0, 0.0]),
                director=np.array([0.0, 0.0, 1.0]),
                id=3,
                type=1,
            ),
            Colloid(
                pos=np.array([0.0, 0.0, 0.0]),
                director=np.array([0.0, 0.0, 1.0]),
                id=4,
                type=0,
            ),
        ]

        colloid_indices = get_colloid_indices(colloid_list, 0)
        npt.assert_equal(colloid_indices, [0, 2, 4])

        colloid_indices = get_colloid_indices(colloid_list, 1)
        npt.assert_equal(colloid_indices, [1, 3])

    def test_compute_distance_matrix(self):
        """
        Test the compute_distance_matrix function.
        """
        set_a = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        set_b = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        distance_matrix = compute_distance_matrix(set_a, set_b)

        npt.assert_array_equal(
            distance_matrix,
            jnp.array(
                [
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                    [[-2.0, -2.0, -2.0], [-1.0, -1.0, -1.0]],
                ]
            ),
        )

    def test_compute_forces(self):
        """
        Test the compute_forces function.

        This test is a good example of Jax's rounding errors.
        You will find that the two arrays differ by ~1e-14, well
        past the 1e-8 precision of a Jax array. This is an issue due
        to the use of the normal numpy testing framework.
        """
        distances = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
        directors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        force_fn = jax.grad(lambda x: (1 / jnp.linalg.norm(x)) ** 12)
        test_forces = jnp.array([force_fn(d) for d in distances])
        test_outs = jnp.linalg.norm(test_forces, axis=-1) * directors

        fn_outs = jnp.array(
            [compute_forces(r, d) for r, d in zip(distances, directors)]
        )
        assert (fn_outs - test_outs).sum() < 1e-8
        npt.assert_array_almost_equal(fn_outs, test_outs)

    def test_compute_torque(self):
        """
        Test the compute_torque function.
        """
        force = jnp.array([1.0, 0.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])

        torque = compute_torque(force, direction)
        npt.assert_array_equal(torque, jnp.array([0.0, 0.0, -1.0]))

    def test_compute_torque_partition_on_rod(self):
        """
        Test the compute_torque_partition_on_rod function.
        """
        colloid_positions = jnp.array(
            [
                [0.0, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ]
        )
        colloid_directors = jnp.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        rod_positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
        )
        rod_directions = jnp.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )

        torque_partition = compute_torque_partition_on_rod(
            colloid_positions, colloid_directors, rod_positions, rod_directions
        )

        npt.assert_array_almost_equal(
            torque_partition, jnp.array([1.0, 0.0]), decimal=5
        )
