"""
Test the utils module.
"""
import jax.numpy as np
import numpy.testing as npt

import swarmrl.utils as utils
from swarmrl.utils.utils import (
    calc_signed_angle_between_directors,
    create_colloids,
    gather_n_dim_indices,
)


class TestUtils:
    """
    Test suite for the utils.
    """

    def test_gather_n_dim_indices(self):
        """
        Test the indices gathering function for even and odd numbers
        """
        # 5 particles, 3 time steps, 2 options
        data = np.array(
            [
                [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
            ]
        )
        indices = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0]])
        true_selection = np.array(
            [[1, 3, 5, 7, 9], [10, 12, 14, 16, 18], [20, 22, 25, 27, 28]]
        )

        test_selection = gather_n_dim_indices(data, indices)
        npt.assert_array_equal(test_selection, true_selection)

        # 2 particles, 3 time steps, 2 options
        data = np.array([[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]])
        indices = np.array([[1, 1], [0, 0], [0, 0]])
        true_selection = np.array([[1, 3], [10, 12], [20, 22]])
        test_selection = gather_n_dim_indices(data, indices)
        npt.assert_array_equal(test_selection, true_selection)

        # 3 particles, 4 time steps, 2 options
        data = np.array(
            [
                [[0, 1], [2, 3], [4, 5]],
                [[10, 11], [12, 13], [14, 15]],
                [[20, 21], [22, 23], [24, 25]],
                [[30, 31], [32, 33], [34, 35]],
            ]
        )
        indices = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 1], [1, 0, 1]])
        true_selection = np.array([[1, 3, 5], [10, 12, 14], [20, 22, 25], [31, 32, 35]])
        test_selection = gather_n_dim_indices(data, indices)
        npt.assert_array_equal(test_selection, true_selection)

        # 4 particles, 4 time steps, 2 options
        data = np.array(
            [
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                [[10, 11], [12, 13], [14, 15], [16, 17]],
                [[20, 21], [22, 23], [24, 25], [26, 27]],
                [[30, 31], [32, 33], [34, 35], [36, 37]],
            ]
        )
        indices = np.array([[1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1]])
        true_selection = np.array(
            [[1, 3, 5, 6], [10, 12, 14, 17], [20, 22, 25, 27], [31, 32, 35, 37]]
        )
        test_selection = gather_n_dim_indices(data, indices)
        npt.assert_array_equal(test_selection, true_selection)

    def test_calc_signed_angle_between_directors(self):
        """
        Test the calc_signed_angle_between_directors
        for positiv and negativ angles in 2D
        """

        my_director1 = np.array([1, 0, 0])
        my_director2 = np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), 0])

        other_director1 = np.array([1, 0, 0])
        other_director2 = np.array([0, 1, 0])
        other_director3 = np.array([1 / 2, np.sqrt(3) / 2, 0])

        # test parallel and antiparallel case
        angle1 = calc_signed_angle_between_directors(my_director1, other_director1)
        angle9 = calc_signed_angle_between_directors(my_director1, -1 * other_director1)

        # test the sign of the angle
        angle2 = calc_signed_angle_between_directors(my_director1, other_director2)
        angle3 = calc_signed_angle_between_directors(my_director1, other_director3)
        angle4 = calc_signed_angle_between_directors(my_director2, other_director1)
        angle5 = calc_signed_angle_between_directors(my_director2, other_director2)
        angle6 = calc_signed_angle_between_directors(my_director2, other_director3)

        # test the normalization
        angle7 = calc_signed_angle_between_directors(
            my_director1, other_director3 * 4.5
        )
        angle8 = calc_signed_angle_between_directors(
            my_director2 * 1.5, other_director3
        )

        assert angle1 == 0
        assert angle2 == np.pi / 2
        assert angle3 == np.pi / 3
        assert angle4 == np.pi * 3 / 4
        assert angle5 == -np.pi * 3 / 4
        assert abs(angle6 + np.pi * 11 / 12) < 10e-6
        assert abs(angle7 - angle3) < 10e-6
        assert abs(angle8 - angle6) < 10e-6
        assert angle9 == np.pi

    def test_create_colloids(self):
        """
        Test the create_colloids function
        """
        center = np.array([100, 100, 0])
        dist = 300
        colloids_0 = create_colloids(
            n_cols=10,
            type_=0,
            center=center,
            dist=dist,
        )

        assert len(colloids_0) == 10
        for col in colloids_0:
            assert col.type == 0
            npt.assert_almost_equal(np.linalg.norm(col.director), 1)
            npt.assert_almost_equal(np.linalg.norm(col.pos - center), dist, decimal=3)

        colloids_1 = create_colloids(
            n_cols=10,
            type_=1,
            center=center,
            dist=2 * dist,
            face_middle=True,
        )

        assert len(colloids_1) == 10
        for col in colloids_1:
            assert col.type == 1
            npt.assert_almost_equal(np.linalg.norm(col.director), 1)
            facing_dir = center - col.pos
            facing_dir /= np.linalg.norm(facing_dir)
            npt.assert_almost_equal(facing_dir, col.director, decimal=3)

    def test_ellipsoid_friction_factors(self):
        """
        Warning: can only test qualitative behaviour, not the full formulae
        """

        # prolate ellipsoid
        axial_semiaxis = 1.6
        equatorial_semiaxis = 0.5
        gamma_rot_ax, gamma_rot_eq = utils.calc_ellipsoid_friction_factors_rotation(
            axial_semiaxis, equatorial_semiaxis, 5
        )
        assert gamma_rot_eq > gamma_rot_ax
        gamma_trans_ax, gamma_trans_eq = (
            utils.calc_ellipsoid_friction_factors_translation(
                axial_semiaxis, equatorial_semiaxis, 5
            )
        )
        assert gamma_trans_eq > gamma_trans_ax

        # oblate ellipsoid
        axial_semiaxis = 0.5
        equatorial_semiaxis = 1.6
        gamma_rot_ax, gamma_rot_eq = utils.calc_ellipsoid_friction_factors_rotation(
            axial_semiaxis, equatorial_semiaxis, 5
        )
        assert gamma_rot_ax > gamma_rot_eq
        gamma_trans_ax, gamma_trans_eq = (
            utils.calc_ellipsoid_friction_factors_translation(
                axial_semiaxis, equatorial_semiaxis, 5
            )
        )
        assert gamma_trans_ax > gamma_trans_eq
