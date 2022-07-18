"""
Test the utils module.
"""
import jax.numpy as np
from numpy.testing import assert_array_equal

from swarmrl.utils.utils import gather_n_dim_indices


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
        assert_array_equal(test_selection, true_selection)

        # 2 particles, 3 time steps, 2 options
        data = np.array([[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]])
        indices = np.array([[1, 1], [0, 0], [0, 0]])
        true_selection = np.array([[1, 3], [10, 12], [20, 22]])
        test_selection = gather_n_dim_indices(data, indices)
        assert_array_equal(test_selection, true_selection)

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
        assert_array_equal(test_selection, true_selection)

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
        assert_array_equal(test_selection, true_selection)
