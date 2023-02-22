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


    def test_gather_n_dim_indices(self):
        """
        Test the indices gathering function for even and odd numbers
        """

        my_director1=np.array([1,0,0])
        my_director2=np.array([-1/np.sqrt(2),-1/np.sqrt(2),0])


        other_director1=np.array([1,0,0])
        other_director2=np.array([0,1,0])
        other_director3=np.array([1/2,np.sqrt(3)/2,0])



        angle1=calc_signed_angle_between_directors(my_director1,other_director1)
        angle2=calc_signed_angle_between_directors(my_director1,other_director2)
        angle3=calc_signed_angle_between_directors(my_director1,other_director3)
        angle4=calc_signed_angle_between_directors(my_director2,other_director1)
        angle5=calc_signed_angle_between_directors(my_director2,other_director2)
        angle6=calc_signed_angle_between_directors(my_director2,other_director3)
        
        assert angle1 == 0
        assert angle2 == np.pi/2
        assert angle3 == np.pi/3
        assert angel4 == np.pi*3/4
        assert angle5 == -np.pi*3/4
        assert angle6 == -np.pi*11/12

 
