# import tempfile
import unittest as ut

import espressomd
import numpy as np

# import pint

# import swarmrl.utils
# from swarmrl.engine import espresso
# from swarmrl.models import dummy_models


def assertNotArrayAlmostEqual(arr0, arr1, atol=1e-6):
    with np.testing.assert_raises(Exception):
        np.testing.assert_array_almost_equal(arr0, arr1, atol=atol)


class EspressoTestRLTrainers(ut.TestCase):
    """
    Tests all of the SwarmRL trainers.
    """

    system = espressomd.System(box_l=[1, 2, 3])

    def test_continuous_training(self):
        """
        Test continuous training.
        """
        pass

    def test_fixed_episodic_training(self):
        """
        Test the episodic training for set episode length.
        """
        pass

    def test_variable_episodic_training(self):
        """
        Test episodic training with engine killing tasks.
        """
        pass

    def test_semi_episodic_training(self):
        """
        Test semi-episodic training.
        """
        pass


if __name__ == "__main__":
    ut.main()
