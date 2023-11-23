"""
Perform an integration test on the harmonic trap class.
"""

import unittest as ut

import numpy as np

from swarmrl.models.harmonic_trap import HarmonicTrap


class TestHarmonicTrap(ut.TestCase):
    """
    Perform an integration test on the harmonic trap class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare the class for the test.

        Returns
        -------
        Sets a class to be operated on.
        """
        cls.model = HarmonicTrap(stiffness=10, center=np.array([0.0, 0.0, 0.0]))
        cls.colloids = [[0.0, 0.0, 0.0], [1.0, 3, 7.9], [-3.6, 3.2, -0.1]]

    def test_harmonic_trap(self):
        """
        Run several loops over the harmonic trap class.

        Returns
        -------
        Asserts that all of the force outputs are correct.
        """
        actual = np.array([[-0.0, -0.0, -0.0], [-10, -30, -79], [36, -32, 1.0]])
        prediction, _ = self.model(np.array(self.colloids))
        np.testing.assert_array_equal(prediction, actual)


if __name__ == "__main__":
    ut.main()
