"""
Unit test for the position-angle observable.
"""
import os

import numpy as np
from numpy.testing import assert_array_equal

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.collist import Collist


class TestAngleObservable:
    """
    Test suite for the position angle observable.
    """

    def test_compute_observable(self):
        """
        Test the computation of the observable for a single colloid.
        """
        observable = Collist()

        cols = []
        for i in range(2):
            position = np.random.random(3)
            direction = np.random.random(3)
            cols.append(Colloid(pos=position, director=direction, type=0, id=i))

        expected = cols
        actual = observable.compute_observable(cols)
        assert_array_equal(expected, actual)

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
