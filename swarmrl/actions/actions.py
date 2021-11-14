"""
Module to house all of the callable operations on a colloid.

Notes
-----
TODO: Add noise to the operations to simulate real environment.
"""
import numpy as np


class TranslateColloid:
    """
    Callable class to translate a colloid.
    """
    def __init__(self, act_force: float = 1.0):
        """
        Constructor for the translation class

        Parameters
        ----------
        act_force : float
                Magnitude of the force to apply.
        """
        self.act_force = act_force

    def __call__(self, colloid):
        """
        Compute the force in the correct direction.

        Parameters
        ----------
        colloid
                Colloid on which to compute the force.

        Returns
        -------
        Force acting on the colloid.
        """
        return self.act_force * colloid.director


class RotateColloid:
    """
    Callable class to rotate a colloid.
    """

    def __init__(self, angle: float = 1.0, clockwise: bool = True):
        """
        Constructor for the rotate class

        Parameters
        ----------
        angle : float
                angle with which to rotate the colloid.
        clockwise : bool
                If true, rotate the particle clockwise.
        """
        if clockwise:
            self.angle = angle
        else:
            self.angle = 2 * np.pi - angle

    def __call__(self, colloid):
        """
        Compute the torque in the correct direction.

        Parameters
        ----------
        colloid
                Colloid on which to compute the force.

        Returns
        -------
        Force acting on the colloid.
        """
        return colloid.director * np.linalg.norm(colloid.director) * np.cos(self.angle)
