"""
Parent class for the observable.
"""


class Observable:
    """
    Parent class for observables.

    Observables act as inputs to the neural networks.
    """
    def compute_observable(self, colloid: object, other_colloids: list):
        """
        Compute the current state observable.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.
        other_colloids
                Other colloids in the system.

        Returns
        -------

        """
        raise NotImplementedError("Implemented in child class.")
