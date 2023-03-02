"""
Parent class for the observable.
"""
from typing import List

from swarmrl.models.interaction_model import Colloid


class Observable:
    """
    Parent class for observables.

    Observables act as inputs to the neural networks.
    """

    def __init__(self, particle_type: int):
        """
        Constructor for the observable.
        """
        self._shape = None
        self.particle_type: int = particle_type

    def initialize(self, colloids: List[Colloid]):
        """
        Initialize the observable with starting positions of the colloids.

        The parent method will just pass. This is because some observables
        might not need to be initialized. Those that do need to be initialized
        will override this method.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids with which to initialize the observable.

        Returns
        -------
        Updates the class state.
        """
        pass

    def get_colloid_indices(self, colloids: List[Colloid], p_type: int = None):
        """
        Get the indices of the colloids in the observable of a specific type.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids from which to get the indices.
        p_type : int (default=None)
                Type of the colloids to get the indices for. If None, the
                particle_type attribute of the class is used.


        Returns
        -------
        indices : List[int]
                List of indices for the colloids of a particular type.
        """
        if p_type is None:
            p_type = self.particle_type

        indices = []
        for i, colloid in enumerate(colloids):
            if colloid.type == p_type:
                indices.append(i)

        return indices

    def compute_observable(self, colloids: List[Colloid]) -> List:
        """
        Compute the current state observable for all colloids.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of all colloids in the system.

        Returns
        -------
        observables : List[np.ndarray] (n_colloids, dimension)
                List of observables, one for each colloid.

        """
        raise NotImplementedError("Implemented in child class.")

    @property
    def observable_shape(self):
        """
        Unchangeable shape of the observable.
        Returns
        -------

        """
        return self._shape
