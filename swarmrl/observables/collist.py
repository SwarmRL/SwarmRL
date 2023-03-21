from typing import List

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable


class Collist(Observable):
    """ """

    def __init__(self):
        pass

    def compute_observable(self, colloids: List[Colloid]) -> List:
        """
        Compute the observable for all colloids.
        """

        return colloids
