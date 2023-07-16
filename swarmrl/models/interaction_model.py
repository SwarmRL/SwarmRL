"""
Model to compute external forces in an espresso simulation.
"""
import dataclasses
import typing

import numpy as np
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Colloid:
    """
    Wrapper class for a colloid object.
    """

    pos: np.ndarray
    director: np.ndarray
    id: int
    velocity: np.ndarray = None
    type: int = 0

    def __eq__(self, other):
        return self.id == other.id

    def tree_flatten(self):
        """
        Method for converting the class into a tuple of a list of children and an
            auxiliary data object.
        This is required for pytrees.
        """
        children = (self.pos, self.director, self.id, self.velocity, self.type)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        A method for converting the tuple of children and auxiliary data back into
            an instance of the class.
        """
        return cls(*children)


@dataclasses.dataclass
class Action:
    """
    Holds the 3 quantities that are applied to the colloid plus an identifier
    """

    id = 0
    force: float = 0.0
    torque: np.ndarray = np.zeros((3,))
    new_direction: np.ndarray = None


class InteractionModel:
    """
    Parent class to compute external forces on colloids in a simulation.
    Inherits from the module class of Torch.
    """

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Compute the next action on colloid.

        Parameters
        ----------
        colloids : list
                List of all Colloids for which an action is being computed

        Returns
        -------
        action : list
                List of Actions for all Colloids in the same order as input colloids
        """
        raise NotImplementedError("Interaction models must define a calc_action method")
