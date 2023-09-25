"""
Data class for the colloid agent.
"""
import dataclasses

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

    def __repr__(self):
        """
        Return a string representation of the colloid.
        """
        return (
            f"Colloid(pos={self.pos}, director={self.director}, id={self.id},"
            f" velocity={self.velocity}, type={self.type})"
        )

    def __eq__(self, other):
        return self.id == other.id

    def tree_flatten(self):
        """
        Flatten the PyTree.
        """
        children = (self.pos, self.director, self.id, self.velocity, self.type)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the PyTree.
        """
        return cls(*children)
