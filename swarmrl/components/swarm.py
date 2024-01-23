"""
Class for the Swarm Pytree Agent
"""

from __future__ import annotations

import dataclasses
from typing import List

import jax.numpy as np
import numpy as onp
from jax.tree_util import register_pytree_node_class

from swarmrl.components.colloid import Colloid
from swarmrl.utils.colloid_utils import get_colloid_indices


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Swarm:
    """
    Wrapper class for a colloid object.
    """

    # Colloid attributes
    pos: np.ndarray
    director: np.ndarray
    id: int
    velocity: np.ndarray = None
    type: int = 0

    # Swarm attributes
    type_indices: dict = None

    def __repr__(self) -> str:
        """
        Return a string representation of the colloid.
        """
        return (
            f"Colloid(pos={self.pos}, director={self.director}, id={self.id},"
            f" velocity={self.velocity}, type={self.type})"
        )

    def __eq__(self, other):
        return self.id == other.id

    def tree_flatten(self) -> tuple:
        """
        Flatten the PyTree.
        """
        children = (
            self.pos,
            self.director,
            self.id,
            self.velocity,
            self.type,
            self.type_indices,
        )
        aux_data = None
        return (children, aux_data)

    def get_species_swarm(self, species: int) -> Swarm:
        """
        Get a swarm of one species.

        Parameters
        ----------
        species : int
            Species index.

        Returns
        -------
        partitioned_swarm : Swarm
            Swarm of one species.
        """
        indices = self.type_indices[species]
        return Swarm(
            pos=np.take(self.pos, indices, axis=0),
            director=np.take(self.director, indices, axis=0),
            id=np.take(self.id, indices, axis=0),
            velocity=np.take(self.velocity, indices, axis=0),
            type=np.take(self.type, indices, axis=0),
            type_indices=None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Swarm:
        """
        Unflatten the PyTree.

        This method is required by Pytrees in Jax.

        Parameters
        ----------
        aux_data : None
            Auxiliary data. Not used in this class.
        children : tuple
            Tuple of children to be unflattened.
        """
        return cls(*children)


def create_swarm(colloids: List[Colloid]) -> Swarm:
    """
    Create a swarm from a list of colloid objects.

    Parameters
    ----------
    colloid : List[Colloid]
        List of colloid objects.

    Returns
    -------
    Swarm
        Swarm object full of all colloids
    """
    # standard colloid attributes
    pos = np.array([c.pos for c in colloids]).reshape(-1, colloids[0].pos.shape[0])
    director = np.array([c.director for c in colloids]).reshape(
        -1, colloids[0].director.shape[0]
    )
    id = np.array([c.id for c in colloids]).reshape(-1, 1)
    velocity = np.array([c.velocity for c in colloids]).reshape(
        -1, colloids[0].velocity.shape[0]
    )
    type = np.array([c.type for c in colloids]).reshape(-1, 1)

    # add species indices to the colloid types.
    type_indices = {}
    types = onp.unique(type)
    for t in types:
        type_indices[t] = np.array(get_colloid_indices(colloids, t))

    return Swarm(pos, director, id, velocity, type, type_indices)
