"""
Class for the Swarm Pytree Agent
"""

from __future__ import annotations

import dataclasses
from typing import List

import jax.numpy as jnp
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
    pos: jnp.ndarray
    director: jnp.ndarray
    id: int
    velocity: jnp.ndarray = None
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
            pos=jnp.take(self.pos, indices, axis=0),
            director=jnp.take(self.director, indices, axis=0),
            id=jnp.take(self.id, indices, axis=0),
            velocity=jnp.take(self.velocity, indices, axis=0),
            type=jnp.take(self.type, indices, axis=0),
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
    pos = jnp.array([c.pos for c in colloids]).reshape(-1, colloids[0].pos.shape[0])
    director = jnp.array([c.director for c in colloids]).reshape(
        -1, colloids[0].director.shape[0]
    )
    id = jnp.array([c.id for c in colloids]).reshape(-1, 1)
    velocity = jnp.array([c.velocity for c in colloids]).reshape(
        -1, colloids[0].velocity.shape[0]
    )
    type = jnp.array([c.type for c in colloids]).reshape(-1, 1)

    # add species indices to the colloid types.
    type_indices = {}
    types = onp.unique(type)
    for t in types:
        type_indices[t] = jnp.array(get_colloid_indices(colloids, t))

    return Swarm(pos, director, id, velocity, type, type_indices)
