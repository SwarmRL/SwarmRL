"""
Various functions for operating on colloids.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from swarmrl.components.colloid import Colloid


@dataclass
class TrajectoryInformation:
    """
    Helper dataclass for training RL models.
    """

    particle_type: int
    features: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    killed: bool = False


@jax.jit
def compute_forces(r: jnp.ndarray, director: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the energy between two colloids.

    This uses a WCA potential to compute a relative force between
    two colloids. It is not physical.
    The method itself implements an energy computation which then uses
    Jax to compute the gradient of the energy with respect to the
    distance between the colloids.

    Parameters
    ----------
    r : jnp.ndarray (dimension, )
        Distance between the two colloids.
    director : jnp.ndarray (dimension, )
        Director of the colloid.

    Returns
    -------
    force : jnp.ndarray (dimension, )
        Force between the two colloids applied along the director
        of the colloid.
    """

    def _sub_compute(r):
        return 1 / jnp.linalg.norm(r) ** 2

    force_fn = jax.grad(_sub_compute)

    return jnp.linalg.norm(force_fn(r)) * director


@jax.jit
def compute_distance_matrix(set_a, set_b):
    """
    Compute a distance matrix between two sets.

    Helper function for computing the distance sets of
    colloids. This is not a commutative operation, if you
    swap a for b you will recieve a different matrix shape.

    Parameters
    ----------
    set_a : jnp.ndarray
        First set of points.
    set_b : jnp.ndarray
        Second set of points.
    """

    def _sub_compute(a, b):
        return b - a

    distance_fn = jax.vmap(_sub_compute, in_axes=(0, None))

    return distance_fn(set_a, set_b)


@jax.jit
def compute_torque(force, direction):
    """
    Compute the torque on a rod.

    Parameters
    ----------

    """
    return jnp.cross(direction, force)


@jax.jit
def compute_torque_partition_on_rod(
    colloid_positions, colloid_directors, rod_positions, rod_directions
):
    """
    Compute the torque partition on a rod using a WCA potential.

    Parameters
    ----------
    colloid_positions : jnp.ndarray (n_colloids, 3)
        Positions of the colloids.
    colloid_directors : jnp.ndarray (n_colloids, 3)
        Directors of the colloids.
    rod_positions : jnp.ndarray (rod_particles, 3)
        Positions of the rod particles.
    rod_directions : jnp.ndarray (rod_particles, 3)
        Directors of the rod particles.
    """
    # (n_colloids, rod_particles, 3)
    distance_matrix = compute_distance_matrix(colloid_positions, rod_positions)
    # distance_matrix = distance_matrix[:, :, :2]

    # Force on the rod
    rod_map_fn = jax.vmap(compute_forces, in_axes=(0, None))  # map over rod particles
    colloid_map_fn = jax.vmap(rod_map_fn, in_axes=(0, 0))  # map over colloids

    # (n_colloids, rod_particles, 3)
    forces = colloid_map_fn(distance_matrix, colloid_directors)

    # Compute torques
    colloid_rod_map = jax.vmap(compute_torque, in_axes=(0, 0))
    colloid_only_map = jax.vmap(colloid_rod_map, in_axes=(0, None))

    torques = colloid_only_map(forces, rod_directions)
    net_rod_torque = torques.sum(axis=1)
    torque_magnitude = jnp.linalg.norm(net_rod_torque, axis=-1) + 1e-8

    return torque_magnitude


def get_colloid_indices(colloids: List["Colloid"], p_type: int) -> List[int]:
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
    indices = []
    for i, colloid in enumerate(colloids):
        if colloid.type == p_type:
            indices.append(i)

    return indices
