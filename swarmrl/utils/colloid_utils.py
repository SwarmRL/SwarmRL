"""
Various functions for operating on colloids.
"""

import jax
import jax.numpy as jnp


def compute_forces(r):
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
    """

    def _sub_compute(r):
        return 1 / jnp.linalg.norm(r) ** 12

    force_fn = jax.grad(_sub_compute)

    return force_fn(r)


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


def compute_torque(force, direction):
    """
    Compute the torque on a rod.

    Parameters
    ----------

    """
    return jnp.cross(direction, force)


def compute_torque_partition_on_rod(colloid_positions, rod_positions, rod_directions):
    """
    Compute the torque on a rod using a WCA potential.
    """
    # (n_colloids, rod_particles, 3)
    distance_matrix = compute_distance_matrix(colloid_positions, rod_positions)
    distance_matrix = distance_matrix[:, :, :2]

    # Force on the rod
    rod_map_fn = jax.vmap(compute_forces, in_axes=(0,))  # map over rod particles
    colloid_map_fn = jax.vmap(rod_map_fn, in_axes=(0,))  # map over colloids

    # (n_colloids, rod_particles, 3)
    forces = colloid_map_fn(distance_matrix)

    # Compute torques
    colloid_rod_map = jax.vmap(compute_torque, in_axes=(0, 0))
    colloid_only_map = jax.vmap(colloid_rod_map, in_axes=(0, None))

    torques = colloid_only_map(forces, rod_directions)
    net_rod_torque = torques.sum(axis=1)
    torque_magnitude = jnp.linalg.norm(net_rod_torque, axis=-1) + 1
    normalization_factors = torque_magnitude.sum()
    torque_partition = torque_magnitude / normalization_factors

    return torque_partition
