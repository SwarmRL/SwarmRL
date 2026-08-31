"""Blueprint action mappers for policy outputs.

These helpers show how to convert a network action vector into a SwarmRL
``Action``. They are meant as small examples for SAC-style agents that produce
continuous action arrays and need a deterministic mapping into the simulation
control space.
"""

from __future__ import annotations

import numpy as np

from swarmrl.actions.actions import Action


def _as_1d_action(action: np.ndarray, expected_size: int) -> np.ndarray:
    """Validate and flatten a single-agent action vector."""
    action_array = np.asarray(action, dtype=float).reshape(-1)
    if action_array.size != expected_size:
        raise ValueError(
            f"Expected action vector of length {expected_size}, got "
            f"shape {np.asarray(action).shape}."
        )
    return action_array


def force_only_mapper(action: np.ndarray) -> Action:
    """Map a single scalar policy output to an ``Action.force`` value."""
    action_array = _as_1d_action(action, expected_size=1)
    return Action(force=float(action_array[0]))


def torque_only_mapper(action: np.ndarray) -> Action:
    """Map a single scalar policy output to z-axis torque."""
    action_array = _as_1d_action(action, expected_size=1)
    return Action(torque=np.array([0.0, 0.0, action_array[0]], dtype=float))


def force_torque_mapper(action: np.ndarray) -> Action:
    """Map ``[force, torque_z]`` to an ``Action`` with both controls set."""
    action_array = _as_1d_action(action, expected_size=2)
    return Action(
        force=float(action_array[0]),
        torque=np.array([0.0, 0.0, action_array[1]], dtype=float),
    )


def direction_force_mapper(action: np.ndarray) -> Action:
    """Map ``[dir_x, dir_y, force]`` to a normalized direction and magnitude."""
    action_array = _as_1d_action(action, expected_size=3)
    direction = action_array[:2]
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        normalized_direction = np.zeros_like(direction)
    else:
        normalized_direction = direction / norm

    return Action(
        force=float(action_array[2]),
        new_direction=normalized_direction,
    )
