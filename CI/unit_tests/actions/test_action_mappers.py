import numpy as np
import pytest

from swarmrl.actions import (
    direction_force_mapper,
    force_only_mapper,
    force_torque_mapper,
    torque_only_mapper,
)


def test_force_only_mapper_sets_force():
    action = force_only_mapper(np.array([1.5]))
    assert action.force == pytest.approx(1.5)
    assert action.torque is None
    assert action.new_direction is None


def test_torque_only_mapper_sets_z_torque():
    action = torque_only_mapper(np.array([-2.0]))
    np.testing.assert_allclose(action.torque, np.array([0.0, 0.0, -2.0]))
    assert action.force == 0.0


def test_force_torque_mapper_sets_both_controls():
    action = force_torque_mapper(np.array([3.0, 4.0]))
    assert action.force == pytest.approx(3.0)
    np.testing.assert_allclose(action.torque, np.array([0.0, 0.0, 4.0]))


def test_direction_force_mapper_normalizes_direction():
    action = direction_force_mapper(np.array([3.0, 4.0, 2.5]))
    assert action.force == pytest.approx(2.5)
    np.testing.assert_allclose(action.new_direction, np.array([0.6, 0.8]))


def test_direction_force_mapper_handles_zero_direction():
    action = direction_force_mapper(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(action.new_direction, np.array([0.0, 0.0]))


def test_action_mappers_validate_input_size():
    with pytest.raises(ValueError):
        force_torque_mapper(np.array([1.0]))
