"""Tests for continuous Gaussian sampling strategy."""

import jax
import jax.numpy as jnp
import pytest

from swarmrl.sampling_strategies.gaussian_distribution import (
    ContinuousGaussianDistribution,
    action_limits_from_bounds,
)


class TestContinuousGaussianDistribution:
    """Unit tests for ContinuousGaussianDistribution."""

    def test_logits_shape_validation(self):
        sampler = ContinuousGaussianDistribution.create(action_dimension=3)
        bad_logits = jnp.zeros((2, 5), dtype=jnp.float32)

        with pytest.raises(ValueError):
            sampler(bad_logits, rng_key=jax.random.PRNGKey(0))

    def test_returns_actions_and_log_probs_in_training_mode(self):
        sampler = ContinuousGaussianDistribution.create(action_dimension=3)
        logits = jnp.zeros((4, 6), dtype=jnp.float32)

        actions, log_probs = sampler(
            logits,
            rng_key=jax.random.PRNGKey(1),
            calculate_log_probs=True,
            deployment_mode=False,
        )

        assert actions.shape == (4, 3)
        assert log_probs is not None
        assert log_probs.shape == (4,)

    def test_returns_none_log_probs_in_deployment_mode(self):
        sampler = ContinuousGaussianDistribution.create(action_dimension=3)
        logits = jnp.zeros((3, 6), dtype=jnp.float32)

        actions, log_probs = sampler(
            logits,
            rng_key=jax.random.PRNGKey(2),
            calculate_log_probs=True,
            deployment_mode=True,
        )

        assert actions.shape == (3, 3)
        assert log_probs is None

    def test_action_limits_are_respected(self):
        limits = jnp.array([[-0.3, 0.3], [-0.2, 0.2], [-1.0, 1.0]], dtype=jnp.float32)
        sampler = ContinuousGaussianDistribution.create(
            action_dimension=3, action_limits=limits
        )
        logits = jnp.zeros((16, 6), dtype=jnp.float32)

        actions, _ = sampler(
            logits,
            rng_key=jax.random.PRNGKey(3),
            calculate_log_probs=False,
            deployment_mode=False,
        )

        assert jnp.all(actions[:, 0] >= -0.3) and jnp.all(actions[:, 0] <= 0.3)
        assert jnp.all(actions[:, 1] >= -0.2) and jnp.all(actions[:, 1] <= 0.2)
        assert jnp.all(actions[:, 2] >= -1.0) and jnp.all(actions[:, 2] <= 1.0)

    def test_create_defaults_to_minus_one_to_one_action_limits(self):
        sampler = ContinuousGaussianDistribution.create(action_dimension=3)

        expected_limits = jnp.array(
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=jnp.float32
        )

        assert sampler.action_limits is not None
        assert sampler.action_limits.shape == (3, 2)
        assert jnp.allclose(sampler.action_limits, expected_limits)

    def test_action_limits_from_bounds_builds_per_dimension_limits(self):
        limits = action_limits_from_bounds(4, 0.0, 1.0)

        expected_limits = jnp.array(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=jnp.float32,
        )

        assert limits.shape == (4, 2)
        assert jnp.allclose(limits, expected_limits)

    def test_log_probs_include_tanh_squash_correction(self):
        sampler = ContinuousGaussianDistribution(action_dimension=2)
        logits = jnp.zeros((3, 4), dtype=jnp.float32)
        key = jax.random.PRNGKey(5)

        actions, log_probs = sampler(
            logits,
            rng_key=key,
            calculate_log_probs=True,
            deployment_mode=False,
        )

        raw_actions = jax.random.normal(key, shape=(3, 2), dtype=jnp.float32)
        gaussian_log_probs = (-0.5 * (raw_actions**2 + jnp.log(2.0 * jnp.pi))).sum(
            axis=-1
        )
        squash_correction = jnp.log(1.0 - jnp.tanh(raw_actions) ** 2 + 1e-6).sum(
            axis=-1
        )

        assert jnp.allclose(actions, raw_actions)
        assert jnp.allclose(log_probs, gaussian_log_probs - squash_correction)

    def test_log_probs_include_affine_action_scale_correction(self):
        action_dimension = 4
        logits = jnp.zeros((3, 2 * action_dimension), dtype=jnp.float32)
        key = jax.random.PRNGKey(6)

        unit_interval_sampler = ContinuousGaussianDistribution.create(
            action_dimension=action_dimension,
            action_limits=action_limits_from_bounds(action_dimension, 0.0, 1.0),
        )
        symmetric_sampler = ContinuousGaussianDistribution.create(
            action_dimension=action_dimension,
            action_limits=action_limits_from_bounds(action_dimension, -1.0, 1.0),
        )

        _, unit_interval_log_probs = unit_interval_sampler(
            logits,
            rng_key=key,
            calculate_log_probs=True,
            deployment_mode=False,
        )
        _, symmetric_log_probs = symmetric_sampler(
            logits,
            rng_key=key,
            calculate_log_probs=True,
            deployment_mode=False,
        )

        expected_offset = action_dimension * jnp.log(2.0)
        assert jnp.allclose(
            unit_interval_log_probs,
            symmetric_log_probs + expected_offset,
        )

    def test_zero_width_action_bounds_are_rejected(self):
        with pytest.raises(ValueError, match="strictly greater"):
            ContinuousGaussianDistribution.create(
                action_dimension=1,
                action_limits=jnp.array([[2.0, 2.0]], dtype=jnp.float32),
            )
