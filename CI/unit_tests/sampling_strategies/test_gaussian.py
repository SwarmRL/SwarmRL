"""Tests for continuous Gaussian sampling strategy."""

import jax
import jax.numpy as jnp
import pytest

from swarmrl.sampling_strategies.gaussian_distribution import (
    ContinuousGaussianDistribution,
)


class TestContinuousGaussianDistribution:
    """Unit tests for ContinuousGaussianDistribution."""

    def test_logits_shape_validation(self):
        sampler = ContinuousGaussianDistribution(action_dimension=3)
        bad_logits = jnp.zeros((2, 5), dtype=jnp.float32)

        with pytest.raises(ValueError):
            sampler(bad_logits, subkey=jax.random.PRNGKey(0))

    def test_returns_actions_and_log_probs_in_training_mode(self):
        sampler = ContinuousGaussianDistribution(action_dimension=3)
        logits = jnp.zeros((4, 6), dtype=jnp.float32)

        actions, log_probs = sampler(
            logits,
            subkey=jax.random.PRNGKey(1),
            calculate_log_probs=True,
            deployment_mode=False,
        )

        assert actions.shape == (4, 3)
        assert log_probs is not None
        assert log_probs.shape == (4,)

    def test_returns_none_log_probs_in_deployment_mode(self):
        sampler = ContinuousGaussianDistribution(action_dimension=3)
        logits = jnp.zeros((3, 6), dtype=jnp.float32)

        actions, log_probs = sampler(
            logits,
            subkey=jax.random.PRNGKey(2),
            calculate_log_probs=True,
            deployment_mode=True,
        )

        assert actions.shape == (3, 3)
        assert log_probs is None

    def test_action_limits_are_respected(self):
        limits = jnp.array([[-0.3, 0.3], [-0.2, 0.2], [-1.0, 1.0]], dtype=jnp.float32)
        sampler = ContinuousGaussianDistribution(action_dimension=3, action_limits=limits)
        logits = jnp.zeros((16, 6), dtype=jnp.float32)

        actions, _ = sampler(
            logits,
            subkey=jax.random.PRNGKey(3),
            calculate_log_probs=False,
            deployment_mode=False,
        )

        assert jnp.all(actions[:, 0] >= -0.3) and jnp.all(actions[:, 0] <= 0.3)
        assert jnp.all(actions[:, 1] >= -0.2) and jnp.all(actions[:, 1] <= 0.2)
        assert jnp.all(actions[:, 2] >= -1.0) and jnp.all(actions[:, 2] <= 1.0)
