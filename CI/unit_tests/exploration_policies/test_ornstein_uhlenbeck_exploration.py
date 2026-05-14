"""Tests for Ornstein-Uhlenbeck exploration policy."""

import jax
import jax.numpy as jnp
import pytest

from swarmrl.exploration_policies.ornstein_uhlenbeck_exploration import (
    GlobalOUExploration,
)


class TestGlobalOUExploration:
    """Unit tests for GlobalOUExploration."""

    @staticmethod
    def _build_policy(action_dimension: int = 4, epsilon: float = 1.0):
        limits = jnp.array([[-1.0, 1.0]] * action_dimension, dtype=jnp.float32)
        return GlobalOUExploration(
            action_limits=limits,
            drift=0.2,
            volatility=0.1,
            long_term_mean=0.0,
            action_dimension=action_dimension,
            epsilon=epsilon,
        )

    def test_output_shape_vector_and_batch(self):
        policy = self._build_policy(action_dimension=4, epsilon=1.0)
        key = jax.random.PRNGKey(0)

        actions_vec = jnp.zeros((4,), dtype=jnp.float32)
        out_vec = policy(actions_vec, key)
        assert out_vec.shape == actions_vec.shape

        actions_batch = jnp.zeros((3, 4), dtype=jnp.float32)
        out_batch = policy(actions_batch, key)
        assert out_batch.shape == actions_batch.shape

    def test_output_is_clipped_to_action_limits(self):
        limits = jnp.array([[-0.5, 0.5], [-0.2, 0.2], [-1.0, 1.0]], dtype=jnp.float32)
        policy = GlobalOUExploration(
            action_limits=limits,
            drift=0.5,
            volatility=2.0,
            long_term_mean=0.0,
            action_dimension=3,
            epsilon=1.0,
        )
        actions = jnp.zeros((5, 3), dtype=jnp.float32)
        out = policy(actions, jax.random.PRNGKey(1))

        assert jnp.all(out[:, 0] >= -0.5) and jnp.all(out[:, 0] <= 0.5)
        assert jnp.all(out[:, 1] >= -0.2) and jnp.all(out[:, 1] <= 0.2)
        assert jnp.all(out[:, 2] >= -1.0) and jnp.all(out[:, 2] <= 1.0)

    def test_reduce_randomness_decays_noise_and_epsilon_with_floor(self):
        policy = self._build_policy(action_dimension=4, epsilon=0.02)
        policy.noise = jnp.ones((4,), dtype=jnp.float32)

        policy.reduce_randomness(decay=0.5)
        assert jnp.allclose(
            policy.noise, jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float32)
        )
        assert float(policy.epsilon) == pytest.approx(0.01)

    def test_ou_state_updates_every_call(self):
        policy = self._build_policy(action_dimension=4, epsilon=1.0)
        before = policy.noise
        _ = policy(jnp.zeros((4,), dtype=jnp.float32), jax.random.PRNGKey(2))
        after = policy.noise

        assert not jnp.allclose(before, after)
