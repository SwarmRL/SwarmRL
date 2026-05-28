"""Unit tests for ActionSelector orchestration."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from swarmrl.action_selection.action_selector import ActionSelector
from swarmrl.exploration_policies.exploration_policy import (
    ContinuousExplorationPolicy,
    DiscreteExplorationPolicy,
)
from swarmrl.sampling_strategies.sampling_strategy import (
    ContinuousSamplingStrategy,
    DiscreteSamplingStrategy,
)


class _DummyDiscreteSampling(DiscreteSamplingStrategy):
    """Deterministic discrete sampler for testing."""

    def __call__(self, logits, rng_key=None):
        del rng_key
        return np.argmax(logits, axis=-1)


class _DummyDiscreteExploration(DiscreteExplorationPolicy):
    """Discrete exploration that shifts indices by +1 modulo action space."""

    def __call__(self, model_action, action_space_length, seed=12345):
        del seed
        return (model_action + 1) % action_space_length


class _DummyContinuousSampling(ContinuousSamplingStrategy):
    """Continuous sampler with deterministic outputs and optional log-probs."""

    def __call__(
        self,
        logits,
        subkey=None,
        calculate_log_probs=True,
        deployment_mode=False,
    ):
        del subkey, deployment_mode
        actions = logits[:, :2]
        if calculate_log_probs:
            return actions, np.zeros((logits.shape[0],), dtype=logits.dtype)
        return actions, None


class _DummyContinuousExploration(ContinuousExplorationPolicy):
    """Continuous exploration that adds +1.0 to actions."""

    def __call__(self, model_actions, rng_key):
        del rng_key
        return model_actions + 1.0


class TestActionSelector:
    """Tests for action selection across discrete and continuous modes."""

    def test_mode_mismatch_raises(self):
        with pytest.raises(ValueError):
            ActionSelector(
                sampling_strategy=_DummyDiscreteSampling(),
                exploration_policy=_DummyContinuousExploration(),
            )

    def test_discrete_training_applies_exploration_and_gathers_log_probs(self):
        selector = ActionSelector(
            sampling_strategy=_DummyDiscreteSampling(),
            exploration_policy=_DummyDiscreteExploration(),
        )

        logits = np.array([[1.0, 3.0, 2.0], [4.0, 1.0, 0.0]], dtype=np.float32)
        sampling_key = jax.random.PRNGKey(0)
        exploration_key = jax.random.PRNGKey(1)

        indices, chosen_log_probs = selector.select(
            logits=logits,
            deployment_mode=False,
            sampling_key=sampling_key,
            exploration_key=exploration_key,
        )

        expected_indices = np.array([2, 1])
        expected_log_probs = np.take_along_axis(
            jax.nn.log_softmax(logits), expected_indices.reshape(-1, 1), axis=-1
        ).reshape(-1)

        onp.testing.assert_array_equal(indices, expected_indices)
        onp.testing.assert_allclose(chosen_log_probs, expected_log_probs)

    def test_discrete_deployment_skips_exploration(self):
        selector = ActionSelector(
            sampling_strategy=_DummyDiscreteSampling(),
            exploration_policy=_DummyDiscreteExploration(),
        )

        logits = np.array([[1.0, 3.0, 2.0], [4.0, 1.0, 0.0]], dtype=np.float32)
        sampling_key = jax.random.PRNGKey(2)
        exploration_key = jax.random.PRNGKey(3)

        indices, chosen_log_probs = selector.select(
            logits=logits,
            deployment_mode=True,
            sampling_key=sampling_key,
            exploration_key=exploration_key,
        )

        expected_indices = np.array([1, 0])
        expected_log_probs = np.take_along_axis(
            jax.nn.log_softmax(logits), expected_indices.reshape(-1, 1), axis=-1
        ).reshape(-1)

        onp.testing.assert_array_equal(indices, expected_indices)
        onp.testing.assert_allclose(chosen_log_probs, expected_log_probs)

    def test_continuous_training_applies_exploration_and_returns_log_probs(self):
        selector = ActionSelector(
            sampling_strategy=_DummyContinuousSampling(),
            exploration_policy=_DummyContinuousExploration(),
        )

        logits = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.6, -0.2]], dtype=np.float32
        )
        sampling_key = jax.random.PRNGKey(4)
        exploration_key = jax.random.PRNGKey(5)

        actions, log_probs = selector.select(
            logits=logits,
            deployment_mode=False,
            sampling_key=sampling_key,
            exploration_key=exploration_key,
        )

        expected_actions = logits[:, :2] + 1.0
        expected_log_probs = np.zeros((2,), dtype=np.float32)

        onp.testing.assert_allclose(actions, expected_actions)
        onp.testing.assert_allclose(log_probs, expected_log_probs)

    def test_continuous_deployment_skips_exploration_and_log_probs(self):
        selector = ActionSelector(
            sampling_strategy=_DummyContinuousSampling(),
            exploration_policy=_DummyContinuousExploration(),
        )

        logits = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.6, -0.2]], dtype=np.float32
        )
        sampling_key = jax.random.PRNGKey(6)
        exploration_key = jax.random.PRNGKey(7)

        actions, log_probs = selector.select(
            logits=logits,
            deployment_mode=True,
            sampling_key=sampling_key,
            exploration_key=exploration_key,
        )

        expected_actions = logits[:, :2]

        onp.testing.assert_allclose(actions, expected_actions)
        assert log_probs is None
