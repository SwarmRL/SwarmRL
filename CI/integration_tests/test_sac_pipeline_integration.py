import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from swarmrl.losses.sac_loss import SoftActorCriticLoss, get_sac_grads
from swarmrl.networks import FlaxModel
from swarmrl.replay_buffer.replay_buffer import ReplayBuffer
from swarmrl.replay_buffer.transition import Transition
from swarmrl.sampling_strategies import ContinuousGaussianDistribution


class TinySacModule(nn.Module):
    action_dim: int = 2

    def setup(self):
        self.actor_hidden = nn.Dense(8)
        self.actor_out = nn.Dense(self.action_dim * 2)
        self.critic_hidden = nn.Dense(8)
        self.critic_q1 = nn.Dense(1)
        self.critic_q2 = nn.Dense(1)

    def __call__(self, feature_data):
        if feature_data.ndim == 1:
            feature_data = feature_data[None, :]
        logits = self.actor(feature_data, rng_key=None)
        dummy_actions = jnp.zeros(
            (feature_data.shape[0], self.action_dim), dtype=feature_data.dtype
        )
        q1, _ = self.critic(feature_data, dummy_actions)
        _ = self.alpha()
        return logits, q1

    def actor(self, feature_data, rng_key=None):
        del rng_key
        hidden = nn.tanh(self.actor_hidden(feature_data))
        return self.actor_out(hidden)

    def critic(self, feature_data, actions):
        x = jnp.concatenate([feature_data, actions], axis=-1)
        hidden = nn.tanh(self.critic_hidden(x))
        q1 = self.critic_q1(hidden)
        q2 = self.critic_q2(hidden)
        return q1, q2

    @nn.compact
    def alpha(self):
        log_alpha = self.param("log_alpha", nn.initializers.zeros, ())
        return jnp.exp(log_alpha)


def build_sac_network(seed=0):
    model = FlaxModel(
        flax_model=TinySacModule(),
        optimizer=optax.adam(1e-3),
        input_shape=(3,),
    )
    model.target_params["critic"] = jax.tree_util.tree_map(
        lambda x: x,
        model.model_state.params,
    )
    return model


def test_full_sac_pipeline_dataflow():
    """
    CI Smoke-Test: tests the full dataflow from environment-transition
    through the ReplayBuffer into the JIT compiled SAC loss function.
    """
    batch_size = 4
    obs_dim = 3
    act_dim = 2

    buffer = ReplayBuffer(capacity=100, seed=42)

    for i in range(10):
        trans = Transition(
            observation=np.random.randn(obs_dim).astype(np.float32),
            action=np.random.randn(act_dim).astype(np.float32),
            reward=float(np.random.rand()),  # test float to array cast
            next_observation=np.random.randn(obs_dim).astype(np.float32),
            terminated=float(i % 5 == 0),  # every 5th step is a done
        )
        buffer.add(trans)

    assert len(buffer) == 10, "Buffer hasn't stored transition."

    batch = buffer.sample(batch_size=batch_size)

    assert batch["observation"].shape == (batch_size, obs_dim)
    assert batch["reward"].shape == (batch_size, 1), "Reward Broadcasting failed!"
    assert batch["terminated"].shape == (
        batch_size,
        1,
    ), "Terminated Broadcasting failed!"

    master_key = jax.random.PRNGKey(99)
    _, actor_rng, next_actor_rng = jax.random.split(master_key, num=3)
    batch["actor_rng"] = actor_rng
    batch["next_actor_rng"] = next_actor_rng

    network = build_sac_network(seed=0)
    sampling_strategy = ContinuousGaussianDistribution.create(action_dimension=act_dim)
    loss_fn = SoftActorCriticLoss(
        sampling_strategy=sampling_strategy,
        target_entropy=-float(act_dim),
    )

    try:
        (total_loss, metrics), grads = get_sac_grads(
            network.model_state.params,
            network.model,
            network.target_params["critic"],
            loss_fn.value_function.__call__,
            loss_fn.sampling_strategy,
            loss_fn.target_entropy,
            batch,
        )
    except Exception as e:
        pytest.fail(f"get_sac_grads failed during compilation/execution: {e}")

    assert jnp.isfinite(total_loss), "Loss is NaN or Inf!"
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
        network.model_state.params
    )
