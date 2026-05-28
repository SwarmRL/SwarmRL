from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from swarmrl.losses.sac_loss import SoftActorCriticLoss, get_sac_grads
from swarmrl.networks.multi_flax_networks import MultiFlaxModel
from swarmrl.replay_buffer.replay_buffer import ReplayBuffer
from swarmrl.replay_buffer.transition import Transition


class TinyActor(nn.Module):
    action_dim: int = 2

    @nn.compact
    def __call__(self, feature_data, rng_key):
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros)(
            feature_data
        )
        raw_action = mean + 0.1 * jax.random.normal(rng_key, mean.shape)
        action = jnp.tanh(raw_action)
        log_prob = -0.5 * jnp.sum(raw_action**2 + jnp.log(2.0 * jnp.pi), axis=-1)
        return action, log_prob


class TinyCritic(nn.Module):
    @nn.compact
    def __call__(self, feature_data, actions):
        x = jnp.concatenate([feature_data, actions], axis=-1)
        hidden = nn.tanh(nn.Dense(8)(x))
        q1 = nn.Dense(1)(hidden)
        q2 = nn.Dense(1)(hidden)
        return q1, q2


class DummyAlphaModule:
    def apply(self, *args, **kwargs):
        return None


@dataclass(frozen=True)
class ScalarState:
    step: int
    params: jax.Array

    def apply_gradients(self, *, grads):
        return ScalarState(step=self.step + 1, params=self.params - 0.01 * grads)


def build_sac_network(batch_size=4, seed=0):
    model = MultiFlaxModel(seed=seed)
    actor = TinyActor()
    critic = TinyCritic()
    obs = jnp.ones((batch_size, 3), dtype=jnp.float32)
    action = jnp.ones((batch_size, 2), dtype=jnp.float32)

    actor_params = actor.init(jax.random.PRNGKey(1), obs, jax.random.PRNGKey(2))[
        "params"
    ]
    critic_params = critic.init(jax.random.PRNGKey(3), obs, action)["params"]

    model.add_network("actor", actor, actor_params, optax.adam(1e-3))
    model.add_network(
        "critic", critic, critic_params, optax.adam(1e-3), has_target=True
    )
    model.networks["log_alpha"] = DummyAlphaModule()
    model.states["log_alpha"] = ScalarState(
        step=0, params=jnp.array(0.0, dtype=jnp.float32)
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

    # init the buffer
    buffer = ReplayBuffer(capacity=100, seed=42)

    # fill the buffer
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

    # 3. sample a batch
    batch = buffer.sample(batch_size=batch_size)

    # check buffer preparation
    assert batch["observation"].shape == (batch_size, obs_dim)
    assert batch["reward"].shape == (batch_size, 1), "Reward Broadcasting failed!"
    assert batch["terminated"].shape == (
        batch_size,
        1,
    ), "Terminated Broadcasting failed!"

    # inject rng_keys as happening in the agent
    master_key = jax.random.PRNGKey(99)
    _, actor_rng, next_actor_rng = jax.random.split(master_key, num=3)
    batch["actor_rng"] = actor_rng
    batch["next_actor_rng"] = next_actor_rng

    # init network and loss fn
    network = build_sac_network(batch_size=batch_size, seed=0)
    loss_fn = SoftActorCriticLoss(target_entropy=-float(act_dim))

    trainable_params = {
        "actor": network.states["actor"].params,
        "critic": network.states["critic"].params,
        "log_alpha": network.states["log_alpha"].params,
    }

    # run jit compiled loss
    try:
        (total_loss, metrics), grads = get_sac_grads(
            trainable_params,
            network.networks["actor"],
            network.networks["critic"],
            network.target_params["critic"],
            loss_fn.value_function.__call__,
            loss_fn.target_entropy,
            batch,
        )
    except Exception as e:
        pytest.fail(f"get_sac_grads failed during compilation/execution: {e}")

    # validate results
    assert jnp.isfinite(total_loss), "Loss is NaN or Inf!"
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics

    # Check if gradients exists for all trainable networks
    assert "actor" in grads
    assert "critic" in grads
    assert "log_alpha" in grads
