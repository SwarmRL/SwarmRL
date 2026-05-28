from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest

from swarmrl.losses.sac_loss import SoftActorCriticLoss, get_sac_grads
from swarmrl.networks.multi_flax_networks import MultiFlaxModel
from swarmrl.sampling_strategies import ContinuousGaussianDistribution
from swarmrl.value_functions.td_return_sac import TDReturnsSAC


class TinyActor(nn.Module):
    action_dim: int = 2

    @nn.compact
    def __call__(self, feature_data, rng_key):
        logits = nn.Dense(self.action_dim * 2, kernel_init=nn.initializers.zeros)(
            feature_data
        )
        return logits


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


def build_sac_network(batch_size=4, seed=0, alpha_params_style="scalar"):
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
    if alpha_params_style == "scalar":
        model.networks["log_alpha"] = DummyAlphaModule()
        model.states["log_alpha"] = ScalarState(
            step=0, params=jnp.array(0.0, dtype=jnp.float32)
        )
    elif alpha_params_style == "dict":
        model.add_network(
            "log_alpha",
            DummyAlphaModule(),
            {"params": jnp.array(0.0, dtype=jnp.float32)},
            optax.adam(1e-3),
        )
    else:
        raise ValueError(f"Unknown alpha_params_style: {alpha_params_style}")
    return model


def build_episode_data(batch_size, seed=0):
    return {
        "observation": jnp.ones((batch_size, 3), dtype=jnp.float32),
        "action": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "reward": jnp.ones((batch_size, 1), dtype=jnp.float32),
        "next_observation": 2.0 * jnp.ones((batch_size, 3), dtype=jnp.float32),
        "terminated": jnp.zeros((batch_size, 1), dtype=jnp.float32),
        "actor_rng": jax.random.PRNGKey(seed + 1),
        "next_actor_rng": jax.random.PRNGKey(seed + 2),
    }


@pytest.mark.parametrize("batch_size", [1, 4, 7])
def test_loss_jit_compilation(batch_size):
    network = build_sac_network(batch_size=batch_size, seed=11)

    # 2. FIXED: Instantiate the strategy via .create()
    loss_fn = SoftActorCriticLoss(
        target_entropy=-2.0,
        sampling_strategy=ContinuousGaussianDistribution.create(action_dimension=2),
    )
    batch = build_episode_data(batch_size, seed=21)

    trainable_params = {
        "actor": network.states["actor"].params,
        "critic": network.states["critic"].params,
        "log_alpha": network.states["log_alpha"].params,
    }

    (total_loss, metrics), grads = get_sac_grads(
        trainable_params,
        network.networks["actor"],
        network.networks["critic"],
        network.target_params["critic"],
        loss_fn.value_function.__call__,
        loss_fn.sampling_strategy.__call__,
        loss_fn.target_entropy,
        batch,
    )

    assert bool(jnp.isfinite(total_loss))
    assert set(metrics) == {
        "critic_loss",
        "actor_loss",
        "temperature_loss",
        "alpha",
        "q1_mean",
    }
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in metrics.values())
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
        trainable_params
    )


@pytest.mark.parametrize("batch_size", [2, 5])
def test_gradient_shapes(batch_size):
    network = build_sac_network(batch_size=batch_size, seed=17)
    loss_fn = SoftActorCriticLoss(
        target_entropy=-2.0,
        sampling_strategy=ContinuousGaussianDistribution.create(action_dimension=2),
    )
    batch = build_episode_data(batch_size, seed=33)

    trainable_params = {
        "actor": network.states["actor"].params,
        "critic": network.states["critic"].params,
        "log_alpha": network.states["log_alpha"].params,
    }

    (_, _), grads = get_sac_grads(
        trainable_params,
        network.networks["actor"],
        network.networks["critic"],
        network.target_params["critic"],
        loss_fn.value_function.__call__,
        loss_fn.sampling_strategy.__call__,
        loss_fn.target_entropy,
        batch,
    )

    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
        trainable_params
    )
    assert jax.tree_util.tree_map(lambda x: x.shape, grads) == jax.tree_util.tree_map(
        lambda x: x.shape, trainable_params
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_target_q_computation(batch_size):
    value_fn = TDReturnsSAC(gamma=0.99, standardize=False)
    rewards = jnp.full((batch_size, 1), 1.0, dtype=jnp.float32)
    q_next_min = jnp.full((batch_size, 1), 2.0, dtype=jnp.float32)
    temperature = jnp.array(0.5, dtype=jnp.float32)
    next_log_probs = jnp.full((batch_size, 1), 1.0, dtype=jnp.float32)
    terminated = jnp.zeros((batch_size, 1), dtype=jnp.float32)

    targets = value_fn(rewards, q_next_min, temperature, next_log_probs, terminated)
    expected = jnp.full((batch_size, 1), 1.0 + 0.99 * (2.0 - 0.5), dtype=jnp.float32)

    assert jnp.allclose(targets, expected)


def test_sac_loss_updates_multiflax_states_with_rng_subkeys():
    network = build_sac_network(batch_size=4, seed=11)
    loss_fn = SoftActorCriticLoss(
        target_entropy=-2.0,
        sampling_strategy=ContinuousGaussianDistribution.create(action_dimension=2),
    )
    batch = build_episode_data(4, seed=21)

    old_actor_step = network.states["actor"].step
    old_critic_step = network.states["critic"].step
    old_alpha_step = network.states["log_alpha"].step
    old_target_critic = jax.tree_util.tree_map(
        lambda x: x.copy(), network.target_params["critic"]
    )

    metrics = loss_fn.compute_loss(network, batch)

    assert network.states["actor"].step == old_actor_step + 1
    assert network.states["critic"].step == old_critic_step + 1
    assert network.states["log_alpha"].step == old_alpha_step + 1
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda new, old: jnp.array_equal(new, old),
            network.target_params["critic"],
            old_target_critic,
        )
    )
    assert set(metrics) == {
        "critic_loss",
        "actor_loss",
        "temperature_loss",
        "alpha",
        "q1_mean",
    }
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in metrics.values())


def test_sac_loss_accepts_dict_style_log_alpha_params():
    network = build_sac_network(batch_size=4, seed=13, alpha_params_style="dict")
    loss_fn = SoftActorCriticLoss(
        target_entropy=-2.0,
        sampling_strategy=ContinuousGaussianDistribution.create(action_dimension=2),
    )
    batch = build_episode_data(4, seed=31)

    metrics = loss_fn.compute_loss(network, batch)

    assert network.states["log_alpha"].step == 1
    assert isinstance(network.states["log_alpha"].params, dict)
    assert jnp.shape(network.states["log_alpha"].params["params"]) == ()
    assert bool(jnp.isfinite(metrics["alpha"]))
