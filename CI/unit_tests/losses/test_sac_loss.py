import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest
from flax.core import freeze, unfreeze

from swarmrl.losses.sac_loss import (
    SoftActorCriticLoss,
    calculate_critic_loss,
    get_sac_grads,
    sac_loss_fn,
)
from swarmrl.networks import FlaxModel
from swarmrl.sampling_strategies import ContinuousGaussianDistribution
from swarmrl.value_functions.td_return_sac import TDReturnsSAC


class TinySacModule(nn.Module):
    action_dim: int = 2

    def setup(self):
        self.actor_hidden = nn.Dense(8)
        self.actor_out = nn.Dense(self.action_dim * 2)
        self.critic_hidden = nn.Dense(8)
        self.critic_q1 = nn.Dense(1)
        self.critic_q2 = nn.Dense(1)
        self.value_head = nn.Dense(1)

    def __call__(self, feature_data):
        if feature_data.ndim == 1:
            feature_data = feature_data[None, :]
        logits = self.actor(feature_data, rng_key=None)
        dummy_actions = jnp.zeros(
            (feature_data.shape[0], self.action_dim), dtype=feature_data.dtype
        )
        q1, _, _, _ = self.full_forward(feature_data, dummy_actions, rng_key=None)
        return logits, q1

    def full_forward(self, feature_data, actions, rng_key=None):
        logits = self.actor(feature_data, rng_key=rng_key)
        q1, q2 = self.critic(feature_data, actions)
        alpha = self.alpha()
        return logits, q1, q2, alpha

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


def build_sac_network(batch_size=4, seed=0):
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


def build_episode_data(batch_size, seed=0):
    return {
        "observation": jnp.ones((batch_size, 3), dtype=jnp.float32),
        "action": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "reward": jnp.ones((batch_size, 1), dtype=jnp.float32),
        "next_observation": 2.0 * jnp.ones((batch_size, 3), dtype=jnp.float32),
        "terminated": jnp.zeros((batch_size, 1), dtype=jnp.float32),
        "truncated": jnp.zeros((batch_size, 1), dtype=jnp.float32),
        "actor_rng": jax.random.PRNGKey(seed + 1),
        "next_actor_rng": jax.random.PRNGKey(seed + 2),
    }


def extract_critic_params(params):
    flat = unfreeze(params)
    return {
        "critic_hidden": flat["critic_hidden"],
        "critic_q1": flat["critic_q1"],
        "critic_q2": flat["critic_q2"],
    }


def replace_critic_params(params, critic_params):
    flat = unfreeze(params)
    flat["critic_hidden"] = critic_params["critic_hidden"]
    flat["critic_q1"] = critic_params["critic_q1"]
    flat["critic_q2"] = critic_params["critic_q2"]
    return freeze(flat)


@pytest.mark.parametrize("batch_size", [1, 4, 7])
def test_loss_jit_compilation(batch_size):
    network = build_sac_network(batch_size=batch_size, seed=11)

    loss_fn = SoftActorCriticLoss(
        target_entropy=-2.0,
        sampling_strategy=ContinuousGaussianDistribution.create(action_dimension=2),
    )
    batch = build_episode_data(batch_size, seed=21)

    trainable_params = network.model_state.params

    (total_loss, metrics), grads = get_sac_grads(
        trainable_params,
        network.model,
        network.target_params["critic"],
        loss_fn.value_function.__call__,
        loss_fn.sampling_strategy,
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

    trainable_params = network.model_state.params

    (_, _), grads = get_sac_grads(
        trainable_params,
        network.model,
        network.target_params["critic"],
        loss_fn.value_function.__call__,
        loss_fn.sampling_strategy,
        loss_fn.target_entropy,
        batch,
    )

    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
        trainable_params
    )
    assert jax.tree_util.tree_map(lambda x: x.shape, grads) == jax.tree_util.tree_map(
        lambda x: x.shape, trainable_params
    )


def test_sac_critic_grads_do_not_include_actor_loss():
    network = build_sac_network(batch_size=4, seed=19)
    loss_fn = SoftActorCriticLoss(
        target_entropy=-2.0,
        sampling_strategy=ContinuousGaussianDistribution.create(action_dimension=2),
    )
    batch = build_episode_data(4, seed=41)
    trainable_params = network.model_state.params

    (_, _), sac_grads = get_sac_grads(
        trainable_params,
        network.model,
        network.target_params["critic"],
        loss_fn.value_function.__call__,
        loss_fn.sampling_strategy,
        loss_fn.target_entropy,
        batch,
    )

    initial_critic_params = extract_critic_params(trainable_params)

    def critic_only_loss(critic_params):
        state_inputs = {"feature_data": jnp.array(batch["observation"])}
        next_state_inputs = {"feature_data": jnp.array(batch["next_observation"])}
        actions = jnp.array(batch["action"])
        rewards = jnp.array(batch["reward"]).reshape(-1, 1)
        terminated = jnp.array(batch["terminated"]).reshape(-1, 1)
        next_network_key, next_sample_key = jax.random.split(batch["next_actor_rng"])
        params_with_updated_critic = replace_critic_params(
            trainable_params, critic_params
        )

        next_logits = network.model.apply(
            {"params": trainable_params},
            rng_key=next_network_key,
            method=network.model.actor,
            **next_state_inputs,
        )
        next_actions, next_log_probs = loss_fn.sampling_strategy(
            logits=next_logits,
            rng_key=next_sample_key,
            calculate_log_probs=True,
            deployment_mode=False,
        )
        q1_next, q2_next = network.model.apply(
            {"params": network.target_params["critic"]},
            actions=next_actions,
            method=network.model.critic,
            **next_state_inputs,
        )
        target_q = loss_fn.value_function(
            rewards=rewards,
            q_next_min=jnp.minimum(q1_next, q2_next),
            temperature=jax.lax.stop_gradient(
                network.model.apply(
                    {"params": trainable_params}, method=network.model.alpha
                )
            ),
            next_log_probs=next_log_probs[..., None],
            terminated=terminated,
        )
        target_q = jax.lax.stop_gradient(target_q)
        q1_pred, q2_pred = network.model.apply(
            {"params": params_with_updated_critic},
            actions=actions,
            method=network.model.critic,
            **state_inputs,
        )
        return calculate_critic_loss(q1_pred, q2_pred, target_q)

    critic_only_grads = jax.grad(critic_only_loss)(initial_critic_params)
    combined_critic_grads = extract_critic_params(sac_grads)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda combined, critic_only: jnp.allclose(
                combined, critic_only, rtol=5e-4, atol=1e-6
            ),
            combined_critic_grads,
            critic_only_grads,
        )
    )


def test_sac_actor_loss_has_zero_gradient_wrt_critic_params():
    network = build_sac_network(batch_size=4, seed=23)
    loss_fn = SoftActorCriticLoss(
        target_entropy=-2.0,
        sampling_strategy=ContinuousGaussianDistribution.create(action_dimension=2),
    )
    batch = build_episode_data(4, seed=43)
    trainable_params = network.model_state.params
    initial_critic_params = extract_critic_params(trainable_params)

    def actor_only_loss_view(critic_params):
        _, metrics = sac_loss_fn(
            replace_critic_params(trainable_params, critic_params),
            network.model,
            network.target_params["critic"],
            loss_fn.value_function.__call__,
            loss_fn.sampling_strategy,
            loss_fn.target_entropy,
            batch,
        )
        return metrics["actor_loss"]

    actor_grad_wrt_critic = jax.grad(actor_only_loss_view)(initial_critic_params)

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda grad: jnp.allclose(grad, 0.0),
            actor_grad_wrt_critic,
        )
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


def test_sac_loss_updates_flax_model_state_and_target_params():
    network = build_sac_network(batch_size=4, seed=11)
    loss_fn = SoftActorCriticLoss(
        target_entropy=-2.0,
        sampling_strategy=ContinuousGaussianDistribution.create(action_dimension=2),
    )
    batch = build_episode_data(4, seed=21)

    old_step = network.model_state.step
    old_target_critic = jax.tree_util.tree_map(
        lambda x: x.copy(), network.target_params["critic"]
    )

    metrics = loss_fn.compute_loss(network, batch)

    assert network.model_state.step == old_step + 1
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


def test_target_q_ignores_truncated_when_terminated_is_zero():
    value_fn = TDReturnsSAC(gamma=0.99, standardize=False)
    rewards = jnp.array([[1.0]], dtype=jnp.float32)
    q_next_min = jnp.array([[2.0]], dtype=jnp.float32)
    temperature = jnp.array(0.5, dtype=jnp.float32)
    next_log_probs = jnp.array([[1.0]], dtype=jnp.float32)
    terminated = jnp.array([[0.0]], dtype=jnp.float32)
    truncated = jnp.array([[1.0]], dtype=jnp.float32)

    del truncated  # truncation is carried in the batch but must not mask SAC targets
    targets = value_fn(rewards, q_next_min, temperature, next_log_probs, terminated)
    expected = jnp.array([[1.0 + 0.99 * (2.0 - 0.5)]], dtype=jnp.float32)

    assert jnp.allclose(targets, expected)


def test_critic_loss_rejects_cross_broadcasting_shapes():
    q1 = jnp.array([1.0, 2.0, 3.0])
    q2 = jnp.array([1.5, 2.5, 3.5])
    target = jnp.array([[1.0], [2.0], [3.0]])

    with pytest.raises(ValueError, match="matching shapes"):
        calculate_critic_loss(q1, q2, target)
