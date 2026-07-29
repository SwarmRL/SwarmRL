import flax.linen as nn
import h5py
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import swarmrl as srl
import swarmrl.agents.sac_agent as sac_agent_module
from swarmrl.agents.sac_agent import SACAgent
from swarmrl.networks import FlaxModel
from swarmrl.replay_buffer.replay_buffer import ReplayBuffer
from swarmrl.replay_buffer.transition import Transition
from swarmrl.sampling_strategies import ContinuousGaussianDistribution
from swarmrl.utils.storage_utils import TransitionStorageConfig


class LossSpy:
    def __init__(self):
        self.calls = []

    def compute_loss(self, network, batch):
        self.calls.append((network, batch))
        return {"critic_loss": 0.0}


class SacActorModule(nn.Module):
    @nn.compact
    def __call__(self, feature_data):
        logits = nn.Dense(4, kernel_init=nn.initializers.zeros)(feature_data)
        value = nn.Dense(1, kernel_init=nn.initializers.zeros)(feature_data)
        return logits, value

    @nn.compact
    def actor(self, feature_data, rng_key):
        del rng_key
        return nn.Dense(4, kernel_init=nn.initializers.zeros)(feature_data)

    @nn.compact
    def critic(self, feature_data, actions):
        x = jnp.concatenate([feature_data, actions], axis=-1)
        hidden = nn.Dense(8)(x)
        q1 = nn.Dense(1)(hidden)
        q2 = nn.Dense(1)(hidden)
        return q1, q2

    @nn.compact
    def alpha(self):
        return self.param("log_alpha", nn.initializers.zeros, ())


class MissingAlphaModule(nn.Module):
    @nn.compact
    def __call__(self, feature_data):
        return nn.Dense(4)(feature_data), nn.Dense(1)(feature_data)

    @nn.compact
    def actor(self, feature_data, rng_key):
        del rng_key
        return nn.Dense(4)(feature_data)

    @nn.compact
    def critic(self, feature_data, actions):
        x = jnp.concatenate([feature_data, actions], axis=-1)
        hidden = nn.Dense(8)(x)
        q = nn.Dense(1)(hidden)
        return q, q


class DummyObservable:
    def initialize(self, colloids):
        self.colloids = colloids

    def compute_observable(self, colloids):
        n = len(colloids)
        base = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        return np.tile(base, (n, 1))


class DummyTask:
    def __init__(self):
        self.kill_switch = False
        self.truncated = False

    def initialize(self, colloids):
        self.colloids = colloids

    def __call__(self, colloids):
        return np.ones(len(colloids), dtype=np.float32)


def build_sac_network(module=None):
    return FlaxModel(
        flax_model=module or SacActorModule(),
        optimizer=optax.adam(learning_rate=0.001),
        input_shape=(3,),
        sampling_strategy=ContinuousGaussianDistribution.create(action_dimension=2),
        exploration_policy=srl.exploration_policies.GlobalOUExploration(
            action_dimension=2,
            action_limits=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32),
            epsilon=1.0,
        ),
    )


def filled_buffer(size=4):
    buffer = ReplayBuffer(capacity=size, seed=0)
    for i in range(size):
        buffer.add(
            Transition(
                observation=np.array([i, i + 1, i + 2], dtype=np.float32),
                action=np.array([0.1, -0.1], dtype=np.float32),
                reward=1.0,
                next_observation=np.array([i + 1, i + 2, i + 3], dtype=np.float32),
                terminated=0.0,
                truncated=0.0,
            )
        )
    return buffer


def make_agent(network, loss=None, replay_buffer=None, **kwargs):
    return SACAgent(
        particle_type=kwargs.pop("particle_type", 0),
        network=network,
        task=kwargs.pop("task", DummyTask()),
        observable=kwargs.pop("observable", DummyObservable()),
        action_mapper=kwargs.pop("action_mapper", lambda action: action),
        loss=loss if loss is not None else LossSpy(),
        replay_buffer=replay_buffer if replay_buffer is not None else filled_buffer(),
        sampling_strategy=kwargs.pop(
            "sampling_strategy",
            ContinuousGaussianDistribution.create(action_dimension=2),
        ),
        batch_size=kwargs.pop("batch_size", 2),
        learning_starts=kwargs.pop("learning_starts", 0),
        gradient_steps=kwargs.pop("gradient_steps", 1),
        train=kwargs.pop("train", True),
        transition_storage_config=kwargs.pop("transition_storage_config", None),
        **kwargs,
    )


def test_sac_agent_rejects_non_flax_network():
    with pytest.raises(TypeError, match="FlaxModel"):
        make_agent(network=object())


def test_sac_agent_rejects_missing_sac_method():
    network = build_sac_network(module=MissingAlphaModule())

    with pytest.raises(ValueError, match="alpha"):
        make_agent(network=network)


def test_sac_agent_initializes_target_critic_params():
    network = build_sac_network()

    assert network.target_params == {}

    make_agent(network=network)

    assert "critic" in network.target_params
    assert jnp.array_equal(
        network.target_params["critic"]["Dense_0"]["kernel"],
        network.model_state.params["Dense_0"]["kernel"],
    )


def test_sac_agent_updates_loss_bridge_with_replay_batch_rng_keys():
    network = build_sac_network()
    loss = LossSpy()
    agent = make_agent(
        network=network,
        loss=loss,
        replay_buffer=filled_buffer(),
        gradient_steps=2,
    )
    agent._step_count = 1
    agent.kill_switch = False

    reward, killed = agent.update_agent()

    assert reward == 0.0
    assert killed is False
    assert len(loss.calls) == 2
    assert all(call[0] is network for call in loss.calls)
    expected_keys = {
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
        "actor_rng",
        "next_actor_rng",
    }
    assert all(set(call[1]) == expected_keys for call in loss.calls)


def test_sac_agent_closes_transition_in_calc_reward():
    network = build_sac_network()
    buffer = ReplayBuffer(capacity=4, seed=0)
    agent = make_agent(
        network=network,
        replay_buffer=buffer,
        batch_size=2,
        learning_starts=0,
        gradient_steps=1,
        train=True,
    )
    colloids = [object(), object(), object()]

    agent.reset_agent(colloids)
    actions = agent.calc_action(colloids)

    assert len(buffer) == 0
    assert len(actions) == 3

    reward = agent.calc_reward(colloids)

    assert reward == 1.0
    assert len(buffer) == 3


def test_sac_agent_closes_each_pending_transition_only_once():
    network = build_sac_network()
    buffer = ReplayBuffer(capacity=4, seed=0)
    agent = make_agent(
        network=network,
        replay_buffer=buffer,
        learning_starts=0,
        train=True,
    )
    colloids = [object()]
    agent.reset_agent(colloids)
    agent.calc_action(colloids)

    agent.calc_reward(colloids)
    agent.calc_reward(colloids)

    assert len(buffer) == 1


def test_sac_agent_warmup_uses_sampling_strategy_action_limits(monkeypatch):
    network = build_sac_network()
    limits = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    strategy = ContinuousGaussianDistribution.create(
        action_dimension=2, action_limits=limits
    )
    agent = make_agent(
        network=network,
        sampling_strategy=strategy,
        replay_buffer=ReplayBuffer(capacity=4, seed=0),
        learning_starts=5,
        batch_size=2,
        train=True,
    )
    colloids = [object(), object(), object()]

    def fail_actor(*args, **kwargs):
        raise AssertionError("warmup must not call the policy actor")

    monkeypatch.setattr(network.model, "actor", fail_actor)

    agent.reset_agent(colloids)
    actions = agent.calc_action(colloids)

    assert len(actions) == 3
    action_array = np.asarray(actions)
    assert np.all(action_array >= 0.0)
    assert np.all(action_array <= 1.0)


def test_sac_agent_logs_when_learning_starts():
    network = build_sac_network()
    agent = make_agent(
        network=network,
        replay_buffer=filled_buffer(),
        learning_starts=1,
        batch_size=2,
        train=True,
    )
    colloids = [object(), object(), object()]
    messages = []
    original_info = sac_agent_module.logger.info

    def capture_info(message, *args, **kwargs):
        messages.append(message)
        return original_info(message, *args, **kwargs)

    sac_agent_module.logger.info = capture_info
    try:
        agent.reset_agent(colloids)
        agent.calc_action(colloids)
        agent.calc_reward(colloids)
        agent.update_agent()
        agent.update_agent()
    finally:
        sac_agent_module.logger.info = original_info

    matching_messages = [
        message for message in messages if "learning starts" in message.lower()
    ]

    assert len(matching_messages) == 1


def test_sac_agent_can_dump_transition_debug_data_with_flax_model(tmp_path):
    network = build_sac_network()

    loss = LossSpy()
    agent = make_agent(
        particle_type=1,
        network=network,
        task=DummyTask(),
        observable=DummyObservable(),
        action_mapper=lambda action: action,
        loss=loss,
        replay_buffer=ReplayBuffer(capacity=4, seed=0),
        batch_size=2,
        learning_starts=0,
        gradient_steps=1,
        train=True,
        transition_storage_config=TransitionStorageConfig(
            out_folder=str(tmp_path),
            storage_preset="verbose",
        ),
    )
    colloids = [object(), object()]

    agent.reset_agent(colloids)
    first_actions = agent.calc_action(colloids)
    agent.calc_reward(colloids)
    second_actions = agent.calc_action(colloids)
    agent.calc_reward(colloids)
    agent.finalize()

    assert len(first_actions) == 2
    assert len(second_actions) == 2

    file_path = tmp_path / "sac_transition_data_1.hdf5"
    with h5py.File(file_path.as_posix(), "r") as h5_file:
        group = h5_file["SAC_1"]
        assert group["observation"].shape[0] == 4
        assert group["action"].shape[0] == 4
        assert group["reward"].shape[0] == 4
        assert group["next_observation"].shape[0] == 4
        assert group["terminated"].shape[0] == 4
        assert group["truncated"].shape[0] == 4


def test_sac_agent_records_truncated_separately_from_terminated():
    network = build_sac_network()
    buffer = ReplayBuffer(capacity=4, seed=0)
    task = DummyTask()
    task.truncated = True
    agent = make_agent(
        network=network,
        replay_buffer=buffer,
        batch_size=2,
        learning_starts=0,
        gradient_steps=1,
        train=True,
        task=task,
    )
    colloids = [object(), object()]

    agent.reset_agent(colloids)
    agent.calc_action(colloids)
    agent.calc_reward(colloids)

    batch = buffer.sample(2)
    assert np.all(batch["terminated"] == 0.0)
    assert np.all(batch["truncated"] == 1.0)
    assert agent.kill_switch is True
