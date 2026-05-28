from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np

from swarmrl.agents.sac_agent import SACAgent
from swarmrl.replay_buffer.replay_buffer import ReplayBuffer
from swarmrl.replay_buffer.transition import Transition
from swarmrl.utils.storage_utils import TransitionStorageConfig


class LossSpy:
    def __init__(self):
        self.calls = []

    def compute_loss(self, network, batch):
        self.calls.append((network, batch))
        return {"critic_loss": 0.0}


class DummyNetwork:
    pass


class DummyActor:
    def apply(self, params, rng_key, feature_data):
        batch_size = feature_data.shape[0]
        action = jnp.zeros((batch_size, 2), dtype=jnp.float32)
        return action, None


class DummyObservable:
    def initialize(self, colloids):
        self.colloids = colloids

    def compute_observable(self, colloids):
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)


class DummyTask:
    def __init__(self):
        self.kill_switch = False

    def initialize(self, colloids):
        self.colloids = colloids

    def __call__(self, colloids):
        return 1.0


def filled_buffer(size=4):
    buffer = ReplayBuffer(capacity=size, seed=0)
    for i in range(size):
        buffer.add(
            Transition(
                observation=np.array([i, i + 1], dtype=np.float32),
                action=np.array([0.1, -0.1], dtype=np.float32),
                reward=1.0,
                next_observation=np.array([i + 1, i + 2], dtype=np.float32),
                terminated=0.0,
            )
        )
    return buffer


def test_sac_agent_updates_multiflax_container_via_loss_bridge():
    network = DummyNetwork()
    loss = LossSpy()
    agent = SACAgent(
        particle_type=0,
        network=network,
        task=None,
        observable=None,
        action_mapper=lambda action: action,
        loss=loss,
        replay_buffer=filled_buffer(),
        batch_size=2,
        learning_starts=0,
        gradient_steps=2,
        train=True,
    )
    agent._step_count = 1

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
        "actor_rng",
        "next_actor_rng",
    }
    assert all(set(call[1]) == expected_keys for call in loss.calls)


def test_sac_agent_can_dump_transition_debug_data(tmp_path):
    network = SimpleNamespace(
        networks={"actor": DummyActor()},
        states={"actor": SimpleNamespace(params=object())},
    )
    loss = LossSpy()
    agent = SACAgent(
        particle_type=1,
        network=network,
        task=DummyTask(),
        observable=DummyObservable(),
        action_mapper=lambda action: [action],
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
    colloids = [object()]

    agent.reset_agent(colloids)
    agent.calc_action(colloids)
    agent.calc_action(colloids)
    agent.finalize()

    file_path = tmp_path / "sac_transition_data_1.hdf5"
    with h5py.File(file_path.as_posix(), "r") as h5_file:
        group = h5_file["SAC_1"]
        assert group["observation"].shape[0] == 1
        assert group["action"].shape[0] == 1
        assert group["reward"].shape[0] == 1
        assert group["next_observation"].shape[0] == 1
        assert group["terminated"].shape[0] == 1
