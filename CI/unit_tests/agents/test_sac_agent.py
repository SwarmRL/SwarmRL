import numpy as np

from swarmrl.agents.sac_agent import SACAgent
from swarmrl.replay_buffer.replay_buffer import ReplayBuffer
from swarmrl.replay_buffer.transition import Transition


class LossSpy:
    def __init__(self):
        self.calls = []

    def compute_loss(self, network, batch):
        self.calls.append((network, batch))
        return {"critic_loss": 0.0}


class DummyNetwork:
    pass


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
