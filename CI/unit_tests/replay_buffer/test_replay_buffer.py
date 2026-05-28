import numpy as np

from swarmrl.replay_buffer import ReplayBuffer, Transition


def test_replay_buffer_add_and_sample_shapes():
    buffer = ReplayBuffer(capacity=8, seed=7)

    for i in range(6):
        buffer.add(
            Transition(
                observation=np.array([i, i + 1], dtype=np.float32),
                action=np.array([0.1 * i], dtype=np.float32),
                reward=float(i),
                next_observation=np.array([i + 1, i + 2], dtype=np.float32),
                terminated=float(bool(i % 2)),
            )
        )

    batch = buffer.sample(4)
    assert batch["observation"].shape == (4, 2)
    assert batch["action"].shape == (4, 1)
    assert batch["reward"].shape == (4, 1)
    assert batch["next_observation"].shape == (4, 2)
    assert batch["terminated"].shape == (4, 1)


def test_replay_buffer_capacity_is_enforced():
    buffer = ReplayBuffer(capacity=3, seed=3)
    for i in range(7):
        buffer.add(
            Transition(
                observation=np.array([i], dtype=np.float32),
                action=np.array([i], dtype=np.float32),
                reward=float(i),
                next_observation=np.array([i + 1], dtype=np.float32),
                terminated=0.0,
            )
        )

    assert len(buffer) == 3
    assert buffer.can_sample(3)
