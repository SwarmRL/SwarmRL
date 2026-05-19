"""Simple ring-buffer replay memory for off-policy algorithms."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from swarmrl.replay_buffer.transition import Transition


class ReplayBuffer:
    """Fixed-size replay buffer with random minibatch sampling."""

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._storage: list[Transition | None] = [None] * self.capacity
        self._size = 0
        self._position = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        self._storage[self._position] = transition
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= int(batch_size)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {self._size}."
            )

        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        transitions = [self._storage[idx] for idx in indices]

        batch = {
            "observation": [],
            "action": [],
            "reward": [],
            "next_observation": [],
            "done": [],
        }
        for transition in transitions:
            data = asdict(transition)
            for key in batch:
                batch[key].append(data[key])

        return {
            "observation": np.asarray(batch["observation"]),
            "action": np.asarray(batch["action"]),
            "reward": np.asarray(batch["reward"], dtype=np.float32).reshape(-1, 1),
            "next_observation": np.asarray(batch["next_observation"]),
            "done": np.asarray(batch["done"], dtype=np.float32).reshape(-1, 1),
        }
