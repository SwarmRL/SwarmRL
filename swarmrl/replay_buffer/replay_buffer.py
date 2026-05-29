"""Simple ring-buffer replay memory for off-policy algorithms."""

from dataclasses import fields

import numpy as np

from swarmrl.replay_buffer.transition import Transition


class ReplayBuffer:
    """
    Fixed-size replay buffer with random minibatch sampling, using lazy-allocated
    NumPy buffers for high-performance vectorized sampling, bypassing slow Python
    loops and dataclass conversions.
    """

    def __init__(self, capacity: int, seed: int | None = None):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity : int
                The maximum number of transitions the buffer can store.
                Must be greater than 0.
            seed : int
                Optional random seed for reproducibility of sample indices.

        Raises:
            ValueError: If `capacity` is less than or equal to 0.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)

        self._size = 0
        self._position = 0

        self._initialized = False
        self._buffers: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self._size

    def _init_buffers(self, transition: Transition) -> None:
        """Dynamically allocates contiguous NumPy arrays based on Transition fields."""
        for field in fields(transition):
            key = field.name
            val = getattr(transition, key)
            val_arr = np.asarray(val)

            # Downcast common 64-bit inputs to reduce buffer memory use.
            dtype = val_arr.dtype
            if dtype == np.float64:
                dtype = np.float32
            elif dtype == np.int64:
                dtype = np.int32

            # Dynamic shape selection: scalars are expanded to (capacity, 1)
            # to maintain standard batch dimensions.

            if val_arr.ndim == 0:
                shape = (self.capacity, 1)
            else:
                shape = (self.capacity,) + val_arr.shape

            self._buffers[key] = np.empty(shape, dtype=dtype)

        self._initialized = True

    def add(self, transition: Transition) -> None:
        if not self._initialized:
            self._init_buffers(transition)

        # Write directly into pre-allocated buffer slots.
        for key in self._buffers:
            val = getattr(transition, key)
            self._buffers[key][self._position] = val

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= int(batch_size)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample {batch_size} transitions "
                f"from buffer of size {self._size}."
            )

        indices = self._rng.choice(self._size, size=batch_size, replace=False)

        return {key: buf[indices] for key, buf in self._buffers.items()}
