"""
Parent class for sampling strategies.
"""

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as np


class SamplingStrategy(ABC):
    """
    Parent class for sampling strategies.
    """

    def compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute the Shannon entropy of the probabilities.

        Parameters
        ----------
        probabilities : np.ndarray (n_colloids, n_actions)
                Probabilities for each colloid to take specific actions.
        """
        eps = 1e-8
        probabilities += eps
        return -np.sum(probabilities * np.log(probabilities))

    @abstractmethod
    def __call__(self, logits: np.ndarray) -> Any:
        """
        Sample from the distribution.

        Parameters
        ----------
        logits : np.ndarray (n_colloids, n_dimensions)
                Logits from the model to use in the computation for each colloid.

        Returns
        -------
        sample : int
                Index of the selected option in the distribution.
        """
        raise NotImplementedError("Implemented in child classes.")


class DiscreteSamplingStrategy(SamplingStrategy, ABC):
    """Parent class for discrete sampling strategies."""


class ContinuousSamplingStrategy(SamplingStrategy, ABC):
    """Parent class for continuous sampling strategies."""
