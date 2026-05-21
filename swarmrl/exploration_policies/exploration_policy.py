"""
Parent class for exploration modules.
"""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as np


class ExplorationPolicy(ABC):
    """
    Parent class for exploration policies.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Apply exploration to model actions.
        """
        raise NotImplementedError


class DiscreteExplorationPolicy(ExplorationPolicy, ABC):
    """
    Parent class for discrete exploration policies.
    """

    @abstractmethod
    def __call__(
        self, model_actions: np.ndarray, action_space_length: int, seed: Any
    ) -> np.ndarray:
        """
        Return an index associated with the chosen action.

        Parameters
        ----------
        model_actions : np.ndarray (n_colloids,)
                Action chosen by the model for each colloid.
        action_space_length : int
                Number of possible actions. Should be 1 higher than the actual highest
                index, i.e if I have actions [0, 1, 2, 3] this number should be 4.

        Returns
        -------
        action : np.ndarray
                Action chosen after the exploration module has operated for
                each colloid.
        """
        raise NotImplementedError


class ContinuousExplorationPolicy(ExplorationPolicy, ABC):
    """
    Parent class for continuous exploration policies.
    """

    @abstractmethod
    def __call__(
        self, model_actions: np.ndarray, rng_key: jax.random.PRNGKey
    ) -> np.ndarray:
        """
        Return an action value

        Parameters
        ----------
        model_actions : np.ndarray (n_colloids,)
                Action chosen by the model for each colloid.
        rng_key : jax.random.PRNGKey
                Key for jax.random module

        Returns
        -------
        action : np.ndarray
                Action chosen after the exploration module has operated.
        """
        raise NotImplementedError
