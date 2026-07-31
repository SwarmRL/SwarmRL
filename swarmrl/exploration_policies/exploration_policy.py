"""
Parent class for exploration modules.
"""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp


class ExplorationPolicy(ABC):
    """
    Parent class for exploration policies.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> jnp.ndarray:
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
        self, model_actions: jnp.ndarray, action_space_length: int, seed: Any
    ) -> jnp.ndarray:
        """
        Return an index associated with the chosen action.

        Parameters
        ----------
        model_actions : jnp.ndarray (n_colloids,)
                Action chosen by the model for each colloid.
        action_space_length : int
                Number of possible actions. Should be 1 higher than the actual highest
                index, i.e if I have actions [0, 1, 2, 3] this number should be 4.

        Returns
        -------
        action : jnp.ndarray
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
        self, model_actions: jnp.ndarray, rng_key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Return an action value

        Parameters
        ----------
        model_actions : jnp.ndarray (n_colloids,)
                Action chosen by the model for each colloid.
        rng_key : jax.random.PRNGKey
                Key for jax.random module

        Returns
        -------
        action : jnp.ndarray
                Action chosen after the exploration module has operated.
        """
        raise NotImplementedError
