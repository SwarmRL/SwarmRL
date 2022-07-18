"""
Random exploration module.
"""
from abc import ABC

import jax
import numpy as onp

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy


class RandomExploration(ExplorationPolicy, ABC):
    """
    Perform exploration by random moves.
    """

    def __init__(self, probability: float):
        """
        Constructor for the random exploration module.

        Parameters
        ----------
        probability : float
                Probability that a random action will be chosen.
                Bound between [0.0, 1.0]
        """
        self.probability = probability

    def __call__(self, model_action: int, action_space_length: int) -> int:
        """
        Check if the exploration should be performed.
        """
        rng = jax.random.PRNGKey(onp.random.randint(0, 1000000000))
        sample = jax.random.uniform(rng)

        if sample <= self.probability:
            rng = jax.random.PRNGKey(onp.random.randint(0, 1000000000))
            exploration_action = jax.random.randint(
                rng, (1,), minval=0, maxval=action_space_length
            )
            return exploration_action[0]
        else:
            return model_action
