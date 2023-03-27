"""
Random exploration module.
"""
from abc import ABC

import jax
import jax.numpy as np
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

    def __call__(
        self, model_actions: np.ndarray, action_space_length: int
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
        rng = jax.random.PRNGKey(onp.random.randint(0, 1000000000))
        sample = jax.random.uniform(rng, shape=model_actions.shape)

        # Get indices to randomize
        exploration_indices = np.where(sample <= self.probability)

        # Choose random actions
        rng = jax.random.PRNGKey(onp.random.randint(0, 1000000000))
        exploration_actions = jax.random.randint(
            rng, shape=(len(exploration_indices),), minval=0, maxval=action_space_length
        )
        model_actions = onp.array(model_actions)

        # Put the new actions in.
        onp.put(model_actions, exploration_indices, exploration_actions)

        return model_actions
