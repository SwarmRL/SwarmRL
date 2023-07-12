"""
Random exploration module.
"""
from abc import ABC
from functools import partial

import jax
import jax.numpy as np

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy


class RandomExploration(ExplorationPolicy, ABC):
    """
    Perform exploration by random moves.
    """

    def __init__(self, probability: float = 0.1):
        """
        Constructor for the random exploration module.

        Parameters
        ----------
        probability : float
                Probability that a random action will be chosen.
                Bound between [0.0, 1.0]
        """
        self.probability = probability

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, model_actions: np.ndarray, action_space_length: int, seed: int = 0
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
        key = jax.random.PRNGKey(seed)
        sample = jax.random.uniform(key, shape=model_actions.shape)

        to_be_changed = np.clip(sample - self.probability, a_min=0, a_max=1)
        to_be_changed = np.clip(to_be_changed * 1e6, a_min=0, a_max=1)
        not_to_be_changed = np.clip(to_be_changed * -10 + 1, 0, 1)

        # Choose random actions
        key, subkey = jax.random.split(key)
        exploration_actions = jax.random.randint(
            subkey,
            shape=(model_actions.shape[0],),
            minval=0,
            maxval=action_space_length,
        )

        # Put the new actions in.
        model_actions = (
            model_actions * to_be_changed + exploration_actions * not_to_be_changed
        ).astype(np.int16)

        return model_actions
