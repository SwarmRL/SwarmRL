"""
Random exploration module.
"""

from functools import partial

import jax
import jax.numpy as jnp

from swarmrl.exploration_policies.exploration_policy import DiscreteExplorationPolicy


class RandomExploration(DiscreteExplorationPolicy):
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
        self, model_actions: jnp.ndarray, action_space_length: int, rng_key: jax.Array
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
        rng_key : jax.Array
                Key for JAX random number generation.

        Returns
        -------
        action : jnp.ndarray
                Action chosen after the exploration module has operated for
                each colloid.
        """
        sample_key, action_key = jax.random.split(rng_key)

        replace_mask = (
            jax.random.uniform(sample_key, shape=model_actions.shape) < self.probability
        )

        exploration_actions = jax.random.randint(
            action_key,
            shape=(model_actions.shape[0],),
            minval=0,
            maxval=action_space_length,
        )

        return jnp.where(
            replace_mask,
            exploration_actions,
            model_actions,
        ).astype(jnp.int16)
