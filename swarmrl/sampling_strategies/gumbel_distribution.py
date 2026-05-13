"""
Module for the Gumbel distribution.
"""

from abc import ABC

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.sampling_strategies.sampling_strategy import DiscreteSamplingStrategy


class GumbelDistribution(DiscreteSamplingStrategy, ABC):
    """
    Class for the Gumbel distribution.
    """

    def __call__(self, logits: np.ndarray, rng_key=None) -> np.ndarray:
        """
        Sample from the distribution.

        Parameters
        ----------
        logits : np.ndarray (n_colloids, n_dimensions)
                Logits from the model to use in the computation for all colloids.
        rng_key : Optional[jax.Array]
                PRNG key for sampling. If ``None``, a random fallback key is created.

        Returns
        -------
        indices : np.ndarray (n_colloids,)
                Indices of chosen actions for all colloids.

        Notes
        -----
        See https://arxiv.org/abs/1611.01144 for more information.
        """
        rng = (
            jax.random.PRNGKey(onp.random.randint(0, 1236534623))
            if rng_key is None
            else rng_key
        )
        noise = jax.random.uniform(rng, shape=logits.shape)

        indices = np.argmax(logits - np.log(-np.log(noise)), axis=-1)

        return indices
