"""
Module for the categorical distribution.
"""

from abc import ABC

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.sampling_strategies.sampling_strategy import DiscreteSamplingStrategy


class CategoricalDistribution(DiscreteSamplingStrategy, ABC):
    """
    Class for the categorical distribution.
    """

    def __init__(self, noise: str = "none"):
        """
        Constructor for the categorical distribution.

        Parameters
        ----------
        noise : str
                Noise method to use, options include none, uniform and gaussian.
        """
        noise_dict = {
            "uniform": jax.random.uniform,
            "gaussian": jax.random.normal,
            "none": None,
        }
        try:
            self.noise = noise_dict[noise]
        except KeyError:
            msg = (
                f"Parsed noise method {noise} is not implemented, please choose"
                "from 'none', 'gaussian' and 'uniform'."
            )
            raise KeyError(msg)

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
                Index of the selected option in the distribution.
        """
        rng = (
            jax.random.PRNGKey(onp.random.randint(0, 1236534623))
            if rng_key is None
            else rng_key
        )

        try:
            noise = self.noise(rng, shape=logits.shape)
        except TypeError:
            # If set to None the noise is just 0
            noise = 0

        indices = jax.random.categorical(rng, logits=logits + noise)

        return indices
