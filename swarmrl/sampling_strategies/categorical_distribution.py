"""
Module for the categorical distribution.
"""
from abc import ABC

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.sampling_strategies.sampling_stratgey import SamplingStrategy


class CategoricalDistribution(SamplingStrategy, ABC):
    """
    Class for the Gumbel distribution.
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

    def __call__(self, logits: np.ndarray, entropy: bool = False):
        """
        Sample from the distribution.

        Parameters
        ----------
        logits : np.ndarray
                Logits from the model to use in the computation.
        entropy : bool
                If true, the Shannon entropy of the distribution is returned.

        Returns
        -------
        sample : int
                Index of the selected option in the distribution.
        entropy : float (optional)
                Shannon entropy of the distribution.
        """
        rng = jax.random.PRNGKey(onp.random.randint(0, 1236534623))

        try:
            noise = self.noise(rng, shape=(len(logits),))
        except TypeError:
            # If set to None the noise is just 0
            noise = 0

        index = jax.random.categorical(rng, logits=logits + noise)

        probabilities = np.exp(logits)
        entropy_val = -1 * (probabilities * np.log(probabilities)).sum()
        max_entropy = -1 * np.log(1 / len(logits))

        if entropy:
            return index, entropy_val / max_entropy
        else:
            return index
