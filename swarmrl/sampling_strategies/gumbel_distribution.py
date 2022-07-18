"""
Module for the Gumbel distribution.
"""
from abc import ABC

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.sampling_strategies.sampling_stratgey import SamplingStrategy


class GumbelDistribution(SamplingStrategy, ABC):
    """
    Class for the Gumbel distribution.
    """

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

        Notes
        -----
        See https://arxiv.org/abs/1611.01144 for more information.
        """
        rng = jax.random.PRNGKey(onp.random.randint(0, 1236534623))
        length = len(logits)
        noise = jax.random.uniform(rng, shape=(length,))

        index = np.argmax(logits - np.log(-np.log(noise)))

        probabilities = np.exp(logits)
        entropy_val = -1 * (probabilities * np.log(probabilities)).sum()
        max_entropy = -1 * np.log(1 / len(logits))

        if entropy:
            return index, entropy_val / max_entropy
        else:
            return index
