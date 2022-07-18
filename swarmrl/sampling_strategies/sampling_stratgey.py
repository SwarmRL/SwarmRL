"""
Parent class for sampling strategies.
"""
import jax.numpy as np


class SamplingStrategy:
    """
    Parent class for sampling strategies.
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
        entropy : float
                Shannon entropy of the distribution.
        """
        raise NotImplementedError("Implemented in child classes.")
