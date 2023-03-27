"""
Parent class for sampling strategies.
"""
import jax.numpy as np


class SamplingStrategy:
    """
    Parent class for sampling strategies.
    """

    def compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute the Shannon entropy of the probabilities.

        Parameters
        ----------
        probabilities : np.ndarray (n_colloids, n_actions)
                Probabilities for each colloid to take specific actions.
        """
        return -np.sum(probabilities * np.log(probabilities))

    def __call__(self, logits: np.ndarray) -> int:
        """
        Sample from the distribution.

        Parameters
        ----------
        logits : np.ndarray (n_colloids, n_dimensions)
                Logits from the model to use in the computation for each colloid.

        Returns
        -------
        sample : int
                Index of the selected option in the distribution.
        """
        raise NotImplementedError("Implemented in child classes.")
