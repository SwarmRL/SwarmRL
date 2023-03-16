"""
Parent class for exploration modules.
"""
import jax.numpy as np


class ExplorationPolicy:
    """
    Parent class for exploration policies.
    """

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
        raise NotImplementedError
