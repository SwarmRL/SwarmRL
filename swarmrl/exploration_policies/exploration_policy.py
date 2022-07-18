"""
Parent class for exploration modules.
"""


class ExplorationPolicy:
    """
    Parent class for exploration policies.
    """

    def __call__(self, model_action: int, action_space_length: int) -> int:
        """
        Return an index associated with the chosen action.

        Parameters
        ----------
        model_action : int
                Action chosen by the model
        action_space_length : int
                Number of possible actions. Should be 1 higher than the actual highest
                index, i.e if I have actions [0, 1, 2, 3] this number should be 4.

        Returns
        -------
        action : int
                Action chosen after the exploration module has operated.
        """
        raise NotImplementedError
