"""
Parent class for the networks.
"""
import jax.numpy as np


class Network:
    """
    A parent class for the networks that will be used.
    """

    def compute_action(self, feature_vector: np.ndarray, explore_mode: bool = False):
        """
        Compute and action from the action space.

        This method will compute and action for an agent and will include the
        exploration vs exploitation strategy.

        Parameters
        ----------
        feature_vector : np.ndarray
                Feature vector to be used in the network.
        explore_mode : bool
                If true, an exploration vs exploitation function is called.

        Returns
        -------
        action : int
                An integer bounded between 0 and the number of output neurons
                corresponding to the action chosen by the agent.
        """
        raise NotImplementedError("Implemented in child class.")

    def __call__(self, feature_vector: np.ndarray):
        """
        Perform the forward pass on the model.

        Parameters
        ----------
        feature_vector : np.ndarray
                Current state of the agent on which actions should be made.

        Returns
        -------

        """
        raise NotImplementedError("Implemented in child class.")

    def export_model(self, directory: str = "Models"):
        """
        Export the model state to a directory.

        Parameters
        ----------
        directory : str (default=Models)
                Directory in which to save the models. If the directory is not
                in the currently directory, it will be created.

        """
        raise NotImplementedError("Implemented in child class")

    def restore_model_state(self, directory):
        """
        Restore the model state from a file.

        Parameters
        ----------
        directory : str
                Path to the model state.

        Returns
        -------
        Updates the model state.
        """
        raise NotImplementedError("Implemented in child class")

    def update_model(
        self,
        grads,
    ):
        """
        Train the model.

        For jax model grads are used to update a model state directly. This method
        takes the grads and updates the params dict corresponding to the relevant
        model.

        Parameters
        ----------
        grads : dict
                Dict of grads from a jax value_and_grad call.
        """
        raise NotImplementedError("Implemented in child class.")
