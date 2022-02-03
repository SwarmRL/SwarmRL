"""
Parent class for the networks.
"""
import torch


class Network(torch.nn.Module):
    """
    A parent class for the networks that will be used.
    """

    def __init__(self, optimizer: torch.optim.Optimizer = None):
        """
        Constructor for the Network parent class.

        This will just call super on the torch Module class.
        """
        super(Network, self).__init__()
        self.optimizer = optimizer
        self.model = torch.nn.Module

    def select_state(self, states: torch.Tensor):
        """
        Select a state based on some probability distribution.

        Parameters
        ----------
        states : torch.Tensor
                States from which to choose.

        Returns
        -------
        state : torch.Tensor
                State to be returned to the simulation.

        Notes
        -----
        TODO: Should be implemented for several distribution.
        """
        return states[0]

    def update_model(self, loss_vector: torch.Tensor):
        """
        Update the model.

        Parameters
        ----------
        loss_vector : torch.Tensor
                Current state of the environment on which predictions should be made.
                The elements of the loss vector MUST be torch tensors in order for the
                backward() method to work.
        """
        raise NotImplementedError("Implemented in child class.")

    def forward(self, state: torch.Tensor):
        """
        Perform the forward pass on the model.

        Parameters
        ----------
        state : torch.Tensor
                Current state of the environment on which predictions should be made.

        Returns
        -------

        """
        raise NotImplementedError("Implemented in child class.")
