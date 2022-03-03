"""
Implement an MLP model.
"""
import numpy as np
import torch

from swarmrl.networks.network import Network


class MLP(Network):
    """
    Implement a multi-layer perceptron.

    Attributes
    ----------
    model : torch.nn.Module
            A stack of torch layers. This is essentially your network but the training
            and selection of the output will be handled by the MLP class.
    loss: torch.nn.module
                Loss function for this model.
    optimizer : torch.nn.Module
                Optimizer for this model.
    """

    def __init__(
        self,
        layer_stack: torch.nn.Module,
    ):
        """
        Construct the model.

        Parameters
        ----------
        layer_stack : torch.nn.Module
                Stack of torch layers e.g.
                torch.nn.Sequential(torch.nn.Linear(), torch.nn.ReLU(), ...)
                The input and output dimension of these models should be correct. The
                output dimension should be > 1, one of these outputs will be selected
                based on some distribution.
        """
        super(MLP, self).__init__()
        self.model = layer_stack

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
        # for loss in loss_vector:
        for loss in loss_vector:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass on the model.

        Parameters
        ----------
        state : torch.Tensor
                Current state of the environment on which predictions should be made.

        Returns
        -------
        state : torch.Tensor
                All possibilities computed.
        """
        possibilities = self.model(state)
        return possibilities
