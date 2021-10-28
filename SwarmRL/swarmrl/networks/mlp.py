"""
Implement an MLP model.
"""
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
            loss_function: torch.nn.Module,
            optimizer: torch.nn.Module
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
        loss_function : torch.nn.module
                Loss function for this model.
        optimizer : torch.nn.Module
                Optimizer for this model.
        """
        super(MLP, self).__init__()
        self.model = layer_stack
        self.loss = loss_function
        self.optimizer = optimizer

    def forward(self, state: torch.Tensor):
        """
        Run the forward pass on the model.

        Parameters
        ----------
        state : torch.Tensor
                Current state of the environment on which predictions should be made.

        Returns
        -------
        state : torch.Tensor
                State to be returned to the simulation.
        """
        possibilities = self.model(state)
        return self.select_state(possibilities)



