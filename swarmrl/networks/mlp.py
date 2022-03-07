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
        self.layer_stack = layer_stack
        self.model = self.layer_stack.apply(self.initialise_weights)

    def update_model(self, loss_vector: torch.Tensor, retain: bool = False):
        """
        Update the model.

        Parameters
        ----------
        retain : bool (default=False)
                If true, retain the graph for further back-propagation on a stale model.
        loss_vector : torch.Tensor
                Current state of the environment on which predictions should be made.
                The elements of the loss vector MUST be torch tensors in order for the
                backward() method to work.
        """
        for i, loss in enumerate(loss_vector):
            if i == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=retain)
        self.optimizer.step()

    def forward(self, state: torch.Tensor):
        """
        Compute the forward pass over the network.
        """
        return self.model(state)

    def initialise_weights(self, model):
        """
        Initialises weights to be in range [-y,y] with y=sqrt(inputs).
        Parameters
        ----------
        model: layer stack

        Returns
        -------
        updated layer stack
        """
        classname = model.__class__.__name__
        # for every Linear layer in a model..
        if classname.find("Linear") != -1:
            # get the number of the inputs
            n = model.in_features
            y = 1.0 / np.sqrt(n)
            model.weight.data.uniform_(-y, y)
            model.bias.data.fill_(0)
