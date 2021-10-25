"""
Implement an MLP model.
"""
import torch


class MLP(torch.nn.Module):
    """
    Implement a multi-layer perceptron.
    """
    def __init__(self, layers: list):
        """
        Construct the model.

        Parameters
        ----------
        """
        super(MLP, self).__init__()