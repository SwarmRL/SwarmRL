"""
Configurations for the Random Network Distillation (RND) intrinsic reward.

Notes
-----
https://arxiv.org/abs/1810.12894
"""

from dataclasses import dataclass, field
from typing import Optional

import optax
from flax import linen as nn
from znnl.distance_metrics import DistanceMetric, OrderNDifference
from znnl.loss_functions import MeanPowerLoss
from znnl.training_strategies.loss_aware_reservoir import LossAwareReservoir
from znnl.training_strategies.simple_training import SimpleTraining


class RNDArchitecture(nn.Module):
    @nn.compact
    def __call__(self, x):
        """
        Call function for the RND architecture.

        This is an example architecture for the RND intrinsic reward.
        It is used as a default architecture in the RNDConfig class, but can be changed
        by passing a different architecture to the RNDConfig constructor.

        Parameters
        ----------
        x : np.ndarray of shape (n_steps, num_colloids, num_features)
                Input to the network.
        """
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        return x


@dataclass
class RNDConfig:
    """
    Example configuration for the RND intrinsic reward.

    Parameters
    ----------
    input_shape : tuple
            Shape of the input to the network.
            It is defined as the shape of the observation.
    target_architecture : nn.Module (default=RNDArchitecture)
            Architecture of the target network. The default architecture is defined in
            the RNDArchitecture class.
    predictor_architecture : nn.Module (default=RNDArchitecture)
            Architecture of the predictor network. The default architecture is defined
            in the RNDArchitecture class.
    training_strategy : SimpleTraining
            Training strategy for training the predictor model from ZnNL.
    distance_metric : DistanceMetric
            Metric to use in the representation comparison.
    n_epochs : int
            Number of epochs to train the predictor model.
    batch_size : int
            Batch size to use in the training.
    clip_rewards : tuple (default=(-5.0, 5.0))
            Tuple of the form (min, max) to clip the rewards from RND.
    training_kwargs : Optional[dict] (default=None)
            Keyword arguments to pass to the training strategy.
    """

    input_shape: tuple
    target_architecture: RNDArchitecture = RNDArchitecture()
    predictor_architecture: RNDArchitecture = RNDArchitecture()
    optimizer = optax.adam(1e-3)
    training_strategy: SimpleTraining = SimpleTraining(
        model=None, loss_fn=MeanPowerLoss(order=2)
    )
    distance_metric: DistanceMetric = OrderNDifference(order=2)
    n_epochs: int = 100
    batch_size: int = 8
    clip_rewards: Optional[tuple] = (-5.0, 5.0)
    training_kwargs: Optional[dict] = field(default_factory=dict)


@dataclass
class RNDLaRConfig:
    """
    Example configuration for the RND intrinsic reward using a loss-aware reservoir.

    More informaion on the loss-aware reservoir can be found in the ZnNL library.

    Parameters
    ----------
    input_shape : tuple
            Shape of the input to the network.
            It is defined as the shape of the observation.
    episode_length : int
            Length of the episode to use in the training. This value is used to
            initialize the reservoir. It defines the number points that have not been
            trained on yet.
    reservoir_size : int (default=1000)
            Size of the reservoir to use in the training.
            The reservoir size defines the number of points that are used as a memory
            buffer for the training.
    target_architecture : nn.Module (default=RNDArchitecture)
            Architecture of the target network. The default architecture is defined in
            the RNDArchitecture class.
    predictor_architecture : nn.Module (default=RNDArchitecture)
            Architecture of the predictor network. The default architecture is defined
            in the RNDArchitecture class.
    distance_metric : DistanceMetric
            Metric to use in the representation comparison.
    n_epochs : int
            Number of epochs to train the predictor model.
    batch_size : int
            Batch size to use in the training.
    clip_rewards : tuple (default=(-5.0, 5.0))
            Tuple of the form (min, max) to clip the rewards from RND.
    training_kwargs : Optional[dict] (default=None)
            Keyword arguments to pass to the training strategy.
    """

    input_shape: tuple
    episode_length: int
    reservoir_size: int
    target_architecture: RNDArchitecture = RNDArchitecture()
    predictor_architecture: RNDArchitecture = RNDArchitecture()
    optimizer = optax.adam(1e-3)
    distance_metric: DistanceMetric = OrderNDifference(order=2)
    n_epochs: int = 100
    batch_size: int = 8
    clip_rewards: Optional[tuple] = (-5.0, 5.0)
    training_kwargs: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        """
        Constructor initializing the training strategy.
        """
        self.training_strategy = LossAwareReservoir(
            model=None,
            loss_fn=MeanPowerLoss(order=2),
            reservoir_size=self.reservoir_size,
            latest_points=self.episode_length,
        )
