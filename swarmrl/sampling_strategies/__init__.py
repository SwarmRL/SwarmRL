"""
Module for sampling strategies.
"""

from swarmrl.sampling_strategies.categorical_distribution import CategoricalDistribution
from swarmrl.sampling_strategies.gaussian_distribution import (
    ContinuousGaussianDistribution,
)
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.sampling_strategies.sampling_strategy import (
    ContinuousSamplingStrategy,
    DiscreteSamplingStrategy,
    SamplingStrategy,
)

__all__ = [
    SamplingStrategy.__name__,
    DiscreteSamplingStrategy.__name__,
    ContinuousSamplingStrategy.__name__,
    CategoricalDistribution.__name__,
    GumbelDistribution.__name__,
    ContinuousGaussianDistribution.__name__,
]
