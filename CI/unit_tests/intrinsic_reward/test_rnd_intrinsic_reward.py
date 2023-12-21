import unittest as ut
from dataclasses import dataclass

import jax.numpy as np
import optax
from flax import linen as nn
from jax import random

from swarmrl.intrinsic_reward.random_network_distillation import *
from swarmrl.intrinsic_reward.rnd_configs import RNDConfig


class RNDNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        return x


@dataclass
class EpisodeDataDummy:
    features: np.ndarray


class RNDTest(ut.TestCase):
    """
    Test class for the RND intrinsic reward.
    """

    @classmethod
    def setup_class(cls):
        """
        Create models and data for the tests.
        """
        # Define a dummy observation of shape (n_steps, num_colloids, num_features)
        cls.observation = random.uniform(random.PRNGKey(0), (10, 5, 3))
        cls.episode_data = EpisodeDataDummy(features=cls.observation)

        # Compute the number of observations
        cls.n_observation = cls.observation.shape[0] * cls.observation.shape[1]

    def test_shapes(self):
        """
        Test that the shapes of the outputs are correct.
        """
        # Define the RND model
        config = RNDConfig(input_shape=(3,))
        intrinsic_reward = RNDReward(config)

        # Test the reshaping of the inputs
        x = intrinsic_reward._reshape_data(self.observation)
        self.assertEqual(x.shape, (self.n_observation, 3))

        # Test the RND model forward pass
        y = intrinsic_reward.predictor_network(x)
        self.assertEqual(y.shape, (self.n_observation, 32))
        self.assertEqual(
            intrinsic_reward.target_network(x).shape, (self.n_observation, 32)
        )

        # Test the RND shape of compute_distance and metric_results
        y = intrinsic_reward.compute_distance(self.observation)
        self.assertEqual(y.shape, ())
        self.assertEqual(intrinsic_reward.metric_results.shape, (self.n_observation,))

        # Test the RND shape of compute_reward
        y = intrinsic_reward.compute_reward(self.episode_data)
        self.assertEqual(y.shape, ())
