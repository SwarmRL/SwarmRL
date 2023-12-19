import unittest as ut
from dataclasses import dataclass

import jax.numpy as np
import optax
from flax import linen as nn
from jax import random

import swarmrl
from swarmrl.intrinsic_reward.random_network_distillation import RNDReward
from swarmrl.intrinsic_reward.rnd_configs import RNDConfig, RNDLaRConfig


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

    def test_reward_decrease(self):
        """
        Test whether the intrinsic reward decreases over training.
        """
        # Define the RND model using simple training
        config = RNDConfig(input_shape=(3,))
        intrinsic_reward = RNDReward(config)

        reward_before = intrinsic_reward.compute_reward(self.episode_data)
        intrinsic_reward.update(self.episode_data)
        reward_after = intrinsic_reward.compute_reward(self.episode_data)
        self.assertGreater(reward_before, reward_after)

        # Define the RND model using loss-aware reservoir training
        config = RNDLaRConfig(
            input_shape=(3,),
            episode_length=10,
            reservoir_size=200,
        )
        intrinsic_reward = RNDReward(config)

        reward_before = intrinsic_reward.compute_reward(self.episode_data)
        intrinsic_reward.update(self.episode_data)
        reward_after = intrinsic_reward.compute_reward(self.episode_data)
