import unittest as ut

import optax

from flax import linen as nn
import jax.numpy as np
from jax import random, vmap

import swarmrl
from dataclasses import dataclass

from swarmrl.intrinsic_reward.random_network_distillation import *


class ActoCriticNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        y = nn.Dense(features=1)(x)
        x = nn.Dense(features=4)(x)
        return x, y


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

        actor_critic = ActoCriticNet()
        # Define an exploration policy
        exploration_policy = swarmrl.exploration_policies.RandomExploration(
            probability=0.0
        )

        # Define a sampling_strategy
        sampling_strategy = swarmrl.sampling_strategies.GumbelDistribution()
        cls.network = swarmrl.networks.FlaxModel(
            flax_model=actor_critic,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(3,),
            sampling_strategy=sampling_strategy,
            exploration_policy=exploration_policy,
        )

    def test_shapes(self):
        """
        Test that the shapes of the outputs are correct.
        """
        # Define the RND model
        config = RNDConfig(agent_network=self.network)
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
