import unittest as ut

import optax

from flax import linen as nn
import jax.numpy as np
from jax import random, vmap

import swarmrl

from swarmrl.intrinsic_reward.random_network_distillation import RNDReward
from swarmrl.intrinsic_reward.rnd_configs import RNDConfig, RNDLaRConfig
from dataclasses import dataclass


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

    def test_reward_decrease(self):
        """
        Test whether the intrinsic reward decreases over training.
        """
        # Define the RND model using simple training
        config = RNDConfig(agent_network=self.network)
        intrinsic_reward = RNDReward(config)

        reward_before = intrinsic_reward.compute_reward(self.episode_data)
        intrinsic_reward.update(self.episode_data)
        reward_after = intrinsic_reward.compute_reward(self.episode_data)
        self.assertGreater(reward_before, reward_after)

        # Define the RND model using loss-aware reservoir training
        config = RNDLaRConfig(
            agent_network=self.network,
            episode_length=10,
            reservoir_size=200,
        )
        intrinsic_reward = RNDReward(config)

        reward_before = intrinsic_reward.compute_reward(self.episode_data)
        intrinsic_reward.update(self.episode_data)
        reward_after = intrinsic_reward.compute_reward(self.episode_data)
