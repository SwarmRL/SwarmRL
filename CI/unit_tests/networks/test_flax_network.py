"""
Uni test for the Flax network
"""
import os
from pathlib import Path

import flax.linen as nn
import jax
import numpy as np
import optax

import swarmrl as srl
from swarmrl.networks import FlaxModel
from swarmrl.rl_protocols.actor_critic import ActorCritic


class TestFlaxNetwork:
    """
    Unit test for the Flax network.
    """

    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """
        # Exploration policy
        cls.exploration_policy = srl.exploration_policies.RandomExploration(
            probability=0.1
        )

        # Sampling strategy
        cls.sampling_strategy = srl.sampling_strategies.GumbelDistribution()

        class ActorNet(nn.Module):
            """A simple dense model."""

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=128)(x)
                x = nn.relu(x)
                x = nn.Dense(features=4)(x)
                return x

        class CriticNet(nn.Module):
            """A simple dense model."""

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=128)(x)
                x = nn.relu(x)
                x = nn.Dense(features=1)(x)
                return x

        cls.actor_network = ActorNet()
        cls.critic_network = CriticNet()

    def test_compute_action(self):
        """
        Test that the compute action method works.
        """
        model = FlaxModel(
            flax_model=self.actor_network,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(2,),
            sampling_strategy=self.sampling_strategy,
            exploration_policy=self.exploration_policy,
        )
        input_data = np.array([[1.0, 2.0], [4.0, 5.0]])

        data_from_call = model(input_data)
        action_indices, action_logits = model.compute_action(input_data)

        # Check shapes
        assert data_from_call.shape == (2, 4)
        assert action_indices.shape == (2,)
        assert action_logits.shape == (2,)

    def test_saving_and_reloading(self):
        """
        Test that one can save and reload the model.
        """
        # Create a model and export it.
        pre_save_model = FlaxModel(
            flax_model=self.actor_network,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(2,),
            sampling_strategy=self.sampling_strategy,
            exploration_policy=self.exploration_policy,
        )
        input_vector = np.array([1.0, 2.0])
        pre_save_output = pre_save_model(input_vector)
        pre_save_model.export_model(filename="model", directory="Models")

        # Check if the model exists
        assert Path("Models/model.pkl").exists()

        # Create a new model
        post_save_model = FlaxModel(
            flax_model=self.actor_network,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(2,),
            sampling_strategy=self.sampling_strategy,
            exploration_policy=self.exploration_policy,
        )
        post_save_output = post_save_model(input_vector)

        # Check that the output is different
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            pre_save_output,
            post_save_output,
        )

        # Load the model state
        post_save_model.restore_model_state(directory="Models", filename="model")
        post_restore_output = post_save_model(input_vector)
        np.testing.assert_array_equal(pre_save_output, post_restore_output)

        # Check that the epoch counts are equal
        pre_count = pre_save_model.epoch_count
        post_count = post_save_model.epoch_count
        np.testing.assert_equal(pre_count, post_count)

        # Check that the optimizer steps are equal
        pre_save_opt_step = pre_save_model.model_state.step
        post_restore_opt_step = post_save_model.model_state.step
        np.testing.assert_equal(pre_save_opt_step, post_restore_opt_step)

        # Check that the optimizer states are equal
        pre_save_opt_state = pre_save_model.model_state.opt_state
        post_restore_opt_state = post_save_model.model_state.opt_state

        def compare_two_opt_states(state1, state2):
            jax.tree_map(
                lambda x, y: np.testing.assert_array_equal(x, y), state1, state2
            )

        compare_two_opt_states(pre_save_opt_state, post_restore_opt_state)

    def test_saving_multiple_models(self):
        rl_protocols = {}

        for i in range(4):
            actor_model = FlaxModel(
                flax_model=self.actor_network,
                optimizer=optax.adam(learning_rate=0.001),
                input_shape=(2,),
                sampling_strategy=self.sampling_strategy,
                exploration_policy=self.exploration_policy,
            )
            critic_model = FlaxModel(
                flax_model=self.actor_network,
                optimizer=optax.adam(learning_rate=0.001),
                input_shape=(2,),
                sampling_strategy=self.sampling_strategy,
                exploration_policy=self.exploration_policy,
            )

            protocol = ActorCritic(
                particle_type=i,
                actor=actor_model,
                critic=critic_model,
                task=None,
                observable=None,
                actions=None,
            )

            rl_protocols[f"{i}"] = protocol

        for item, val in rl_protocols.items():
            val.actor.export_model(
                filename=f"ActorModel_{item}", directory="MultiModels"
            )
            val.critic.export_model(
                filename=f"CriticModel_{item}", directory="MultiModels"
            )

        count = 0
        # Iterate directory
        for path in os.listdir("MultiModels"):
            # check if current path is a file
            if os.path.isfile(os.path.join("MultiModels", path)):
                count += 1
        np.testing.assert_equal(count, 8)
