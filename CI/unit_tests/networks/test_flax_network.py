"""
Uni test for the Flax network
"""
from pathlib import Path

import flax.linen as nn
import jax
import numpy as np
import optax

import swarmrl as srl
from swarmrl.networks import FlaxModel


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

        cls.network = ActorNet()

    def test_saving_and_reloading(self):
        """
        Test that one can save and reload the model.
        """
        # Create a model and export it.
        pre_save_model = FlaxModel(
            flax_model=self.network,
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
            flax_model=self.network,
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
