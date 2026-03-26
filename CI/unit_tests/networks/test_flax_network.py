"""
Uni test for the Flax network
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import flax.linen as nn
import jax
import numpy as np
import optax

import swarmrl as srl
from swarmrl.agents.actor_critic import ActorCriticAgent
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
            probability=0.0
        )

        # Sampling strategy
        cls.sampling_strategy = srl.sampling_strategies.GumbelDistribution()

        class Network(nn.Module):
            """A simple dense model."""

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=128)(x)
                x = nn.relu(x)
                y = nn.Dense(features=1)(x)
                x = nn.Dense(features=4)(x)
                return x, y

        cls.network = Network()

    def test_compute_action(self):
        """
        Test that the compute action method works.
        """
        model = FlaxModel(
            flax_model=self.network,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(2,),
            sampling_strategy=self.sampling_strategy,
            exploration_policy=self.exploration_policy,
        )
        input_data = np.array([[1.0, 2.0], [4.0, 5.0]])

        data_from_call, value_from_call = model.apply_fn(
            {"params": model.model_state.params}, input_data
        )
        action_indices, action_logits = model.compute_action(input_data)

        # Check shapes
        assert data_from_call.shape == (2, 4)
        assert value_from_call.shape == (2, 1)
        assert action_indices.shape == (2,)
        assert action_logits.shape == (2,)

    def test_call_method(self):
        """
        Test that the call method works.
        """
        model = FlaxModel(
            flax_model=self.network,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(2,),
            sampling_strategy=self.sampling_strategy,
            exploration_policy=self.exploration_policy,
        )
        input_data = np.random.uniform(size=(10, 100, 2))

        data_from_call, value_from_call = model(model.model_state.params, input_data)

        # Check shapes
        assert data_from_call.shape == (10, 100, 4)
        assert value_from_call.shape == (10, 100, 1)

    def test_saving_and_reloading(self):
        """
        Test that one can save and reload the model.
        """
        with TemporaryDirectory() as temp_directory:
            # Create a model and export it.
            pre_save_model = FlaxModel(
                flax_model=self.network,
                optimizer=optax.adam(learning_rate=0.001),
                input_shape=(2,),
                sampling_strategy=self.sampling_strategy,
                exploration_policy=self.exploration_policy,
            )
            input_vector = np.array([[1.0, 2.0]])
            pre_save_logits, pre_save_value = pre_save_model.apply_fn(
                {"params": pre_save_model.model_state.params}, input_vector
            )
            pre_save_model.export_model(filename="model", directory=temp_directory)

            # Check if the model exists
            assert Path(temp_directory, "model.pkl").exists()

            # Create a new model
            post_save_model = FlaxModel(
                flax_model=self.network,
                optimizer=optax.adam(learning_rate=0.001),
                input_shape=(2,),
                sampling_strategy=self.sampling_strategy,
                exploration_policy=self.exploration_policy,
            )
            post_save_logits, post_save_value = post_save_model.apply_fn(
                {"params": post_save_model.model_state.params}, input_vector
            )
            # Check that the logits are different
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                pre_save_logits,
                post_save_logits,
            )

            # Check that the values are different
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                pre_save_value,
                post_save_value,
            )

            # Load the model state
            post_save_model.restore_model_state(
                directory=temp_directory, filename="model"
            )
            post_restore_logits, post_restore_value = post_save_model.apply_fn(
                {"params": post_save_model.model_state.params}, input_vector
            )

            np.testing.assert_array_equal(pre_save_logits, post_restore_logits)
            np.testing.assert_array_equal(pre_save_value, post_restore_value)

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
                jax.tree_util.tree_map(
                    lambda x, y: np.testing.assert_array_equal(x, y), state1, state2
                )

            compare_two_opt_states(pre_save_opt_state, post_restore_opt_state)

    def test_saving_multiple_models(self):
        with TemporaryDirectory() as temp_directory:
            rl_protocols = {}

            for i in range(4):
                network = FlaxModel(
                    flax_model=self.network,
                    optimizer=optax.adam(learning_rate=0.001),
                    input_shape=(2,),
                    sampling_strategy=self.sampling_strategy,
                    exploration_policy=self.exploration_policy,
                )

                protocol = ActorCriticAgent(
                    particle_type=i,
                    network=network,
                    task=None,
                    observable=None,
                    actions=None,
                )

                rl_protocols[f"{i}"] = protocol

            for item, val in rl_protocols.items():
                val.network.export_model(
                    filename=f"ActorModel_{item}", directory=temp_directory
                )

            saved_models = [
                path for path in Path(temp_directory).iterdir() if path.is_file()
            ]
            np.testing.assert_equal(len(saved_models), 4)
