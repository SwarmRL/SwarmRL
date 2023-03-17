"""
Test the ML based interaction model.
"""
import os
from pathlib import Path

import flax.linen as nn
import numpy as np
import optax
from numpy.testing import assert_array_almost_equal, assert_array_equal

import swarmrl as srl
from swarmrl.models.interaction_model import Action, Colloid
from swarmrl.models.ml_model import MLModel
from swarmrl.networks.flax_network import FlaxModel
from swarmrl.sampling_strategies.categorical_distribution import CategoricalDistribution


def _action_to_index(action):
    """
    Convert an action to an index for this test.
    """
    if action.force != 0.0:
        return 0
    elif action.torque[-1] == 0.1:
        return 1
    elif action.torque[-1] == -0.1:
        return 2
    else:
        return 3


class FlaxNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)
        return x


class DummyTask:
    """
    Dummy task for the test
    """

    def __call__(self, data):
        """
        Dummy call method.
        """
        return [1.0 for item in data if item.type == 1]


class SecondDummyTask:
    """
    Dummy task for the test
    """

    def __call__(self, data):
        """
        Dummy call method.
        """
        return [5.0 for item in data if item.type == 1]


class TestMLModel:
    """
    Test the ML interaction model to ensure it is functioning correctly.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        observable = srl.observables.PositionObservable(
            box_length=np.array([1000, 1000, 1000])
        )
        network = FlaxModel(
            flax_model=FlaxNet(),
            input_shape=(3,),
            optimizer=optax.sgd(0.001),
            rng_key=6862168,
            sampling_strategy=CategoricalDistribution(),
        )
        second_network = FlaxModel(
            flax_model=FlaxNet(),
            input_shape=(3,),
            optimizer=optax.sgd(0.001),
            rng_key=6862168,
            sampling_strategy=CategoricalDistribution(),
        )
        translate = Action(force=10.0)
        rotate_clockwise = Action(torque=np.array([0.0, 0.0, 15.0]))
        rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -15.0]))
        do_nothing = Action()

        cls.action_space = {
            "RotateClockwise": rotate_clockwise,
            "Translate": translate,
            "RotateCounterClockwise": rotate_counter_clockwise,
            "DoNothing": do_nothing,
        }

        cls.interaction = MLModel(
            models={"0": network},
            observables={"0": observable},
            tasks={"0": DummyTask()},
            actions={"0": cls.action_space},
        )
        cls.multi_interaction = MLModel(
            models={"0": network, "2": second_network},
            observables={"0": observable, "2": observable},
            tasks={"0": DummyTask(), "2": SecondDummyTask()},
            actions={"0": cls.action_space, "2": cls.action_space},
            record_traj=True,
        )

    def test_species_and_order_handling(self):
        """
        Test species and paricle actions are returned correctly.
        """
        colloid_1 = Colloid(np.array([3, 7, 1]), np.array([0, 0, 1]), 0, 1)
        colloid_2 = Colloid(np.array([1, 1, 0]), np.array([0, 0, -1]), 1, 0)
        colloid_3 = Colloid(np.array([100, 27, 0.222]), np.array([0, 0, 1]), 2, 2)

        actions = self.multi_interaction.calc_action(
            [colloid_1, colloid_2, colloid_3],
        )

        # Check that the second action is correct
        actions[1].force == 0.0
        assert_array_equal(actions[0].torque, np.array([0.0, 0.0, 0.0]))

        # Check reward data
        loaded_data_0 = np.load(".traj_data_0.npy", allow_pickle=True)
        loaded_data_2 = np.load(".traj_data_2.npy", allow_pickle=True)
        loaded_data_0 = loaded_data_0.item()["rewards"]
        loaded_data_2 = loaded_data_2.item()["rewards"]
        assert loaded_data_2 == 5.0
        assert loaded_data_0 == 1.0

        # Clean up files
        os.remove(".traj_data_0.npy")
        os.remove(".traj_data_2.npy")

    def test_file_saving(self):
        """
        Test that classes are saved correctly.
        """
        colloid_1 = Colloid(np.array([3, 7, 1]), np.array([0, 0, 1]), 0, 0)
        colloid_2 = Colloid(np.array([1, 1, 0]), np.array([0, 0, -1]), 1, 0)
        colloid_3 = Colloid(np.array([100, 27, 0.222]), np.array([0, 0, 1]), 2, 0)

        self.interaction.record_traj = True
        self.interaction.calc_action(
            [colloid_1, colloid_2, colloid_3], explore_mode=False
        )

        # Check if the file exists
        data_file = Path(".traj_data_0.npy")
        assert data_file.exists()

        # Check that data is stored correctly
        data = np.load(".traj_data_0.npy", allow_pickle=True)
        data = data.item().get("features")

        # Colloid 1
        assert_array_almost_equal(data[0][0], colloid_1.pos / 1000.0)
        # assert_array_equal(data[0][0].director, colloid_1.director)

        # Colloid 2
        assert_array_almost_equal(data[0][1], colloid_2.pos / 1000.0)
        # assert_array_equal(data[0][1].director, colloid_2.director)

        # Colloid 3
        assert_array_almost_equal(data[0][2], colloid_3.pos / 1000.0)
        # assert_array_equal(data[0][2].director, colloid_3.director)

        # Check for additional colloid addition
        colloid_1 = Colloid(np.array([9, 1, 6]), np.array([0, 0, -1.0]), 0, 0)
        colloid_2 = Colloid(np.array([8, 8, 8]), np.array([0, 0, 1.0]), 1, 0)
        colloid_3 = Colloid(np.array([-4.7, 3, -0.222]), np.array([0, 0, -1.0]), 2, 0)
        self.interaction.calc_action(
            [colloid_1, colloid_2, colloid_3], explore_mode=False
        )

        # Check if the file exists
        data_file = Path(".traj_data_0.npy")
        assert data_file.exists()

        # Check that data is stored correctly
        data = np.load(".traj_data_0.npy", allow_pickle=True)
        data = data.item().get("features")

        # Colloid 1
        assert_array_almost_equal(data[1][0], colloid_1.pos / 1000.0)
        # assert_array_equal(data[1][0].director, colloid_1.director)

        # Colloid 2
        assert_array_almost_equal(data[1][1], colloid_2.pos / 1000.0)
        # assert_array_equal(data[1][1].director, colloid_2.director)

        # Colloid 3
        assert_array_almost_equal(data[1][2], colloid_3.pos / 1000.0)
        # assert_array_equal(data[1][2].director, colloid_3.director)

        self.interaction.record_traj = False
        os.remove(".traj_data_0.npy")
