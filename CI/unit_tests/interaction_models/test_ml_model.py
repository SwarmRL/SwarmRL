"""
Test the ML based interaction model.
"""
import os
from pathlib import Path

import flax.linen as nn
import numpy as np
import optax
from numpy.testing import assert_array_almost_equal

import swarmrl as srl
from swarmrl.models.interaction_model import Action
from swarmrl.models.ml_model import MLModel
from swarmrl.networks.flax_network import FlaxModel
from swarmrl.sampling_strategies.categorical_distribution import CategoricalDistribution


class DummyColloid:
    """
    Dummy colloid class for the test.
    """

    def __init__(self, pos=np.array([1, 1, 0]), director=np.array([0, 0, 1])):
        self.pos = pos
        self.director = director
        self.type = 0


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

    def __call__(
        self,
        observable: np.ndarray,
        colloid: object,
        colloids: list,
        other_colloids: list,
    ):
        """
        Dummy call method.
        """
        return 1.0


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
        translate = Action(force=10.0)
        rotate_clockwise = Action(torque=np.array([0.0, 0.0, 15.0]))
        rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -15.0]))
        do_nothing = Action()

        action_space = {
            "RotateClockwise": rotate_clockwise,
            "Translate": translate,
            "RotateCounterClockwise": rotate_counter_clockwise,
            "DoNothing": do_nothing,
        }

        cls.interaction = MLModel(
            models={"0": network},
            observables={"0": observable},
            tasks={"0": DummyTask()},
            actions={"0": action_space},
        )

    def test_file_saving(self):
        """
        Test that classes are saved correctly.
        """
        colloid_1 = DummyColloid(np.array([3, 7, 1]), np.array([0, 0, 1]))
        colloid_2 = DummyColloid(np.array([1, 1, 0]), np.array([0, 0, -1]))
        colloid_3 = DummyColloid(np.array([100, 27, 0.222]), np.array([0, 0, 1]))

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
        colloid_1 = DummyColloid(np.array([9, 1, 6]), np.array([0, 0, -1.0]))
        colloid_2 = DummyColloid(np.array([8, 8, 8]), np.array([0, 0, 1.0]))
        colloid_3 = DummyColloid(np.array([-4.7, 3, -0.222]), np.array([0, 0, -1.0]))
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

    def _test_force_selection(self):
        """
        Test that the model returns actions with equal probability.

        At first initialization, the actor should output roughly even actions. In this
        test, we compute the action 1000 times and check the distribution turns out to
        be approximately 0.25 for each.

        Notes
        -----
        We could make this stricter but numpy array equal allows only for this decimal
        check. If we used a difference the value could be constrained to around
        0.23 - 0.27.
        """

        colloid = DummyColloid()

        actions = []
        for _ in range(1000):
            action = self.interaction.calc_action([colloid], explore_mode=False)
            actions.append(_action_to_index(action[0]))

        actions = np.array(actions)

        action_0_freq = np.count_nonzero(actions == 0)
        action_1_freq = np.count_nonzero(actions == 1)
        action_2_freq = np.count_nonzero(actions == 2)
        action_3_freq = np.count_nonzero(actions == 3)

        freq_array = (
            np.array([action_0_freq, action_1_freq, action_2_freq, action_3_freq])
            / 1000
        )
        ref_array = np.array([0.25, 0.25, 0.25, 0.25])
        assert_array_almost_equal(freq_array, ref_array, decimal=1)
