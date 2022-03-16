"""
Test the ML based interaction model.
"""
import numpy as np
import torch

import swarmrl as srl
from swarmrl.models.ml_model import MLModel
from swarmrl.networks import MLP


class DummyColloid:
    """
    Dummy colloid class for the test.
    """

    pos = np.array([1, 1, 1])


class TestMLModel:
    """
    Test the ML interaction model to ensure it is functioning correctly.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        torch.manual_seed(0)  # set seed for reproducibility.

        # simple model for testing.
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 4),
        )

        model = model.double()
        network = MLP(layer_stack=model)
        observable = srl.observables.PositionObservable()
        cls.interaction = MLModel(
            model=network, observable=observable
        )

    def test_force_selection(self):
        """
        Test that the expected force output is returned by the model.
        """
        colloid = DummyColloid()
        action = self.interaction.calc_action([colloid])

        assert action[0].force == 10.0

    def test_negative_torque(self):
        """
        Test that I can always get a negative torque for a fixed input.
        """
        torch.manual_seed(5)
        colloid = DummyColloid()
        action = self.interaction.calc_action([colloid])

        np.testing.assert_array_equal(action[0].torque, [0.0, 0.0, -0.1])

    def test_positive_torque(self):
        """
        Test that I can always get a positive torque for a fixed input.
        """
        torch.manual_seed(3)
        colloid = DummyColloid()
        action = self.interaction.calc_action([colloid])

        np.testing.assert_array_equal(action[0].torque, [0.0, 0.0, 0.1])

    def test_no_action(self):
        """
        Test that I can get a no action for a fixed input.
        """
        torch.manual_seed(2)
        colloid = DummyColloid()
        action = self.interaction.calc_action([colloid])

        assert action[0].force == 0.0
        np.testing.assert_array_equal(action[0].torque, [0.0, 0.0, 0.0])
        assert action[0].new_direction is None
