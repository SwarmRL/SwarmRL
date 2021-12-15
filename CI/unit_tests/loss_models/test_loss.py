"""
Run a unit test on the loss module.
"""
import unittest
from swarmrl.loss_models.loss import Loss
import torch


class TestLoss(unittest.TestCase):
    """
    Test the loss functions for RL models.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test class.

        Returns
        -------

        """
        cls.loss = Loss(n_colloids=10)
        cls.rewards = torch.tensor(
            [
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3,
                4, 4, 4, 4, 4,
                5, 5, 5, 5, 5,
                6, 6, 6, 6, 6,
                7, 7, 7, 7, 7,
                8, 8, 8, 8, 8,
                9, 9, 9, 9, 9,
                10, 10, 10, 10, 10
            ]
        )

    def test_expected_returns(self):
        """
        Test the true values of the standard returns.

        Returns
        -------

        """
        true_values = torch.tensor([[5., 4., 3., 2., 1.],
                                    [10., 8., 6., 4., 2.],
                                    [15., 12., 9., 6., 3.],
                                    [20., 16., 12., 8., 4.],
                                    [25., 20., 15., 10., 5.],
                                    [30., 24., 18., 12., 6.],
                                    [35., 28., 21., 14., 7.],
                                    [40., 32., 24., 16., 8.],
                                    [45., 36., 27., 18., 9.],
                                    [50., 40., 30., 20., 10.]])
        discounted_returns = self.loss.compute_discounted_returns(
            rewards=self.rewards, standardize=False, gamma=1
        )
        torch.testing.assert_allclose(true_values, discounted_returns)

    def test_expected_returns_standardized(self):
        """
        Test the expected return method for standardized data.

        Notes
        -----
        Test that the expected returns are correct.
        """
        discounted_returns = self.loss.compute_discounted_returns(rewards=self.rewards)
        self.assertAlmostEqual(torch.mean(discounted_returns[0]).numpy(), 0.0)
        self.assertAlmostEqual(torch.mean(discounted_returns[1]).numpy(), 0.0)
        self.assertAlmostEqual(torch.mean(discounted_returns[2]).numpy(), 0.0)
        self.assertAlmostEqual(torch.mean(discounted_returns[3]).numpy(), 0.0)
        self.assertAlmostEqual(torch.mean(discounted_returns[4]).numpy(), 0.0)
        self.assertAlmostEqual(torch.std(discounted_returns[0]).numpy(), 1.0)
        self.assertAlmostEqual(torch.std(discounted_returns[1]).numpy(), 1.0)
        self.assertAlmostEqual(torch.std(discounted_returns[2]).numpy(), 1.0)
        self.assertAlmostEqual(torch.std(discounted_returns[3]).numpy(), 1.0)
        self.assertAlmostEqual(torch.std(discounted_returns[4]).numpy(), 1.0)
