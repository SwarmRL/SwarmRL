"""
Test the find origin task.
"""
import unittest
import torch
from swarmrl.tasks.find_origin import FindOrigin


class SimulatorEngine:
    """
    A simulator for the engine class for use in testing.
    """

    def __init__(self, particles: int = 10, time_steps: int = 50):
        """"
        Constructor for the simulated engine.

        Parameters
        ----------
        particles : int
                Number of particles in the system.
        time_steps : int
                Number of time steps in the episode.
        """
        self.particles = particles
        self.time_steps = time_steps

    def get_particle_data(self):
        """
        Simulation the get_particle_data method of a real engine.

        Returns
        -------
        data : dict
                A simulated dict of data.
        """
        return {'Unwrapped_Positions': torch.zeros((self.particles, 3))}


class TestFindOrigin(unittest.TestCase):
    """
    Test the find origin task.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare the test class.

        Returns
        -------
        Builds an engine object.
        """
        engine = SimulatorEngine()
        cls.task = FindOrigin(engine=engine)

    def test_compute_particle_reward(self):
        """
        Test the compute_particle_reward method.

        Returns
        -------

        """
        expected_rewards = torch.tensor([[1., 1., 1., -0.],
                                         [1., 1., 1., -0.],
                                         [1., 1., -0., -0.],
                                         [1., 1., -0., 0.],
                                         [0., 0., 0., 0.],
                                         [0., 0., 0., 0.],
                                         [-0., -0., -0., 1.],
                                         [1., 1., 0., 0.],
                                         [0., 1., 0., 1.],
                                         [0., 1., 1., 1.]], dtype=torch.float64)
        particle_positions = torch.tensor(
            [
                [5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [2, 0, 0],
                [0, 5, 0], [0, 4, 0], [0, 3, 0], [0, 2, 0], [0, 2, 0],
                [1, 1, 2], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0],
                [3, 3, 3], [3, 2, 2], [3, 1, 2], [3, 2, 1], [3, 4, 4],
                [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [6, 0, 0],
                [0, 1, 0], [0, 1, 1], [1, 1, 1], [2, 1, 1], [2, 5, 1],
                [1, 2, 4], [4, 2, 1], [2, 4, 1], [4, 2, 1], [1, 1, 1],
                [5, 5, 6], [6, 1, 2], [2, 2, 2], [4, 6, 1], [9, 9, 9],
                [8, 8, 8], [20, 1000, 500000000], [1, 1, 1], [20, 20, 20], [10, 1, 1],
                [1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 1, 1], [0, 0, 0]
            ]
        )
        particle_reward = self.task.compute_particle_reward(particle_positions, 10)
        torch.testing.assert_allclose(particle_reward, expected_rewards)

    def test_compute_reward(self):
        """
        Test compute reward method.

        Returns
        -------

        """
        expected_rewards = torch.tensor([[1., 1., 1., -0.],
                                         [1., 1., 1., -0.],
                                         [1., 1., -0., -0.],
                                         [1., 1., -0., 0.],
                                         [0., 0., 0., 0.],
                                         [0., 0., 0., 0.],
                                         [-0., -0., -0., 1.],
                                         [1., 1., 0., 0.],
                                         [0., 1., 0., 1.],
                                         [0., 1., 1., 1.]], dtype=torch.float64)
        particle_positions = torch.tensor(
            [
                [5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [2, 0, 0],
                [0, 5, 0], [0, 4, 0], [0, 3, 0], [0, 2, 0], [0, 2, 0],
                [1, 1, 2], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0],
                [3, 3, 3], [3, 2, 2], [3, 1, 2], [3, 2, 1], [3, 4, 4],
                [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [6, 0, 0],
                [0, 1, 0], [0, 1, 1], [1, 1, 1], [2, 1, 1], [2, 5, 1],
                [1, 2, 4], [4, 2, 1], [2, 4, 1], [4, 2, 1], [1, 1, 1],
                [5, 5, 6], [6, 1, 2], [2, 2, 2], [4, 6, 1], [9, 9, 9],
                [8, 8, 8], [20, 1000, 500000000], [1, 1, 1], [20, 20, 20], [10, 1, 1],
                [1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 1, 1], [0, 0, 0]
            ]
        )
        rewards = self.task.compute_reward(particle_positions)
        torch.testing.assert_allclose(rewards, expected_rewards)
