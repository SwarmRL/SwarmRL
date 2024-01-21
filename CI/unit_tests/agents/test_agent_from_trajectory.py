import os
import unittest as ut

import numpy as np

import swarmrl.agents
from swarmrl.agents.agent_from_trajectory import AgentFromTrajectory
from swarmrl.components import Colloid


class TestAgentFromTrajectory(ut.TestCase):
    def setUp(self):
        self.harmonic_2d = swarmrl.agents.agent_from_trajectory.harmonic_2d
        self.harmonic_1d = swarmrl.agents.agent_from_trajectory.harmonic_1d

        self.agent_force_function = AgentFromTrajectory(
            force_function=self.harmonic_2d,
            time_slice=0.01,
            gammas=[10, 10],
            acts_on_types=[1],
            params=[1, 1, 0],
        )

        self.trajectory = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
        )
        self.agent_trajectory = AgentFromTrajectory(
            trajectory=self.trajectory,
            time_slice=0.01,
            gammas=[10, 10],
            acts_on_types=[1],
        )

    def test_update_force_function(self):
        self.agent_force_function.update_force_function(self.harmonic_1d)
        self.assertEqual(self.agent_force_function.force_function, self.harmonic_1d)

    def test_load_trajectory(self):
        script_path = os.path.dirname(__file__)
        self.loading_test = AgentFromTrajectory(trajectory=script_path)
        self.assertTrue(np.array_equal(self.loading_test.wanted_pos, self.trajectory))

    def test_force_function(self):
        coll0 = Colloid(
            pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]), id=1, type=1
        )
        coll1 = Colloid(
            pos=np.array([1.0, 0, 0]), director=np.array([1, 0, 0]), id=2, type=0
        )
        actions = self.agent_force_function.calc_action([coll0, coll1])
        self.assertGreater(actions[0].force, 0)
        self.assertEqual(actions[1].force, 0)

    def test_force_trajectory(self):
        coll0 = Colloid(
            pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]), id=1, type=1
        )
        coll1 = Colloid(
            pos=np.array([1.0, 0, 0]), director=np.array([1, 0, 0]), id=2, type=0
        )
        actions = self.agent_trajectory.calc_action([coll0, coll1])
        self.assertGreater(actions[0].force, 1)
        self.assertEqual(actions[1].force, 0)


if __name__ == "__main__":
    ut.main()
