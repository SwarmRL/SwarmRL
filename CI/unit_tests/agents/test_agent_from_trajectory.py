import pkgutil
import unittest as ut

import swarmrl.agents

submodule_names = [name for _, name, _ in pkgutil.iter_modules(swarmrl.agents.__path__)]
print(submodule_names)


class TestAgentFromTrajectory(ut.TestCase):
    def setUp(self):
        self.harmonic_2d = swarmrl.agents.agent_from_trajectory.harmonic_2d
        self.harmonic_1d = swarmrl.agents.agent_from_trajectory.harmonic_1d

        self.agent_force_function = (
            swarmrl.agents.agent_from_trajectory.AgentFromTrajectory(
                force_function=self.harmonic_2d,
                time_slice=0.01,
                gammas=[1e5, 1e5],
                acts_on_types=1,
                params=[1, 1, 0],
            )
        )

        self.trajectory = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
        self.agent_trajectory = swarmrl.agents.AgentFromTrajectory(
            trajectory=self.trajectory,
            time_slice=0.01,
            gammas=[1e5, 1e5],
            acts_on_types=1,
        )

    def test_change_force_function(self):
        self.agent_force_function.update_force_function(self.harmonic_1d)
        self.assertEqual(self.agent_force_function.force_function, self.harmonic_1d)


if __name__ == "__main__":
    ut.main()
