import numpy as np

from swarmrl.actions import Action
from swarmrl.agents.actor_critic import ActorCriticAgent
from swarmrl.components import Colloid
from swarmrl.force_functions import ForceFunction
from swarmrl.tasks import Task
from swarmrl.utils import TrajectoryInformation


class _Network:
    def compute_action(self, observables):
        n_particles = len(observables)
        return np.zeros(n_particles, dtype=int), np.zeros(n_particles)


class _StatefulObservable:
    def __init__(self):
        self.calls = 0

    def initialize(self, colloids):
        self.calls = 0

    def compute_observable(self, colloids):
        self.calls += 1
        return np.array([[colloid.pos[0]] for colloid in colloids])


class _Task(Task):
    def __init__(self, terminate=False):
        super().__init__(particle_type=0)
        self.terminate = terminate

    def __call__(self, colloids):
        self.kill_switch = self.terminate
        return np.ones(len(colloids))


def _colloids(x):
    return [
        Colloid(
            pos=np.array([x, 0.0, 0.0]),
            director=np.array([1.0, 0.0, 0.0]),
            id=0,
            velocity=np.zeros(3),
            type=0,
        )
    ]


def _agent(terminate=False):
    return ActorCriticAgent(
        particle_type=0,
        network=_Network(),
        task=_Task(terminate=terminate),
        observable=_StatefulObservable(),
        actions={"idle": Action()},
    )


def test_trajectory_information_has_transition_boundary_fields():
    trajectory = TrajectoryInformation(particle_type=0)

    assert trajectory.terminated == []
    assert trajectory.truncated == []
    assert trajectory.final_observation is None


def test_reward_records_resulting_observation_and_aligned_status():
    agent = _agent(terminate=False)
    agent.reset_agent(_colloids(0.0))

    agent.calc_action(_colloids(0.0))
    agent.calc_reward(_colloids(1.0))

    assert agent.trajectory.terminated == [False]
    assert agent.trajectory.truncated == [False]
    np.testing.assert_array_equal(agent.trajectory.final_observation, [[1.0]])
    assert agent.observable.calls == 2


def test_final_observation_is_reused_for_next_rollout_action():
    agent = _agent(terminate=False)
    agent.reset_agent(_colloids(0.0))
    agent.calc_action(_colloids(0.0))
    agent.calc_reward(_colloids(1.0))
    final_observation = agent.trajectory.final_observation

    agent.reset_trajectory(preserve_pending_observation=True)
    agent.calc_action(_colloids(1.0))

    assert agent.observable.calls == 2
    np.testing.assert_array_equal(agent.trajectory.features[0], final_observation)


def test_rollout_end_prioritizes_termination_over_truncation():
    agent = _agent(terminate=False)
    agent.reset_agent(_colloids(0.0))
    agent.calc_action(_colloids(0.0))
    agent.calc_reward(_colloids(1.0))
    agent.update_agent = lambda: [2.0]

    agent.on_rollout_end(terminated=True, truncated=True)

    assert agent.trajectory.terminated == [True]
    assert agent.trajectory.truncated == [False]


def test_task_termination_updates_force_function_immediately():
    agent = _agent(terminate=True)
    agent.reset_agent(_colloids(0.0))
    force_fn = ForceFunction(agents={"0": agent})
    force_fn.calc_action(_colloids(0.0))

    force_fn.calc_reward(_colloids(1.0))

    assert force_fn.kill_switch is True
    assert agent.trajectory.terminated == [True]


def test_rollout_end_hook_owns_actor_critic_boundary_and_update():
    agent = _agent(terminate=False)
    agent.reset_agent(_colloids(0.0))
    agent.calc_action(_colloids(0.0))
    agent.calc_reward(_colloids(1.0))
    agent.update_agent = lambda: [2.0]

    result = agent.on_rollout_end(terminated=False, truncated=True)

    assert agent.trajectory.terminated == [False]
    assert agent.trajectory.truncated == [True]
    assert result == [2.0]
