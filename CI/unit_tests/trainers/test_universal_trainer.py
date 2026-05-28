from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from swarmrl.trainers.universal_trainer import UniversalTrainer


@dataclass
class FakeColloid:
    id: int
    type: int


@dataclass
class FakeEngine:
    colloids: list[FakeColloid]
    integrate_calls: list[int] = field(default_factory=list)
    finalized: bool = False

    def integrate(self, n_slices, force_model):
        self.integrate_calls.append(n_slices)
        force_model.calc_action(self.colloids)

    def finalize(self):
        self.finalized = True


@dataclass
class FakeAgent:
    particle_type: int
    reward: float
    reset_calls: int = 0
    calc_action_calls: int = 0
    update_agent_calls: int = 0
    _last_reward: float = 0.0
    kill_switch: bool = False
    trajectory: object = None

    def reset_agent(self, colloids):
        self.reset_calls += 1

    def calc_action(self, colloids):
        self.calc_action_calls += 1
        self._last_reward = self.reward
        return [object() for _ in colloids if str(_.type) == str(self.particle_type)]

    def update_agent(self):
        self.update_agent_calls += 1
        return self.reward, self.kill_switch


def test_universal_trainer_helper_methods_follow_duck_typing():
    trainer = UniversalTrainer([])

    on_policy_agent = SimpleNamespace(trajectory=SimpleNamespace(rewards=[1.25]))

    off_policy_agent = SimpleNamespace(replay_buffer=object(), _last_reward=2.5)

    explicit_none_agent = SimpleNamespace(replay_buffer=None)

    assert trainer._is_off_policy_agent(on_policy_agent) is False
    assert trainer._is_off_policy_agent(off_policy_agent) is True
    assert trainer._is_off_policy_agent(explicit_none_agent) is False
    assert trainer._get_agent_reward(on_policy_agent) == 1.25
    assert trainer._get_agent_reward(off_policy_agent) == 2.5


def test_universal_trainer_helper_returns_trajectory_reward_when_no_last_reward():
    trainer = UniversalTrainer([])

    agent = SimpleNamespace(trajectory=SimpleNamespace(rewards=[0.75, 1.25]))

    assert trainer._get_agent_reward(agent) == 1.25


def test_universal_trainer_continuous_loop_handles_on_and_off_policy_agents():
    on_policy_agent = FakeAgent(particle_type=1, reward=1.5, trajectory=object())
    on_policy_agent.trajectory = type("Trajectory", (), {"rewards": [1.5]})()
    off_policy_agent = FakeAgent(particle_type=2, reward=2.5)
    off_policy_agent.replay_buffer = object()
    engine = FakeEngine(colloids=[FakeColloid(id=1, type=1), FakeColloid(id=2, type=2)])

    trainer = UniversalTrainer([on_policy_agent, off_policy_agent])

    rewards = trainer.perform_rl_training(
        n_episodes=2,
        episode_length=3,
        load_bar=False,
        system_runner=engine,
    )

    assert np.array_equal(rewards, np.array([12.0, 12.0]))
    assert engine.finalized is True
    assert engine.integrate_calls == [1, 1, 1, 1, 1, 1]
    assert on_policy_agent.reset_calls == 2
    assert off_policy_agent.reset_calls == 2
    assert on_policy_agent.calc_action_calls == 6
    assert off_policy_agent.calc_action_calls == 6
    assert off_policy_agent.update_agent_calls == 6
    assert on_policy_agent.update_agent_calls == 2


def test_universal_trainer_episodic_reset_uses_cycle_tags_and_finalizes_engines():
    agent = FakeAgent(particle_type=1, reward=3.0, trajectory=object())
    agent.trajectory = type("Trajectory", (), {"rewards": [3.0]})()
    created_engines = []

    def get_engine(system, cycle_tag=None):
        engine = FakeEngine([FakeColloid(id=1, type=1)])
        engine.cycle_tag = cycle_tag
        created_engines.append(engine)
        return engine

    trainer = UniversalTrainer([agent])

    rewards = trainer.perform_rl_training(
        n_episodes=2,
        episode_length=1,
        load_bar=False,
        get_engine=get_engine,
        system=object(),
        reset_frequency=1,
        save_episodic_data=True,
    )

    assert np.array_equal(rewards, np.array([3.0, 3.0]))
    assert [engine.cycle_tag for engine in created_engines] == ["0", "1"]
    assert all(engine.finalized for engine in created_engines)
    assert agent.reset_calls == 2
    assert agent.update_agent_calls == 2
