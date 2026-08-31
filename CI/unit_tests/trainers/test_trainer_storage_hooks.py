"""Tests for trainer-level agent lifecycle hooks."""

from swarmrl.agents.agent import Agent
from swarmrl.trainers.episodic_trainer import EpisodicTrainer
from swarmrl.trainers.trainer import Trainer


class _DummyAgent(Agent):
    def __init__(self, particle_type):
        self.particle_type = particle_type
        self.finalize_calls = 0

    def calc_action(self, colloids):
        return []

    def finalize(self):
        self.finalize_calls += 1


def test_trainer_finalize_agents_delegates_to_agents():
    agent_0 = _DummyAgent(particle_type=0)
    agent_1 = _DummyAgent(particle_type=1)
    trainer = Trainer([agent_0, agent_1])

    trainer.finalize_agents()

    assert agent_0.finalize_calls == 1
    assert agent_1.finalize_calls == 1


class _BoundaryAgent(Agent):
    def __init__(self, particle_type=0):
        self.particle_type = particle_type
        self.boundaries = []
        self.finalize_calls = 0

    def on_rollout_end(self, *, terminated, truncated):
        self.boundaries.append((terminated, truncated))
        return [1.0]

    def reset_agent(self, colloids):
        pass

    def finalize(self):
        self.finalize_calls += 1


class _Engine:
    colloids = []

    def integrate(self, episode_length, force_fn):
        pass

    def finalize(self):
        pass


def test_trainer_marks_requested_rollout_truncation_before_update():
    agent = _BoundaryAgent()
    trainer = Trainer([agent])

    trainer.update_rl(truncated=True)

    assert agent.boundaries == [(False, True)]


def test_episodic_trainer_only_truncates_scheduled_reset_boundaries():
    agent = _BoundaryAgent()
    trainer = EpisodicTrainer([agent])

    trainer.perform_rl_training(
        get_engine=lambda system: _Engine(),
        system=object(),
        n_episodes=3,
        episode_length=4,
        reset_frequency=2,
        load_bar=False,
        save_episodic_data=False,
    )

    assert agent.boundaries == [(False, False), (False, True), (False, False)]


def test_one_agent_termination_marks_all_agents_terminal():
    terminating_agent = _BoundaryAgent(particle_type=0)
    other_agent = _BoundaryAgent(particle_type=1)
    trainer = Trainer([terminating_agent, other_agent])

    trainer.update_rl(terminated=True)

    assert terminating_agent.boundaries == [(True, False)]
    assert other_agent.boundaries == [(True, False)]
