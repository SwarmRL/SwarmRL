"""Tests for trainer-level storage lifecycle hooks."""

from swarmrl.agents.agent import Agent
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
