"""Tests for generic storage hooks on the Agent base class."""

from swarmrl.agents.agent import Agent


class _DummyAgent(Agent):
    def calc_action(self, colloids):
        return []


class _DummyStorage:
    def __init__(self):
        self.writes = []

    def write(self, trajectory):
        self.writes.append(trajectory)


class TestAgentStorageHooks:
    def test_storage_hooks_are_noop_without_backend(self):
        agent = _DummyAgent()

        agent.persist_trajectory({"step": 1})

    def test_storage_hooks_delegate_to_backend(self):
        agent = _DummyAgent()
        storage = _DummyStorage()
        trajectory_data = {"step": 3}

        agent.trajectory_storage = storage
        agent.persist_trajectory(trajectory_data)

        assert storage.writes == [trajectory_data]
