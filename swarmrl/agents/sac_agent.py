"""SAC agent with replay-buffer based off-policy updates."""

from __future__ import annotations

import typing

import numpy as np

from swarmrl.actions.actions import Action
from swarmrl.agents.agent import Agent
from swarmrl.components.colloid import Colloid
from swarmrl.losses.sac_loss import SoftActorCriticLoss
from swarmrl.observables.observable import Observable
from swarmrl.replay_buffer.replay_buffer import ReplayBuffer
from swarmrl.replay_buffer.transition import Transition
from swarmrl.tasks.task import Task


class SACAgent(Agent):
    """Continuous-control SAC agent skeleton compatible with SwarmRL trainers."""

    def __init__(
        self,
        particle_type: int,
        actor_network,
        critic_network,
        task: Task,
        observable: Observable,
        action_mapper: typing.Callable[[np.ndarray], Action],
        loss: SoftActorCriticLoss | None = None,
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int = 256,
        learning_starts: int = 1000,
        gradient_steps: int = 1,
        train: bool = True,
    ):
        self.particle_type = particle_type
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.task = task
        self.observable = observable
        self.action_mapper = action_mapper
        self.loss = loss or SoftActorCriticLoss()
        self.replay_buffer = replay_buffer or ReplayBuffer(capacity=100_000)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.gradient_steps = int(gradient_steps)
        self.train = train

        self._step_count = 0
        self._pending_observation = None
        self._pending_action = None
        self._last_reward = 0.0

    def __name__(self) -> str:
        return "SACAgent"

    def reset_agent(self, colloids: typing.List[Colloid]):
        self.observable.initialize(colloids)
        self.task.initialize(colloids)
        self._pending_observation = None
        self._pending_action = None

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        observation = np.asarray(self.observable.compute_observable(colloids))
        policy_action, _ = self.actor_network.compute_action(observables=observation)
        reward = self.task(colloids)

        # Finalize transition once next observation is available.
        if self.train and self._pending_observation is not None:
            self.replay_buffer.add(
                Transition(
                    observation=np.asarray(self._pending_observation),
                    action=np.asarray(self._pending_action),
                    reward=float(np.mean(reward)),
                    next_observation=np.asarray(observation),
                    done=bool(self.task.kill_switch),
                )
            )

        self._pending_observation = observation
        self._pending_action = policy_action
        self._last_reward = float(np.mean(reward))

        action_vectors = np.asarray(policy_action)
        if action_vectors.ndim == 1:
            action_vectors = action_vectors.reshape(1, -1)
        chosen_actions = [self.action_mapper(vec) for vec in action_vectors]

        self.kill_switch = self.task.kill_switch
        self._step_count += 1
        return chosen_actions

    def update_agent(self) -> tuple[float, bool]:
        if not self.train:
            return self._last_reward, self.task.kill_switch

        if self._step_count < self.learning_starts:
            return self._last_reward, self.task.kill_switch

        if not self.replay_buffer.can_sample(self.batch_size):
            return self._last_reward, self.task.kill_switch

        for _ in range(self.gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size)
            if hasattr(self.critic_network, "sac_train_step"):
                self.critic_network.sac_train_step(
                    actor_network=self.actor_network,
                    loss=self.loss,
                    batch=batch,
                )
            else:
                raise NotImplementedError(
                    "critic_network must implement "
                    "sac_train_step(actor_network, loss, batch)."
                )

        return self._last_reward, self.task.kill_switch

    def initialize_network(self):
        if hasattr(self.actor_network, "reinitialize_network"):
            self.actor_network.reinitialize_network()
        if hasattr(self.critic_network, "reinitialize_network"):
            self.critic_network.reinitialize_network()

    def save_agent(self, directory: str):
        self.actor_network.export_model(
            filename=f"{self.__name__()}_actor_{self.particle_type}",
            directory=directory,
        )
        self.critic_network.export_model(
            filename=f"{self.__name__()}_critic_{self.particle_type}",
            directory=directory,
        )

    def restore_agent(self, directory: str):
        self.actor_network.restore_model_state(
            filename=f"{self.__name__()}_actor_{self.particle_type}",
            directory=directory,
        )
        self.critic_network.restore_model_state(
            filename=f"{self.__name__()}_critic_{self.particle_type}",
            directory=directory,
        )
