"""SAC agent with replay-buffer based off-policy updates."""

import typing

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from swarmrl.actions.actions import Action
from swarmrl.agents.agent import Agent
from swarmrl.components.colloid import Colloid
from swarmrl.losses.sac_loss import SoftActorCriticLoss
from swarmrl.networks.multi_flax_networks import MultiFlaxModel
from swarmrl.observables.observable import Observable
from swarmrl.replay_buffer.replay_buffer import ReplayBuffer
from swarmrl.replay_buffer.transition import Transition
from swarmrl.tasks.task import Task
from swarmrl.utils.storage_utils import (
    TransitionStorageConfig,
    TransitionTrajectoryStorage,
)


class SACAgent(Agent):
    """
    Continuous-control SAC agent with a MultiFlaxModel container.
    Handles Off-Policy Transition storage and JAX-native RNG splitting.
    """

    def __init__(
        self,
        particle_type: int,
        network: MultiFlaxModel,
        task: Task,
        observable: Observable,
        action_mapper: typing.Callable[[np.ndarray], typing.List[Action]],
        loss: SoftActorCriticLoss,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
        learning_starts: int = 1000,
        gradient_steps: int = 1,
        train: bool = True,
        seed: int = 42,
        transition_storage_config: TransitionStorageConfig | None = None,
    ):
        # SwarmRL Core
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.action_mapper = action_mapper

        # SAC / JAX Core
        self.network = network
        self.loss = loss
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gradient_steps = gradient_steps
        self.train = train

        self.transition_storage_config = transition_storage_config
        self.trajectory_storage = None
        if self.transition_storage_config is not None:
            self.trajectory_storage = TransitionTrajectoryStorage(
                particle_type=self.particle_type,
                out_folder=self.transition_storage_config.out_folder,
                preset=self.transition_storage_config.storage_preset,
                stored_attributes=self.transition_storage_config.stored_attributes,
                allow_existing_file=self.transition_storage_config.allow_existing_file,
                write_chunk_size=self.transition_storage_config.write_chunk_size,
            )

        # Master JAX PRNG Key for this agent
        self.rng = jax.random.PRNGKey(seed)

        # Off-Policy State Tracking
        self._step_count = 0
        self._pending_observation = None
        self._pending_action = None
        self._last_reward = 0.0

    def __name__(self) -> str:
        return "SACAgent"

    def reset_agent(self, colloids: typing.List[Colloid]):
        """Resets the observable, tasks, and clears the pending step memory."""
        self.observable.initialize(colloids)
        self.task.initialize(colloids)
        self._pending_observation = None
        self._pending_action = None
        self.kill_switch = False

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Computes the state, stores the previous transition, and samples a new action.
        """
        # 1. Get current state (s_t) and environment feedback
        current_obs = self.observable.compute_observable(colloids)
        reward = self.task(colloids)
        terminated = float(self.task.kill_switch)
        self.kill_switch = self.task.kill_switch

        # 2. Store full Transition (s_{t-1}, a_{t-1}, r_t, s_t, terminated)
        if self.train and self._pending_observation is not None:
            transition = Transition(
                observation=self._pending_observation,
                action=self._pending_action,
                reward=float(reward),
                next_observation=current_obs,
                terminated=terminated,
            )
            self.replay_buffer.add(transition)
            self.persist_trajectory(transition)

        # 3. Sample new action (a_t)
        self.rng, sample_key = jax.random.split(self.rng)
        # Create 2D array for jax, dummy batch size of 1
        state_inputs = {"feature_data": jnp.expand_dims(current_obs, axis=0)}

        if self._step_count < self.learning_starts:
            # Optional: Random sampling for the first X steps for better exploration.
            # You could call your sampling strategy directly here with a completely
            # uniform logits array, or let the uninitialized network act randomly.
            pass

        actor_params = self.network.states["actor"].params

        actions_jax, _ = self.network.networks["actor"].apply(
            {"params": actor_params},
            rng_key=sample_key,
            **state_inputs,
        )

        action_np = np.asarray(jax.device_get(actions_jax))[0]

        # 4. Update pending state for the next step
        self._pending_observation = current_obs
        self._pending_action = action_np
        self._last_reward = reward
        self._step_count += 1

        chosen_actions = self.action_mapper(action_np)
        return chosen_actions

    def update_agent(self) -> tuple[float, bool]:
        """
        Samples from the ReplayBuffer and triggers the compiled Loss/Update step.
        """
        killed = self.kill_switch

        if not self.train or not self.replay_buffer.can_sample(self.batch_size):
            return self._last_reward, killed

        for _ in range(self.gradient_steps):
            # 1. Sample Replay buffer
            batch = self.replay_buffer.sample(self.batch_size)

            # 2. Inject fresh RNG keys into batch payload
            self.rng, actor_rng, next_actor_rng = jax.random.split(self.rng, num=3)
            batch["actor_rng"] = actor_rng
            batch["next_actor_rng"] = next_actor_rng

            # 3. Compute loss and immediately apply updates
            #   (encapsulated in SACLoss)
            metrics = self.loss.compute_loss(self.network, batch)
            logger.debug(metrics)
        return self._last_reward, killed

    def initalize_network(self):
        if hasattr(self.network, "reinitialize_network"):
            self.network.reinitialize_network()

    def save_agent(self, directory: str):
        self.network.export_model(
            filename=f"{self.__name__()}_{self.particle_type}",
            directory=directory,
        )

    def restore_agent(self, directory: str):
        self.network.restore_model_state(
            filename=f"{self.__name__()}_{self.particle_type}",
            directory=directory,
        )
