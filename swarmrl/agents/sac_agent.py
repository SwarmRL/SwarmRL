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
from swarmrl.networks.flax_network import FlaxModel
from swarmrl.observables.observable import Observable
from swarmrl.replay_buffer.replay_buffer import ReplayBuffer
from swarmrl.replay_buffer.transition import Transition
from swarmrl.sampling_strategies.sampling_strategy import ContinuousSamplingStrategy
from swarmrl.tasks.task import Task
from swarmrl.utils.storage_utils import (
    TransitionStorageConfig,
    TransitionTrajectoryStorage,
)


class SACAgent(Agent):
    """
    Continuous-control SAC agent using a single FlaxModel-managed module.
    Handles off-policy transition storage and JAX-native RNG splitting.

    The provided FlaxModel must wrap a user-defined Flax module that exposes
    explicit `actor(...)`, `critic(...)`, and `alpha()` methods, because SAC
    invokes these subpaths separately during action selection, critic updates,
    and temperature updates.
    """

    def __init__(
        self,
        particle_type: int,
        network: FlaxModel,
        task: Task,
        observable: Observable,
        action_mapper: typing.Callable[[np.ndarray], list[Action]],
        loss: SoftActorCriticLoss,
        replay_buffer: ReplayBuffer,
        sampling_strategy: ContinuousSamplingStrategy,
        batch_size: int = 256,
        learning_starts: int = 1000,
        gradient_steps: int = 1,
        train: bool = True,
        seed: int = 42,
        transition_storage_config: TransitionStorageConfig | None = None,
    ):
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.action_mapper = action_mapper
        self.sampling_strategy = sampling_strategy

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
        self.rng = jax.random.PRNGKey(seed)

        self._step_count = 0
        self._pending_observation = None
        self._pending_action = None
        self._last_reward = 0.0

        self._validate_sac_network_contract()
        if "critic" not in self.network.target_params:
            self.network.target_params["critic"] = jax.tree_util.tree_map(
                lambda x: x, self.network.model_state.params
            )

    def __name__(self) -> str:
        return "SACAgent"

    def _validate_sac_network_contract(self):
        """
        Validate that the wrapped Flax module exposes the SAC method contract.
        SAC does not use a single __call__ path. It must be able to invoke the
        policy, twin critic, and temperature separately from one shared module.
        """
        if not isinstance(self.network, FlaxModel):
            raise TypeError("SACAgent requires a FlaxModel network.")

        required_methods = ("actor", "critic", "alpha")
        for method_name in required_methods:
            if not hasattr(self.network.model, method_name):
                raise ValueError(
                    f"SAC requires the Flax module to define '{method_name}(...)'."
                )

    def reset_agent(self, colloids: list[Colloid]):
        """Resets the observable, tasks, and clears the pending step memory."""
        self.observable.initialize(colloids)
        self.task.initialize(colloids)
        self._pending_observation = None
        self._pending_action = None
        self.kill_switch = False

    def calc_action(self, colloids: list[Colloid]) -> list[Action]:
        """
        Computes the current state, samples a new action, and stages the transition.
        """
        # 1. Get current state (s_t)
        current_obs = self.observable.compute_observable(colloids)

        # 2. Sample new action (a_t)
        self.rng, network_key, sample_key, warmup_key = jax.random.split(
            self.rng, num=4
        )
        if isinstance(current_obs, dict):
            state_inputs = jax.tree_util.tree_map(
                lambda x: jnp.expand_dims(x, axis=0), current_obs
            )
        else:
            state_inputs = {"feature_data": jnp.expand_dims(current_obs, axis=0)}

        if self._step_count < self.learning_starts:
            # Use uniform random actions during the learning_starts warm-up.
            action_dim = self.sampling_strategy.action_dimension
            actions_jax = jax.random.uniform(
                warmup_key,
                shape=(1, action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        else:
            logits_jax = self.network.model.apply(
                {"params": self.network.model_state.params},
                rng_key=network_key,
                method=self.network.model.actor,
                **state_inputs,
            )

            # Sample new actions
            actions_jax, _ = self.sampling_strategy(
                logits=logits_jax,
                rng_key=sample_key,
                calculate_log_probs=False,
                deployment_mode=not self.train,
            )

        action_np = np.asarray(jax.device_get(actions_jax))[0]

        # 3. Stage (s_t, a_t); reward and next state are attached in calc_reward().
        self._pending_observation = current_obs
        self._pending_action = action_np
        self._step_count += 1

        chosen_actions = self.action_mapper(action_np)
        return chosen_actions

    def calc_reward(
        self, colloids: list[Colloid], external_reward: float = 0.0
    ) -> float:
        """
        Computes the post-step reward and closes the staged replay transition.
        """
        reward = float(self.task(colloids) + external_reward)
        terminated = float(self.task.kill_switch)
        self.kill_switch = self.task.kill_switch

        # We might cache the next observation as well to reuse for next calc_action
        next_observation = self.observable.compute_observable(colloids)

        if (
            self.train
            and self._pending_observation is not None
            and self._pending_action is not None
        ):
            transition = Transition(
                observation=self._pending_observation,
                action=self._pending_action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
            )
            self.replay_buffer.add(transition)
            self.persist_trajectory(transition)

        self._last_reward = reward
        return reward

    def update_agent(self) -> tuple[float, bool]:
        """
        Samples from the ReplayBuffer and triggers the compiled Loss/Update step.
        """
        killed = self.kill_switch

        if (
            not self.train
            or self._step_count < self.learning_starts
            or not self.replay_buffer.can_sample(self.batch_size)
        ):
            return self._last_reward, killed

        for _ in range(self.gradient_steps):
            # 1. Sample Replay buffer
            batch = self.replay_buffer.sample(self.batch_size)

            # 2. Inject fresh RNG keys into batch payload
            self.rng, actor_rng, next_actor_rng = jax.random.split(self.rng, num=3)
            batch["actor_rng"] = actor_rng
            batch["next_actor_rng"] = next_actor_rng

            # 3. Compute loss and immediately apply updates
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
