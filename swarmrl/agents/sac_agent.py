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
    AgentStorageConfig,
    AgentTrajectoryStorage,
    TransitionStorageConfig,
    TransitionTrajectoryStorage,
)


class SACAgent(Agent):
    """
    Continuous-control SAC agent using a single FlaxModel-managed module.
    Handles off-policy transition storage and JAX-native RNG splitting.

    Each particle experience is stored as an individual replay-buffer transition.

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
        action_mapper: typing.Callable[[np.ndarray], Action | list[Action]],
        loss: SoftActorCriticLoss,
        replay_buffer: ReplayBuffer,
        sampling_strategy: ContinuousSamplingStrategy,
        batch_size: int = 256,
        learning_starts: int = 1000,
        gradient_steps: int = 1,
        train: bool = True,
        seed: int = 42,
        storage_config: AgentStorageConfig | None = None,
        transition_storage_config: TransitionStorageConfig | None = None,
    ):
        """
        Constructor for the soft-actor-critic protocol.

        Parameters
        ----------
        particle_type : int
                Particle ID this RL protocol applies to.
        observable : Observable
                Observable for this particle type and network input
        task : Task
                Task for this particle type to perform.
        action_mapper : Callable
                Maps actions into the needed format.
        loss : Loss (default=SoftActorCriticLoss)
                Loss function to use to update the networks.
        replay_buffer : ReplayBuffer
                Stores the sampled (s,a,r,s') sets for sampling.
        sampling_strategy : ContinuousSamplingStrategy
                Samples actions from the policy output
        batch_size : int
                Number of samples per network update step
        learning_starts : int
                Number of samples to collect before the first update
        gradient_steps : int
                Number of batches drawn per network update call
        train : bool (default=True)
                Flag to indicate if the agent is training.
        seed : int
        storage_config : AgentStorageConfig | None (default=None)
                Optional storage configuration for agent data (log_probs, features...)
                If None, no trajectory data is persisted to file.
        transition_storage_config : TransitionStorageConfig | None (default=None)
                Optional storage configuration for transitions (s,a,r,s') stored in
                the ReplayBuffer. If None, no trajectory data is persisted to file.

        """
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
        self.transition_trajectory_storage = None
        if self.transition_storage_config is not None:
            self.transition_trajectory_storage = TransitionTrajectoryStorage(
                particle_type=self.particle_type,
                out_folder=self.transition_storage_config.out_folder,
                preset=self.transition_storage_config.storage_preset,
                stored_attributes=self.transition_storage_config.stored_attributes,
                allow_existing_file=self.transition_storage_config.allow_existing_file,
                write_chunk_size=self.transition_storage_config.write_chunk_size,
            )

        self.storage_config = storage_config
        self.trajectory_storage = None
        if self.storage_config is not None:
            self.trajectory_storage = AgentTrajectoryStorage(
                particle_type=self.particle_type,
                out_folder=self.storage_config.out_folder,
                preset=self.storage_config.storage_preset,
                stored_attributes=self.storage_config.stored_attributes,
                allow_existing_file=self.storage_config.allow_existing_file,
                write_chunk_size=self.storage_config.write_chunk_size,
            )
        self.rng = jax.random.PRNGKey(seed)

        self._step_count = 0
        self._pending_observation = None
        self._pending_action = None
        self._last_reward = 0.0
        self._learning_starts_logged = False

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
        Computes the current state, samples new per-particle actions, and stages
        the transition batch.
        """
        # 1. Get current state (s_t)
        current_obs = np.asarray(self.observable.compute_observable(colloids))
        if current_obs.ndim == 1:
            current_obs = current_obs[None, :]
        elif current_obs.ndim != 2:
            raise ValueError(
                "SACAgent expects observable arrays with "
                "shape (n_particles, n_features)."
            )

        # 2. Sample new action (a_t)
        self.rng, network_key, sample_key, warmup_key = jax.random.split(
            self.rng, num=4
        )
        state_inputs = {"feature_data": jnp.asarray(current_obs)}
        n_particles = int(current_obs.shape[0])

        if self._step_count < self.learning_starts:
            # Use uniform random actions during the learning_starts warm-up.
            action_dim = self.sampling_strategy.action_dimension
            action_limits = getattr(self.sampling_strategy, "action_limits", None)
            if action_limits is None:
                minval = -1.0
                maxval = 1.0
            else:
                action_limits = jnp.asarray(action_limits)
                minval = action_limits[:, 0]
                maxval = action_limits[:, 1]
            actions_jax = jax.random.uniform(
                warmup_key,
                shape=(n_particles, action_dim),
                minval=minval,
                maxval=maxval,
            )
        else:
            logits_jax = self.network.model.apply(
                {"params": self.network.model_state.params},
                rng_key=network_key,
                method=self.network.model.actor,
                **state_inputs,
            )
            actions_jax, _ = self.sampling_strategy(
                logits=logits_jax,
                rng_key=sample_key,
                calculate_log_probs=False,
                deployment_mode=not self.train,
            )

        action_np = np.asarray(jax.device_get(actions_jax))
        if action_np.ndim == 1:
            action_np = action_np[None, :]

        # 3. Stage (s_t, a_t); reward and next state are attached in calc_reward().
        self._pending_observation = current_obs
        self._pending_action = action_np
        self._step_count += 1

        chosen_actions = []
        for particle_action in action_np:
            mapped_action = self.action_mapper(particle_action)
            if isinstance(mapped_action, list):
                chosen_actions.extend(mapped_action)
            else:
                chosen_actions.append(mapped_action)
        return chosen_actions

    def calc_reward(
        self, colloids: list[Colloid], external_reward: float = 0.0
    ) -> float:
        """
        Computes post-step rewards and closes one replay transition per particle.

        Returns the mean reward across particles as a reporting summary only.
        SAC training uses the per-transition rewards written into the replay
        buffer below, not this aggregated return value.
        """
        rewards = np.asarray(self.task(colloids) + external_reward)
        if rewards.ndim == 0:
            rewards = rewards[None]
        terminated = float(self.task.kill_switch)
        truncated = float(getattr(self.task, "truncated", False))
        self.kill_switch = bool(self.task.kill_switch or truncated)

        # We might cache the next observation as well to reuse for next calc_action
        next_observation = np.asarray(self.observable.compute_observable(colloids))
        if next_observation.ndim == 1:
            next_observation = next_observation[None, :]
        elif next_observation.ndim != 2:
            raise ValueError(
                "SACAgent expects observable arrays with "
                "shape (n_particles, n_features)."
            )

        if (
            self.train
            and self._pending_observation is not None
            and self._pending_action is not None
        ):
            n_particles = int(self._pending_observation.shape[0])
            if rewards.shape[0] != n_particles:
                raise ValueError(
                    "Reward array length must match the number of "
                    "staged particle observations."
                )
            if next_observation.shape[0] != n_particles:
                raise ValueError(
                    "Next-observation batch size must match the number of "
                    "staged particle observations."
                )
            if self._pending_action.shape[0] != n_particles:
                raise ValueError(
                    "Pending action batch size must match the number of "
                    "staged particle observations."
                )

            for particle_idx in range(n_particles):
                transition = Transition(
                    observation=self._pending_observation[particle_idx],
                    action=self._pending_action[particle_idx],
                    reward=float(rewards[particle_idx]),
                    next_observation=next_observation[particle_idx],
                    terminated=terminated,
                    truncated=truncated,
                )
                self.replay_buffer.add(transition)
                if self.transition_trajectory_storage is not None:
                    self.transition_trajectory_storage.write(transition)

        self._pending_observation = None
        self._pending_action = None

        # For logging purposes
        self._last_reward = float(np.mean(rewards))
        return self._last_reward

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

        if not self._learning_starts_logged:
            logger.info(
                f"Learning starts at step {self._step_count} "
                f"(learning_starts={self.learning_starts}).",
            )
            self._learning_starts_logged = True

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

    def finalize(self) -> None:
        """Finalize any configured agent and transition trajectory storages."""
        super().finalize()
        if self.transition_trajectory_storage is not None:
            self.transition_trajectory_storage.finalize()

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
