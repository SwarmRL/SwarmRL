"""Action selection orchestration for discrete and continuous stacks."""

import jax
import jax.numpy as np

from swarmrl.exploration_policies.exploration_policy import (
    ContinuousExplorationPolicy,
    DiscreteExplorationPolicy,
    ExplorationPolicy,
)
from swarmrl.sampling_strategies.sampling_strategy import (
    ContinuousSamplingStrategy,
    DiscreteSamplingStrategy,
    SamplingStrategy,
)


class ActionSelector:
    """Compose sampling + exploration for action selection."""

    def __init__(
        self,
        sampling_strategy: SamplingStrategy,
        exploration_policy: ExplorationPolicy,
    ) -> None:
        self.sampling_strategy = sampling_strategy
        self.exploration_policy = exploration_policy
        self.mode = self._infer_mode(sampling_strategy, exploration_policy)

    @staticmethod
    def _infer_mode(
        sampling_strategy: SamplingStrategy,
        exploration_policy: ExplorationPolicy,
    ) -> str:
        sampling_is_discrete = isinstance(sampling_strategy, DiscreteSamplingStrategy)
        sampling_is_continuous = isinstance(
            sampling_strategy, ContinuousSamplingStrategy
        )
        explore_is_discrete = isinstance(exploration_policy, DiscreteExplorationPolicy)
        explore_is_continuous = isinstance(
            exploration_policy, ContinuousExplorationPolicy
        )

        if sampling_is_discrete and explore_is_discrete:
            return "discrete"
        if sampling_is_continuous and explore_is_continuous:
            return "continuous"

        raise ValueError(
            "Sampling strategy and exploration policy modes must match. "
            f"Got sampling={type(sampling_strategy).__name__}, "
            f"exploration={type(exploration_policy).__name__}."
        )

    def select(
        self,
        logits: np.ndarray,
        deployment_mode: bool,
        sampling_key,
        exploration_key,
    ) -> tuple:
        if self.mode == "discrete":
            indices = self.sampling_strategy(logits, rng_key=sampling_key)
            log_probs = jax.nn.log_softmax(logits)

            if not deployment_mode:
                exploration_seed = int(
                    jax.random.randint(exploration_key, (), 0, 2**31 - 1)
                )
                indices = self.exploration_policy(
                    indices, logits.shape[-1], exploration_seed
                )
            expanded_indices = np.expand_dims(indices, axis=-1)
            chosen_log_probs = np.take_along_axis(log_probs, expanded_indices, axis=-1)
            chosen_log_probs = np.squeeze(chosen_log_probs, axis=-1)

            return indices, chosen_log_probs

        if self.mode == "continuous":
            calculate_log_probs = not deployment_mode
            sampled_actions, log_probs = self.sampling_strategy(
                logits=logits,
                subkey=sampling_key,
                calculate_log_probs=calculate_log_probs,
                deployment_mode=deployment_mode,
            )
            actions = sampled_actions
            if not deployment_mode:
                actions = self.exploration_policy(sampled_actions, exploration_key)
            return actions, log_probs

        raise ValueError(f"Unsupported action selection mode: {self.mode}")
