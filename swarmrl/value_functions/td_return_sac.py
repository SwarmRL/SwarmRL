"""
Module for computing entropy-regularized Temporal Difference (TD)
returns for Soft Actor-Critic (SAC).

This implementation is based on "Soft Actor-Critic Algorithms and
Applications" by Haarnoja et al. (arXiv:1812.05905, 2018/2019).
"""

from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as np


class TDReturnsSAC:
    """
    Computes the 1-step entropy-regularized TD target for Soft Actor-Critic (SAC) as
    presented in Haarnoja et al., "Soft Actor-Critic Algorithms and Applications"
    (arXiv:1812.05905, 2018/2019).

    Mathematical target formulation:
        y_t = r(s_t, a_t) + gamma * (1 - d_t) *
            [ q_next_min - alpha * log pi(a_{t+1}|s_{t+1}) ]

    Where:
        - r(s_t, a_t) is the immediate reward.
        - gamma is the discount factor.
        - d_t is the terminal flag (1.0 if state is terminal, 0.0 otherwise).
        - q_next_min is the minimum of target twin Q-networks at s_{t+1}
          (clipped double-Q).
        - alpha is the temperature parameter (entropy scaling factor).
        - log pi(a_{t+1}| s_{t+1}) is the log-probability of the next action.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        standardize: bool = False,
        standardize_axis: Optional[Union[int, Tuple[int, ...]]] = 0,
    ):
        """
        Constructor for the SAC TD returns class.

        Parameters
        ----------
        gamma : float
            The discount factor for future rewards.
        standardize : bool
            If True, standardize the calculated TD targets. Generally defaulted to False
            in SAC to maintain stable absolute Q-value scales, but kept as an option.
        standardize_axis : int, tuple of ints, or None
            The axis or axes along which to compute the mean and standard deviation
            during standardization. Defaults to 0.
        """
        self.gamma = gamma
        self.standardize = standardize
        self.standardize_axis = standardize_axis

        # Set by us to stabilize division operations.
        self.eps = np.finfo(np.float32).eps.item()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        rewards: jax.Array,
        q_next_min: jax.Array,
        temperature: Union[float, jax.Array],
        next_log_probs: jax.Array,
        dones: jax.Array,
    ) -> jax.Array:
        """
        Computes the 1-step entropy-regularized Bellman target for SAC.

        Parameters
        ----------
        rewards : jax.Array
            Immediate rewards earned by the agent.
            Expected shape: (batch_size,) or (batch_size, 1)
        q_next_min : jax.Array
            The minimum predicted Q-value from the target twin Q-networks at state
            s_{t+1} (min(Q_target1, Q_target2)). Expected shape: Matches `rewards`.
        temperature : float or jax.Array
            The entropy temperature parameter (alpha) regulating exploration.
            Expected shape: Scalar or matching `rewards` batch dimension.
        next_log_probs : jax.Array
            Log-probabilities of the action chosen in the next state s_{t+1}.
            Expected shape: Matches `rewards`.
        dones : jax.Array
            Terminal flags (1.0 for terminal transition, 0.0 otherwise).
            Expected shape: Matches `rewards`. Must be float type for math operations.

        Returns
        -------
        targets : jax.Array
            The computed 1-step soft TD targets. Same shape as `rewards`.
        """
        # Calculate the soft state value of the next state:
        # V(s_{t+1}) = min(Q_target1, Q_target2) - alpha * log_pi(a_{t+1}|s_{t+1})
        soft_value_next = q_next_min - temperature * next_log_probs

        # Calculate the TD target: y_t = R_t + gamma * (1 - d_t) * V(s_{t+1})
        # If dones is 1.0 (terminal state), the future soft value is masked to 0.0.
        targets = rewards + (1.0 - dones) * self.gamma * soft_value_next
        if self.standardize:
            # We use keepdims=True to preserve the dimensions of the original array.
            # This ensures robust broadcasting regardless of which axes are selected
            # for standardization.
            mean_vector = np.mean(targets, axis=self.standardize_axis, keepdims=True)
            std_vector = (
                np.std(targets, axis=self.standardize_axis, keepdims=True) + self.eps
            )

            targets = (targets - mean_vector) / std_vector

        return targets
