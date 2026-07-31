"""
Module for the expected returns value function.
"""

from functools import partial

import jax
import jax.numpy as jnp


class GAE:
    """
    Class for the expected returns.
    """

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        """
        Constructor for the generalized advantage estimate  class

        Parameters
        ----------
        gamma : float
                A decay factor for the values of the task each time step.
        lambda_ : float
                A decay factor that describes the amount of bias included in the
                advantage calculation.

        Notes
        -----
        See https://arxiv.org/pdf/1506.02438.pdf for more information.
        """
        self.gamma = gamma
        self.lambda_ = lambda_

        # Set by us to stabilize division operations.
        self.eps = jnp.finfo(jnp.float32).eps.item()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        rewards: jax.Array,
        values: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Call function for the advantage.
        Parameters
        ----------
        rewards : jax.Array (n_time_steps, n_particles)
                A numpy array of rewards to use in the calculation.
        values : jax.Array (n_time_steps + 1, n_particles)
                Critic predictions for each action state and the final observation.
        terminated : jax.Array (n_time_steps,)
                Whether each transition ended in a terminal task state.
        truncated : jax.Array (n_time_steps,)
                Whether each transition ended because the environment was reset.
        Returns
        -------
        advantages : jax.Array (n_time_steps, n_particles)
                Standardized generalized advantages.
        returns : jax.Array (n_time_steps, n_particles)
                Boundary-aware critic targets before advantage standardization.
        """
        done = jnp.logical_or(terminated, truncated)

        def advantage_step(next_advantage, transition):
            """
            Boundary-aware generalized advantage estimate:
            $delta_t = r_t + gamma * (1 - terminated_t) * V_(t+1) - V_t$.
            $A_t = delta_t + gamma * lambda * (1 - done_t) * A_(t+1)$.
            Terminations use zero bootstrap; both environment boundaries stop
            advantage propagation.
            """
            reward, value, next_value, is_terminated, is_done = transition
            bootstrap_value = jnp.where(
                is_terminated,
                jnp.zeros_like(next_value),
                next_value,
            )
            delta = reward + self.gamma * bootstrap_value - value
            continued_advantage = jnp.where(
                is_done,
                jnp.zeros_like(next_advantage),
                next_advantage,
            )
            advantage = delta + self.gamma * self.lambda_ * continued_advantage
            return advantage, advantage

        # There is no advantage beyond the rollout (trainer episode). The final delta
        # still bootstraps from V(s_T), unless the final transition terminated.
        _, advantages = jax.lax.scan(
            advantage_step,
            jnp.zeros_like(values[-1], dtype=jnp.result_type(rewards, values)),
            (rewards, values[:-1], values[1:], terminated, done),
            reverse=True,
        )
        # V(s_T) is bootstrap-only; returns align with the T action-producing states.
        returns = advantages + values[:-1]

        advantages = (advantages - jnp.mean(advantages)) / (
            jnp.std(advantages) + self.eps
        )
        return advantages, returns
