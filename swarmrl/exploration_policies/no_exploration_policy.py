"""No-op exploration policy for algorithms with built-in stochasticity (like SAC)."""

import jax
import jax.numpy as jnp

from swarmrl.exploration_policies.exploration_policy import ContinuousExplorationPolicy


class NoExplorationPolicy(ContinuousExplorationPolicy):
    """
    An identity exploration policy that returns actions exactly as sampled.

    Useful for stochastic-policy algorithms where extra exploration noise
    should not be applied.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Accepts any arguments to remain compatible with factory patterns,
        but completely ignores them.
        """
        pass

    def reduce_randomness(self, decay: float = 0.95) -> None:
        """Does nothing, as there is no randomness to decay."""
        pass

    def __call__(self, model_actions: jnp.ndarray, rng_key: jax.Array) -> jnp.ndarray:
        """
        Returns the model actions completely unmodified.

        Parameters
        ----------
        model_actions : jnp.ndarray
            The actions produced by the sampling strategy.
        rng_key : jax.Array
            The PRNG key (ignored).

        Returns
        -------
        jnp.ndarray
            The identical model_actions.
        """
        return model_actions
