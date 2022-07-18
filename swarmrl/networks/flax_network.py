"""
Jax model for reinforcement learning.
"""
import logging
from abc import ABC
from typing import Callable

import jax
import jax.numpy as np
import numpy as onp
from flax import linen as nn
from flax.training.train_state import TrainState
from optax._src.base import GradientTransformation

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
from swarmrl.networks.network import Network
from swarmrl.sampling_strategies.sampling_stratgey import SamplingStrategy

logger = logging.getLogger(__name__)


class FlaxModel(Network, ABC):
    """
    Class for the Flax model in ZnRND.
    """

    def __init__(
        self,
        flax_model: nn.Module,
        input_shape: tuple,
        optimizer: GradientTransformation,
        loss_fn: Callable = None,
        exploration_policy: ExplorationPolicy = None,
        sampling_strategy: SamplingStrategy = None,
        rng_key: int = onp.random.randint(0, 500),
    ):
        """
        Constructor for a Flax model.

        Parameters
        ----------
        flax_model : nn.Module
                Flax model as a neural network.
        loss_fn : Callable
                A function to use in the loss computation.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        rng_key : int
                Key to seed the model with. Default is a randomly generated key but
                the parameter is here for testing purposes.
        """
        self.sampling_strategy: SamplingStrategy
        self.exploration_policy: ExplorationPolicy
        self.model_state: TrainState
        self.rng = jax.random.PRNGKey(onp.random.randint(0, 500))

        self.model = flax_model
        self.apply_fn = jax.jit(self.model.apply)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.exploration_policy = exploration_policy
        self.sampling_strategy = sampling_strategy

        # initialize the model state
        init_rng = jax.random.PRNGKey(rng_key)
        state = self._create_train_state(init_rng)
        self.model_state = state

    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        """
        Compute the current metrics of the training.

        Parameters
        ----------
        predictions : np.ndarray
                Predictions made by the network.
        targets : np.ndarray
                Targets from the training data.

        Returns
        -------
        metrics : dict
                A dict of current training metrics, e.g. {"loss": ..., "accuracy": ...}
        """
        loss = self.loss_fn(predictions, targets)

        metrics = {"loss": loss}

        return metrics

    def _create_train_state(self, init_rng: int) -> TrainState:
        """
        Create a training state of the model.

        Parameters
        ----------
        init_rng : int
                Initial rng for train state that is immediately deleted.

        Returns
        -------
        state : TrainState
                initial state of model to then be trained.
        """
        params = self.model.init(init_rng, np.ones(list(self.input_shape)))["params"]

        return TrainState.create(
            apply_fn=self.apply_fn, params=params, tx=self.optimizer
        )

    def update_model(
        self,
        grads,
    ):
        """
        Train the model.

        See the parent class for a full doc-string.
        """
        logger.debug(f"{grads=}")
        logger.debug(f"{self.model_state=}")
        self.model_state = self.model_state.apply_gradients(grads=grads)
        logger.debug(f"{self.model_state=}")

    def compute_action(self, feature_vector: np.ndarray, explore_mode: bool = False):
        """
        Compute the action.
        """
        # Compute state
        model_output = self.apply_fn(
            {"params": self.model_state.params}, feature_vector
        )
        logger.debug(f"{model_output=}")

        # Compute the action
        index = self.sampling_strategy(model_output)

        if explore_mode:
            index = self.exploration_policy(index, len(model_output))

        return index, model_output[index]

    def __call__(self, feature_vector: np.ndarray):
        """
        See parent class for full doc string.

        Parameters
        ----------
        feature_vector : np.ndarray
                Observable to be passed through the network on which a decision is made.
        """

        return self.apply_fn({"params": self.model_state.params}, feature_vector)
