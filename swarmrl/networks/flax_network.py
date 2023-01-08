"""
Jax model for reinforcement learning.
"""
import logging
import os
import pickle
from abc import ABC

import jax
import jax.numpy as np
import numpy as onp
from flax import linen as nn

# from flax.training import checkpoints
from flax.training.train_state import TrainState
from optax._src.base import GradientTransformation

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
from swarmrl.networks.network import Network
from swarmrl.sampling_strategies.sampling_stratgey import SamplingStrategy

logger = logging.getLogger(__name__)


class FlaxModel(Network, ABC):
    """
    Class for the Flax model in ZnRND.

    Attributes
    ----------
    epoch_count : int
            Current epoch stage. Used in saving the models.
    """

    def __init__(
        self,
        flax_model: nn.Module,
        input_shape: tuple,
        optimizer: GradientTransformation = None,
        exploration_policy: ExplorationPolicy = None,
        sampling_strategy: SamplingStrategy = None,
        rng_key: int = None,
        deployment_mode: bool = False,
    ):
        """
        Constructor for a Flax model.

        Parameters
        ----------
        flax_model : nn.Module
                Flax model as a neural network.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        rng_key : int
                Key to seed the model with. Default is a randomly generated key but
                the parameter is here for testing purposes.
        deployment_mode : bool
                If true, the model is a shell for the network and nothing else. No
                training can be performed, this is only used in deployment.
        """
        if rng_key is None:
            rng_key = onp.random.randint(0, 1027465782564)
        self.sampling_strategy = sampling_strategy
        self.model = flax_model
        self.apply_fn = jax.jit(self.model.apply)
        self.input_shape = input_shape
        self.model_state = None

        if not deployment_mode:
            self.optimizer = optimizer
            self.exploration_policy = exploration_policy

            # initialize the model state
            init_rng = jax.random.PRNGKey(rng_key)
            _, subkey = jax.random.split(init_rng)
            self.model_state = self._create_train_state(subkey)

            self.epoch_count = 0

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

        self.epoch_count += 1

    def compute_action(self, feature_vector: np.ndarray, explore_mode: bool = False):
        """
        Compute the action.
        """
        # Compute state
        try:
            model_output = self.apply_fn(
                {"params": self.model_state.params}, feature_vector
            )
        except AttributeError:
            model_output = self.apply_fn(
                {"params": self.model_state["params"]}, feature_vector
            )
        logger.debug(f"{model_output=}")

        # Compute the action
        index = self.sampling_strategy(model_output)

        if explore_mode:
            index = self.exploration_policy(index, len(model_output))

        return index, model_output[index]

    def export_model(self, filename: str = "model", directory: str = "Models"):
        """
        Export the model state to a directory.

        Parameters
        ----------
        filename : str (default=models)
                Name of the file the models are saved in.
        directory : str (default=Models)
                Directory in which to save the models. If the directory is not
                in the currently directory, it will be created.

        """
        model_params = self.model_state.params
        opt_state = self.model_state.opt_state
        opt_step = self.model_state.step
        epoch = self.epoch_count

        os.makedirs(directory, exist_ok=True)

        with open(directory + "/" + filename + ".pkl", "wb") as f:
            pickle.dump((model_params, opt_state, opt_step, epoch), f)

    def restore_model_state(self, filename, directory):
        """
        Restore the model state from a file.

        Parameters
        ----------
        filename : str
                Name of the model state file
        directory : str
                Path to the model state file.

        Returns
        -------
        Updates the model state.
        """

        with open(directory + "/" + filename + ".pkl", "rb") as f:
            model_params, opt_state, opt_step, epoch = pickle.load(f)

        self.model_state = self.model_state.replace(
            params=model_params, opt_state=opt_state, step=opt_step
        )
        self.epoch_count = epoch

    def __call__(self, feature_vector: np.ndarray):
        """
        See parent class for full doc string.

        Parameters
        ----------
        feature_vector : np.ndarray
                Observable to be passed through the network on which a decision is made.
        """

        return self.apply_fn({"params": self.model_state.params}, feature_vector)
