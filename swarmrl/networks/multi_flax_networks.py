"""
Unified manager for multi-network configurations in JAX/Flax.
"""

import os
import pickle
from abc import ABC
from typing import Any

import jax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from loguru import logger

from swarmrl.networks.network import Network


class MultiFlaxModel(Network, ABC):
    """
    Unified manager for multi-network configurations.
    Maintains flat registries for modules, parameters, and targets.
    """

    def __init__(self, seed: int | None = None, deployment_mode: bool = False):
        super().__init__(seed=seed)
        self.deployment_mode = deployment_mode
        self.epoch_count = 0

        self.networks: dict[str, nn.Module] = {}
        self.states: dict[str, TrainState] = {}
        self.target_params: dict[str, Any] = {}

    def add_network(
        self,
        name: str,
        flax_module: nn.Module,
        init_params: Any,
        optimizer: optax.GradientTransformation | None = None,
        has_target: bool = False,
    ) -> None:
        """
        Registers a pure network module (outputs logits), instantiates its
        TrainState, and handles target assignments.
        """
        self.networks[name] = flax_module

        # In deployment mode, or if no optimizer is provided, freeze updates
        tx = (
            optax.identity()
            if (self.deployment_mode or optimizer is None)
            else optimizer
        )

        self.states[name] = TrainState.create(
            apply_fn=flax_module.apply, params=init_params, tx=tx
        )

        if has_target:
            self.target_params[name] = jax.tree_util.tree_map(lambda x: x, init_params)

    def export_model(self, filename: str = "multi_model", directory: str = "Models"):
        """Exports all parameter maps, optimizer states, and steps simultaneously."""
        os.makedirs(directory, exist_ok=True)

        export_data = {
            "states": {
                k: {"params": s.params, "opt_state": s.opt_state, "step": s.step}
                for k, s in self.states.items()
            },
            "target_params": self.target_params,
            "epoch_count": self.epoch_count,
        }

        with open(os.path.join(directory, f"{filename}.pkl"), "wb") as f:
            pickle.dump(export_data, f)
        logger.info(
            "Successfully exported multi-network states to "
            f"{directory}/{filename}.pkl"
        )

    def restore_model_state(self, filename: str, directory: str):
        """Restores every tracking registry out of an exported payload file."""
        with open(os.path.join(directory, f"{filename}.pkl"), "rb") as f:
            payload = pickle.load(f)

        for k, state_data in payload["states"].items():
            if k in self.states:
                self.states[k] = self.states[k].replace(
                    params=state_data["params"],
                    opt_state=state_data["opt_state"],
                    step=state_data["step"],
                )

        self.target_params = payload["target_params"]
        self.epoch_count = payload["epoch_count"]

        logger.info("Restored all structural networks from tracking save state.")
