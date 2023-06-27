import os
import pickle
from abc import ABC
from typing import List

import flax.linen as nn
import jax
import jax.numpy as np
import jax.tree_util as tree
import numpy as onp
import optax
from flax.training.train_state import TrainState
from optax import GradientTransformation

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
from swarmrl.networks.network import Network
from swarmrl.observables.col_graph import GraphObservable
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy


class EncodeNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(12)(x)
        return x


class CritNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class ActNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(4)(x)
        return x


class InfluenceNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class GraphNet(nn.Module):
    edge_encoder: EncodeNet
    channel_encoder: EncodeNet
    edge_influencer: InfluenceNet
    message_influencer: InfluenceNet
    actress: ActNet
    criticer: CritNet

    @nn.compact
    def __call__(self, graph: GraphObservable):
        nodes, edges, channels, receivers, senders, globals_, n_node, n_edge = graph

        # Encode the nodes.
        # embedding_vec = self.encoder(nodes)

        print(np.shape(edges))
        channel_embedding = self.channel_encoder(channels)
        edge_embedding = self.edge_encoder(edges)
        edge_scores = self.edge_influencer(edge_embedding)
        edge_scores = jax.nn.softmax(edge_scores, axis=1)

        message = (
            np.mean(
                tree.tree_map(lambda c, c_s: c * c_s, edge_embedding, edge_scores),
                axis=0,
            )
            + channel_embedding
        )
        message_score = jax.nn.softmax(self.message_influencer(message), axis=1)
        graph_representation = np.mean(
            tree.tree_map(
                lambda m, m_s: m * m_s,
                message,
                message_score,
            ),
            axis=0,
        )

        logits = self.actress(graph_representation)
        value = self.criticer(graph_representation)
        return logits, value


class GraphModel(Network, ABC):
    """
    Abstract class for graph models.
    """

    def __init__(
        self,
        edge_encoder: EncodeNet = EncodeNet(),
        channel_encoder: EncodeNet = EncodeNet(),
        edge_influencer: InfluenceNet = InfluenceNet(),
        message_influencer: InfluenceNet = InfluenceNet(),
        actress: ActNet = ActNet(),
        criticer: CritNet = CritNet(),
        optimizer: GradientTransformation = optax.adam(1e-3),
        exploration_policy: ExplorationPolicy = None,
        sampling_strategy: SamplingStrategy = GumbelDistribution(),
        rng_key: int = 42,
        deployment_mode: bool = False,
        record_memory: bool = False,
        example_graph: GraphObservable = None,
    ):
        if rng_key is None:
            rng_key = onp.random.randint(0, 1027465782564)
        self.sampling_strategy = sampling_strategy
        self.model = GraphNet(
            edge_encoder=edge_encoder,
            channel_encoder=channel_encoder,
            edge_influencer=edge_influencer,
            message_influencer=message_influencer,
            actress=actress,
            criticer=criticer,
        )
        self.apply_fn = jax.jit(self.model.apply)
        self.model_state = None

        if not deployment_mode:
            self.optimizer = optimizer
            self.exploration_policy = exploration_policy

            # initialize the model state
            init_rng = jax.random.PRNGKey(rng_key)
            _, subkey = jax.random.split(init_rng)
            self.model_state = self._create_train_state(subkey, example_graph)

            self.epoch_count = 0

        self.record_memory = record_memory
        self.memory = {
            "file_name": "graph_memory.npy",
            "observables": [],
            "graph_representations": [],
            "influences": [],
            "logits": [],
            "log_probs": [],
            "indices": [],
            "taken_log_probs": [],
        }
        self.kind = "network"

    def _create_train_state(self, init_rng: int, example_graph) -> TrainState:
        params = self.model.init(init_rng, example_graph)["params"]

        return TrainState.create(
            apply_fn=self.apply_fn, params=params, tx=self.optimizer
        )

    def reinitialize_network(self):
        """
        Initialize the neural network.
        """
        rng_key = onp.random.randint(0, 1027465782564)
        init_rng = jax.random.PRNGKey(rng_key)
        _, subkey = jax.random.split(init_rng)
        self.model_state = self._create_train_state(subkey)

    def update_model(
        self,
        grads,
    ):
        """
        Train the model.

        See the parent class for a full doc-string.
        """
        self.model_state = self.model_state.apply_gradients(grads=grads)

        self.epoch_count += 1

    def compute_action(self, observables: List, explore_mode: bool = False):
        """
        Compute and action from the action space.

        This method computes an action on all colloids of the relevant type.

        Parameters
        ----------
        observables : List
                Observable for each colloid for which the action should be computed.
        explore_mode : bool
                If true, an exploration vs exploitation function is called.

        Returns
        -------
        tuple : (np.ndarray, np.ndarray)
                The first element is an array of indices corresponding to the action
                taken by the agent. The value is bounded between 0 and the number of
                output neurons. The second element is an array of the corresponding
                log_probs (i.e. the output of the network put through a softmax).
        """
        logits_list = []
        # Compute state
        for obs in observables:
            try:
                logits, _ = self.apply_fn({"params": self.model_state.params}, obs)
            except AttributeError:  # We need this for loaded models.
                logits, _ = self.apply_fn({"params": self.model_state["params"]}, obs)
            logits_list.append(logits)

        # Compute the action
        indices = self.sampling_strategy(np.array(logits_list))
        # Add a small value to the log_probs to avoid log(0) errors.
        eps = 0
        log_probs = np.log(jax.nn.softmax(np.array(logits_list)) + eps)
        taken_log_probs = np.take_along_axis(
            log_probs, indices.reshape(-1, 1), axis=1
        ).reshape(-1)

        if explore_mode:
            indices = self.exploration_policy(indices, len(logits_list))
        return indices, taken_log_probs

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

    def __call__(self, graph_obs: GraphObservable):
        """
        See parent class for full doc string.

        Parameters
        ----------
        feature_vector : np.ndarray
                Observable to be passed through the network on which a decision is made.

        Returns
        -------
        logits : np.ndarray
                Output of the network.
        """

        try:
            return self.apply_fn({"params": self.model_state.params}, graph_obs)
        except AttributeError:  # We need this for loaded models.
            return self.apply_fn({"params": self.model_state["params"]}, graph_obs)
