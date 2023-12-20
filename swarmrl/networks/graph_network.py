"""
Graph model for reinforcement learning.
"""

import logging
import os
import pickle
from abc import ABC

import flax.linen as nn
import jax
import jax.numpy as np
import jax.tree_util as tree
import numpy as onp
import optax
from flax.training.train_state import TrainState
from jraph._src import utils
from optax import GradientTransformation

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
from swarmrl.networks.network import Network
from swarmrl.observables.col_graph import GraphObservable
from swarmrl.sampling_strategies import GumbelDistribution


class EncodeNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        return x


class EmbedNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(12)(x)
        return x


class ActNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        y = nn.Dense(1)(x)
        x = nn.Dense(4)(x)

        return x, y


class InfluenceNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x


class GraphNet2(nn.Module):
    """
    Graph network class.
    Node features are encoded.
    Weighted messages are computed between nodes and aggregated.
    The network can learn to ignore messages from certain nodes.
    The aggregated messages are added to the node features and then aggregated using
    a softmax over the influence of each node.
    The influence of each node is computed using a simple dense network and is
    a measure of the importance of each node in the graph.
    """

    node_encoder: EncodeNet
    node_embedder: EmbedNet
    node_influence: InfluenceNet
    actress: ActNet
    temperature: float = 10.0

    @nn.compact
    def __call__(self, graph: GraphObservable):
        nodes, _, destinations, receivers, senders, _, n_node, n_edge = graph

        n_nodes = n_node[0]
        vodes = self.node_encoder(nodes)
        vodes = compute_pure_message(vodes, senders, receivers, n_nodes)
        vodes = np.concatenate([vodes, nodes[:, -1:]], axis=1)
        influence = self.node_influence(vodes)
        attention = nn.softmax(influence, axis=0)
        # stack the last entry of the node features to the vodes.
        graph_representation = np.sum(self.node_embedder(nodes) * attention, axis=0)
        logits, value = self.actress(graph_representation)

        return logits / self.temperature, value


class GraphNet(nn.Module):
    """
    Graph network class.
    Node features are encoded.
    Weighted messages are computed between nodes and aggregated.
    The network can learn to ignore messages from certain nodes.
    The aggregated messages are added to the node features and then aggregated using
    a softmax over the influence of each node.
    The influence of each node is computed using a simple dense network and is
    a measure of the importance of each node in the graph.
    """

    node_encoder: EncodeNet
    node_embedder: EmbedNet
    node_influence: InfluenceNet
    actress: ActNet
    temperature: float = 10.0

    @nn.compact
    def __call__(self, graph: GraphObservable):
        nodes, _, destinations, receivers, senders, _, n_node, n_edge = graph

        n_nodes = n_node[0]

        vodes = self.node_encoder(nodes)
        vodes = compute_pure_message(vodes, senders, receivers, n_nodes)
        vodes = np.concatenate([vodes, nodes[:, -1:]], axis=1)
        influence = self.node_influence(vodes)
        attention = nn.softmax(influence, axis=0)
        # stack the last entry of the node features to the vodes.
        graph_representation = np.sum(vodes * attention, axis=0)
        graph_representation = self.node_embedder(graph_representation)
        logits, value = self.actress(graph_representation)

        return logits / self.temperature, value


def compute_pure_message(nodes, senders, receivers, n_nodes, message_passing_steps=1):
    """Compute the message for each node based on the influence between
    sender and receiver.
    """
    # for _ in range(message_passing_steps):
    send_messages = tree.tree_map(lambda n: n[senders], nodes)
    message = utils.segment_sum(send_messages, receivers, n_nodes)
    nodes = tree.tree_map(lambda n: n + message, nodes)
    return nodes


logger = logging.getLogger(__name__)


def compute_weighted_message(nodes, senders, receivers, n_nodes):
    """Compute the message for each node based on the influence between
    sender and receiver.
    """
    # compute sender and receiver nodes
    sending_nodes = tree.tree_map(lambda n: n[senders], nodes)
    receiving_nodes = tree.tree_map(lambda n: n[receivers], nodes)
    # compute influence (inspired by VAIN)
    influences = utils.segment_softmax(
        np.exp(-np.linalg.norm(sending_nodes - receiving_nodes, axis=-1)),
        receivers,
        n_nodes,
    )
    # compute message. The message is the influence times the sending node
    send_messages = influences[:, np.newaxis] * sending_nodes
    # aggregate messages
    message = utils.segment_sum(send_messages, receivers, n_nodes)
    nodes = tree.tree_map(lambda n: n + message, nodes)
    return nodes


class GraphModel(Network, ABC):
    """
    Abstract class for graph models.
    """

    def __init__(
        self,
        init_graph: GraphObservable = None,
        node_encoder: EncodeNet = EncodeNet(),
        node_embedding: EmbedNet = EmbedNet(),
        node_influence: InfluenceNet = InfluenceNet(),
        actress: ActNet = ActNet(),
        # criticer: CritNet = CritNet(),
        optimizer: GradientTransformation = optax.adam(1e-4),
        exploration_policy: ExplorationPolicy = None,
        sampling_strategy=GumbelDistribution(),
        rng_key: int = onp.random.randint(0, 2**32 - 1),
        deployment_mode: bool = False,
    ):
        """
        Constructor for a Graph model.

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

        self.sampling_strategy = sampling_strategy
        self.model = GraphNet(
            node_encoder=node_encoder,
            node_embedder=node_embedding,
            node_influence=node_influence,
            actress=actress,
        )

        self.apply_fn = jax.vmap(
            self.model.apply,
            in_axes=(None, GraphObservable(0, None, 0, 0, 0, None, None, None)),
        )

        self.swarm_apply_fn = jax.vmap(
            self.apply_fn,
            in_axes=(None, GraphObservable(0, None, 0, 0, 0, None, None, None)),
        )

        self.model_state = None

        if not deployment_mode:
            self.optimizer = optimizer
            self.exploration_policy = exploration_policy

            # initialize the model state
            init_rng = jax.random.PRNGKey(rng_key)
            _, subkey = jax.random.split(init_rng)
            self.model_state = self._create_train_state(subkey, init_graph)

            self.epoch_count = 0

    def _create_custom_train_state(self, optimizer: dict):
        """
        Deal with the optimizers in case of complex configuration.
        """
        return type("TrainState", (TrainState,), optimizer)

    def _create_train_state(
        self, init_rng: int, init_graph: GraphObservable
    ) -> TrainState:
        """
        Create a training state of the model.

        Parameters
        ----------
        init_rng : int
                Initial rng for train state that is immediately deleted.

        Returns
        -------
        state : TrainState / CustomTrainState
                initial state of model to then be trained.
                If you have multiple optimizers, this will create a custom train state.
        """
        params = self.model.init(init_rng, init_graph)["params"]

        if isinstance(self.optimizer, dict):
            CustomTrainState = self._create_custom_train_state(self.optimizer)

            return CustomTrainState.create(
                apply_fn=self.model.apply, params=params, tx=self.optimizer
            )
        else:
            return TrainState.create(
                apply_fn=self.model.apply, params=params, tx=self.optimizer
            )

    def reinitialize_network(self):
        """
        Initialize the neural network.
        """
        rng_key = onp.random.randint(0, 1027465782564)
        init_rng = jax.random.PRNGKey(rng_key)
        _, subkey = jax.random.split(init_rng)
        self.model_state = self._create_train_state(subkey)

    def update_model(self, grads):
        """
        Train the model.

        See the parent class for a full doc-string.
        """
        # Logging for grads and pre-train model state
        logger.debug(f"{grads=}")
        logger.debug(f"{self.model_state=}")

        if isinstance(self.optimizer, dict):
            pass

        else:
            self.model_state = self.model_state.apply_gradients(grads=grads)

        # Logging for post-train model state
        logger.debug(f"{self.model_state=}")

        self.epoch_count += 1

    def compute_action(self, observables: GraphObservable):
        """
        Compute and action from the action space.

        This method computes an action on all colloids of the relevant type.

        Parameters
        ----------
        observable : GraphObservable
                The graph observation. It is a named tuple with the following fields:
                nodes, edges, destinations, receivers, senders, n_node, n_edge.
                Each of the nodes is a batched version of the corresponding
                attribute of the graph. The batch size is the number of colloids
                of the relevant type.

        Returns
        -------
        tuple : (np.ndarray, np.ndarray)
                The first element is an array of indices corresponding to the action
                taken by the agent. The value is bounded between 0 and the number of
                output neurons. The second element is an array of the corresponding
                log_probs (i.e. the output of the network put through a softmax).
        """
        try:
            logits, _ = self.apply_fn({"params": self.model_state.params}, observables)
        except AttributeError:  # We need this for loaded models.
            logits, _ = self.apply_fn(
                {"params": self.model_state["params"]}, observables
            )
        logger.debug(f"{logits=}")  # (n_colloids, n_actions)

        # Compute the action
        indices = np.array(self.sampling_strategy(np.array(logits)))
        # Add a small value to the log_probs to avoid log(0) errors.
        eps = 0
        log_probs = np.log(jax.nn.softmax(np.array(logits)) + eps)
        taken_log_probs = np.take_along_axis(
            log_probs, indices.reshape(-1, 1), axis=1
        ).reshape(-1)

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

    def __call__(self, params, sequenced_graph):
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
        return self.swarm_apply_fn(params, sequenced_graph)
        # except AttributeError:  # We need this for loaded models.
        #     return self.swarm_apply_fn({"params": params}, sequenced_graph)
