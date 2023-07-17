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
from jax import config
from jraph._src import utils
from optax import GradientTransformation

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
from swarmrl.networks.network import Network
from swarmrl.observables.col_graph_V0 import GraphObservable
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy

config.update("jax_enable_x64", True)


class EncodeNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(12)(x)
        x = nn.relu(x)
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
        x = nn.Dense(3)(x)
        return x


class InfluenceNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x


class GraphNetV0(nn.Module):
    """
    This is the first version of the graph network.
    Node features are encoded and then aggregated using a softmax over
    the influence of each node.
    The influence of each node is computed using a simple dense network and is
    a measure of the importance of each node in the graph.
    There are no interactions or messages between nodes.
    """

    node_encoder: EncodeNet
    node_influence: InfluenceNet
    actress: ActNet
    criticer: CritNet

    @nn.compact
    def __call__(self, graph: GraphObservable):
        nodes, _, destinations, _, _, _, n_node, _ = graph

        vodes = self.node_encoder(nodes)
        influence = self.node_influence(vodes)
        padding_mask = np.where(destinations == -1, np.array([-15]), np.array([0]))
        alpha = nn.softmax(influence + padding_mask[:, np.newaxis], axis=0)
        graph_representation = np.sum(vodes * alpha, axis=0)
        logits = self.actress(graph_representation)
        value = self.criticer(graph_representation)
        return logits, value


class GraphNetV1(nn.Module):
    """
    This is the second version of the graph network.
    Node features are encoded.
    Pure messages are computed between nodes and aggregated without any weighting.
    The aggregated messages are added to the node features and then aggregated using
    a softmax over the influence of each node.
    The influence of each node is computed using a simple dense network and is
    a measure of the importance of each node in the graph.
    """

    node_encoder: EncodeNet
    node_influence: InfluenceNet
    actress: ActNet
    criticer: CritNet

    @nn.compact
    def __call__(self, graph: GraphObservable):
        nodes, _, destinations, receivers, senders, _, n_node, n_edge = graph
        n_nodes = n_node[0]
        vodes = self.node_encoder(nodes)
        messages = compute_pure_message(
            nodes=vodes,
            senders=senders,
            receivers=receivers,
            n_nodes=n_nodes,
        )

        vodes = vodes + messages
        influence = self.node_influence(vodes)
        # padding_mask = np.where(destinations == -1, np.array([0]), np.array([0]))
        # alpha = nn.softmax(influence + padding_mask[:, np.newaxis], axis=0)
        alpha = nn.softmax(influence, axis=0)
        graph_representation = np.sum(vodes * alpha, axis=0)
        logits = self.actress(graph_representation)
        value = self.criticer(graph_representation)
        return logits, value


class GraphNetV2(nn.Module):
    """
    This is the second version of the graph network.
    Node features are encoded.
    Weighted messages are computed between nodes and aggregated.
    The network can learn to ignore messages from certain nodes.
    The aggregated messages are added to the node features and then aggregated using
    a softmax over the influence of each node.
    The influence of each node is computed using a simple dense network and is
    a measure of the importance of each node in the graph.
    """

    node_encoder: EncodeNet
    node_influence: InfluenceNet
    actress: ActNet
    criticer: CritNet

    @nn.compact
    def __call__(self, graph: GraphObservable):
        nodes, _, destinations, receivers, senders, _, n_node, n_edge = graph
        n_nodes = n_node[0]
        vodes = self.node_encoder(nodes)

        sending_nodes = tree.tree_map(lambda n: n[senders], vodes)
        receiving_nodes = tree.tree_map(lambda n: n[receivers], vodes)

        attention_score = 10 * self.node_influence(receiving_nodes)
        # influences = utils.segment_softmax(
        #     attention_score.squeeze(),
        #     receivers,
        #     n_nodes,
        # )
        # compute message. The message is the influence times the sending node
        # send_messages = influences[:, None] * sending_nodes
        send_messages = attention_score.squeeze()[:, None] * sending_nodes
        # # aggregate messages
        messages = utils.segment_sum(send_messages, receivers, n_nodes)
        vodes = vodes + messages
        influence = self.node_influence(vodes)
        # padding_mask = np.where(destinations == -1, np.array([0]), np.array([0]))
        # alpha = nn.softmax(influence + padding_mask[:, np.newaxis], axis=0)
        alpha = nn.softmax(influence, axis=0)
        graph_representation = np.sum(vodes * alpha, axis=0)
        logits = self.actress(graph_representation)
        value = self.criticer(graph_representation)
        return logits, value


def compute_pure_message(nodes, senders, receivers, n_nodes):
    received_messages = tree.tree_map(lambda v: v[senders], nodes)
    message = utils.segment_sum(received_messages, receivers, n_nodes)
    return message


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
    return message


class GraphModel_V0(Network, ABC):
    """
    Abstract class for graph models.
    """

    def __init__(
        self,
        version=0,
        node_encoder: EncodeNet = EncodeNet(),
        node_influence: InfluenceNet = InfluenceNet(),
        actress: ActNet = ActNet(),
        criticer: CritNet = CritNet(),
        optimizer: GradientTransformation = optax.sgd(1e-3),
        exploration_policy: ExplorationPolicy = None,
        sampling_strategy: SamplingStrategy = GumbelDistribution(),
        rng_key: int = 42,
        deployment_mode: bool = False,
        example_graph: GraphObservable = None,
    ):
        if rng_key is None:
            rng_key = onp.random.randint(0, 1027465782564)
        self.sampling_strategy = sampling_strategy
        if version == 0:
            self.model = GraphNetV0(
                node_encoder=node_encoder,
                node_influence=node_influence,
                actress=actress,
                criticer=criticer,
            )
        elif version == 1:
            self.model = GraphNetV1(
                node_encoder=node_encoder,
                node_influence=node_influence,
                actress=actress,
                criticer=criticer,
            )
        elif version == 2:
            self.model = GraphNetV2(
                node_encoder=node_encoder,
                node_influence=node_influence,
                actress=actress,
                criticer=criticer,
            )
        else:
            raise ValueError("Invalid version number")

        self.apply_fn = jax.vmap(
            self.model.apply,
            in_axes=(None, GraphObservable(0, None, 0, 0, 0, None, None, None)),
        )

        self.swarm_apply_fn = jax.vmap(
            self.apply_fn,
            in_axes=(None, GraphObservable(0, None, 0, 0, 0, None, None, None)),
        )

        # self.apply_fn = jax.jit(self.apply_fn)
        self.model_state = None

        if not deployment_mode:
            self.optimizer = optimizer
            self.exploration_policy = exploration_policy

            # initialize the model state
            init_rng = jax.random.PRNGKey(rng_key)
            _, subkey = jax.random.split(init_rng)
            self.model_state = self._create_train_state(subkey, example_graph)

            self.epoch_count = 0

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

    def compute_action(self, observables: GraphObservable, explore_mode: bool = False):
        """
        Compute and action from the action space.

        This method computes an action on all colloids of the relevant type.

        Parameters
        ----------
        graph_obs : GraphObservable
                The graph observation. It is a named tuple with the following fields:
                nodes, edges, destinations, receivers, senders, n_node, n_edge.
                Each of the nodes is a batched version of the corresponding
                attribute of the graph. The batch size is the number of colloids
                of the relevant type.
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
        try:
            logits, _ = self.apply_fn({"params": self.model_state.params}, observables)
        except AttributeError:  # We need this for loaded models.
            logits, _ = self.apply_fn(
                {"params": self.model_state["params"]}, observables
            )

        # Compute the action
        indices = np.array(self.sampling_strategy(np.array(logits)))
        # Add a small value to the log_probs to avoid log(0) errors.
        eps = 0
        log_probs = np.log(jax.nn.softmax(np.array(logits)) + eps)
        taken_log_probs = np.take_along_axis(
            log_probs, indices.reshape(-1, 1), axis=1
        ).reshape(-1)

        if explore_mode:
            indices = self.exploration_policy(indices, 3)

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

    def __call__(self, sequenced_graph, params):
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

        # try:
        #     return list(map(lambda graph_obs: self.apply_fn({"params": params}
        #     , graph_obs), sequenced_graph))
        # except AttributeError:  # We need this for loaded models.
        #     return list(map(lambda graph_obs: self.apply_fn({"params": params}
        #     , graph_obs), sequenced_graph))
        try:
            return self.swarm_apply_fn({"params": params}, sequenced_graph)
        except AttributeError:  # We need this for loaded models.
            return self.swarm_apply_fn({"params": params}, sequenced_graph)
