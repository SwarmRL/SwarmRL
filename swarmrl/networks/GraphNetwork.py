# """
# Jax model for reinforcement learning.
# """
# import logging
# import os
# import pickle
# from abc import ABC
# from typing import List
#
# import jax
# import jax.numpy as np
# import numpy as onp
# from flax import linen as nn
#
# # from flax.training import checkpoints
# from flax.training.train_state import TrainState
# from optax._src.base import GradientTransformation
#
# from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
# from swarmrl.networks.network import Network
# from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy
#
# logger = logging.getLogger(__name__)
#
#
# class ColEmbedding(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         return nn.Dense(features=128)(x)
#
#
# class RodEmbedding(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         return nn.Dense(features=128)(x)
#
#
# class ActorNet(nn.Module):
#     """A simple dense model."""
#
#     def setup(self):
#         self.col_embed = ColEmbedding()
#         self.rod_embed = RodEmbedding()
#
#     @nn.compact
#     def __call__(self, x):
#         colloid_embedding = self.col_embed(x[:, 0])
#         rod_embedding = self.rod_embed(x[:, 1])
#
#         x = colloid_embedding + rod_embedding
#         x = nn.relu(x)
#         x = nn.Dense(features=128)(x)
#         x = nn.relu(x)
#         x = nn.Dense(features=4)(x)
#         return x
#
#
# class GraphModel(Network, ABC):
#     """
#     Class for the Flax model in ZnRND.
#
#     Attributes
#     ----------
#     epoch_count : int
#             Current epoch stage. Used in saving the models.
#     """
#
#     def __init__(
#         self,
#         flax_model: nn.Module,
#         input_shape: tuple,
#         optimizer: GradientTransformation = None,
#         exploration_policy: ExplorationPolicy = None,
#         sampling_strategy: SamplingStrategy = None,
#         rng_key: int = None,
#         deployment_mode: bool = False,
#     ):
#         """
#         Constructor for a Flax model.
#
#         Parameters
#         ----------
#         flax_model : nn.Module
#                 Flax model as a neural network.
#         optimizer : Callable
#                 optimizer to use in the training. OpTax is used by default and
#                 cross-compatibility is not assured.
#         input_shape : tuple
#                 Shape of the NN input.
#         rng_key : int
#                 Key to seed the model with. Default is a randomly generated key but
#                 the parameter is here for testing purposes.
#         deployment_mode : bool
#                 If true, the model is a shell for the network and nothing else. No
#                 training can be performed, this is only used in deployment.
#         """
#         if rng_key is None:
#             rng_key = onp.random.randint(0, 1027465782564)
#         self.sampling_strategy = sampling_strategy
#         self.model = flax_model
#         self.apply_fn = jax.jit(self.model.apply)
#         self.input_shape = input_shape
#         self.model_state = None
#
#         if not deployment_mode:
#             self.optimizer = optimizer
#             self.exploration_policy = exploration_policy
#
#             # initialize the model state
#             init_rng = jax.random.PRNGKey(rng_key)
#             _, subkey = jax.random.split(init_rng)
#             self.model_state = self._create_train_state(subkey)
#
#             self.epoch_count = 0
#
#     def _create_train_state(self, init_rng: int) -> TrainState:
#         """
#         Create a training state of the model.
#
#         Parameters
#         ----------
#         init_rng : int
#                 Initial rng for train state that is immediately deleted.
#
#         Returns
#         -------
#         state : TrainState
#                 initial state of model to then be trained.
#         """
#         params = self.model.init(init_rng, np.ones(list(self.input_shape)))["params"]
#
#         return TrainState.create(
#             apply_fn=self.apply_fn, params=params, tx=self.optimizer
#         )
#
#     def update_model(
#         self,
#         grads,
#     ):
#         """
#         Train the model.
#
#         See the parent class for a full doc-string.
#         """
#         logger.debug(f"{grads=}")
#         logger.debug(f"{self.model_state=}")
#         self.model_state = self.model_state.apply_gradients(grads=grads)
#         logger.debug(f"{self.model_state=}")
#
#         self.epoch_count += 1
#
#     def compute_action(self, observables: List, explore_mode: bool = False):
#         """
#         Compute and action from the action space.
#
#         This method computes an action on all colloids of the relevant type.
#
#         Parameters
#         ----------
#         observables : List
#                 Observable for each colloid for which the action should be computed.
#         explore_mode : bool
#                 If true, an exploration vs exploitation function is called.
#
#         Returns
#         -------
#         tuple : (np.ndarray, np.ndarray)
#                 The first element is an array of indices corresponding to the action
#                 taken by the agent. The value is bounded between 0 and the number of
#                 output neurons. The second element is an array of the corresponding
#                 log_probs (i.e. the output of the network put through a softmax).
#         """
#         # Compute state
#         try:
#             logits = self.apply_fn(
#                 {"params": self.model_state.params}, np.array(observables)
#             )
#         except AttributeError:  # We need this for loaded models.
#             logits = self.apply_fn(
#                 {"params": self.model_state["params"]}, np.array(observables)
#             )
#         logger.debug(f"{logits=}")  # (n_colloids, n_actions)
#
#         # Compute the action
#         indices = self.sampling_strategy(logits)
#
#         # Add a small value to the log_probs to avoid log(0) errors.
#         eps = 1e-8
#         log_probs = np.log(jax.nn.softmax(logits) + eps)
#         if explore_mode:
#             indices = self.exploration_policy(indices, len(logits))
#         return (
#             indices,
#             np.take_along_axis(log_probs, indices.reshape(-1, 1), axis=1).reshape(-1),
#         )
#
#     def export_model(self, filename: str = "model", directory: str = "Models"):
#         """
#         Export the model state to a directory.
#
#         Parameters
#         ----------
#         filename : str (default=models)
#                 Name of the file the models are saved in.
#         directory : str (default=Models)
#                 Directory in which to save the models. If the directory is not
#                 in the currently directory, it will be created.
#
#         """
#         model_params = self.model_state.params
#         opt_state = self.model_state.opt_state
#         opt_step = self.model_state.step
#         epoch = self.epoch_count
#
#         os.makedirs(directory, exist_ok=True)
#
#         with open(directory + "/" + filename + ".pkl", "wb") as f:
#             pickle.dump((model_params, opt_state, opt_step, epoch), f)
#
#     def restore_model_state(self, filename, directory):
#         """
#         Restore the model state from a file.
#
#         Parameters
#         ----------
#         filename : str
#                 Name of the model state file
#         directory : str
#                 Path to the model state file.
#
#         Returns
#         -------
#         Updates the model state.
#         """
#
#         with open(directory + "/" + filename + ".pkl", "rb") as f:
#             model_params, opt_state, opt_step, epoch = pickle.load(f)
#
#         self.model_state = self.model_state.replace(
#             params=model_params, opt_state=opt_state, step=opt_step
#         )
#         self.epoch_count = epoch
#
#     def __call__(self, feature_vector: np.ndarray):
#         """
#         See parent class for full doc string.
#
#         -------
#         logits : np.ndarray
#                 Output of the network.
#         """
#
#         try:
#             return self.apply_fn({"params": self.model_state.params}, feature_vector)
#         except AttributeError:  # We need this for loaded models.
#             return self.apply_fn({"params": self.model_state["params"]},
#             feature_vector)
#
#
# """
# graph obs implementation computer.
# """
# import functools
# from abc import ABC
# from typing import Any, Callable, Dict, List, Optional, Tuple
#
# import flax.linen as nn
# import haiku as hk
# import jax
# import jax.numpy as np
# import jax.tree_util as tree
# import jraph
# import networkx as nx
# import numpy as onp
# import optax
# from flax.training.train_state import TrainState
# from jax import random
# from jraph._src import graph as gn_graph
# from jraph._src import utils
# from optax._src.base import GradientTransformation
#
# from .observable import Observable
#
#
# def _angle_and_dist(colloid, colloids):
#     # angles between the colloid director and line of sight to other colloids
#     angles = []
#     # angles between colloid director and director of other colloids
#     angles2 = []
#     dists = []
#     my_director = colloid.director[:2]
#     for col in colloids:
#         if col is not colloid:
#             my_col_vec = col.pos[:2] - colloid.pos[:2]
#             my_col_dist = np.linalg.norm(my_col_vec)
#
#             # compute angle 1
#             my_col_vec = my_col_vec / my_col_dist
#             angle = np.arccos(np.dot(my_col_vec, my_director))
#             orthogonal_dot = np.dot(
#                 my_col_vec, np.array([-my_director[1], my_director[0]])
#             )
#             angle *= np.sign(orthogonal_dot) / np.pi
#
#             # compute angle 2
#             other_director = col.director[:2]
#             angle2 = np.arccos(np.dot(other_director, my_director))
#             orthogonal_dot2 = np.dot(
#                 other_director, np.array([-my_director[1], my_director[0]])
#             )
#             angle2 *= np.sign(orthogonal_dot2) / np.pi
#             angles2.append(angle2)
#             angles.append(angle)
#             dists.append(my_col_dist)
#     return np.array(angles), np.array(angles2), np.array(dists)
#
#
# class GraphObs(Observable, ABC):
#     """
#     Implementation of the GraphOps observable.
#     """
#
#     def __init__(
#         self,
#         box_size,
#         r_cut: float,
#         encoder_network: nn.Module,
#         node_updater_network: nn.Module,
#         influencer_network: nn.Module,
#         obs_shape: int = 8,
#         relate=False,
#         attention_normalize_fn=utils.segment_softmax,
#         seed=42,
#     ):
#         self.box_size = box_size
#         self.r_cut = r_cut
#         self.obs_shape = obs_shape
#         self.relate = relate
#         self.attention_normalize_fn = attention_normalize_fn
#
#         # ML part of the observable
#         self.rngkey = random.PRNGKey(seed)
#         self.networks = {
#             "encoder": encoder_network,
#             "node_updater": node_updater_network,
#             "influencer": influencer_network,
#         }
#         self.states = {"encoder": None, "node_updater": None, "influencer": None}
#
#         self.encode_fn, self.update_node_fn, self.influence_eval_fn = (
#             self._init_models()
#         )
#
#     def initialize(self, colloids: list):
#         pass
#
#     def _init_models(self):
#         split, self.rngkey = random.split(key=self.rngkey)
#         for key, item in self.networks.items():
#             rngkey, split = random.split(key=split)
#             params = item.init(rngkey, np.ones(4))["params"]
#             self.states[key] = TrainState.create(
#                 apply_fn=jax.jit(item.apply),
#                 params=params,
#                 tx=optax.adam(learning_rate=0.001),
#             )
#
#         encoder = jax.jit(self.networks["encoder"].apply)
#
#         def encode_fn(features):
#             encoded = encoder({"params": self.states["encoder"].params}, features)
#             return encoded
#
#         node_updater = jax.jit(self.networks["node_updater"].apply)
#
#         def node_update_fn(features):
#             updated = node_updater(
#                 {"params": self.states["node_updater"].params}, features
#             )
#             return updated
#
#         influencer = jax.jit(self.networks["influencer"].apply)
#
#         def influencer_fn(features):
#             influenced = influencer(
#                 {"params": self.states["influencer"].params}, features
#             )
#             return influenced
#
#         return encode_fn, node_update_fn, influencer_fn
#
#     def _update_models(
#         self,
#         grads,
#     ):
#         # questionable how the grads will come back! Let's see at gradient computation
#         for key, item in self.states:
#             self.states[key] = self.states[key].apply_gradients(grads=grads)
#
#     def _build_graph(self, colloid, colloids):
#         nodes = []
#         angles, angles2, dists = _angle_and_dist(colloid, colloids)
#         node_index = 0
#         for i, col in enumerate(colloids):
#             if col is not colloid:
#                 r = dists[i]
#                 if r < self.r_cut:
#                     node_index += 1
#                     node = np.hstack(
#                         ((dists[i] / self.box_size), angles[i], angles2[i], col.type)
#                     )
#                     nodes.append(node)
#         graph = utils.get_fully_connected_graph(
#             n_node_per_graph=len(nodes),
#             n_graph=1,
#             node_features=np.array(nodes),
#             add_self_edges=False,
#         )
#         return graph
#
#     def compute_observable(
#         self, colloid: object, other_colloids: list, return_graph=False
#     ):
#         graph = self._build_graph(colloid, other_colloids)
#
#         nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
#
#         # encode node features n_i to attention_vec a_i
#         attention_vectors = tree.tree_map(lambda n: self.encode_fn(n), nodes)
#
#         if self.relate:
#             # function 1
#             sent_attributes, sent_attention = tree.tree_map(
#                 lambda n, a: (n[senders], a[senders]), nodes, attention_vectors
#             )
#             # function 2
#             received_attributes, received_attention = tree.tree_map(
#                 lambda n, a: (n[receivers], a[receivers]), nodes, attention_vectors
#             )
#             # function 3
#             # this can be made to a learnable matrix norm.
#             edges = tree.tree_map(
#                 lambda r, s: np.exp(-np.linalg.norm(r - s, axis=1) ** 2),
#                 received_attention,
#                 sent_attention,
#             )
#             # function 4
#             # softmax
#             tree_calculate_weights = functools.partial(
#                 utils.segment_softmax, segment_ids=receivers, num_segments=n_node
#             )
#             weights = tree.tree_map(tree_calculate_weights, edges)
#
#             # function 5
#             received_weighted_attributes = tree.tree_map(
#                 lambda r, w: r * w[:, None], nodes[receivers], weights
#             )
#             # function 6
#             received_message = utils.segment_sum(
#                 received_weighted_attributes, receivers, num_segments=n_node
#             )
#             # function 7
#             nodes = self.update_node_fn(received_message)
#
#             influence_score = self.influence_eval_fn(attention_vectors)
#         else:
#             influence_score = self.influence_eval_fn(nodes)
#
#         influence = jax.nn.softmax(influence_score)
#
#         # computes the actual feature
#         graph_representation = np.sum(
#             tree.tree_map(lambda n, i: n * i, nodes, influence), axis=0
#         )
#
#         if not return_graph:
#             return graph_representation
#
#         else:
#             return gn_graph.GraphsTuple(
#                 nodes=nodes,
#                 edges=edges,
#                 receivers=receivers,
#                 senders=senders,
#                 globals=graph_representation,
#                 n_node=n_node,
#                 n_edge=n_edge,
#             )
#
#
# def export_model(states: dict, filename: str = "graph_obs",
# directory: str = "Models"):
#     """
#     Export the model state to a directory.
#
#     Parameters
#     ----------
#     filename : str (default=models)
#             Name of the file the models are saved in.
#     directory : str (default=Models)
#             Directory in which to save the models. If the directory is not
#             in the currently directory, it will be created.
#
#     """
#     model_params = self.model_state.params
#     opt_state = self.model_state.opt_state
#     opt_step = self.model_state.step
#     epoch = self.epoch_count
#
#     os.makedirs(directory, exist_ok=True)
