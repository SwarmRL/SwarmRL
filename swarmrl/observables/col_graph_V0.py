from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Union

import jax.numpy as np
import numpy as onp
from jax import vmap

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable
from swarmrl.utils.utils import calc_signed_angle_between_directors

# from jax.tree_util import register_pytree_node


ArrayTree = Union[np.ndarray, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]


class GraphObservable(NamedTuple):
    nodes: Optional[ArrayTree]
    edges: Optional[ArrayTree]
    destinations: Optional[np.ndarray]
    receivers: Optional[np.ndarray]
    senders: Optional[np.ndarray]
    globals_: Optional[ArrayTree]
    n_node: np.ndarray
    n_edge: np.ndarray


#     def tree_flatten(self):
#         return (
#             self.nodes, self.edges, self.destinations, self.receivers, self.senders,
#             self.globals_, self.n_node, self.n_edge,
#         )
#
#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children)
#
#
# register_pytree_node(
#     GraphObservable,
#     GraphObservable.tree_flatten,
#     GraphObservable.tree_unflatten,
# )
#


class ColGraphV0(Observable):
    """ """

    def __init__(
        self,
        cutoff: float = 0.1,
        relation_angle: float = 0.2,
        box_size=None,
        record_memory=False,
        particle_type: int = 0,
    ):
        """
        Parameters
        ----------
        cutoff : float
            Cutoff distance for the graph.
        box_size : ndarray
            Size of the box.
        """
        self.cutoff = cutoff
        self.relation_angle = relation_angle
        self.box_size = box_size
        self.eps = 10e-8
        self.vangle = vmap(calc_signed_angle_between_directors, in_axes=(None, 0))
        self.record_memory = record_memory
        self.particle_type = particle_type

    def compute_observable(self, colloids: List[Colloid]) -> GraphObservable:
        """
        Builds a graph for each colloid in the system. In the graph, each node is a
        representation of a colloid within the cutoff distance.
        """
        # normalize the positions by the box size.
        positions = np.array([col.pos for col in colloids]) / self.box_size
        # directions = np.array([col.director for col in colloids])
        types = np.array([col.type for col in colloids])
        # delta_types = types[:, None] - types
        # compute the direction between all pais of colloids.
        part_part_vec = positions[:, None] - positions
        distances = np.linalg.norm(part_part_vec, axis=-1)
        part_part_vec = -1 * part_part_vec / (distances[:, :, None] + self.eps)

        nodes_list = []
        destinations_list = []
        n_nodes_list = []
        globals_list = []

        max_num_nodes = 0

        for col in colloids:
            if col.type == self.particle_type:
                # mask for the colloids within the cutoff distance. without itself.
                mask = (distances[col.id] < self.cutoff) & (distances[col.id] > 0)
                # get the indices of sender and receiver within the cutoff distance.
                num_nodes = np.sum(mask)

                if num_nodes == 0:
                    nodes_list.append(np.array([np.array([1.0, 1.0, 1.0])]))
                    globals_list.append([None])
                    n_nodes_list.append([1])
                    destinations_list.append([0])
                    continue

                director = np.copy(col.director)
                relevant_distances = distances[col.id][mask]

                relevant_part_part_vec = part_part_vec[col.id][mask]
                relevant_types = types[mask]
                pos_angles = self.vangle(director, relevant_part_part_vec)
                delta_type = relevant_types - col.type

                # stack the features of the nodes.
                nodes = np.hstack(
                    (
                        (
                            relevant_distances[:, None],
                            pos_angles[:, None],
                            delta_type[:, None],
                        )
                    )
                )
                nodes_list.append(nodes)
                globals_list.append([None])
                n_nodes_list.append([num_nodes])
                destinations_list.append([i for i in range(num_nodes)])
                # update the maximum number of nodes, edges and channels for the
                # padding later.
                max_num_nodes = np.maximum(max_num_nodes, num_nodes)

            else:
                pass

        # pad the graphs to the maximum number of nodes, edges and channels.
        num_graphs = len(globals_list)

        node_pad = onp.zeros((num_graphs, max_num_nodes, 3))
        destinations_pad = -1 * onp.ones((num_graphs, max_num_nodes))

        # pad the graphs to the maximum number of nodes, edges and channels.
        for i in range(num_graphs):
            node_pad[i, : len(nodes_list[i]), :] = nodes_list[i]
            destinations_pad[i, : len(destinations_list[i])] = destinations_list[i]

        graph = GraphObservable(
            nodes=node_pad,
            edges=[None] * num_graphs,
            destinations=destinations_pad.astype(int),
            globals_=globals_list,
            receivers=[None] * num_graphs,
            senders=[None] * num_graphs,
            n_node=np.array(n_nodes_list).astype(int),
            n_edge=np.zeros(num_graphs).astype(int),
        )

        return graph


class ColGraphV1(Observable):
    """ """

    def __init__(
        self,
        num_nodes: int,
        cutoff: float = 0.1,
        relation_angle: float = 0.1,
        box_size=None,
        record_memory=False,
        particle_type: int = 0,
    ):
        """
        Parameters
        ----------
        cutoff : float
            Cutoff distance for the graph.
        box_size : ndarray
            Size of the box.
        """
        self.cutoff = cutoff
        self.relation_angle = relation_angle
        if box_size is None:
            self.box_size = 1000 * np.array([1.0, 1.0, 1.0])
        else:
            self.box_size = box_size
        self.eps = 10e-8
        self.vangle = vmap(calc_signed_angle_between_directors, in_axes=(None, 0))
        self.record_memory = record_memory
        self.particle_type = particle_type
        self.num_nodes = num_nodes - 1

    def initialize(self, colloids: List[Colloid]):
        self.num_nodes = len(colloids) - 1

    def compute_observable(self, colloids: List[Colloid]) -> GraphObservable:
        """
        Builds a graph for each colloid in the system. In the graph, each node is a
        representation of a colloid within the cutoff distance.
        """
        # normalize the positions by the box size.
        positions = np.array([col.pos for col in colloids]) / self.box_size
        # directions = np.array([col.director for col in colloids])
        types = np.array([col.type for col in colloids])
        # delta_types = types[:, None] - types
        # compute the direction between all pais of colloids.
        part_part_vec = positions[:, None] - positions
        distances = np.linalg.norm(part_part_vec, axis=-1)
        part_part_vec = -1 * part_part_vec / (distances[:, :, None] + self.eps)

        nodes_list = []
        destinations_list = []
        receivers_list = []
        senders_list = []
        n_nodes_list = []
        n_edges_list = []
        globals_list = []

        for col in colloids:
            if col.type == self.particle_type:
                # mask for the colloids within the cutoff distance. without itself.
                mask = (distances[col.id] < self.cutoff) & (distances[col.id] > 0)
                # get the indices of sender and receiver within the cutoff distance.
                num_nodes = np.sum(mask)

                if num_nodes == 0:
                    nodes_list.append(np.array([np.array([1.0, 1.0, 1.0])]))
                    globals_list.append([None])
                    n_nodes_list.append([1])
                    receivers_list.append([None])
                    senders_list.append([None])
                    destinations_list.append([0])
                    continue

                director = np.copy(col.director)
                relevant_distances = distances[col.id][mask]

                relevant_part_part_vec = part_part_vec[col.id][mask]
                relevant_types = types[mask]
                pos_angles = self.vangle(director, relevant_part_part_vec)
                delta_type = relevant_types - col.type

                pair_wise_angle = (
                    np.abs(np.abs(pos_angles[:, None]) - np.abs(pos_angles)) / np.pi % 1
                )

                edge_mask = (pair_wise_angle < self.relation_angle) & (
                    pair_wise_angle > 0.0
                )
                edge_list = np.argwhere(edge_mask).T
                sender = edge_list[0]
                receiver = edge_list[1]

                num_edges = len(sender)

                receivers_list.append(receiver)
                senders_list.append(sender)

                # stack the features of the nodes.
                nodes = np.hstack(
                    (
                        (
                            relevant_distances[:, None],
                            pos_angles[:, None],
                            delta_type[:, None],
                        )
                    )
                )
                nodes_list.append(nodes)
                globals_list.append([None])
                n_nodes_list.append([num_nodes])
                n_edges_list.append([num_edges])
                destinations_list.append([i for i in range(num_nodes)])
                # update the maximum number of nodes, edges and channels for the
                # padding later.
                # max_num_nodes = np.maximum(max_num_nodes, num_nodes)
                # max_num_edges = np.maximum(max_num_edges, num_edges)

            else:
                pass

        # pad the graphs to the maximum number of nodes, edges and channels.
        num_graphs = len(globals_list)

        node_pad = onp.zeros((num_graphs, self.num_nodes, 3))
        sender_pad = -1 * onp.ones((num_graphs, self.num_nodes**2))
        receiver_pad = -1 * onp.ones((num_graphs, self.num_nodes**2))
        destinations_pad = -1 * onp.ones((num_graphs, self.num_nodes))

        # pad the graphs to the maximum number of nodes, edges and channels.
        for i in range(num_graphs):
            node_pad[i, : len(nodes_list[i]), :] = nodes_list[i]
            receiver_pad[i, : len(receivers_list[i])] = receivers_list[i]
            sender_pad[i, : len(senders_list[i])] = senders_list[i]
            destinations_pad[i, : len(destinations_list[i])] = destinations_list[i]

        graph = GraphObservable(
            nodes=node_pad,
            edges=np.zeros_like(node_pad[0]),
            destinations=destinations_pad.astype(int),
            globals_=np.zeros((num_graphs, 1)),
            receivers=receiver_pad.astype(int),
            senders=sender_pad.astype(int),
            n_node=np.array(n_nodes_list).astype(int),
            n_edge=np.array(n_edges_list).astype(int),
        )

        return graph
