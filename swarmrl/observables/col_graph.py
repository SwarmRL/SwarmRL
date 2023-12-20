from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Union

import jax.numpy as np
import numpy as onp
from jax import config, jit, vmap

from swarmrl.components import Colloid
from swarmrl.observables.observable import Observable
from swarmrl.utils.utils import calc_signed_angle_between_directors

ArrayTree = Union[np.ndarray, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
config.update("jax_enable_x64", True)


class GraphObservable(NamedTuple):
    nodes: Optional[ArrayTree]
    edges: Optional[ArrayTree]
    destinations: Optional[np.ndarray]
    receivers: Optional[np.ndarray]
    senders: Optional[np.ndarray]
    globals_: Optional[ArrayTree]
    n_node: np.ndarray
    n_edge: np.ndarray


@jit
def compute_dists_vecs(positions):
    """
    Compute the distances and vectors between all pairs of particles.
    """
    part_part_vec = positions[:, None] - positions
    distances = np.linalg.norm(part_part_vec, axis=-1)
    part_part_vec = -1 * part_part_vec / (distances[:, :, None] + 10e-8)
    return distances, part_part_vec


@jit
def compute_pair_wise_angle(pos_angles):
    """
    Compute the angle between all pairs of particles.
    """
    return np.abs(np.abs(pos_angles[:, None]) - np.abs(pos_angles)) / np.pi % 1


class ColGraph(Observable):
    """ """

    def __init__(
        self,
        colloids: List[Colloid],
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

        self.num_nodes = None
        self.initialize(colloids)

    def initialize(self, colloids: List[Colloid]):
        self.num_nodes = len(colloids) - 1

    def compute_initialization_input(self, colloids: List[Colloid]) -> GraphObservable:
        """
        A method that computes the input for the initialization of a graph model.
        It is used to create the train-state of the graph model.
        Instead of a graph with padded nodes for all colloids, it only
        contains a single colloid. So to speak a unvectorized version of the
        compute_observable method.

        Parameters
        ----------
        colloids : List[Colloid]
            List of colloids in the system.

        Returns
        -------
        GraphObservable
            A graph of a single colloid's observation.
        """
        system_graph = self.compute_observable(colloids)
        # get the graph for the first colloid.
        single_graph = GraphObservable(
            nodes=system_graph.nodes[0],
            edges=system_graph.edges[0],
            destinations=system_graph.destinations[0],
            receivers=system_graph.receivers[0],
            senders=system_graph.senders[0],
            globals_=system_graph.globals_[0],
            n_node=system_graph.n_node[0],
            n_edge=system_graph.n_edge[0],
        )
        return single_graph

    def compute_observable(self, colloids: List[Colloid]) -> GraphObservable:
        """
        Builds a graph for each colloid in the system. In the graph, each node is a
        representation of a colloid within the cutoff distance.
        """
        # normalize the positions by the box size.
        positions = np.array([col.pos for col in colloids]) / self.box_size
        types = np.array([col.type for col in colloids])
        # compute the direction between all pais of colloids.
        distances, part_part_vec = compute_dists_vecs(positions)

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

                director = np.copy(col.director)
                relevant_distances = distances[col.id][mask]

                relevant_part_part_vec = part_part_vec[col.id][mask]
                relevant_types = types[mask]
                pos_angles = self.vangle(director, relevant_part_part_vec)
                delta_type = relevant_types - col.type

                pair_wise_angle = compute_pair_wise_angle(pos_angles)

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


@jit
def create_batch_graphs(features) -> GraphObservable:
    """
    Create a batch of graphs from a list of graphs.

    Parameters
    ----------
    features : list(GraphObservable)
            List of graphs.

    Returns
    -------
    new_graph : GraphObservable
            Batch of graphs.
    """
    new_nodes = np.array([graph.nodes for graph in features])
    new_edges = np.array([graph.edges for graph in features])
    new_destinations = np.array([graph.destinations for graph in features])
    new_receivers = np.array([graph.receivers for graph in features])
    new_senders = np.array([graph.senders for graph in features])
    new_globals = np.array([graph.globals_ for graph in features])
    new_n_node = np.array([graph.n_node for graph in features])
    new_n_edge = np.array([graph.n_edge for graph in features])
    batched_graph = GraphObservable(
        nodes=new_nodes,
        edges=new_edges,
        destinations=new_destinations.astype(int),
        receivers=new_receivers.astype(int),
        senders=new_senders.astype(int),
        globals_=new_globals,
        n_node=new_n_node.astype(int)[0],
        n_edge=new_n_edge.astype(int),
    )
    return batched_graph
