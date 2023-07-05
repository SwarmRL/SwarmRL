from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Union

import jax.numpy as np
import numpy as onp
from jax import vmap

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable
from swarmrl.utils.utils import calc_signed_angle_between_directors

ArrayTree = Union[np.ndarray, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]


class GraphObservable(NamedTuple):
    nodes: Optional[ArrayTree]
    edges: Optional[ArrayTree]
    channels: Optional[ArrayTree]
    destinations: Optional[ArrayTree]
    receivers: Optional[np.ndarray]
    senders: Optional[np.ndarray]
    globals: Optional[ArrayTree]
    n_node: np.ndarray
    n_edge: np.ndarray


class ColGraph2(Observable):
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

    def compute_observable(self, colloids: List[Colloid]) -> List[GraphObservable]:
        """
        Builds a graph for each colloid in the system. In the graph, each node is a
        representation of a colloid within the cutoff distance.
        """
        # normalize the positions by the box size.
        positions = np.array([col.pos for col in colloids]) / self.box_size
        # directions = np.array([col.director for col in colloids])
        types = np.array([col.type for col in colloids])
        delta_types = types[:, None] - types
        # compute the direction between all pais of colloids.
        part_part_vec = positions[:, None] - positions
        distances = np.linalg.norm(part_part_vec, axis=-1)
        part_part_vec = -1 * part_part_vec / (distances[:, :, None] + self.eps)

        nodes_list = []
        edges_list = []
        channels_list = []
        destinations_list = []
        receivers_list = []
        senders_list = []
        globals_list = []
        n_node_list = []
        n_edge_list = []

        max_num_nodes = 0
        max_num_edges = 0
        max_num_channels = 0

        for col in colloids:
            if col.type == self.particle_type:
                # mask for the colloids within the cutoff distance. without itself.
                mask = (distances[col.id] < self.cutoff) & (distances[col.id] > 0)
                # get the indices of sender and receiver within the cutoff distance.
                num_nodes = np.sum(mask)

                if num_nodes == 0:
                    nodes_list.append(None)
                    edges_list.append(np.array([np.array([0, 0, 0])]))
                    channels_list.append(np.array([np.array([0, 0, 0])]))
                    destinations_list.append(np.arange(1))
                    receivers_list.append([0])
                    senders_list.append([0])
                    globals_list.append(None)
                    n_node_list.append(np.array([1]))
                    n_edge_list.append(np.array([1]))
                    continue

                director = np.copy(col.director)
                relevant_distances = distances[col.id][mask]

                relevant_part_part_vec = part_part_vec[col.id][mask]
                relevant_types = types[mask]
                pos_angles = self.vangle(director, relevant_part_part_vec)

                # compute pairwise absolute difference between the angles.
                pair_wise_angle = (
                    np.abs(np.abs(pos_angles[:, None]) - np.abs(pos_angles)) / np.pi % 1
                )

                edge_mask = (pair_wise_angle < self.relation_angle) & (
                    pair_wise_angle > 0.0
                )

                edge_list = np.argwhere(edge_mask).T
                sender = edge_list[0]
                receiver = edge_list[1]
                edges_angles = pair_wise_angle[edge_mask]
                edges_angles = np.reshape(edges_angles, (edges_angles.shape[0]))
                delta_type = relevant_types - col.type

                # stack the features of the nodes.
                channels = np.hstack(
                    (
                        (
                            relevant_distances[:, None],
                            pos_angles[:, None],
                            delta_type[:, None],
                        )
                    )
                )

                edges = np.vstack(
                    (
                        distances[mask][:, mask][edge_mask],
                        edges_angles,
                        delta_types[mask][:, mask][edge_mask],
                    )
                ).T

                nodes_list.append(None)
                edges_list.append(edges)
                channels_list.append(channels)
                destinations_list.append(np.arange(num_nodes))
                receivers_list.append(receiver)
                senders_list.append(sender)
                globals_list.append(None)
                n_node_list.append(np.array([num_nodes]))
                n_edge_list.append(np.array([edges.shape[0]]))

                # update the maximum number of nodes, edges and channels for the
                # padding later.
                max_num_nodes = np.maximum(max_num_nodes, num_nodes)
                max_num_edges = np.maximum(max_num_edges, edges.shape[0])
                max_num_channels = np.maximum(max_num_channels, channels.shape[0])

            else:
                pass

        # pad the graphs to the maximum number of nodes, edges and channels.
        num_graphs = len(globals_list)

        edge_pad = onp.zeros((num_graphs, max_num_edges, 3))
        channel_pad = onp.zeros((num_graphs, max_num_channels, 3))
        sender_pad = -1 * onp.ones((num_graphs, max_num_edges))
        receiver_pad = -1 * onp.ones((num_graphs, max_num_edges))
        destinations_pad = -1 * onp.ones((num_graphs, max_num_nodes))

        # pad the graphs to the maximum number of nodes, edges and channels.
        for i in range(num_graphs):
            edge_pad[i, : len(edges_list[i]), :] = edges_list[i]
            channel_pad[i, : channels_list[i].shape[0], :] = channels_list[i]
            receiver_pad[i, : len(receivers_list[i])] = receivers_list[i]
            sender_pad[i, : len(senders_list[i])] = senders_list[i]
            destinations_pad[i, : destinations_list[i].shape[0]] = destinations_list[i]

        graph = GraphObservable(
            nodes=nodes_list,
            edges=edge_pad,
            channels=channel_pad,
            destinations=destinations_pad,
            globals=globals_list,
            receivers=receiver_pad.astype(int),
            senders=sender_pad.astype(int),
            n_node=np.array(n_node_list).astype(int),
            n_edge=np.array(n_edge_list).astype(int),
        )

        return graph
