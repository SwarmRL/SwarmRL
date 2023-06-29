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


class ColGraph(Observable):
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
        graph_obs = []
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
                    print("No nodes within the cutoff distance")
                    graph_obs.append(
                        GraphObservable(
                            nodes=None,
                            edges=np.array([np.array([0, 0, 0])]),
                            channels=np.array([np.array([0, 0, 0])]),
                            destinations=None,
                            globals=None,
                            receivers=[0],
                            senders=[0],
                            n_node=np.array([1]),
                            n_edge=np.array([1]),
                        )
                    )
                    continue

                director = np.copy(col.director)
                relevant_distances = distances[col.id][mask]

                relevant_part_part_vec = part_part_vec[col.id][mask]
                relevant_types = types[mask]
                pos_angles = self.vangle(director, relevant_part_part_vec)

                # compute pairwise absolute difference between the angles.
                pair_wise_angle = np.abs(pos_angles[:, None] - pos_angles) / np.pi % 1

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

                if len(edges) == 0:
                    edges = np.array([np.array([0, 0, 0])])

                graph_obs.append(
                    GraphObservable(
                        nodes=None,
                        edges=edges,
                        channels=channels,
                        destinations=None,
                        globals=None,
                        receivers=receiver,
                        senders=sender,
                        n_node=np.array([num_nodes]),
                        n_edge=np.array([edges.shape[0]]),
                    )
                )
                nodes_list.append(None)
                edges_list.append(edges)
                channels_list.append(channels)
                destinations_list.append(None)
                receivers_list.append(receiver)
                senders_list.append(sender)
                globals_list.append(None)
                n_node_list.append(np.array([num_nodes]))
                n_edge_list.append(np.array([edges.shape[0]]))

                if num_nodes > max_num_nodes:
                    max_num_nodes = num_nodes
                if edges.shape[0] > max_num_edges:
                    max_num_edges = edges.shape[0]
                if channels.shape[0] > max_num_channels:
                    max_num_channels = channels.shape[0]

                graph_obs[-1].senders.astype(np.float32)
                graph_obs[-1].receivers.astype(np.float32)
            else:
                pass

        num_graphs = len(graph_obs)
        edge_pad = onp.zeros((num_graphs, max_num_edges, 3))
        channel_pad = onp.zeros((num_graphs, max_num_channels, 3))
        # node_pad = np.zeros((num_graphs, max_num_nodes, 3))
        sender_pad = -1 * onp.ones((num_graphs, max_num_edges))
        receiver_pad = -1 * onp.ones((num_graphs, max_num_edges))

        for i in range(len(graph_obs)):
            edge_pad[i, : graph_obs[i].edges.shape[0], :] = graph_obs[i].edges
            channel_pad[i, : graph_obs[i].channels.shape[0], :] = graph_obs[i].channels
            receiver_pad[i, : len(graph_obs[i].receivers)] = graph_obs[i].receivers
            sender_pad[i, : len(graph_obs[i].senders)] = graph_obs[i].senders

        second_return = [
            nodes_list,
            edge_pad,
            channel_pad,
            destinations_list,
            receiver_pad,
            sender_pad,
            globals_list,
            n_node_list,
            n_edge_list,
        ]

        print("max_num_nodes", max_num_nodes)
        print("max_num_edges", max_num_edges)
        print("max_num_channels", max_num_channels)

        return graph_obs, second_return
