from typing import List

import jax.numpy as np
from jax import vmap
from jraph._src import utils as gn_utils
from jraph._src.graph import GraphsTuple

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable
from swarmrl.utils.utils import calc_signed_angle_between_directors, save_memory


class ColGraph(Observable):
    """ """

    def __init__(self, cutoff: float = 0.1, box_size=None, record_memory=False):
        """
        Parameters
        ----------
        cutoff : float
            Cutoff distance for the graph.
        box_size : ndarray
            Size of the box.
        """
        self.cutoff = cutoff
        self.box_size = box_size
        self.eps = 10e-8
        self.vangle = vmap(calc_signed_angle_between_directors, in_axes=(None, 0))
        self.record_memory = record_memory
        self.memory = {
            "file_name": "col_graph.npy",
            "colloids": [],
            "graph_obs": [],
            "masks": [],
            "relevant_distances_memo": [],
            "relevant_part_part_vec_memo": [],
            "relevant_directions_memo": [],
        }

    def compute_observable(self, colloids: List[Colloid]) -> List[GraphsTuple]:
        """
        Builds a graph for each colloid in the system. In the graph, each node is a
        representation of a colloid within the cutoff distance.
        """
        graph_obs = []
        # normalize the positions by the box size.
        positions = np.array([col.pos for col in colloids]) / self.box_size
        directions = np.array([col.director for col in colloids])
        types = np.array([col.type for col in colloids])
        # compute the direction between all pais of colloids.
        part_part_vec = positions[:, None] - positions
        distances = np.linalg.norm(part_part_vec, axis=-1)
        part_part_vec = -1 * part_part_vec / (distances[:, :, None] + self.eps)
        for col in colloids:
            # mask for the colloids within the cutoff distance. without itself.
            mask = (distances[col.id] < self.cutoff) & (distances[col.id] > 0)

            num_nodes = np.sum(mask)
            if num_nodes == 0:
                graph_obs.append(
                    gn_utils.get_fully_connected_graph(
                        n_node_per_graph=1,
                        n_graph=1,
                        node_features=np.array([np.array([-1, -1, -1, -1])]),
                        add_self_edges=False,
                    )
                )
                continue
            director = np.copy(col.director)
            relevant_distances = distances[col.id][mask]
            relevant_part_part_vec = part_part_vec[col.id][mask]
            relevant_directions = directions[mask]
            relevant_types = types[mask]
            pos_angles = self.vangle(director, relevant_part_part_vec)
            # sight_angles = self.vangle(director, relevant_directions)
            delta_type = relevant_types - col.type

            # stack the features of the nodes.
            nodes = np.hstack(
                (
                    (
                        relevant_distances[:, None],
                        pos_angles[:, None],
                        # sight_angles[:, None],
                        delta_type[:, None],
                    )
                )
            )
            graph_obs.append(
                gn_utils.get_fully_connected_graph(
                    n_node_per_graph=len(nodes),
                    n_graph=1,
                    node_features=nodes,
                    add_self_edges=False,
                )
            )
            graph_obs[-1].senders.astype(np.float32)
            graph_obs[-1].receivers.astype(np.float32)
            if self.memory:
                self.memory["colloids"] = colloids
                self.memory["masks"].append(mask)
                self.memory["relevant_distances_memo"].append(relevant_distances)
                self.memory["relevant_part_part_vec_memo"].append(
                    relevant_part_part_vec
                )
                self.memory["relevant_directions_memo"].append(relevant_directions)

        if self.record_memory:
            self.memory["graph_obs"] = graph_obs
            self.memory = save_memory(self.memory)
        return graph_obs
