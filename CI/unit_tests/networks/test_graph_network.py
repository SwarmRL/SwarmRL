import flax.linen as nn
import jax
import jax.numpy as np
import jax.tree_util as tree
import numpy as onp

from swarmrl.models.interaction_model import Colloid
from swarmrl.networks.graph_network import GraphModel
from swarmrl.observables.col_graph import ColGraph


def build_cols(collist):
    """
    Helper function that builds a list of colloids from a list.

    Parameters
    ----------
    collist : list
        List of the number of colloids of each type.

    Returns
    -------
    cols : list

    """
    cols = []
    i = 0
    for type_cols, num_cols in enumerate(collist):
        for _ in range(num_cols):
            position = 1000 * onp.random.random(3)
            position[-1] = 0
            direction = onp.random.random(3)
            direction[-1] = 0
            direction = direction / onp.linalg.norm(direction)
            cols.append(Colloid(pos=position, director=direction, type=type_cols, id=i))
            i += 1
    return cols


class EncodeNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(8)(x)
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
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class GraphNet(nn.Module):
    encoder: EncodeNet
    actress: ActNet
    influencer: InfluenceNet

    @nn.compact
    def __call__(self, graph):
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph

        # Encode the nodes.
        embedding_vec = self.encoder(nodes)
        influence_scores = self.influencer(nodes)
        influence = jax.nn.softmax(influence_scores, axis=0)
        graph_representation = np.mean(
            tree.tree_map(lambda n, i: n * i, embedding_vec, influence), axis=0
        )
        logits = self.actress(graph_representation)
        return logits, graph_representation, influence


class TestGraphNetwork:
    def test_graph_network(self):
        col_graph = ColGraph(cutoff=0.7, box_size=np.array([1000, 1000, 1000]))

        encoder = EncodeNet()
        actress = ActNet()
        influencer = InfluenceNet()

        rng_key = 42
        graph_actor = GraphModel(
            encoder=encoder, actress=actress, influencer=influencer, rng_key=rng_key
        )
        assert graph_actor is not None

        log_probabs = []
        for i in range(150):
            cols = build_cols([3, 3])
            graph_obs = col_graph.compute_observable(cols)
            _, log_probs = graph_actor.compute_action(graph_obs)
            log_probabs.append(log_probs)
        print(log_probabs)

    def test_create_trainstate(self):
        encoder = EncodeNet()
        actress = ActNet()
        influencer = InfluenceNet()
        actor = GraphNet(encoder=encoder, actress=actress, influencer=influencer)

        cols = build_cols([5, 5])
        # build the graph.
        col_graph = ColGraph(cutoff=0.7, box_size=np.array([1000, 1000, 1000]))
        graph_obs = col_graph.compute_observable(cols)

        rng = jax.random.PRNGKey(10)
        params = actor.init(rng, graph_obs[0])["params"]

        apply_fn = jax.jit(actor.apply)

        probabs = []
        graph_reps = []
        influences = []
        for i in range(100):
            cols = build_cols([3, 3])
            graph_obs = col_graph.compute_observable(cols)
            logits, graph_rep, influence = apply_fn({"params": params}, graph_obs[0])
            probs = jax.nn.softmax(logits)
            probabs.append(probs)
            graph_reps.append(graph_rep)
            influences.append(influence)

    def test_batch_deployment(self):
        col_graph = ColGraph(cutoff=0.7, box_size=np.array([1000, 1000, 1000]))

        # create a single graph for initialization.
        cols = build_cols([10])
        graph_obs = col_graph.compute_observable(cols)
        init_graph = graph_obs[0]

        encoder = EncodeNet()
        actress = ActNet()
        influencer = InfluenceNet()
        actor = GraphNet(encoder=encoder, actress=actress, influencer=influencer)

        rng = jax.random.PRNGKey(10)
        params = actor.init(rng, init_graph)["params"]

        assert params is not None
        for key in params.keys():
            assert key in ["encoder", "actress", "influencer"]

    # def test_things(self):
    #     graph_obs = ColGraph(cutoff=3.0, box_size=np.array([1000, 1000, 1000]))
    #
    #     # nodes = onp.squeeze(nodes)
    #     nan_cols = []
    #
    #     for m in range(1000):
    #         colls = build_cols([10])
    #         graphs = graph_obs.compute_observable(colls)
    #         nodes = []
    #         for graph in graphs:
    #             nodes.append(graph.nodes)
    #
    #         nodes = onp.array(nodes)
    #         # wirte a funciton that finde "nan" in the nodes.
    #         for i, node in enumerate(nodes):
    #             for j, n in enumerate(node):
    #                 for k, obs in enumerate(n):
    #                     if np.isnan(obs):
    #                         info_dict = {"colls": colls,
    #                                      "graph": graphs[i],
    #                                      "location": (i, j, k)}
    #                         nan_cols.append(info_dict)
    #                         break
    #                     else:
    #                         pass
    #     print(f"number of nans: {len(nan_cols)}")
    #     np.save("nan_cols.npy", nan_cols, allow_pickle=True)
    #
    # def test_evaluate(self):
    #     graph_obs = ColGraph(cutoff=3.0, box_size=np.array([1000, 1000, 1000]))
    #     test_data = np.load("nan_cols.npy", allow_pickle=True)
    #     colloids = test_data[0]["colls"]
    #     graph = test_data[0]["graph"]
    #     location = test_data[0]["location"]
    #     colloid_id = location[0]
    #     new_graphs = graph_obs.compute_observable(colloids)
    #     print(new_graphs[colloid_id].nodes)
    #     print(graph.nodes)
    #     assert new_graphs[colloid_id] == graph
