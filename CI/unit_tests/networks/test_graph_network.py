import flax.linen as nn
import jax.numpy as np
import jax.tree_util as tree
import numpy as onp
import numpy.testing as npt
from jraph._src import utils

from swarmrl.models.interaction_model import Colloid
from swarmrl.networks.graph_network_V0 import GraphModel_V0
from swarmrl.observables.col_graph_V0 import ColGraphV1, GraphObservable


def build_circle_cols(n_cols, dist=300):
    cols = []
    pos_0 = 1000 * onp.random.random(3)
    pos_0[-1] = 0
    direction_0 = onp.random.random(3)
    direction_0[-1] = 0
    for i in range(n_cols):
        theta = onp.random.random(1)[0] * 2 * np.pi
        position = pos_0 + dist * onp.array([onp.cos(theta), onp.sin(theta), 0])
        direction = onp.random.random(3)
        direction[-1] = 0
        direction = direction / onp.linalg.norm(direction)
        cols.append(Colloid(pos=position, director=direction, type=0, id=i))
    return cols


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


class TestGraphNetwork:
    @classmethod
    def setup_class(cls):
        cls.cols = build_circle_cols(10)
        cls.graph_obs = ColGraphV1(
            colloids=cls.cols, cutoff=2.0, box_size=np.array([1000, 1000, 1000])
        )

        cls.init_graph = cls.graph_obs.compute_initialization_input(cls.cols)

        cls.graph_model = GraphModel_V0(
            node_encoder=EncodeNet(),
            node_influence=InfluenceNet(),
            actress=ActNet(),
            criticer=CritNet(),
            version=2,
            init_graph=cls.init_graph,
        )

    def test_network_architecture(self):
        test_graph = GraphObservable(
            nodes=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
            edges=[None],
            receivers=np.array([0, 0, 1, 2, 2]).astype(int),
            senders=np.array([1, 2, 0, 0, 1]).astype(int),
            destinations=np.array([0, 1, 2]).astype(int),
            globals_=[None],
            n_node=np.array([3]).astype(int),
            n_edge=np.array([5]).astype(int),
        )

        graph_model = GraphModel_V0(
            node_encoder=EncodeNet(),
            node_influence=InfluenceNet(),
            actress=ActNet(),
            criticer=CritNet(),
            version=2,
            init_graph=test_graph,
        )

        graph_net = graph_model.model
        params = graph_model.model_state.params
        encoder = EncodeNet()
        actress = ActNet()
        influencer = InfluenceNet()
        criticer = CritNet()

        # unpack the graph
        nodes, _, destinations, receivers, senders, _, n_node, n_edge = test_graph

        n_nodes = n_node[0]
        # embed the nodes
        vodes = encoder.apply({"params": params["node_encoder"]}, nodes)
        npt.assert_equal(np.shape(vodes), (3, 12))
        # select the sending and receiving nodes
        sending_nodes = tree.tree_map(lambda x: x[senders], nodes)
        receiving_nodes = tree.tree_map(lambda x: x[receivers], nodes)

        npt.assert_equal(np.shape(sending_nodes), (5, 3))
        npt.assert_equal(np.shape(receiving_nodes), (5, 3))

        # this is only done to show that the sending and receiving nodes are
        # selected correctly
        # the sending nodes should be the nodes-features of nodes with
        # index 1, 2, 0, 0, 1
        # the receiving nodes should be the nodes-features of nodes
        # with index 0, 0, 1, 2, 2
        npt.assert_array_equal(
            sending_nodes,
            np.array([[2, 2, 2], [3, 3, 3], [1, 1, 1], [1, 1, 1], [2, 2, 2]]),
        )
        npt.assert_array_equal(
            receiving_nodes,
            np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 3]]),
        )

        # select the sending and receiving vodes (embedded nodes) like it is
        # actually done in the graph network
        sending_vodes = tree.tree_map(lambda x: x[senders], vodes)

        # compute the attention score and check the dimensions
        attention_score = influencer.apply(
            {"params": params["node_influence"]}, sending_vodes
        )
        npt.assert_equal(np.shape(attention_score), (5, 1))
        actual_send_messages = 10 * attention_score.squeeze()[:, None] * sending_vodes

        # compute dummy attention and dummy message to show that the graph network works
        dummy_attention_score = 0.5 * np.ones((5, 1))
        dummy_send_messages = dummy_attention_score.squeeze()[:, None] * sending_nodes

        # the send messages should be the sending nodes multiplied by the attention
        # score it should be 1/2 of the sending nodes
        npt.assert_equal(np.shape(dummy_send_messages), (5, 3))
        npt.assert_array_equal(
            dummy_send_messages,
            np.array(
                [
                    [1, 1, 1],
                    [1.5, 1.5, 1.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [1, 1, 1],
                ]
            ),
        )

        # aggregate the received dummy messages.
        # the receivers are 0, 0, 1, 2, 2
        # the received messages should be the sum of messages sent to 0, 1, 2
        # the received messages should be 2.5, 0.5, 1.5
        dummy_messages = utils.segment_sum(dummy_send_messages, receivers, n_nodes)
        npt.assert_equal(np.shape(dummy_messages), (3, 3))
        npt.assert_array_equal(
            dummy_messages,
            np.array([[2.5, 2.5, 2.5], [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]),
        )

        actual_messages = utils.segment_sum(actual_send_messages, receivers, n_nodes)
        npt.assert_equal(np.shape(actual_messages), (3, 12))

        vodes = vodes + actual_messages
        influence = influencer.apply({"params": params["node_influence"]}, vodes)
        alpha = nn.softmax(influence, axis=0)

        graph_representation = np.sum(vodes * alpha, axis=0)
        logits = actress.apply({"params": params["actress"]}, graph_representation)
        value = criticer.apply({"params": params["criticer"]}, graph_representation)

        graph_model_output = graph_net.apply({"params": params}, test_graph)

        npt.assert_array_almost_equal(graph_model_output[0], logits)
        npt.assert_array_almost_equal(graph_model_output[1], value)

    def test_init_graph_network(self):
        assert self.graph_model is not None
        assert self.graph_model.apply_fn is not None
        assert self.graph_model.model_state is not None
        assert self.graph_model.kind == "network"

    # def test_call(self):
    #     # input_graph = self.graph_obs.compute_observable(self.cols)
    #     # output = self.graph_model.compute_action(input_graph)

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
