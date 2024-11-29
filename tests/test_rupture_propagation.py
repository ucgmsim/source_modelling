import collections

import networkx as nx
import numpy as np
import pytest
import hypothesis_networkx as hnx
from hypothesis import strategies as st

from hypothesis import given, settings
from source_modelling import rupture_propagation
from networkx.algorithms.tree import mst


def to_weighted_directed(graph: nx.Graph) -> nx.DiGraph:
    digraph = graph.to_directed()
    for u, v in digraph.edges:
        digraph[u][v]["weight"] = digraph[u][v]["weight_uv"]
        digraph[v][u]["weight"] = digraph[v][u]["weight_vu"]
    return digraph


random_strongly_connected_graph = hnx.graph_builder(
    graph_type=nx.Graph,
    node_keys=st.text(),
    node_data=st.fixed_dictionaries(
        {"node_prior": st.floats(min_value=0.01, max_value=0.99)}
    ),
    edge_data=st.fixed_dictionaries(
        {
            "weight": st.floats(min_value=0.01, max_value=0.99),
        }
    ),
    connected=True,
    min_nodes=5,
    max_nodes=10,
    self_loops=False,
)


def graph_repr(graph: nx.Graph) -> str:
    return repr(sorted([sorted(edge) for edge in graph.edges]))


@given(graph=random_strongly_connected_graph)
@settings(deadline=None, max_examples=50)
def test_random_sampling_root(graph: nx.Graph):
    """Test that the random sampling of spanning trees has a distribution of
    root nodes that matches the expectation from the Matrix-Tree theorem."""
    n = 10000
    probability_of_spanning_trees = 0.0
    tree_probabilities: dict[str, float] = {}
    for tree in mst.SpanningTreeIterator(graph):
        p_tree = 1.0
        for u, v in graph.edges:
            if tree.has_edge(u, v):
                p_tree *= graph[u][v]["weight"]
            else:
                p_tree *= 1 - graph[u][v]["weight"]

        tree_probabilities[graph_repr(tree)] = p_tree
        probability_of_spanning_trees += p_tree

    trees = collections.Counter(
        [
            graph_repr(tree)
            for tree in rupture_propagation.sampled_spanning_tree(graph, n)
        ]
    )
    kl_divergence = 0.0
    for tree_repr, count in trees.items():
        p_tree = tree_probabilities[tree_repr] / probability_of_spanning_trees
        q_tree = count / n
        kl_divergence += p_tree * np.log(p_tree / q_tree)

    assert kl_divergence < 0.01


@pytest.mark.parametrize(
    "tree",
    [
        # Linear case, must have order [a, b, c]
        {"a": None, "b": "a", "c": "b"},
        # Splay cases (here multiple orders are ok)
        {"a": None, "b": "a", "c": "a"},
        {"a": None, "b": "a", "c": "b", "d": "a"},
    ],
)
def test_tree_nodes_in_order(tree: dict[str, str]):
    """Check that `rupture_propagation.tree_nodes_in_order` produces a list of tree nodes in-order.

    The phrase 'in-order' here means:

    1. Each node is presented after it's parent and,
    2. All nodes are eventually listed.

    The reason a simple input -> output test is not used is because
    there are multiple correct orders for a given tree and later
    implementations of this method might change the order used (and we
    don't care about which we use).
    """
    seen_nodes = set()
    for node in rupture_propagation.tree_nodes_in_order(tree):
        # Check that the node's parent was before it in the node list
        assert tree[node] in seen_nodes or tree[node] is None
        seen_nodes.add(node)

    # Check that every node is seen eventually
    assert len(seen_nodes) == len(tree)
