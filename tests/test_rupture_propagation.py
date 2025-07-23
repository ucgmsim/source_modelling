import collections
import random
from typing import Optional

import hypothesis_networkx as hnx
import networkx as nx
import numpy as np
import pytest
import scipy as sp
from hypothesis import strategies as st
from networkx.algorithms.tree import mst

from source_modelling import rupture_propagation, sources

random_strongly_connected_graph = hnx.graph_builder(
    graph_type=nx.Graph,
    node_keys=st.text(
        alphabet=st.characters(
            codec="utf-8", min_codepoint=ord("a"), max_codepoint=ord("z")
        ),
        min_size=1,
        max_size=1,
    ),
    node_data=st.fixed_dictionaries(
        {"node_prior": st.floats(min_value=0.01, max_value=0.99)}
    ),
    edge_data=st.fixed_dictionaries(
        {
            "weight": st.floats(min_value=0.01, max_value=0.99),
        }
    ),
    connected=True,
    min_nodes=3,
    max_nodes=6,
    self_loops=False,
)


def graph_repr(graph: nx.Graph) -> str:
    return repr(sorted([sorted(edge) for edge in graph.edges]))


@pytest.mark.parametrize(
    "graph",
    [
        nx.from_edgelist(
            [
                (0, 1, {"weight": 0.482}),
                (0, 2, {"weight": 0.387}),
                (3, 4, {"weight": 0.473}),
            ]
        )
    ],
)
def test_disconnected_graph(graph: nx.Graph):
    with pytest.raises(
        ValueError, match="The graph must be connected to find a spanning tree."
    ):
        rupture_propagation.most_likely_spanning_tree(graph)
    with pytest.raises(
        ValueError, match="The graph must be connected to find a spanning tree."
    ):
        rupture_propagation.sampled_spanning_tree(graph, 1)


@pytest.mark.parametrize(
    "graph,n",
    [
        (
            nx.from_edgelist(
                [
                    (0, 1, {"weight": 0.482}),
                    (0, 2, {"weight": 0.387}),
                    (0, 3, {"weight": 0.473}),
                    (0, 4, {"weight": 0.140}),
                    (0, 5, {"weight": 0.248}),
                    (0, 6, {"weight": 0.554}),
                    (0, 7, {"weight": 0.279}),
                    (0, 8, {"weight": 0.322}),
                    (5, 7, {"weight": 0.329}),
                    (6, 7, {"weight": 0.366}),
                ]
            ),
            1000,
        ),
        (
            nx.from_edgelist(
                [
                    (0, 1, {"weight": 0.325}),
                    (0, 2, {"weight": 0.135}),
                    (0, 3, {"weight": 0.450}),
                    (0, 4, {"weight": 0.407}),
                    (0, 5, {"weight": 0.985}),
                    (0, 6, {"weight": 0.727}),
                    (0, 7, {"weight": 0.367}),
                    (1, 7, {"weight": 0.695}),
                    (2, 6, {"weight": 0.855}),
                    (2, 7, {"weight": 0.728}),
                ]
            ),
            1000,
        ),
        (
            nx.from_edgelist(
                [
                    (0, 1, {"weight": 0.105}),
                    (0, 2, {"weight": 0.667}),
                    (0, 3, {"weight": 0.963}),
                    (0, 4, {"weight": 0.564}),
                    (0, 5, {"weight": 0.909}),
                    (0, 6, {"weight": 0.727}),
                    (0, 7, {"weight": 0.664}),
                    (0, 8, {"weight": 0.407}),
                    (0, 9, {"weight": 0.858}),
                    (2, 4, {"weight": 0.457}),
                ]
            ),
            1000,
        ),
        (
            nx.from_edgelist(
                [
                    (0, 1, {"weight": 0.894916027316404}),
                    (0, 2, {"weight": 0.896814610789705}),
                    (0, 3, {"weight": 0.030232124838620922}),
                    (0, 4, {"weight": 0.3630489907627853}),
                    (1, 3, {"weight": 0.7509864160244057}),
                    (1, 2, {"weight": 0.21939697409137915}),
                    (2, 4, {"weight": 0.9921565707388508}),
                    (2, 3, {"weight": 0.8573802355482458}),
                    (3, 4, {"weight": 0.2873980590446522}),
                ]
            ),
            2000,
        ),
        (
            nx.from_edgelist(
                [
                    (0, 1, {"weight": 0.0559606383770036}),
                    (0, 2, {"weight": 0.701652339081683}),
                    (0, 3, {"weight": 0.6520275886642523}),
                    (0, 4, {"weight": 0.9252156741656073}),
                    (1, 3, {"weight": 0.6702484199066396}),
                    (2, 3, {"weight": 0.715224241484411}),
                    (3, 4, {"weight": 0.7295751595679032}),
                ]
            ),
            1000,
        ),
    ],
)
def test_random_sampling_root(graph: nx.Graph, n: int):
    """Test that the random sampling of spanning trees has a distribution of
    root nodes that matches the expectation from the Matrix-Tree theorem."""
    random.seed(1)
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
    eps = 1e-20  # realllyyy small constant to ensure KL-divergence works
    kl_divergence = sp.stats.entropy(
        [p_tree for tree_repr, p_tree in tree_probabilities.items()],
        [trees.get(tree_repr, 0) / n + eps for tree_repr in tree_probabilities],
    )

    assert kl_divergence < 0.1


@pytest.mark.parametrize(
    "graph, probability_threshold",
    [
        # Test case 1: Simple graph with a low probability threshold
        (
            nx.Graph(
                [
                    (0, 1, {"weight": 0.8}),
                    (1, 2, {"weight": 0.6}),
                    (2, 0, {"weight": 0.7}),
                ]
            ),
            0.1,
        ),
        # Test case 2: Simple graph with a higher probability threshold
        (
            nx.Graph(
                [
                    (0, 1, {"weight": 0.8}),
                    (1, 2, {"weight": 0.6}),
                    (2, 0, {"weight": 0.7}),
                ]
            ),
            0.9,
        ),
        # Test case 3: Larger graph with a moderate probability threshold
        (
            nx.Graph(
                [
                    (0, 1, {"weight": 0.5}),
                    (1, 2, {"weight": 0.4}),
                    (2, 3, {"weight": 0.3}),
                    (3, 0, {"weight": 0.2}),
                    (0, 2, {"weight": 0.1}),
                ]
            ),
            0.5,
        ),
        # Test case 4: Graph with edges having very low probabilities
        (
            nx.Graph(
                [
                    (0, 1, {"weight": 0.01}),
                    (1, 2, {"weight": 0.02}),
                    (2, 0, {"weight": 0.03}),
                ]
            ),
            0.05,
        ),
        # Test case 5: Graph with edges having very high probabilities
        (
            nx.Graph(
                [
                    (0, 1, {"weight": 0.99}),
                    (1, 2, {"weight": 0.98}),
                    (2, 0, {"weight": 0.97}),
                ]
            ),
            0.95,
        ),
    ],
)
def test_select_top_spanning_trees(graph: nx.Graph, probability_threshold: float):
    # Compute all spanning trees and their probabilities using the brute force method

    all_trees, all_probabilities = rupture_propagation.spanning_tree_with_probabilities(
        graph
    )
    result_trees = rupture_propagation.select_top_spanning_trees(
        graph, probability_threshold
    )
    result_trees_probabilities = []

    for tree in result_trees:
        p_tree = 1.0
        q_tree = 1.0
        for u, v in graph.edges:
            if tree.has_edge(u, v):
                p_tree *= graph[u][v]["weight"]
                q_tree *= graph[u][v]["weight"] / (1 - graph[u][v]["weight"])
            else:
                p_tree *= 1 - graph[u][v]["weight"]
        result_trees_probabilities.append(p_tree)

    total_probability = sum(all_probabilities)

    assert sum(result_trees_probabilities) / total_probability >= probability_threshold
    # Sort trees by probability
    sorted_trees = sorted(
        zip(all_trees, all_probabilities), key=lambda kv: kv[1], reverse=True
    )
    # Check that the probabilities of the trees found by the result trees matches the brute force method.
    # NOTE: We cannot compare the trees directly because equally likely trees could be in either order.
    assert [
        p_tree for _, p_tree in sorted_trees[: len(result_trees_probabilities)]
    ] == result_trees_probabilities


@pytest.mark.parametrize(
    "graph",
    [
        # Test case 1: Simple graph with a low probability threshold
        nx.Graph(
            [
                (0, 1, {"weight": 0.8}),
                (1, 2, {"weight": 0.6}),
                (2, 0, {"weight": 0.7}),
            ]
        ),
        # Test case 2: Simple graph with a higher probability threshold
        nx.Graph(
            [
                (0, 1, {"weight": 0.8}),
                (1, 2, {"weight": 0.6}),
                (2, 0, {"weight": 0.7}),
            ]
        ),
        # Test case 3: Larger graph with a moderate probability threshold
        nx.Graph(
            [
                (0, 1, {"weight": 0.5}),
                (1, 2, {"weight": 0.4}),
                (2, 3, {"weight": 0.3}),
                (3, 0, {"weight": 0.2}),
                (0, 2, {"weight": 0.1}),
            ]
        ),
        # Test case 4: Graph with edges having very low probabilities
        nx.Graph(
            [
                (0, 1, {"weight": 0.01}),
                (1, 2, {"weight": 0.02}),
                (2, 0, {"weight": 0.03}),
            ]
        ),
        # Test case 5: Graph with edges having very high probabilities
        nx.Graph(
            [
                (0, 1, {"weight": 0.99}),
                (1, 2, {"weight": 0.98}),
                (2, 0, {"weight": 0.97}),
            ]
        ),
    ],
)
def test_most_likely_spanning_tree(graph: nx.Graph):
    # Compute all spanning trees and their probabilities using the brute force method

    all_trees, all_probabilities = rupture_propagation.spanning_tree_with_probabilities(
        graph
    )
    selected_top_tree = rupture_propagation.most_likely_spanning_tree(graph)
    mst_probability = 1

    for u, v in graph.edges:
        if selected_top_tree.has_edge(u, v):
            mst_probability *= graph[u][v]["weight"]
        else:
            mst_probability *= 1 - graph[u][v]["weight"]

    # Sort trees by probability
    top_tree, top_probability = max(
        zip(all_trees, all_probabilities), key=lambda kv: kv[1]
    )

    assert (top_tree == mst) or (np.isclose(top_probability, mst_probability))


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
def test_tree_nodes_in_order(tree: rupture_propagation.Tree):
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


@pytest.mark.parametrize(
    "sources_map, initial_source, initial_source_distribution, jump_impossibility_limit_distance, expected_root_distribution",
    [
        # Test case 1: Simple sources map with equal initial source distribution
        (
            {
                "A": sources.Point(
                    np.array([-41.2865, 174.7762, 0]), 1000, 1000, 0, 0, 0
                ),
                "B": sources.Point(
                    np.array([-41.2865, 174.7862, 0]), 1000, 1000, 0, 0, 0
                ),
                "C": sources.Point(
                    np.array([-41.2865, 174.7962, 0]), 1000, 1000, 0, 0, 0
                ),
            },
            None,
            None,
            15000,
            {"A": 0.33, "B": 0.33, "C": 0.34},
        ),
        # Test case 2: Simple sources map with skewed initial source distribution
        (
            {
                "A": sources.Point(
                    np.array([-41.2865, 174.7762, 0]), 1000, 1000, 0, 0, 0
                ),
                "B": sources.Point(
                    np.array([-41.2865, 174.7862, 0]), 1000, 1000, 0, 0, 0
                ),
                "C": sources.Point(
                    np.array([-41.2865, 174.7962, 0]), 1000, 1000, 0, 0, 0
                ),
            },
            None,
            {"A": 0.1, "B": 0.2, "C": 0.7},
            15000,
            {"A": 0.1, "B": 0.2, "C": 0.7},
        ),
        # Test case 3: Initial source given
        (
            {
                "A": sources.Point(
                    np.array([-41.2865, 174.7762, 0]), 1000, 1000, 0, 0, 0
                ),
                "B": sources.Point(
                    np.array([-41.2865, 174.7862, 0]), 1000, 1000, 0, 0, 0
                ),
                "C": sources.Point(
                    np.array([-41.2865, 174.7962, 0]), 1000, 1000, 0, 0, 0
                ),
            },
            "A",
            None,
            15000,
            {"A": 1.0, "B": 1e-21, "C": 1e-21},
        ),
    ],
)
def test_sample_rupture_propagation(
    sources_map: dict[str, sources.Point],
    initial_source: str | None,
    initial_source_distribution: dict[str, float] | None,
    jump_impossibility_limit_distance: int,
    expected_root_distribution: dict[str, float],
):
    n_samples = 100
    sampled_trees = [
        rupture_propagation.sample_rupture_propagation(
            sources_map,
            initial_source=initial_source,
            initial_source_distribution=initial_source_distribution,
            jump_impossibility_limit_distance=jump_impossibility_limit_distance,
        )
        for _ in range(n_samples)
    ]

    # Extract the roots from the sampled trees
    sampled_roots = [
        node
        for tree in sampled_trees
        for node, parent in tree.items()
        if parent is None
    ]

    # Count the occurrences of each root
    root_counts = collections.Counter(sampled_roots)

    # Calculate the empirical probabilities
    empirical_probabilities = {
        root: count / n_samples for root, count in root_counts.items()
    }

    # Ensure all nodes are in empirical_probabilities
    for root in expected_root_distribution:
        if root not in empirical_probabilities:
            empirical_probabilities[root] = 0.0

    # Convert dictionaries to lists of probabilities
    empirical_probs = np.array(
        [
            empirical_probabilities[root]
            for root in sorted(expected_root_distribution.keys())
        ]
    )
    expected_probs = np.array(
        [
            expected_root_distribution[root]
            for root in sorted(expected_root_distribution.keys())
        ]
    )

    # Calculate KL divergence
    kl_divergence = sp.stats.entropy(empirical_probs, expected_probs)

    # Assert that the KL divergence is small
    assert kl_divergence < 0.1

    # Check that no sources are connected greater than the jump impossibility limit distance
    for tree in sampled_trees:
        for node, parent in tree.items():
            if parent is not None:
                distance = sources_map[node].rrup_distance(
                    sources_map[parent].coordinates
                )
                assert distance <= jump_impossibility_limit_distance


@pytest.mark.parametrize(
    "source_map, rupture_causality_tree, expected_jump_points",
    [
        # Test case 1: Simple rupture causality tree
        (
            {
                "A": sources.Point(
                    np.array([-41.2865, 174.7762, 0]), 1000, 1000, 0, 0, 0
                ),
                "B": sources.Point(
                    np.array([-41.2865, 174.7862, 0]), 1000, 1000, 0, 0, 0
                ),
                "C": sources.Point(
                    np.array([-41.2865, 174.7962, 0]), 1000, 1000, 0, 0, 0
                ),
            },
            {"A": None, "B": "A", "C": "B"},
            {
                "B": rupture_propagation.JumpPair(
                    np.array([0.5, 0.5]), np.array([0.5, 0.5])
                ),
                "C": rupture_propagation.JumpPair(
                    np.array([0.5, 0.5]), np.array([0.5, 0.5])
                ),
            },
        ),
        # Test case 2: Another simple rupture causality tree
        (
            {
                "A": sources.Point(
                    np.array([-41.2865, 174.7762, 0]), 1000, 1000, 0, 0, 0
                ),
                "B": sources.Point(
                    np.array([-41.2865, 174.7862, 0]), 1000, 1000, 0, 0, 0
                ),
                "C": sources.Point(
                    np.array([-41.2865, 174.7962, 0]), 1000, 1000, 0, 0, 0
                ),
            },
            {"A": None, "B": "A", "C": "A"},
            {
                "B": rupture_propagation.JumpPair(
                    np.array([0.5, 0.5]), np.array([0.5, 0.5])
                ),
                "C": rupture_propagation.JumpPair(
                    np.array([0.5, 0.5]), np.array([0.5, 0.5])
                ),
            },
        ),
    ],
)
def test_jump_points_from_rupture_tree(
    source_map: dict[str, sources.Point],
    rupture_causality_tree: dict[str, str | None],
    expected_jump_points: dict[str, rupture_propagation.JumpPair],
):
    result_jump_points = rupture_propagation.jump_points_from_rupture_tree(
        source_map, rupture_causality_tree
    )

    # Check if the jump points match the expected values
    for fault, expected_jump in expected_jump_points.items():
        assert np.allclose(
            result_jump_points[fault].from_point, expected_jump.from_point
        )
        assert np.allclose(result_jump_points[fault].to_point, expected_jump.to_point)
