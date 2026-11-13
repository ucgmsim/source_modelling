import collections
import random

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
        ValueError, match=r"^The graph must be connected to find a spanning tree\.$"
    ):
        rupture_propagation.most_likely_spanning_tree(graph)
    with pytest.raises(
        ValueError, match=r"^The graph must be connected to find a spanning tree\.$"
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
    for tree in mst.SpanningTreeIterator(graph):  # ty: ignore[invalid-argument-type]
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
        # Test case 2: Larger graph with a moderate probability threshold
        nx.Graph(
            [
                (0, 1, {"weight": 0.5}),
                (1, 2, {"weight": 0.4}),
                (2, 3, {"weight": 0.3}),
                (3, 0, {"weight": 0.2}),
                (0, 2, {"weight": 0.1}),
            ]
        ),
        # Test case 3: Graph with edges having very low probabilities
        nx.Graph(
            [
                (0, 1, {"weight": 0.01}),
                (1, 2, {"weight": 0.02}),
                (2, 0, {"weight": 0.03}),
            ]
        ),
        # Test case 4: Graph with edges having very high probabilities
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
            sources_map,  # ty: ignore[invalid-argument-type]
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


def test_sample_rupture_propagation_solves_each_pair_once(
    monkeypatch: pytest.MonkeyPatch,
):
    """Closest points should be solved once per unordered pair of sources."""
    sources_map = {
        name: sources.Point(np.array([-41.2865, lon, 0]), 1000, 1000, 0, 0, 0)
        for name, lon in zip("ABCD", [174.7762, 174.7862, 174.7962, 174.8062])
    }
    closest_point_between_sources = sources.closest_point_between_sources
    calls = []

    def counting_closest_point_between_sources(
        source_a: sources.IsSource, source_b: sources.IsSource
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((source_a, source_b))
        return closest_point_between_sources(source_a, source_b)

    monkeypatch.setattr(
        sources,
        "closest_point_between_sources",
        counting_closest_point_between_sources,
    )
    tree = rupture_propagation.sample_rupture_propagation(
        sources_map,  # ty: ignore[invalid-argument-type]
        initial_source="A",
    )
    n = len(sources_map)
    assert len(calls) == n * (n - 1) // 2
    assert set(tree) == set(sources_map)


@pytest.mark.parametrize(
    "source_map, rupture_causality_tree, min_depth, expected_jump_points",
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
            None,
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
            None,
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
    min_depth: float | None,
    expected_jump_points: dict[str, rupture_propagation.JumpPair],
):
    result_jump_points = rupture_propagation.jump_points_from_rupture_tree(
        source_map,  # ty: ignore[invalid-argument-type]
        rupture_causality_tree,
        min_depth,
    )

    # Check if the jump points match the expected values
    for fault, expected_jump in expected_jump_points.items():
        assert np.allclose(
            result_jump_points[fault].from_point, expected_jump.from_point
        )
        assert np.allclose(result_jump_points[fault].to_point, expected_jump.to_point)


# The bottom depth (in metres) of the sources used in the minimum depth
# tests below (planes 10km wide dipping at 45 degrees from the surface).
_DIPPING_PLANE_BOTTOM_M = 10_000 * np.sin(np.radians(45))


@pytest.mark.parametrize(
    "min_depth, expected_jump_depth_m",
    [
        # No minimum depth: the sources dip away from each other, so the
        # closest points are at the surface.
        (None, 0),
        # The minimum depth lies within both sources and so is respected exactly.
        (5, 5000),
        # The minimum depth is below the bottom of both sources and so is
        # clamped to just above their bottom depth.
        (100, 0.99 * _DIPPING_PLANE_BOTTOM_M),
    ],
)
def test_jump_points_from_rupture_tree_min_depth(
    min_depth: float | None, expected_jump_depth_m: float
):
    # Two planes dipping away from each other, so that the distance
    # between them increases with depth.
    source_map = {
        "A": sources.Plane.from_centroid_strike_dip(
            np.array([-41.2865, 174.7762]),
            dip=45,
            length=10,
            width=10,
            dtop=0,
            strike_nztm=0,
            dip_dir_nztm=270,
        ),
        "B": sources.Plane.from_centroid_strike_dip(
            np.array([-41.2865, 174.8762]),
            dip=45,
            length=10,
            width=10,
            dtop=0,
            strike_nztm=0,
            dip_dir_nztm=90,
        ),
    }

    result_jump_points = rupture_propagation.jump_points_from_rupture_tree(
        source_map,  # ty: ignore[invalid-argument-type]
        {"A": None, "B": "A"},
        min_depth,
    )

    jump = result_jump_points["B"]
    from_depth_m = source_map["A"].fault_coordinates_to_wgs_depth_coordinates(
        jump.from_point
    )[-1]
    to_depth_m = source_map["B"].fault_coordinates_to_wgs_depth_coordinates(
        jump.to_point
    )[-1]

    # The sources dip away from each other, so the jump is made as
    # shallow as the minimum depth allows.
    assert from_depth_m == pytest.approx(expected_jump_depth_m, abs=1)
    assert to_depth_m == pytest.approx(expected_jump_depth_m, abs=1)


def test_shaw_dieterich_distance_model_at_delta():
    assert rupture_propagation.shaw_dieterich_distance_model(1.0, 3, 1) == 1


def test_shaw_dieterich_distance_model_decay():
    assert rupture_propagation.shaw_dieterich_distance_model(
        4.0, 3, 1
    ) == pytest.approx(np.exp(-1))


def test_shaw_dieterich_distance_model_clipped_below_delta():
    assert rupture_propagation.shaw_dieterich_distance_model(0.0, 3, 1) == 1


def test_prune_distance_graph_removes_edges_beyond_cutoff():
    assert rupture_propagation.prune_distance_graph(
        {"a": {"b": 10.0, "c": 20.0}}, cutoff=15.0, tolerance=0.0
    ) == {"a": {"b": 10.0}}


def test_prune_distance_graph_cutoff_is_inclusive():
    # The cutoff is a maximum separation, so a pair exactly at it is kept.
    assert rupture_propagation.prune_distance_graph(
        {"a": {"b": 15.0}}, cutoff=15.0, tolerance=0.0
    ) == {"a": {"b": 15.0}}


def test_prune_distance_graph_keeps_isolated_faults():
    assert rupture_propagation.prune_distance_graph(
        {"a": {"b": 20.0}, "b": {"a": 20.0}}, cutoff=15.0, tolerance=0.0
    ) == {"a": {}, "b": {}}


def test_probability_graph_converts_metres_to_kilometres():
    graph = rupture_propagation.probability_graph(
        {"a": {"b": 4000.0}, "b": {"a": 4000.0}}
    )
    assert graph["a"]["b"]["weight"] == pytest.approx(np.exp(-1))


def test_probability_graph_caps_weight():
    graph = rupture_propagation.probability_graph({"a": {"b": 0.0}, "b": {"a": 0.0}})
    assert graph["a"]["b"]["weight"] == pytest.approx(0.99)


def test_probability_graph_keeps_isolated_nodes():
    graph = rupture_propagation.probability_graph(
        {"a": {"b": 4000.0}, "b": {"a": 4000.0}, "c": {}}
    )
    assert "c" in graph.nodes
    assert graph.degree("c") == 0


def test_probability_graph_records_distance():
    graph = rupture_propagation.probability_graph(
        {"a": {"b": 4000.0}, "b": {"a": 4000.0}}
    )
    assert graph["a"]["b"]["distance"] == pytest.approx(4000.0)


def test_most_likely_spanning_tree_breaks_cap_ties_by_distance():
    # AB, AC and BC are all at the cap. Inserted in the order AB, AC, BC,
    # an insertion-order tie-break keeps AC and drops BC, although BC is
    # the shorter jump. With the distance tie-break, BC is kept.
    distances = {
        "a": {"b": 500.0, "c": 1030.0},
        "b": {"a": 500.0, "c": 800.0},
        "c": {"a": 1030.0, "b": 800.0},
    }
    graph = rupture_propagation.probability_graph(distances, d0=5, delta=1)
    assert all(
        data["weight"] == pytest.approx(0.99) for *_, data in graph.edges(data=True)
    )
    tree = rupture_propagation.most_likely_spanning_tree(graph)
    assert {frozenset(edge) for edge in tree.edges} == {
        frozenset(("a", "b")),
        frozenset(("b", "c")),
    }


@pytest.mark.parametrize("d0", [2.0, 3.0, 5.0, 20.0])
def test_most_likely_spanning_tree_is_independent_of_d0(d0: float):
    # The same distances under every length scale give the same tree: the
    # distance model preserves the ordering of the separations, and the
    # cap ties are broken by separation.
    distances = {
        "a": {"b": 500.0, "c": 1030.0, "d": 6000.0},
        "b": {"a": 500.0, "c": 800.0, "d": 4000.0},
        "c": {"a": 1030.0, "b": 800.0, "d": 1045.0},
        "d": {"a": 6000.0, "b": 4000.0, "c": 1045.0},
    }
    reference = rupture_propagation.most_likely_spanning_tree(
        rupture_propagation.probability_graph(distances, d0=2.0, delta=1)
    )
    tree = rupture_propagation.most_likely_spanning_tree(
        rupture_propagation.probability_graph(distances, d0=d0, delta=1)
    )
    assert {frozenset(edge) for edge in tree.edges} == {
        frozenset(edge) for edge in reference.edges
    }


def test_most_likely_spanning_tree_without_distance_keeps_insertion_order():
    graph = nx.Graph(
        [
            ("a", "b", {"weight": 0.99}),
            ("a", "c", {"weight": 0.99}),
            ("b", "c", {"weight": 0.99}),
        ]
    )
    tree = rupture_propagation.most_likely_spanning_tree(graph)
    assert {frozenset(edge) for edge in tree.edges} == {
        frozenset(("a", "b")),
        frozenset(("a", "c")),
    }


def test_prune_distance_graph_tolerance():
    # The default allowance of 100 m keeps a pair 57 m beyond the cutoff
    # (Alpine: Jacksons to Kaniere to Hunter Valley in the NZ NSHM 2022,
    # 15.057 km against a 15 km limit) and drops one 101 m beyond it.
    distances = {"a": {"b": 15_057.0, "c": 15_101.0}}
    assert rupture_propagation.prune_distance_graph(distances, cutoff=15_000.0) == {
        "a": {"b": 15_057.0}
    }
    assert rupture_propagation.prune_distance_graph(
        distances, cutoff=15_000.0, tolerance=0.0
    ) == {"a": {}}


def _triangle(weight_01: float, weight_12: float, weight_02: float) -> nx.Graph:
    return nx.Graph(
        [
            (0, 1, {"weight": weight_01}),
            (1, 2, {"weight": weight_12}),
            (0, 2, {"weight": weight_02}),
        ]
    )


def _edge_set(graph: nx.Graph) -> set[frozenset]:
    return {frozenset(edge) for edge in graph.edges}


def test_spanning_tree_with_probabilities_exact():
    # Odds of the edges are 4, 1 and 1. Each tree's unnormalised weight is
    # the product of w for its edges and (1 - w) for the excluded edge.
    graph = _triangle(0.8, 0.5, 0.5)
    trees, probabilities = rupture_propagation.spanning_tree_with_probabilities(graph)
    assert len(trees) == len(probabilities) == 3
    result = {
        frozenset(_edge_set(tree)): probability
        for tree, probability in zip(trees, probabilities)
    }
    assert result == pytest.approx(
        {
            frozenset({frozenset((0, 1)), frozenset((1, 2))}): 0.8 * 0.5 * 0.5,
            frozenset({frozenset((0, 1)), frozenset((0, 2))}): 0.8 * 0.5 * 0.5,
            frozenset({frozenset((1, 2)), frozenset((0, 2))}): 0.5 * 0.5 * 0.2,
        }
    )
    for tree in trees:
        assert nx.is_tree(tree)
        assert set(tree.nodes) == set(graph.nodes)


@pytest.mark.parametrize(
    "probability_threshold, expected_number_of_trees",
    [
        # The two trees with edge (0, 1) each carry 4/9 of the weight, the
        # remaining tree 1/9.
        (0.4, 1),
        (0.85, 2),
        (0.95, 3),
    ],
)
def test_select_top_spanning_trees_exact_count(
    probability_threshold: float, expected_number_of_trees: int
):
    graph = _triangle(0.8, 0.5, 0.5)
    trees = rupture_propagation.select_top_spanning_trees(graph, probability_threshold)
    assert len(trees) == expected_number_of_trees
    # Trees are returned in descending order of probability, so the
    # likely trees (containing the 0.8 edge) come first.
    for tree in trees[:2]:
        assert frozenset((0, 1)) in _edge_set(tree)
    # The input graph is not modified.
    assert graph[0][1]["weight"] == 0.8


@pytest.mark.parametrize(
    "probability_threshold, expected_number_of_trees",
    [
        # All three trees have weight 1/16 out of a total of 3/16, and these
        # ratios are exact in floating point, so the threshold is met exactly.
        (1 / 3, 1),
        (2 / 3, 2),
        (1.0, 3),
    ],
)
def test_select_top_spanning_trees_threshold_is_inclusive(
    probability_threshold: float, expected_number_of_trees: int
):
    graph = _triangle(0.2, 0.2, 0.2)
    trees = rupture_propagation.select_top_spanning_trees(graph, probability_threshold)
    assert len(trees) == expected_number_of_trees


def test_most_likely_spanning_tree_returns_log_odds_and_attributes():
    graph = nx.Graph()
    graph.add_node("a", node_prior=0.1)
    graph.add_node("b", node_prior=0.2)
    graph.add_node("c", node_prior=0.3)
    graph.add_edge("a", "b", weight=0.8, distance=100.0)
    graph.add_edge("b", "c", weight=0.6, distance=200.0)
    graph.add_edge("a", "c", weight=0.3, distance=300.0)

    tree = rupture_propagation.most_likely_spanning_tree(graph)

    assert dict(tree.nodes(data=True)) == {
        "a": {"node_prior": 0.1},
        "b": {"node_prior": 0.2},
        "c": {"node_prior": 0.3},
    }
    assert _edge_set(tree) == {frozenset(("a", "b")), frozenset(("b", "c"))}
    assert tree["a"]["b"]["weight"] == pytest.approx(np.log(0.8 / 0.2))
    assert tree["b"]["c"]["weight"] == pytest.approx(np.log(0.6 / 0.4))
    assert tree["a"]["b"]["distance"] == 100.0
    assert tree["b"]["c"]["distance"] == 200.0
    # The input graph is not modified.
    assert graph["a"]["b"]["weight"] == 0.8


def test_most_likely_spanning_tree_mixed_distance_attributes():
    # Equal weights where only some edges carry a distance must still
    # produce a spanning tree rather than failing to compare the keys.
    graph = nx.Graph(
        [
            ("a", "b", {"weight": 0.99, "distance": 500.0}),
            ("a", "c", {"weight": 0.99}),
            ("b", "c", {"weight": 0.99, "distance": 100.0}),
        ]
    )
    tree = rupture_propagation.most_likely_spanning_tree(graph)
    assert nx.is_tree(tree)
    assert set(tree.nodes) == {"a", "b", "c"}
    # The longest jump is the one left out.
    assert frozenset(("a", "b")) not in _edge_set(tree)


def test_sample_rupture_propagation_rejects_source_and_distribution():
    sources_map = {
        name: sources.Point(np.array([-41.2865, lon, 0]), 1000, 1000, 0, 0, 0)
        for name, lon in zip("AB", [174.7762, 174.7862])
    }
    with pytest.raises(
        ValueError,
        match="Cannot specify an initial source and initial source distribution",
    ):
        rupture_propagation.sample_rupture_propagation(
            sources_map,  # ty: ignore[invalid-argument-type]
            initial_source="A",
            initial_source_distribution={"A": 0.5, "B": 0.5},
        )
