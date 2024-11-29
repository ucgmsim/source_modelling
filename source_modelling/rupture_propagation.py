"""
Rupture Propagation Module

This module provides functions for computing likely rupture paths from
information about the distances between faults.

Functions:
    - shaw_dieterich_distance_model:
      Compute fault jump probabilities using the Shaw-Dieterich distance model.
    - prune_distance_graph: Prune the distance graph based on a cutoff value.
    - probability_graph:
      Convert a graph of distances between faults into a graph of jump
      probabilities using the Shaw-Dieterich model.
    - sample_rupture_propagation: Sample a rupture propagation tree from a set of sources.

Typing Aliases:
    - DistanceGraph: A graph representing distances between faults.
    - RuptureCausalityTree: A tree representing the causality of ruptures between faults.
"""

import random
from collections import defaultdict, namedtuple
from typing import Generator, Optional

import networkx as nx
from networkx.algorithms.tree import mst
import numpy as np

from qcore import coordinates
from source_modelling import sources

DistanceGraph = dict[str, dict[str, float]]
Tree = dict[str, Optional[str]]


def spanning_tree_with_probabilities(
    graph: nx.DiGraph,
) -> tuple[list[nx.DiGraph], list[float]]:
    r"""
    Compute all spanning trees of a graph and their probabilities.

    Trees from have probability:

    P(T) = \prod_{(u, v) \in T} w(u, v) * \prod_{(u, v) \notin T}(1 - w(u, v)),

    where w(u, v) is the weight of the edge between u and v.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph with weighted edges.

    Returns
    -------
    list[nx.DiGraph]
        A list of spanning trees of the graph.
    list[float]
        An array of probabilities corresponding to each spanning tree. Each
        probability represents the likelihood of the corresponding spanning tree
        based on edge weights in the graph.
    """

    trees = []
    probabilities = []
    for tree in mst.SpanningTreeIterator(graph.to_undirected()):
        p_tree = 1.0
        for u, v in graph.edges:
            if tree.has_edge(u, v):
                p_tree *= graph[u][v]["weight"]
            else:
                p_tree *= 1 - graph[u][v]["weight"]
        trees.append(tree)
        probabilities.append(p_tree)

    return trees, probabilities


def sampled_spanning_tree(graph: nx.DiGraph, n_samples: int = 1) -> Tree | list[Tree]:
    r"""
    Sample spanning trees from a graph based on edge weights.

    Trees are sampled according to their probabilities, which are defined as:

    P(T) = \prod_{(u, v) \in T} w(u, v) * \prod_{(u, v) \notin T} (1 - w(u, v)),

    where w(u, v) is the weight of the edge between nodes u and v.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph from which to sample the spanning trees. Higher weighted edges
        are more likely to be included.
    n_samples : int, optional
        The number of trees to sample. Default is 1.

    Returns
    -------
    Tree or list[Tree]
        A single sampled spanning tree if `n_samples == 1`, otherwise a list of
        sampled spanning trees.
    """

    trees, probabilities = spanning_tree_with_probabilities(graph)
    return random.choices(trees, k=n_samples, weights=probabilities)


def graph_to_tree(tree: nx.DiGraph) -> Tree:
    """
    Convert a directed graph to a tree representation.

    Parameters
    ----------
    tree : nx.DiGraph
        Directed acyclic graph to be converted into a tree.

    Returns
    -------
    Tree
        A dictionary where each key is a node, and the value is its parent node.
        The root node is mapped to `None`.
    """
    root = tree_root(tree)
    return {v: u for u, v in tree.edges} | {root: None}


def tree_root(tree: nx.DiGraph) -> str:
    """
    Identify the root of a directed tree.

    The root node is defined as the node with no incoming edges.

    Parameters
    ----------
    tree : nx.DiGraph
        Directed acyclic graph representing the tree.

    Returns
    -------
    str
        The root node of the tree.
    """
    return next(n for n in tree.nodes if tree.in_degree(n) == 0)


def sample_tree_with_root_probabilities(
    graph: nx.Graph, root_probabilities: dict[str, float], n_samples: int = 1
) -> Tree | list[Tree]:
    """
    Sample spanning trees with root selection based on probabilities.

    This method combines the sampling of spanning trees with prior probabilities
    assigned to the root nodes. Trees are sampled according to their probabilities,
    and the root node is selected proportionally to the given `root_probabilities`.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph with weighted edges.
    root_probabilities : dict[str, float]
        Prior probabilities for each node to be selected as the root.
    n_samples : int, optional
        Number of trees to sample. Default is 1.

    Returns
    -------
    Tree or list[Tree]
        A single sampled spanning tree if `n_samples == 1`, otherwise a list of
        sampled spanning trees.
    """
    trees, probabilities = spanning_tree_with_probabilities(graph)

    sampled_trees: list[nx.DiGraph] = random.choices(
        trees, weights=probabilities, k=n_samples
    )
    roots = random.choices(
        list(root_probabilities), k=n_samples, weights=list(root_probabilities.values())
    )
    rooted_trees = [
        graph_to_tree(nx.dfs_tree(tree, root))
        for tree, root in zip(sampled_trees, roots)
    ]

    if n_samples == 1:
        return rooted_trees[0]
    return rooted_trees


JumpPair = namedtuple("JumpPair", ["from_point", "to_point"])


def shaw_dieterich_distance_model(distance: float, d0: float, delta: float) -> float:
    """
    Calculate fault jump probabilities using the Shaw-Dieterich distance model.

    This model computes the likelihood of a rupture propagating between two faults
    based on their separation distance and physical parameters.

    Parameters
    ----------
    distance : float
        The distance between two faults (in metres).
    d0 : float
        The characteristic distance parameter (in metres).
    delta : float
        The characteristic slip distance parameter (in metres).

    Returns
    -------
    float
        The calculated probability, clipped to a maximum of 1.

    References
    ----------
    [0] Shaw, B. E., & Dieterich, J. H. (2007). Probabilities for jumping fault
        segment stepovers. *Geophysical Research Letters*, 34(1).
    """
    return min(1, np.exp(-(distance - delta) / d0))


def prune_distance_graph(distances: DistanceGraph, cutoff: int) -> DistanceGraph:
    """
    Remove edges from a distance graph that exceed a cutoff.

    Parameters
    ----------
    distances : DistanceGraph
        A dictionary representing the graph of distances between faults.
    cutoff : int
        The maximum allowed distance (in metres) for edges in the graph.

    Returns
    -------
    DistanceGraph
        A pruned copy of the input distance graph, containing only edges with
        distances less than the cutoff.
    """
    return {
        fault_u: {
            fault_v: distance
            for fault_v, distance in neighbours_fault_u.items()
            if distance < cutoff
        }
        for fault_u, neighbours_fault_u in distances.items()
    }


def probability_graph(
    distances: DistanceGraph, d0: float = 3, delta: float = 1
) -> nx.DiGraph:
    """
    Convert a distance graph into a probability graph.

    The edge probabilities are computed using the Shaw-Dieterich model, and they
    represent the likelihood of rupture propagation between faults. Each edge
    probability is capped at 0.99 for numerical stability.

    Parameters
    ----------
    distances : DistanceGraph
        A dictionary representing distances between faults.
    d0 : float, optional
        The `d0` parameter for the Shaw-Dieterich model (in kilometres). Default is 3.
    delta : float, optional
        The `delta` parameter for the Shaw-Dieterich model (in kilometres). Default is 1.

    Returns
    -------
    nx.DiGraph
        A directed graph where edges are weighted by probabilities of rupture
        propagation between faults.
    """

    return nx.from_dict_of_dicts(
        {
            fault_u: {
                fault_v: {
                    "weight": min(
                        shaw_dieterich_distance_model(distance / 1000, d0, delta), 0.99
                    )
                }
                for fault_v, distance in neighbours_fault_u.items()
            }
            for fault_u, neighbours_fault_u in distances.items()
        },
        create_using=nx.Graph,
    )


def distance_between(
    source_a: sources.IsSource,
    source_b: sources.IsSource,
    source_a_point: np.ndarray,
    source_b_point: np.ndarray,
) -> float:
    global_point_a = source_a.fault_coordinates_to_wgs_depth_coordinates(source_a_point)
    global_point_b = source_b.fault_coordinates_to_wgs_depth_coordinates(source_b_point)
    return coordinates.distance_between_wgs_depth_coordinates(
        global_point_a, global_point_b
    )


def sample_rupture_propagation(
    sources_map: dict[str, sources.IsSource],
    initial_source: Optional[str] = None,
    initial_source_distribution: Optional[dict[str, float]] = None,
    jump_impossibility_limit_distance: int = 15000,
) -> Tree:
    """
    Generate a rupture causality tree from source faults.

    The rupture tree is constructed by sampling a spanning tree from a graph of
    faults, where edge probabilities represent the likelihood of rupture
    propagation between faults. The Shaw-Dieterich distance model is used to
    compute these probabilities.

    Parameters
    ----------
    sources_map : dict[str, sources.IsSource]
        A mapping of fault names to their corresponding source objects.
    initial_source : str, optional
        The fault to use as the root of the rupture tree. If not provided, the root
        is selected randomly based on the possible valid rupture trees.
    initial_source_distribution : dict[str, float], optional
        A probability distribution over the initial sources. Cannot be specified
        alongside `initial_source`.
    jump_impossibility_limit_distance : int, optional
        The maximum distance (in metres) for a fault-to-fault jump to be considered
        possible. Default is 15,000.

    Returns
    -------
    Tree
        A tree representing rupture causality, where each node's value is its
        parent in the tree. The root has a value of `None`.
    """

    if initial_source and initial_source_distribution:
        raise ValueError(
            "Cannot specify an initial source and initial source distribution at the same time."
        )
    distance_graph = {
        source_a_name: {
            source_b_name: distance_between(
                source_a,
                source_b,
                *sources.closest_point_between_sources(source_a, source_b),
            )
            for source_b_name, source_b in sources_map.items()
            if source_a_name != source_b_name
        }
        for source_a_name, source_a in sources_map.items()
    }
    # Prune the distance graph to remove physically impossible jumps.
    distance_graph = prune_distance_graph(
        distance_graph, jump_impossibility_limit_distance
    )
    # Convert the distance graph to a probability graph.
    jump_probability_graph = probability_graph(distance_graph)

    if not initial_source and not initial_source_distribution:
        return sample_tree_with_root_probabilities(
            jump_probability_graph,
            root_probabilities={
                source: 1.0 / len(sources_map) for source in sources_map
            },
        )

    if initial_source:
        return graph_to_tree(nx.dfs_tree(sampled_spanning_tree(jump_probability_graph)))

    return graph_to_tree(
        sample_tree_with_root_probabilities(
            jump_probability_graph, root_probabilities=initial_source_distribution
        )
    )


def jump_points_from_rupture_tree(
    source_map: dict[str, sources.IsSource],
    rupture_causality_tree: Tree,
) -> dict[str, JumpPair]:
    """
    Extract jump points between faults from a rupture causality tree.

    Jump points indicate the locations on each fault where a rupture propagates
    from the parent fault to the child fault.

    Parameters
    ----------
    source_map : dict[str, sources.IsSource]
        A mapping of fault names to their corresponding source objects.
    rupture_causality_tree : Tree
        A rupture causality tree.

    Returns
    -------
    dict[str, JumpPair]
        A dictionary mapping fault names to their jump points. Each jump point is
        represented as a `JumpPair` containing the source and destination points.
    """
    jump_points = {}
    for source, parent in rupture_causality_tree.items():
        if parent is None:
            continue
        source_point, parent_point = sources.closest_point_between_sources(
            source_map[source], source_map[parent]
        )
        jump_points[source] = JumpPair(parent_point, source_point)
    return jump_points


def tree_nodes_in_order(tree: Tree) -> Generator[str, None, None]:
    """
    Generate nodes of a tree in topologically sorted order.

    Parameters
    ----------
    tree : Tree
        A rupture causality tree where keys are nodes, and values are their parent nodes.

    Yields
    ------
    str
        The next node in the topologically sorted order.
    """
    tree_child_map = defaultdict(list)
    for cur, parent in tree.items():
        if parent:
            tree_child_map[parent].append(cur)

    def in_order_traversal(
        node: str,
    ) -> Generator[str, None, None]:
        yield node
        for child in tree_child_map[node]:
            yield from in_order_traversal(child)

    initial_fault = next(cur for cur, parent in tree.items() if not parent)
    yield from in_order_traversal(initial_fault)
