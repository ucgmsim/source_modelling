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

import collections
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
    graph: nx.DiGraph, roots: set[str]
) -> tuple[list[nx.DiGraph], np.ndarray]:
    trees = []
    probabilities = []
    for undirected_tree in mst.SpanningTreeIterator(graph.to_undirected()):
        for trial_root in roots:
            tree = nx.dfs_tree(undirected_tree, trial_root)
            p_tree = 1.0
            for u, v in graph.edges:
                if tree.has_edge(u, v):
                    p_tree *= graph[u][v]["weight"]
                else:
                    p_tree *= 1 - graph[u][v]["weight"]
            trees.append(tree)
            probabilities.append(p_tree)
    return trees, np.array(probabilities)


def sampled_spanning_tree(
    graph: nx.DiGraph, root: Optional[str] = None, n_samples: int = 1
) -> dict[str, Optional[str]] | list[dict[str, Optional[str]]]:
    r"""Fairly sample a random spanning tree from a graph.

    Trees from this function are sampled proportionally to the weight of the
    tree, defined as:

    P(T) = \prod_{(u, v) \in T} w(u, v) * \prod_{(u, v) \notin T}(1 - w(u, v)),

    where w(u, v) is the weight of the edge between u and v.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to sample the tree from. Higher weighted edges are included in
        the graph more often.
    root: str, optional
        If provided, then the randomly sampled tree is drawn from the
        population of random trees having `root` as their root.  Otherwise,
        the random tree will have a random root (though, not uniformly random).
    n_samples: int, optional
        If provided, then return

    Returns
    -------
    dict[str, Optional[str]]
        The sampled spanning tree if `n_samples == 1`.
    list[dict[str, Optional[str]]]
        The sampled spanning tree if `n_samples > 1`.
    """
    roots = set(graph.nodes)
    if root:
        roots = {root}
    trees, probabilities = spanning_tree_with_probabilities(graph, roots)
    tree_samples = random.choices(trees, k=n_samples, weights=probabilities)
    tree_roots = [
        next(node for node in tree.nodes if tree.in_degree(node) == 0)
        for tree in tree_samples
    ]
    samples_as_tree_functions: list[dict[str, Optional[str]]] = [
        {v: u for u, v in tree.edges} | {root: None}
        for root, tree in zip(tree_roots, tree_samples)
    ]
    if n_samples == 1:
        return samples_as_tree_functions[0]

    return samples_as_tree_functions


def graph_to_tree(tree: nx.DiGraph) -> Tree:
    root = tree_root(tree)
    return {v: u for u, v in tree.edges} | {root: None}


def tree_root(tree: nx.DiGraph) -> str:
    return next(n for n in tree.nodes if n.in_degree() == 0)


def sample_tree_with_prior(
    graph: nx.DiGraph, prior: dict[str, float], n_samples: int = 1
) -> Tree | list[Tree]:
    """Sample spanning trees with a prior probability for roots.

    Trees are sampled according to the tree sampling algorithm in
    `sampled_spanning_tree`, but are conditioned on `prior`: where P(R)
    is some prior probability of selecting a node R. Note that the prior
    dictionary probabilities do not have to sum to 1, these probabilities
    are used to condition the selection of roots via bayes theorem:

    P(tree has root R | we sample a spanning tree) =
        P(sample a spanning tree | tree has root R) * P(tree has root R)
        / P(we sample a spanning tree)

    P(tree has root R) is the prior.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph with weighted edges.
    prior : dict[str, float]
        Prior probabilities for each node being the root.
    n_samples : int, optional
        Number of trees to sample. Default is 1.

    Returns
    -------
    Tree or list[Tree]
        A sampled spanning tree if `n_samples` is 1, otherwise a list of sampled trees.
    """
    trees, probabilities = spanning_tree_with_probabilities(graph, set(graph.nodes))
    tree_roots = [tree_root(tree) for tree in trees]

    p_rupture_propagates: dict[str, float] = {}

    rooted_trees: defaultdict[str, list[nx.DiGraph]] = defaultdict(list)
    rooted_probabilities: defaultdict[str, list[float]] = defaultdict(list)
    for root, tree, probability in zip(tree_roots, trees, probabilities):
        p_rupture_propagates[root] += probability
        rooted_trees[root].append(tree)
        rooted_probabilities[root].append(probability)

    p_rupture = probabilities.sum()
    posterior_initial = {
        root: p_rupture_propagates[root] * prior[root] / p_rupture for root in prior
    }
    roots = collections.Counter(
        random.choices(
            list(posterior_initial),
            k=n_samples,
            weights=list(posterior_initial.values()),
        )
    )

    sampled_trees: list[nx.DiGraph] = []
    for root, n_root_samples in roots.items():
        sampled_trees.extend(
            random.choices(
                rooted_trees[root], weights=rooted_probabilities[root], k=n_root_samples
            )
        )

    if n_samples == 1:
        return graph_to_tree(sampled_trees[0])
    return [graph_to_tree(graph) for graph in sampled_trees]


JumpPair = namedtuple("JumpPair", ["from_point", "to_point"])


def shaw_dieterich_distance_model(distance: float, d0: float, delta: float) -> float:
    """
    Compute fault jump probabilities using the Shaw-Dieterich distance model[0].

    Parameters
    ----------
    distance : float
        The distance between two faults.
    d0 : float
        The characteristic distance parameter.
    delta : float
        The characteristic slip distance parameter.

    Returns
    -------
    float
        The calculated probability.

    References
    ----------
    [0]: Shaw, B. E., & Dieterich, J. H. (2007). Probabilities for jumping fault
         segment stepovers. Geophysical Research Letters, 34(1).
    """
    return min(1, np.exp(-(distance - delta) / d0))


def prune_distance_graph(distances: DistanceGraph, cutoff: int) -> DistanceGraph:
    """
    Prune the distance graph based on a cutoff value.

    Parameters
    ----------
    distances : DistanceGraph
        The graph of distances between faults.
    cutoff : int
        The cutoff distance in metres.

    Returns
    -------
    DistanceGraph
        A copy of the input distance graph, keeping only edges that are less
        than the cutoff.
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
    Convert a graph of distances between faults into a graph of jump
    probabilities using the Shaw-Dieterich model.

    This model assumes that the probability of a rupture jumping from fault_u to
    fault_v is independent of its jumping to any other adjacent fault.

    Parameters
    ----------
    distances : DistanceGraph
        The distance graph between faults.
    d0 : float, optional
        The d0 parameter for the Shaw-Dieterich model. See `shaw_dieterich_distance_model`.
    delta : float, optional
        The delta parameter for the Shaw-Dieterich model. See `shaw_dieterich_distance_model`.

    Returns
    -------
    nx.DiGraph
        The graph with faults as vertices. Each edge (fault_u, fault_v)
        has a probability as a weight. The probability here
        is the likelihood a rupture propagates from fault_u to fault_v,
        relative to the probability it propagates to any of the other neighbours of fault_u.
    """
    probabilities_raw = {
        fault_u: {
            fault_v: shaw_dieterich_distance_model(distance / 1000, d0, delta)
            for fault_v, distance in neighbours_fault_u.items()
        }
        for fault_u, neighbours_fault_u in distances.items()
    }
    normalising_constants = {
        fault_u: sum(probabilities.values())
        for fault_u, probabilities in probabilities_raw.items()
    }
    return nx.from_dict_of_dicts(
        {
            fault_u: {
                fault_v: {"weight": float(probability / normalising_constants[fault_u])}
                for fault_v, probability in probabilities.items()
            }
            for fault_u, probabilities in probabilities_raw.items()
        },
        create_using=nx.DiGraph,
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
    jump_impossibility_limit_distance: int = 15000,
) -> Tree:
    """Sample a rupture propagation tree from a set of sources.

    This function samples a rupture propagation tree from a set of sources. The
    tree is sampled fairly according to jump probabilities between sources,
    which are calculated using the Shaw-Dieterich distance model (see `shaw_dieterich_distance_model`).

    Parameters
    ----------
    sources_map : dict[str, sources.IsSource]
        A mapping of source names to source objects.
    initial_source : str, optional
        The name of the source to use as the root of the tree. If not provided,
        the root is chosen randomly (but not uniformly so, the root will depend
        on the number of possible valid rupture trees starting from it).
    jump_impossibility_limit_distance : int, optional
        The maximum distance between faults for which a jump is considered
        possible. Faults further than this distance apart will not be
        connected in the tree.

    Returns
    -------
    RuptureCausalityTree
        A tree representing the causality of ruptures between faults. Each key
        in the tree is a fault name, and the value is the name of the fault that
        caused the rupture of the key fault. The root of the tree is the fault
        that caused the rupture of all other faults.
    """
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

    if initial_source is None:
        return random_spanning_tree(jump_probability_graph)

    return random_spanning_tree_with_root(jump_probability_graph, initial_source)


def jump_points_from_rupture_tree(
    source_map: dict[str, sources.IsSource],
    rupture_causality_tree: Tree,
) -> dict[str, JumpPair]:
    jump_points = {}
    for source, parent in rupture_causality_tree.items():
        if parent is None:
            continue
        source_point, parent_point = sources.closest_point_between_sources(
            source_map[source], source_map[parent]
        )
        jump_points[source] = JumpPair(parent_point, source_point)
    return jump_points


def tree_nodes_in_order(
    tree: dict[str, str],
) -> Generator[str, None, None]:
    """Generate faults in topologically sorted order.

    Parameters
    ----------
    tree : dict[str, str]
        The rupture causality tree.

    Yields
    ------
    str
        The next fault in the topologically sorted order.
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
