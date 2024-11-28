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

import itertools
import random
from collections import defaultdict, namedtuple
from typing import Generator, Optional

import networkx as nx
import numpy as np

from qcore import coordinates
from source_modelling import sources

DistanceGraph = dict[str, dict[str, float]]
RuptureCausalityTree = dict[str, Optional[str]]


def loop_erased_random_walk(
    graph: nx.DiGraph, root: str, hitting_set: set[str]
) -> list[str]:
    """Perform a loop-erased random walk on the graph.

    A loop-erased random walk is a random walk that does not repeat itelf.

    If a graph looks like

    A -- B
    | \/ |
    | /\ |
    D -- C

    Then a *walk* in this graph would be a sequence of nodes connected by edges that may repeat, for example:

    A -> B -> C -> A -> D

    The *loop-erased random walk* is the same walk, but with the loops erased:

    A -> D

    Notice that the loop A -> B -> C -> A has been erased.

    Obviously a walk could continue forever, but usually we have a set of nodes
    we're interested in reaching, a so-called *hitting set*. For example if the
    hitting set in the above walk was C we would stopped at:

    A -> B -> C

    Of course, this walk has no loops, so this is what would be returned.

    A *loop-erased random walk* is a random walk that is then loop-erased. The
    walk randomly traverses the graph according to the weights of the edges, and
    then the loops are erased. The walk continues until it reaches a node in the
    `hitting_set`.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to perform the walk on.
    root : str
        The starting node of the walk.
    hitting_set : set[str]
        The set of nodes that the loop erased walk must end at.

    Returns
    -------
    list[str]
        A loop-erased random walk beginning at `root` that hits the `hitting_set`.

    References
    ----------
    [0]: https://en.wikipedia.org/wiki/Loop-erased_random_walk for the
    definition of the algorithm. Many other text books may have this.
    """
    # First we generate a random walk
    walk = [root]
    while walk[-1] not in hitting_set:
        neighbours = list(graph[walk[-1]])
        probabilities = [
            graph[walk[-1]][neighbour]["weight"] for neighbour in neighbours
        ]
        neighbour = np.random.choice(neighbours, p=probabilities)
        walk.append(neighbour)

    # Now we erase loops

    loop_erased_walk = [root]
    while loop_erased_walk[-1] != walk[-1]:
        loop_erased_walk.append(
            walk[
                max(i for i in range(len(walk)) if walk[i] == loop_erased_walk[-1]) + 1
            ]
        )

    return loop_erased_walk


def random_spanning_tree_with_root(
    graph: nx.DiGraph, root: str
) -> dict[str, Optional[str]]:
    r"""Fairly sample a random spanning tree with a given root from a graph.

    Trees from this function are sampled proportionally to the weight of the
    tree, defined as:

    w(T) = \prod_{(u, v) \in T} w(u, v),

    where w(u, v) is the weight of the edge between u and v. If w(u, v) is the
    probability of transitition from a vertex u to a vertex v (i.e. the graph is
    a markov chain), then w(T) is proportional to the probability of sampling
    this tree from the distribution of all weighted spanning trees.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to sample the tree from. Higher weighted edges are included in
        the graph more often.
    root : str
        The root of the tree.

    Returns
    -------
    dict[str, Optional[str]]
        The sampled spanning tree.
    """
    random_spanning_tree: dict[str, Optional[str]] = {root: None}
    while set(random_spanning_tree) != set(graph.nodes):
        to_include_vertex = random.choice(
            list(set(graph.nodes) - set(random_spanning_tree))
        )
        walk = loop_erased_random_walk(
            graph, to_include_vertex, set(random_spanning_tree)
        )
        for now, next in itertools.pairwise(walk):
            random_spanning_tree[now] = next
    return random_spanning_tree


def random_spanning_tree(graph: nx.DiGraph) -> dict[str, Optional[str]]:
    r"""Fairly sample a random spanning tree from a graph.

    Trees from this function are sampled proportionally to the weight of the
    tree, defined as:

    w(T) = \prod_{(u, v) \in T} w(u, v),

    where w(u, v) is the weight of the edge between u and v. If w(u, v) is the
    probability of transitition from a vertex u to a vertex v (i.e. the graph is
    a markov chain), then w(T) is proportional to the probability of sampling
    this tree from the distribution of all weighted spanning trees.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to sample the tree from. Higher weighted edges are included in
        the graph more often.

    Returns
    -------
    dict[str, Optional[str]]
        The sampled spanning tree.
    """
    epsilon = 1
    death_state = "__DEATH_STATE__"
    while True:
        epsilon /= 2
        graph_epsilon = graph.copy()
        # divide all the weights by 1 - epsilon
        for u, v in graph_epsilon.edges:
            graph_epsilon[u][v]["weight"] *= 1 - epsilon
        graph_epsilon.add_node(death_state)
        # add a transition from all nodes to the death state with weight epsilon
        for node in graph_epsilon.nodes:
            graph_epsilon.add_edge(node, death_state, weight=epsilon)
        tree = random_spanning_tree_with_root(graph_epsilon, death_state)
        if sum(1 for node in tree if tree[node] == death_state) == 1:
            break
    tree.pop(death_state)
    tree = {
        node: parent if parent != death_state else None for node, parent in tree.items()
    }
    return tree


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
    probablities using the Shaw-Dieterich model.

    This model assumes that the probability of a rupture jumping from fault_u to
    fault_v is independent of it's jumping to any other adjacent fault.

    Parameters
    ----------
    distances : DistanceGraph
        The distance graph between faults.
    d0 : float, optional
        The d0 parameter for the Shaw_Dieterich model. See `shaw_dieterich_distance_model`.
    delta : float, optional
        The delta parameter for the Shaw_Dieterich model. See `shaw_dieterich_distance_model`.

    Returns
    -------
    nx.DiGraph
        The graph with faults as vertices. Each edge (fault_u, fault_v)
        has a log-probability -p as a weight. The log-probability -p here
        is the negative of the log-probability a rupture propogates from
        fault_u to fault_v, relative to the probability it propogates to
        any of the other neighbours of fault_u.
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
) -> RuptureCausalityTree:
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
        impossible. Faults further than this distance apart will not be
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
    rupture_causality_tree: RuptureCausalityTree,
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
    faults : list[RealisationFault]
        List of RealisationFault objects.

    Yields
    ------
    RealisationFault
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
