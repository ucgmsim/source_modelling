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
    - probabilistic_minimum_spanning_tree: Generate a probabilistic minimum spanning tree.

Typing Aliases:
    - DistanceGraph: A graph representing distances between faults.
    - ProbabilityGraph: A graph representing jump probabilities between faults.
    - RuptureCausalityTree: A tree representing the causality of ruptures between faults.
"""

from collections import defaultdict, namedtuple
from typing import Generator, Optional

import numpy as np

from qcore import coordinates
from source_modelling import sources

DistanceGraph = dict[str, dict[str, int]]
ProbabilityGraph = defaultdict[str, dict[str, float]]
RuptureCausalityTree = dict[str, Optional[str]]

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
) -> ProbabilityGraph:
    """
    Convert a graph of distances between faults into a graph of jump
    probablities using the Shaw-Dieterich model.

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
    ProbabilityGraph
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
    probabilities_log = defaultdict(dict)
    for fault_u, neighbours_fault_u in probabilities_raw.items():
        normalising_constant = sum(neighbours_fault_u.values())
        if normalising_constant == 0:
            for fault_v, _ in neighbours_fault_u.items():
                probabilities_log[fault_u][fault_v] = 1 / len(neighbours_fault_u)
        for fault_v, prob in neighbours_fault_u.items():
            probabilities_log[fault_u][fault_v] = prob / normalising_constant
    return probabilities_log


def probabilistic_minimum_spanning_tree(
    probability_graph: ProbabilityGraph, initial_fault: str
) -> RuptureCausalityTree:
    """
    Generate a probabilistic minimum spanning tree.

    The minimum spanning tree of the probability graph represents rupture
    causality tree with the highest likelihood of occuring assuming that
    rupture jumps are independent.

    NOTE: While the overall probability of the minimum spanning tree is high,
    the paths from the initial fault may not be the most likely.

    Parameters
    ----------
    probability_graph : ProbabilityGraph
        The probability graph.
    initial_fault : str
        The initial fault.

    Returns
    -------
    RuptureCausalityTree
        The probabilistic minimum spanning tree.
    """
    rupture_causality_tree = {initial_fault: None}
    path_probabilities = defaultdict(lambda: np.inf)
    path_probabilities[initial_fault] = 0
    processed_faults = set()
    for _ in range(len(probability_graph)):
        current_fault = min(
            (fault for fault in probability_graph if fault not in processed_faults),
            key=lambda fault: path_probabilities[fault],
        )
        processed_faults.add(current_fault)
        for fault_neighbour, probability in probability_graph[current_fault].items():
            if (
                fault_neighbour not in processed_faults
                and probability < path_probabilities[fault_neighbour]
            ):
                path_probabilities[fault_neighbour] = probability
                rupture_causality_tree[fault_neighbour] = current_fault
    return rupture_causality_tree


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


def estimate_most_likely_rupture_propagation(
    sources_map: dict[str, sources.IsSource],
    initial_source: str,
    jump_impossibility_limit_distance: int = 15000,
) -> RuptureCausalityTree:
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
    distance_graph = prune_distance_graph(
        distance_graph, jump_impossibility_limit_distance
    )
    jump_probability_graph = probability_graph(distance_graph)
    return probabilistic_minimum_spanning_tree(
        jump_probability_graph, initial_fault=initial_source
    )


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
