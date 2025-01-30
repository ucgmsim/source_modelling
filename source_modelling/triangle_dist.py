from typing import NamedTuple

import gpytoolbox
import numba
import numpy as np


@numba.njit
def edge_edge_distance(P1, Q1, P2, Q2):
    """
    Computes the distance between two edges (segments) in 3D space.

    Parameters
    ----------
    P1 : (3,) numpy array
        start point of first edge
    Q1 : (3,) numpy array
        end point of first edge
    P2 : (3,) numpy array
        start point of second edge
    Q2 : (3,) numpy array
        end point of second edge

    Returns
    -------
    d : float
        The distance between the two edges.
    R1 : (3,) numpy array
        The closest point on the first edge to the second edge.
    R2 : (3,) numpy array
        The closest point on the second edge to the first edge.

    Notes
    -----
    This function is based on the algorithm from the "Proximity Query Pack" by Eric Larsen and Stefan Gottschalk

    Examples
    --------
    ```python
    import numpy as np
    from gpytoolbox import edge_edge_distance
    P1 = np.array([0.0,0.0,0.0])
    P2 = np.array([1.0,0.0,0.0])
    Q1 = np.array([0.0,1.0,0.0])
    Q2 = np.array([1.0,1.0,0.0])
    dist,R1,R2 = gpytoolbox.edge_edge_distance(P1,Q1,P2,Q2)
    ```
    """

    P = P1
    Q = P2
    A = Q1 - P1
    B = Q2 - P2
    T = Q - P
    A_dot_A = np.dot(A, A)
    B_dot_B = np.dot(B, B)
    A_dot_B = np.dot(A, B)
    A_dot_T = np.dot(A, T)
    B_dot_T = np.dot(B, T)

    denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B
    if denom == 0:
        t = 0
    else:
        t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom

    if (t < 0) or np.isnan(t):
        t = 0
    elif t > 1:
        t = 1

    if B_dot_B == 0:
        u = 0
    else:
        u = (t * A_dot_B - B_dot_T) / B_dot_B

    if (u <= 0) or np.isnan(u):
        Y = Q.copy()
        t = A_dot_T / A_dot_A
        if (t <= 0) or np.isnan(t):
            X = P.copy()
        elif t >= 1:
            X = P + A
        else:
            X = P + t * A
    elif u >= 1:
        Y = Q + B
        t = (A_dot_T + A_dot_B) / A_dot_A
        if (t <= 0) or np.isnan(t):
            X = P.copy()
        elif t >= 1:
            X = P + A
        else:
            X = P + t * A
    else:
        Y = Q + u * B
        if (t <= 0) or np.isnan(t):
            X = P.copy()
        elif t >= 1:
            X = P + A
        else:
            X = P + t * A

    R1 = X
    R2 = Y
    dist = np.square(R1 - R2).sum()
    return dist, R1, R2


@numba.njit
def triangle_triangle_distance(s0, s1, s2, t0, t1, t2):
    """Compute the distance between two triangles.
    Parameters
    ----------
    s0 : (3,) array
        First vertex of first triangle.
    s1 : (3,) array
        Second vertex of first triangle.
    s2 : (3,) array
        Third vertex of first triangle.
    t0 : (3,) array
        First vertex of second triangle.
    t1 : (3,) array
        Second vertex of second triangle.
    t2 : (3,) array
        Third vertex of second triangle.
    Returns
    -------
    d : float
        Distance between the two triangles.
    s : (3,) array
        Closest point on first triangle.
    t : (3,) array
        Closest point on second triangle.
    Notes
    -----
    This function is based on the algorithm from the "Proximity Query Pack" by Eric Larsen and Stefan Gottschalk. It would be nice to also output the closest point but this is hard in the case where the triangles intersect each other. Whenever we have a pure python triangle-triangle intersection function, we can use it here.

    Examples
    --------
    ```python
    import numpy as np
    from gpytoolbox import triangle_triangle_distance
    s0 = np.array([0.0,0.0,0.0])
    s1 = np.array([1.0,0.0,0.0])
    s2 = np.array([0.0,1.0,0.0])
    t0 = np.array([0.0,0.0,1.0])
    t1 = np.array([1.0,0.0,1.0])
    t2 = np.array([0.0,1.0,1.0])
    dist,s,t = triangle_triangle_distance(s0,s1,s2,t0,t1,t2)
    ```
    """
    shown_disjoint = False
    triangle_s = np.vstack((s0, s1, s2))
    triangle_t = np.vstack((t0, t1, t2))
    s_triangle_edges = np.vstack((s1 - s0, s2 - s1, s0 - s2))
    t_triangle_edges = np.vstack((t1 - t0, t2 - t1, t0 - t2))

    min_dist = np.inf
    min_p = None
    min_q = None

    for i in range(3):
        for j in range(3):
            dist_sq, edge_p, edge_q = edge_edge_distance(
                triangle_s[i],
                triangle_s[i] + s_triangle_edges[i],
                triangle_t[j],
                triangle_t[j] + t_triangle_edges[j],
            )
            vec_distance = edge_p - edge_q
            if dist_sq <= min_dist:
                min_p = edge_p
                min_q = edge_q
                min_dist = dist_sq
                a = np.dot(triangle_s[(i + 2) % 3] - edge_p, vec_distance)
                b = np.dot(triangle_t[(j + 2) % 3] - edge_q, vec_distance)
                if a <= 0 and b >= 0:
                    return np.sqrt(min_dist), min_p, min_q
                p = np.dot(vec_distance, vec_distance)
                if a < 0:
                    a = 0
                if b > 0:
                    b = 0
                if (p - a + b) > 0:
                    shown_disjoint = True

    s_triangle_normal = np.cross(s_triangle_edges[0], s_triangle_edges[1])
    s_triangle_norm_length = np.dot(s_triangle_normal, s_triangle_normal)

    if s_triangle_norm_length > 1e-15:
        t_triangle_project_onto_s = np.dot(s_triangle_normal, triangle_s - triangle_t)
        point = -1
        if np.all(t_triangle_project_onto_s > 0):
            if t_triangle_project_onto_s[0] < t_triangle_project_onto_s[1]:
                point = 0
            else:
                point = 1
            if t_triangle_project_onto_s[2] < t_triangle_project_onto_s[point]:
                point = 2
        elif np.all(t_triangle_project_onto_s < 0):
            if t_triangle_project_onto_s[0] > t_triangle_project_onto_s[1]:
                point = 0
            else:
                point = 1
            if t_triangle_project_onto_s[2] > t_triangle_project_onto_s[point]:
                point = 2
        if point >= 0:
            shown_disjoint = True
            t_edge = triangle_t[point] - triangle_s[0]
            s_edge_norm = np.cross(s_triangle_normal, s_triangle_edges[0])
            if np.dot(t_edge, s_edge_norm) > 0:
                t_edge = triangle_t[point] - triangle_s[1]
                s_edge_norm = np.cross(s_triangle_normal, s_triangle_edges[1])
                if np.dot(t_edge, s_edge_norm) > 0:
                    t_edge = triangle_t[point] - triangle_s[2]
                    s_edge_norm = np.cross(s_triangle_normal, s_triangle_edges[2])
                    if np.dot(t_edge, s_edge_norm) > 0:
                        edge_p = (
                            triangle_t[point]
                            + s_triangle_normal
                            * (t_triangle_project_onto_s[point])
                            / s_triangle_norm_length
                        )
                        edge_q = triangle_t[point].copy()
                        vec_distance = edge_p - edge_q
                        return (
                            np.sqrt(np.dot(vec_distance, vec_distance)),
                            edge_p,
                            edge_q,
                        )

    t_triangle_normal = np.cross(t_triangle_edges[0], t_triangle_edges[1])
    t_triangle_normal_length = np.dot(t_triangle_normal, t_triangle_normal)

    if t_triangle_normal_length > 1e-15:
        s_triangle_project_onto_t = np.dot(t_triangle_normal, triangle_t - triangle_s)
        point = -1
        if np.all(s_triangle_project_onto_t > 0):
            if s_triangle_project_onto_t[0] < s_triangle_project_onto_t[1]:
                point = 0
            else:
                point = 1
            if s_triangle_project_onto_t[2] < s_triangle_project_onto_t[point]:
                point = 2
        elif np.all(s_triangle_project_onto_t < 0):
            if s_triangle_project_onto_t[0] > s_triangle_project_onto_t[1]:
                point = 0
            else:
                point = 1
            if s_triangle_project_onto_t[2] > s_triangle_project_onto_t[point]:
                point = 2

        if point >= 0:
            shown_disjoint = True
            t_edge = triangle_s[point] - triangle_t[0]
            s_edge_norm = np.cross(t_triangle_normal, t_triangle_edges[0])
            if np.dot(t_edge, s_edge_norm) > 0:
                t_edge = triangle_s[point] - triangle_t[1]
                s_edge_norm = np.cross(t_triangle_normal, t_triangle_edges[1])
                if np.dot(t_edge, s_edge_norm) > 0:
                    t_edge = triangle_s[point] - triangle_t[2]
                    s_edge_norm = np.cross(t_triangle_normal, t_triangle_edges[2])
                    if np.dot(t_edge, s_edge_norm) > 0:
                        edge_q = (
                            triangle_s[point]
                            + t_triangle_normal
                            * (s_triangle_project_onto_t[point])
                            / t_triangle_normal_length
                        )
                        edge_p = triangle_s[point].copy()
                        return (
                            np.sqrt(np.dot(vec_distance, vec_distance)),
                            edge_p,
                            edge_q,
                        )

    if shown_disjoint:
        return np.sqrt(min_dist), min_p, min_q
    else:
        return 0.0, min_p, min_q


class AABBTree(NamedTuple):
    centres: np.ndarray
    widths: np.ndarray
    children: np.ndarray
    parents: np.ndarray
    depths: np.ndarray
    tri_indices: np.ndarray
    split_dir: np.ndarray


class Mesh(NamedTuple):
    vertices: np.ndarray
    faces: np.ndarray


def minimum_distance(mesh_1: Mesh, mesh_2: Mesh):
    """
    Compute the minimum distance between two triangle meshes in 3D.

    Parameters
    ----------
    v1 : (n1,3) array
        Vertices of first mesh.
    f1 : (m1,3) array
        Faces of first mesh.
    v2 : (n2,3) array
        Vertices of second mesh.
    f2 : (m2,3) array
        Faces of second mesh.

    Returns
    -------
    d : float
        Minimum distance value.

    Notes
    -----
    This function could be extended with polyline and pointcloud functionality without much trouble.

    Examples
    --------
    ```python
    # meshes in v,f and u,g
    # Minimum distance value
    d = gpytoolbox.minimum_distance(v,f,u,g)
    ```
    """

    # Initialize AABB tree for mesh 1
    tree_1 = AABBTree(*gpytoolbox.initialize_aabbtree(*mesh_1))
    # Initialize AABB tree for mesh 2
    tree_2 = AABBTree(*gpytoolbox.initialize_aabbtree(*mesh_2))

    first_queue_pair = [0, 0]
    queue = [first_queue_pair]
    current_best_guess = np.inf, None, None
    while len(queue) > 0:
        q1, q2 = queue.pop()
        is_leaf1 = tree_1.children[q1, 1] == -1
        is_leaf2 = tree_2.children[q2, 1] == -1
        if is_leaf1 and is_leaf2:
            t1 = tree_1.tri_indices[q1].item()
            t2 = tree_2.tri_indices[q2].item()
            d, p, q = triangle_triangle_distance(
                mesh_1.vertices[mesh_1.faces[t1, 0], :],
                mesh_1.vertices[mesh_1.faces[t1, 1], :],
                mesh_1.vertices[mesh_1.faces[t1, 2], :],
                mesh_2.vertices[mesh_2.faces[t2, 0], :],
                mesh_2.vertices[mesh_2.faces[t2, 1], :],
                mesh_2.vertices[mesh_2.faces[t2, 2], :],
            )
            if d < current_best_guess[0]:
                current_best_guess = d, p, q
        else:
            d = np.max(
                np.abs(tree_1.centres[q1, :] - tree_2.centres[q2, :])
                - (tree_1.widths[q1, :] + tree_2.widths[q2, :]) / 2
            )
            if d < current_best_guess[0]:
                if not is_leaf1 and is_leaf2:
                    queue.extend(
                        [
                            [tree_1.children[q1, 0], q2],
                            [tree_1.children[q1, 1], q2],
                        ]
                    )
                elif not is_leaf2 and is_leaf1:
                    queue.extend(
                        [
                            [q1, tree_2.children[q2, 0]],
                            [q1, tree_2.children[q2, 1]],
                        ]
                    )
                elif not is_leaf1 and not is_leaf2:
                    queue.extend(
                        [
                            [tree_1.children[q1, 0], tree_2.children[q2, 0]],
                            [tree_1.children[q1, 1], tree_2.children[q2, 0]],
                            [tree_1.children[q1, 0], tree_2.children[q2, 1]],
                            [tree_1.children[q1, 1], tree_2.children[q2, 1]],
                        ]
                    )

    return current_best_guess
