import numpy as np
import numba
import gpytoolbox


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
    dist = np.linalg.norm(R1 - R2)
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
    S = [s0, s1, s2]
    T = [t0, t1, t2]
    Sv = [s1 - s0, s2 - s1, s0 - s2]
    Tv = [t1 - t0, t2 - t1, t0 - t2]
    # Sv1 = s2 - s1
    # Sv2 = s0 - s2

    mindd = np.sum((S[0] - T[0]) ** 2) + 10.0

    for i in range(3):
        for j in range(3):
            # print("S[i] : ", S[i])
            # print("Sv[i] : ", S[i]+Sv[i])
            # print("T[j] : ", T[j])
            # print("Tv[j] : ", T[j]+Tv[j])
            _, P, Q = edge_edge_distance(S[i], S[i] + Sv[i], T[j], T[j] + Tv[j])
            # print("P : ", P)
            # print("Q : ", Q)
            # # print(P)
            # # print(Q)
            VEC = Q - P
            V = Q - P
            dd = np.dot(V, V)
            # # print("BB")
            if dd <= mindd:
                minP = P.copy()
                minQ = Q.copy()
                mindd = dd
                Z = S[(i + 2) % 3] - P
                a = np.dot(Z, VEC)
                Z = T[(j + 2) % 3] - Q
                b = np.dot(Z, VEC)
                if (a <= 0) and (b >= 0):
                    # print("Here0")
                    return np.sqrt(mindd), minP, minQ
                p = np.dot(V, VEC)
                if a < 0:
                    a = 0
                if b > 0:
                    b = 0
                if (p - a + b) > 0:
                    shown_disjoint = True
    # .....
    # # print("Sv[0] : ", Sv[0])
    # # print("Sv[1] : ", Sv[1])
    Sn = np.cross(Sv[0], Sv[1])
    # # print("Sn : ", Sn)
    Snl = np.dot(Sn, Sn)

    if Snl > 1e-15:
        # V = S[0] - T[0]
        Tp = [np.dot(Sn, S[0] - T[0]), np.dot(Sn, S[0] - T[1]), np.dot(Sn, S[0] - T[2])]
        point = -1
        if (Tp[0] > 0) and (Tp[1] > 0) and (Tp[2] > 0):
            if Tp[0] < Tp[1]:
                point = 0
            else:
                point = 1
            if Tp[2] < Tp[point]:
                point = 2
        elif (Tp[0] < 0) and (Tp[1] < 0) and (Tp[2] < 0):
            if Tp[0] > Tp[1]:
                point = 0
            else:
                point = 1
            if Tp[2] > Tp[point]:
                point = 2
        if point >= 0:
            shown_disjoint = True
            V = T[point] - S[0]
            Z = np.cross(Sn, Sv[0])
            if np.dot(V, Z) > 0:
                V = T[point] - S[1]
                Z = np.cross(Sn, Sv[1])
                if np.dot(V, Z) > 0:
                    V = T[point] - S[2]
                    Z = np.cross(Sn, Sv[2])
                    if np.dot(V, Z) > 0:
                        # # print("T[point] : ", T[point])
                        # # print("Tp[point] : ", Tp[point])
                        # # print("Sn : ", Sn)
                        # # print("Snl : ", Snl)
                        P = T[point] + Sn * (Tp[point]) / Snl
                        Q = T[point].copy()
                        # print("Here1")
                        return np.sqrt(np.dot(P - Q, P - Q)), P, Q

    Tn = np.cross(Tv[0], Tv[1])
    Tnl = np.dot(Tn, Tn)

    if Tnl > 1e-15:
        # V = T[0] - S[0]
        # Sp = [T[0] - S[0], T[0] - S[1], T[0] - S[2]]
        Sp = [np.dot(Tn, T[0] - S[0]), np.dot(Tn, T[0] - S[1]), np.dot(Tn, T[0] - S[2])]
        point = -1
        if (Sp[0] > 0) and (Sp[1] > 0) and (Sp[2] > 0):
            if Sp[0] < Sp[1]:
                point = 0
            else:
                point = 1
            if Sp[2] < Sp[point]:
                point = 2
        elif (Sp[0] < 0) and (Sp[1] < 0) and (Sp[2] < 0):
            if Sp[0] > Sp[1]:
                point = 0
            else:
                point = 1
            if Sp[2] > Sp[point]:
                point = 2
        if point >= 0:
            shown_disjoint = True
            V = S[point] - T[0]
            Z = np.cross(Tn, Tv[0])
            if np.dot(V, Z) > 0:
                V = S[point] - T[1]
                Z = np.cross(Tn, Tv[1])
                if np.dot(V, Z) > 0:
                    V = S[point] - T[2]
                    Z = np.cross(Tn, Tv[2])
                    if np.dot(V, Z) > 0:
                        Q = S[point] + Tn * (Sp[point]) / Tnl
                        P = S[point].copy()
                        # print("Here2")
                        return np.sqrt(np.dot(P - Q, P - Q)), P, Q

    if shown_disjoint:
        # print("Here3")
        return np.sqrt(mindd), minP, minQ
    else:
        # print("Here4")
        return 0.0, minP, minQ


def minimum_distance(v1, f1, v2, f2):
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

    dim = v1.shape[1]
    # Initialize AABB tree for mesh 1
    C1, W1, CH1, PAR1, D1, tri_ind1, _ = gpytoolbox.initialize_aabbtree(v1, f1)
    # Initialize AABB tree for mesh 2
    C2, W2, CH2, PAR2, D2, tri_ind2, _ = gpytoolbox.initialize_aabbtree(v2, f2)

    first_queue_pair = [0, 0]
    queue = [first_queue_pair]
    current_best_guess = np.inf, None, None
    while len(queue) > 0:
        q1, q2 = queue.pop()
        # print("-----------")
        # print("Queue length : {}".format(len(queue)))
        # print("q1: ",q1)
        # print("q2: ",q2)
        # print("CH1[q1,]: ",CH1[q1,:])
        # print("CH2[q2,]: ",CH2[q2,:])
        # print("current_best_guess: ",current_best_guess)
        is_leaf1 = CH1[q1, 1] == -1
        is_leaf2 = CH2[q2, 1] == -1
        if is_leaf1 and is_leaf2:
            # Compute distance between triangles
            t1 = tri_ind1[q1].item()
            t2 = tri_ind2[q2].item()
            # print("t1: ",t1)
            # print("t2: ",t2)
            d, p, q = triangle_triangle_distance(
                v1[f1[t1, 0], :],
                v1[f1[t1, 1], :],
                v1[f1[t1, 2], :],
                v2[f2[t2, 0], :],
                v2[f2[t2, 1], :],
                v2[f2[t2, 2], :],
            )
            # print("d: ",d)
            if d < current_best_guess[0]:
                current_best_guess = d, p, q
        else:
            # Find distance between boxes
            d = np.max(np.abs(C1[q1, :] - C2[q2, :]) - (W1[q1, :] + W2[q2, :]) / 2)
            # print("d: ",d)
            if d < current_best_guess[0]:
                # Add children to queue

                if (not is_leaf1) and (is_leaf2):
                    queue.append([CH1[q1, 0], q2])
                    queue.append([CH1[q1, 1], q2])
                    # queue.append([CH1[q1,2],q2])
                    # queue.append([CH1[q1,3],q2])
                    # if dim==3:
                    #     queue.append([CH1[q1,4],q2])
                    #     queue.append([CH1[q1,5],q2])
                    #     queue.append([CH1[q1,6],q2])
                    #     queue.append([CH1[q1,7],q2])
                if (not is_leaf2) and (is_leaf1):
                    queue.append([q1, CH2[q2, 0]])
                    queue.append([q1, CH2[q2, 1]])
                    # queue.append([q1,CH2[q2,2]])
                    # queue.append([q1,CH2[q2,3]])
                    # if dim==3:
                    #     queue.append([q1,CH2[q2,4]])
                    #     queue.append([q1,CH2[q2,5]])
                    #     queue.append([q1,CH2[q2,6]])
                    #     queue.append([q1,CH2[q2,7]])
                if (not is_leaf1) and (not is_leaf2):
                    queue.append([CH1[q1, 0], CH2[q2, 0]])
                    queue.append([CH1[q1, 1], CH2[q2, 0]])
                    queue.append([CH1[q1, 0], CH2[q2, 1]])
                    queue.append([CH1[q1, 1], CH2[q2, 1]])

    return current_best_guess
