"""Implementation of gc2 distance metrics from NGA-West-3.

All functions from this module are derived from the following paper:

Spudich, P. A., & Chiou, B. (2015). Strike-parallel and strike-normal
coordinate system around geometrically complicated rupture traces: Use
by NGA-West2 and further improvements (No. 2015-1028). US Geological
Survey.

All referenced pages and equations are in this paper.
"""

import itertools

import numpy as np
import shapely
from numba import float64, int64, njit, uint64


def segment_rx_ry(
    bounds: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate segment rx and ry distances.

    Parameters
    ----------
    bounds : np.ndarray
        Bounds for n segments, an array of compatible shape to (m, 2,
        2) = (num_segments, num_trace_points, x & y).
    points : np.ndarray
        Points to measure rx and ry. Has shape (n, 2).

    Returns
    -------
    rx : np.ndarray
        The Rx distance measure, an array of shape (m, n).
    ry : np.ndarray
        The Ry distance measure, an array of shape (m, n).
    """
    bounds = np.atleast_3d(bounds).reshape((-1, 2, 2))
    points = np.atleast_2d(points)

    q = bounds[:, 0, :]
    r = bounds[:, 1, :]

    qr = (r - q)[:, np.newaxis, :]

    qp = points[np.newaxis, :, :] - q[:, np.newaxis, :]

    dot_qp_qr = np.vecdot(qp, qr)
    dot_qr_qr = np.vecdot(qr, qr)

    t = dot_qp_qr / dot_qr_qr

    closest_point = q[:, np.newaxis, :] + t[:, :, np.newaxis] * qr

    rx = np.linalg.norm(points[np.newaxis, :, :] - closest_point, axis=-1)

    sign = np.sign(qp[..., 1] * qr[..., 0] - qp[..., 0] * qr[..., 1])
    rx *= sign

    ry = t * np.sqrt(dot_qr_qr)

    return rx, ry


def segment_weights(
    trace_lengths: np.ndarray, rx: np.ndarray, ry: np.ndarray
) -> np.ndarray:
    """Calculate segment weights from trace lengths.

    Segment weights implement Equation (1) of Section 3.4.

    Parameters
    ----------
    trace_lengths : np.ndarray
        Length of each segment trace in strike order, an array of shape (m,).
    rx : np.ndarray
        The rx distances to each point (n points in total), an array of shape (m, n).
    ry : np.ndarray
        The ry distances to each point (n points in total), an array of shape (m, n).

    Returns
    -------
    np.ndarray
        An array of shape (m, n) containing distance weights for each segment point pair.
    """
    trace_lengths = trace_lengths[:, np.newaxis]

    mask_zero = np.isclose(rx, 0.0)

    theta = np.arctan(
        np.divide(trace_lengths - ry, rx, where=~mask_zero, out=np.zeros_like(rx))
    ) - np.arctan(np.divide(-ry, rx, where=~mask_zero, out=np.zeros_like(rx)))

    w = np.divide(theta, rx, where=~mask_zero, out=np.zeros_like(rx))

    if np.any(mask_zero):
        special_case = np.divide(
            1.0,
            ry - trace_lengths,
            where=(ry != trace_lengths),
            out=np.full_like(w, np.nan),
        ) - np.divide(1.0, ry, where=(ry != 0), out=np.full_like(w, np.nan))

        w = np.where(mask_zero, special_case, w)

    return w


@njit([float64[:](float64[:], uint64[:]), float64[:](float64[:], int64[:])], cache=True)
def cumulative_reduction(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Calculate sum between indices.

    Parameters
    ----------
    data : np.ndarray
        The data to sum.
    indices : np.ndarray
        The indices to sum between.

    Returns
    -------
    np.ndarray
        The equivalent to ``np.cumulative_sum(data[indices[i]:indices[i + 1]], include_initial=True)`` for each index i.
    """
    out = np.zeros_like(data)
    # indices defines the start of each trace.
    # We append the total length to handle the last block.
    for j in range(len(indices) - 1):
        start = indices[j]
        end = indices[j + 1]

        acc = 0.0
        for i in range(start, end):
            out[i] = acc
            acc += data[i]
    return out


@njit(
    [float64[:, :](float64[:, :], uint64[:]), float64[:, :](float64[:, :], int64[:])],
    cache=True,
)
def diff_reduction(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Calculate a diff reduction points between the given indices.

    Parameters
    ----------
    points : np.ndarray
        The points take a difference for. Differences are done in pairs so that ``diff[i] = points[2*i + 1] - points[2*i]``.
    indices : np.ndarray
        Indices of the boundary. Differences are not computed between boundaries.

    Returns
    -------
    np.ndarray
        The differences between pairs of elements in ``points`` considering the boundaries ``indices``.
    """
    n_segments = len(points) // 2
    out = np.zeros((n_segments, points.shape[1]))

    write_ptr = 0
    for j in range(len(indices) - 1):
        start = indices[j]
        end = indices[j + 1]
        for i in range(start, end, 2):
            out[write_ptr] = points[i + 1] - points[i]
            write_ptr += 1
    return out


def calculate_gc2_u_origins(
    segment_lengths: np.ndarray,  # List of lengths per trace j
    segment_indices: np.ndarray,  # Index of segments
    trace_starts: np.ndarray,  # p_1,j for each trace, shape (num_traces, 2)
    p_origin: np.ndarray,  # The back-end antipodal point
    b_hat: np.ndarray,  # Nominal strike unit vector
) -> np.ndarray:
    """Calculates shifted origins for GC2 Rx/Ry calculations in multi-trace systems.

    Follows Equation (12), page 6 of the GC2 distance metric specification to transform
    local segment distances into a globalised U-coordinate system.

    Parameters
    ----------
    segment_lengths : np.ndarray
        The lengths of every individual segment across all traces,
        shape (m,).
    segment_indices : np.ndarray
        The indices marking the start of each trace within the flattened
        segment array, shape (num_traces + 1,). This must include the
        total number of segments as the final element.
    trace_starts : np.ndarray
        The (x, y) coordinates for the first point of each trace,
        shape (num_traces, 2).
    p_origin : np.ndarray
        The global origin point, shape (2,).
    b_hat : np.ndarray
        The nominal strike unit vector for the entire system, shape (2,).

    Returns
    -------
    np.ndarray
        The cumulative U-coordinate distance at the start of each segment
        relative to p_origin, shape (m,).
    """
    local_shifts = cumulative_reduction(segment_lengths, segment_indices)
    segment_counts = np.diff(segment_indices)

    segment_counts = segment_counts.astype(np.int64, casting="unsafe")

    global_shifts = np.repeat(np.dot(trace_starts - p_origin, b_hat), segment_counts)
    return global_shifts + local_shifts


def generalised_t_u_coordinates(
    trace_lengths: np.ndarray,
    rx: np.ndarray,
    ry: np.ndarray,
    segment_u_origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised rx and ry calculations for multiple segments.

    Implements Equations (3) and (9) of page 4.

    Parameters
    ----------
    trace_lengths : np.ndarray
        Lengths of each segment, an array of shape (m,).
    rx : np.ndarray
        The rx distances to each point (n points in total), an array of shape (m, n).
    ry : np.ndarray
        The ry distances to each point (n points in total), an array of shape (m, n).
    segment_u_origins : np.ndarray
        The U-coordinate origin shift for each segment, shape (m,).
        Calculated via Equation 12.

    Returns
    -------
    t : np.ndarray
        The weighted average rx distance, an array of shape (m, n).
    u : np.ndarray
        The weighted average ry distance, an array of shape (m, n).
    """
    w = segment_weights(trace_lengths, rx, ry)

    # Apply the Equation 12 shifts to ry
    ry_global = ry + segment_u_origins[:, np.newaxis]

    # Calculate weighted averages
    t = np.average(rx, axis=0, weights=w)
    u = np.average(ry_global, axis=0, weights=w)

    return t, u


def antipodal_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the farthest pair of points in a given set of points.

    Parameters
    ----------
    points : np.ndarray
        The points to search.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The farthest pair of points in the set.

    Raises
    ------
    ValueError
        If ``len(points) == 1``.
    """
    # Implementation following the observation that complex algorithms
    # => many complex bugs. Therefore, only pay the cost of a complex
    # implementation when the bugs are worth the speed increase. We
    # could use the rotating calipers method here but it would only be
    # marginally faster given the input size and we cannot implement
    # it from well-tested libraries.

    if len(points) == 1:
        raise ValueError("Cannot find antipodal pair with only one point.")
    hull = shapely.convex_hull(shapely.multipoints(points))
    hull_points = shapely.get_coordinates(hull)

    # Shapely polygons repeat the first point at the end to close the
    # exterior ring. Removing this doesn't change the result because
    # the point is included twice, but it is faster.
    if isinstance(hull, shapely.Polygon):
        hull_points = hull_points[:-1]

    return max(
        (pair for pair in itertools.combinations(hull_points, 2)),
        key=lambda pair: np.square(pair[1] - pair[0]).sum(),
    )


def trial_strike_vector(trace_endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a trial strike vector between all trace endpoints.

    Implements the algorithm described in page 6.

    Parameters
    ----------
    trace_endpoints : np.ndarray
        The trace endpoints of the rupture.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The pair (a, b) of farthest endpoints ordered such that a -> b is in the
        east direction.
    """
    a, b = antipodal_points(trace_endpoints)
    # NZTM (y, x) coordinate system, b east of a ideally.
    if b[1] < a[1]:
        a, b = b, a
    return a, b


def strike_corrected_directions(
    trace_directions: np.ndarray, trial_unit_vector: np.ndarray
) -> np.ndarray:
    """Correct strike directions by reversing discordant strikes.

    Implements the algorithm described in page 6.

    Parameters
    ----------
    trace_directions : np.ndarray
        The trace directions to correct.
    trial_unit_vector : np.ndarray
        The unit vector for directions.

    Returns
    -------
    np.ndarray
        Trace endpoints re-ordered so that they always point in or against the
        direction of the unit vector.
    """
    trace_directions = trace_directions.copy()
    e = np.vecdot(trial_unit_vector, trace_directions)
    e_sum = e.sum()
    trace_directions[np.sign(e) != np.sign(e_sum)] *= -1
    return trace_directions


def multi_trace_rx_ry(
    trace_points: np.ndarray,
    trace_indices: np.ndarray,
    rx: np.ndarray,
    ry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-trace rx and ry calculations.

    Parameters
    ----------
    trace_points : np.ndarray
        Trace points.
    trace_indices : np.ndarray
        Trace indices for end of segments
    rx : np.ndarray
        The rx values calculated to points.
    ry : np.ndarray
        The ry values calculated to points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The (T, U) generalised coordinates for Rx and Ry calculations.
    """
    direction_vectors = diff_reduction(trace_points, trace_indices)
    segment_lengths = np.linalg.norm(direction_vectors, axis=1)

    directions_per_trace = np.diff(trace_indices) // 2
    directions_per_trace = directions_per_trace.astype(np.int64, casting="unsafe")
    trace_direction_start_indices = np.cumulative_sum(
        directions_per_trace, include_initial=True
    )

    trace_starts = trace_points[trace_indices[:-1]]
    end_indices = trace_indices[1:] - 1
    trace_ends = trace_points[end_indices]
    trace_endpoints = np.concatenate((trace_starts, trace_ends))

    p_origin, p_end = trial_strike_vector(trace_endpoints)
    trial_unit_vector = p_end - p_origin
    trial_unit_vector /= np.linalg.norm(trial_unit_vector)
    direction_vectors = strike_corrected_directions(
        direction_vectors, trial_unit_vector
    )
    b_hat = np.sum(direction_vectors, axis=0)
    b_hat /= np.linalg.norm(b_hat)

    u_shift_origins = calculate_gc2_u_origins(
        segment_lengths,
        trace_direction_start_indices,
        trace_starts,
        p_origin,
        b_hat,
    )

    return generalised_t_u_coordinates(segment_lengths, rx, ry, u_shift_origins)
