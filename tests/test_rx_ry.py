import numpy as np
import pytest
import scipy as sp
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from source_modelling import gc2_distances, sources


def test_segment_rx_ry_output_shapes() -> None:
    m, n = 5, 10
    bounds = np.random.rand(m, 2, 2)
    points = np.random.rand(n, 2)

    rx, ry = gc2_distances.segment_rx_ry(bounds, points)

    assert rx.shape == (m, n), f"Expected rx shape ({m}, {n}), got {rx.shape}"
    assert ry.shape == (m, n), f"Expected ry shape ({m}, {n}), got {ry.shape}"


@given(
    bounds=arrays(
        np.float64,
        (1, 2, 2),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    ),
    t=st.floats(min_value=-100, max_value=100),
    u=st.floats(min_value=-100, max_value=100),
)
def test_rx_ry(bounds: np.ndarray, t: float, u: float) -> None:
    q = bounds[0, 0, :]
    r = bounds[0, 1, :]
    qr = r - q
    segment_length = np.linalg.norm(qr)

    assume(segment_length > 1e-7)

    norm = np.array([-qr[1], qr[0]]) / segment_length

    point = q + t * qr + u * norm

    rx_arr, ry_arr = gc2_distances.segment_rx_ry(bounds, point)
    rx = rx_arr.item()
    ry = ry_arr.item()

    assert rx == pytest.approx(u, abs=1e-5, rel=1e-5)

    expected_ry = t * segment_length
    assert ry == pytest.approx(expected_ry, abs=1e-5, rel=1e-5)


@given(
    length=st.floats(min_value=0.1, max_value=1000.0),
    # Set rx != 0 to avoid the edge case where equation 5 applies.
    rx=st.one_of(
        st.floats(min_value=0.1, max_value=500.0),
        st.floats(min_value=-500, max_value=-0.1),
    ),
    # ry: longitudinal distance from the start of the segment
    ry=st.floats(min_value=-500.0, max_value=1500.0),
)
def test_segment_weights_integral_match(length: float, rx: float, ry: float) -> None:
    trace_length = np.array([length])
    rx_arr = np.array([[rx]])
    ry_arr = np.array([[ry]])

    def integrand(u: float) -> float:
        dist_sq = rx**2 + (u - ry) ** 2
        return 1.0 / dist_sq

    expected_w, _ = sp.integrate.quad(integrand, 0, length)

    actual_w_arr = gc2_distances.segment_weights(trace_length, rx_arr, ry_arr)
    actual_w = actual_w_arr.item()

    assert actual_w == pytest.approx(expected_w, rel=1e-4, abs=1e-8)


@given(
    length=st.floats(min_value=0.1, max_value=1000.0),
    ry=st.floats(min_value=-500.0, max_value=1500.0),
)
def test_segment_weights_at_zero_off_segment(length: float, ry: float) -> None:
    assume(
        (ry < 0 and not np.isclose(ry, 0.0))
        or (ry > length and not np.isclose(ry, length))
    )
    trace_length = np.array([length])
    rx = 0.0
    rx_arr = np.array([[rx]])
    ry_arr = np.array([[ry]])

    expected_w = 1 / (ry - length) - (1.0 / ry)

    actual_w_arr = gc2_distances.segment_weights(trace_length, rx_arr, ry_arr)
    actual_w = actual_w_arr.item()

    assert actual_w == pytest.approx(expected_w, rel=1e-4, abs=1e-8)


@st.composite
def point_on_trace_strategy(draw: st.DrawFn):
    # Draw an array of lengths (at least one segment)
    lengths = draw(
        st.lists(st.floats(min_value=0.1, max_value=1000.0), min_size=1, max_size=50)
    )
    num_segments = len(lengths)

    # Pick the index that will be inside
    target_idx = draw(st.integers(min_value=0, max_value=num_segments - 1))

    ry_values = []
    for i, length in enumerate(lengths):
        if i == target_idx:
            # Inside the segment: [0, length]
            val = draw(st.floats(min_value=0.0, max_value=length))
        else:
            # Outside: Either negative or greater than length
            # We use a small buffer to avoid floating point edge cases at exactly 0 or length
            outside_below = st.floats(max_value=-0.1)
            outside_above = st.floats(min_value=length + 0.1)
            val = draw(st.one_of(outside_below, outside_above))

        ry_values.append(val)

    return np.array(lengths), target_idx, np.array(ry_values)


@given(value=point_on_trace_strategy())
def test_segment_weights_at_zero_on_segment(
    value: tuple[np.ndarray, int, np.ndarray],
) -> None:
    length, idx, ry = value
    rx = np.zeros_like(ry)

    actual_w = gc2_distances.segment_weights(
        length, rx[:, np.newaxis], ry[:, np.newaxis]
    )

    assert actual_w[idx] == pytest.approx(1.0)
    assert actual_w.sum() == pytest.approx(1.0)


def test_segment_weights_fail_for_overlapping_segments() -> None:
    ry = np.array([0.0, 0.0])
    rx = np.zeros_like(ry)
    length = np.ones_like(ry)
    with pytest.raises(ValueError, match="Points lie on overlapping traces"):
        _ = gc2_distances.segment_weights(length, rx[:, np.newaxis], ry[:, np.newaxis])


def test_rx_ry_plane() -> None:
    """Test that strike order is correctly interpreted in plane context"""
    plane = sources.Plane.from_centroid_strike_dip(
        np.array([-43.538, 172.6474, 5.0]), 90.0, 10.0, width=5.0, strike=0.0
    )
    # Conveniently, strike = 0 implies that west = left and east = right
    point_left = np.array([-43.538, 172.6174])
    # meanwhile, south of the bottom edge implies negative ry.
    point_right = np.array([-43.60, 172.6574])
    rx_left, ry_left = plane.rx_ry_distance(point_left)
    rx_right, ry_right = plane.rx_ry_distance(point_right)

    assert rx_left < 0
    assert rx_right > 0

    assert ry_left > 0
    assert ry_right < 0


def test_rx_ry_fault() -> None:
    """Test that strike order is correctly interpreted in fault context"""

    # Deliberately setup as an extension of the plane case with the
    # fault striking 0 degrees initially and then tending 15 degrees
    # east.
    fault = sources.Fault.from_trace_points(
        np.array(
            [
                [-43.58302058, 172.64739978],
                [-43.49297907, 172.6474],
                [-43.44832311516278, 172.7160675056574],
            ]
        ),
        dtop=0.0,
        dbottom=10.0,
        dip=90.0,
        dip_dir=90.0,
    )

    # Conveniently, strike = 0 implies that west = left and east = right
    point_left = np.array([-43.538, 172.6174])
    # meanwhile, south of the bottom edge implies negative ry.
    point_right = np.array([-43.60, 172.6574])
    rx_left, ry_left = fault.rx_ry_distance(point_left)
    rx_right, ry_right = fault.rx_ry_distance(point_right)

    assert rx_left < 0
    assert rx_right > 0

    assert ry_left > 0
    assert ry_right < 0


def test_rx_ry_fault_single_plane() -> None:
    """Test that fault rx, ry == plane rx, ry in the single plane fault case"""
    plane = sources.Plane.from_centroid_strike_dip(
        np.array([-43.538, 172.6474, 5.0]), 90.0, 10.0, width=5.0, strike=0.0
    )
    fault = sources.Fault([plane])

    point = np.array([-43.538, 172.6174])

    rx_plane, ry_plane = plane.rx_ry_distance(point)
    rx_fault, ry_fault = fault.rx_ry_distance(point)

    assert rx_plane == pytest.approx(rx_fault)
    assert ry_plane == pytest.approx(ry_fault)


def test_antipodal_points_simple_square() -> None:
    """Tests a unit square; antipodal points should be diagonal corners."""
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    p1, p2 = gc2_distances.antipodal_points(points)

    distance = np.linalg.norm(p1 - p2)
    # Diagonal of a 1x1 square is sqrt(2)
    assert distance == pytest.approx(np.sqrt(2))


def test_antipodal_points_with_interior_points() -> None:
    """Tests that points inside the hull don't affect the result."""
    # A triangle with a point right in the middle
    points = np.array([[0, 0], [10, 0], [5, 10], [5, 5]])
    p1, p2 = gc2_distances.antipodal_points(points)

    distance = np.linalg.norm(p1 - p2)
    # The furthest distance is between (0,0) and (5,10) or (10,0) and (5,10)
    # both are sqrt(125) ~ 11.18
    assert distance == pytest.approx(np.sqrt(125))


def test_antipodal_points_collinear_points() -> None:
    """Tests points on a straight line."""
    points = np.array([[0, 0], [1, 1], [2, 2], [5, 5]])
    p1, p2 = gc2_distances.antipodal_points(points)

    # Furthest points are the ends of the line
    assert np.linalg.norm(p1 - p2) == pytest.approx(np.sqrt(50))


def test_antipodal_points_single_point_error() -> None:
    """Tests that a single point raises an error (as combinations requires 2)."""
    points = np.array([[1, 1]])
    with pytest.raises(ValueError):
        gc2_distances.antipodal_points(points)


@given(
    data=arrays(np.float64, st.integers(2, 20), elements=st.floats(0.1, 100)),
)
def test_cumulative_reduction_resets_per_trace(data: np.ndarray) -> None:
    # Split data into two traces at a random point
    split = len(data) // 2
    indices = np.array([0, split, len(data)], dtype=np.uint64)

    out = gc2_distances.cumulative_reduction(data, indices)

    # Invariant: Each trace starts at 0.0
    assert out[0] == 0.0
    assert out[split] == 0.0
    # Invariant: It is a prefix sum within the trace
    assert out[split - 1] == pytest.approx(np.sum(data[0 : split - 1]))
    assert out[-1] == pytest.approx(np.sum(data[split:-1]))


def test_diff_reduction_isolates_traces() -> None:
    # 6 points, 2 traces: [P0, P1, P2, P3] and [P4, P5]
    points = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 0],
            [2, 0],  # Trace 1
            [10, 10],
            [11, 11],  # Trace 2
        ],
        dtype=np.float64,
    )
    indices = np.array([0, 4, len(points)], dtype=np.uint64)

    # Expected segments: (P1-P0), (P2-P1), (P4-P3)
    expected = np.array([[1, 0], [1, 0], [1, 1]])
    actual = gc2_distances.diff_reduction(points, indices)

    assert actual == pytest.approx(expected)


@given(
    points=arrays(
        np.float64,
        shape=st.tuples(st.integers(2, 10), st.just(2)),
        elements=st.floats(-1e5, 1e5),
    ),
    pair_indices=st.lists(
        st.tuples(st.integers(0, 9), st.integers(0, 9)),
        min_size=20,
        max_size=20,
    ),
)
def test_trial_strike_vector_invariants(
    points: np.ndarray, pair_indices: list[tuple[int, int]]
) -> None:
    assume(len(np.unique(points, axis=0)) == len(points))
    # Antipodal distance should be max distance in set
    a, b = gc2_distances.trial_strike_vector(points)
    dist_sq = np.sum((a - b) ** 2)

    # Check against a few Hypothesis-generated pairs
    for i, j in pair_indices:
        p1 = points[i % len(points)]
        p2 = points[j % len(points)]
        assert dist_sq >= np.sum((p1 - p2) ** 2) - 1e-7

    # Invariant: Canonical orientation (NZTM x-axis/Easting is index 1)
    assert b[1] >= a[1]


def test_strike_corrected_directions_flips_appropriately() -> None:
    trial_unit = np.array([1.0, 0.0])  # East
    # Trace 1: East, Trace 2: Slightly North East, Trace 3: West (Opposing)
    dirs = np.array([[1.0, 0], [1.0, 0.2], [-1.0, 0]])

    # Invariant: Majority direction wins. West should flip to East.
    corrected = gc2_distances.strike_corrected_directions(dirs, trial_unit)

    assert corrected[2, 0] > 0  # The third segment should have flipped
    # Invariant: Dot product sum of corrected dirs with trial vector must be positive
    assert np.sum(np.vecdot(corrected, trial_unit)) > 0


def test_calculate_gc2_u_origins_logic() -> None:
    # Setup: 2 traces. Trace 1 is 10 units long. Trace 2 starts 50 units East.
    seg_lengths = np.array([5.0, 5.0, 3.0, 3.0])
    seg_indices = np.array([0, 2, 4], dtype=np.uint64)
    trace_starts = np.array([[0.0, 0.0], [0.0, 50.0]])
    p_origin = np.array([0.0, 0.0])
    b_hat = np.array([0.0, 1.0])  # Strike is East (x-axis)

    # Trace 1 global: 0.0. Local: [0, 5]
    # Trace 2 global: 50.0. Local: [0, 3]
    expected = np.array([0.0, 5.0, 50.0, 53.0])

    actual = gc2_distances.calculate_gc2_u_origins(
        seg_lengths, seg_indices, trace_starts, p_origin, b_hat
    )

    assert actual == pytest.approx(expected)


def test_multi_trace_rx_ry_origin_alignment() -> None:
    """
    Test that origin shifts correctly align colinear traces.

    Setup:
    Trace 1: (0, 0) to (10, 0)  [Length 10]
    Gap:    (10, 0) to (20, 0) [Length 10]
    Trace 2: (20, 0) to (30, 0) [Length 10]

    Observation Point P: (25, 5)
    """
    # 1. Define geometry
    trace_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],  # Trace 1 (2 points)
            [20.0, 0.0],
            [30.0, 0.0],  # Trace 2 (2 points)
        ]
    )
    trace_indices = np.array([0, 2, 4], dtype=np.uint64)

    # 2. Mock rx and ry for point P(25, 5) relative to each segment
    # Segment 1 (0,0 -> 10,0):
    #   P is 25 units 'along' the infinite line (ry=25)
    #   P is 5 units 'away' (rx=5)
    # Segment 2 (20,0 -> 30,0):
    #   P is 5 units 'along' the start of this segment (ry=5)
    #   P is 5 units 'away' (rx=5)
    rx = np.array([[5.0], [5.0]])  # Shape (m=2 segments, n=1 point)
    ry = np.array([[25.0], [5.0]])

    # 3. Calculate
    t, u = gc2_distances.multi_trace_rx_ry(trace_points, trace_indices, rx, ry)

    # 4. Assertions
    # T (weighted rx) should remain 5.0
    assert t.item() == pytest.approx(5.0)

    # U (weighted global ry) calculation check:
    # Trace 1 origin shift: 0 (start) + 0 (global) = 0. Global U = 25 + 0 = 25.
    # Trace 2 origin shift: 0 (start) + 20 (global) = 20. Global U = 5 + 20 = 25.
    # Since both segments agree the global U is 25, the average must be 25.
    assert u.item() == pytest.approx(25.0)


def test_multi_trace_rx_ry_local_shift_accumulation() -> None:
    """Test that multiple segments within a single trace accumulate local shifts."""
    # Single trace with two segments: (0,0) -> (5,0) -> (15,0)
    # Traces are represented in the format [start0, end0, start1, end1],
    # ergo the repeat of [5.0, 0.0] is intentional and necessary for the
    # tests to pass.
    trace_points = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 0.0], [15.0, 0.0]])
    trace_indices = np.array([0, 4], dtype=np.uint64)

    # Point P(10, 2)
    # Seg 1: ry = 10, rx = 2
    # Seg 2: ry = 5,  rx = 2 (10 units from start of Seg 2, which is at x=5)
    rx = np.array([[2.0], [2.0]])
    ry = np.array([[10.0], [5.0]])

    _, u = gc2_distances.multi_trace_rx_ry(trace_points, trace_indices, rx, ry)

    # Local shift for Seg 2 should be length of Seg 1 (5.0)
    # Seg 1 Global U: 10 + 0 = 10
    # Seg 2 Global U: 5 + 5 = 10
    assert u.item() == pytest.approx(10.0)
