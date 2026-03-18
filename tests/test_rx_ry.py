import numpy as np
import pytest
import scipy as sp
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from source_modelling import sources


def test_segment_rx_ry_output_shapes():
    m, n = 5, 10
    bounds = np.random.rand(m, 2, 2)
    points = np.random.rand(n, 2)

    rx, ry = sources.segment_rx_ry(bounds, points)

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
def test_rx_ry(bounds, t, u):
    q = bounds[0, 0, :]
    r = bounds[0, 1, :]
    qr = r - q
    segment_length = np.linalg.norm(qr)

    assume(segment_length > 1e-7)

    norm = np.array([-qr[1], qr[0]]) / segment_length

    point = q + t * qr + u * norm

    rx_arr, ry_arr = sources.segment_rx_ry(bounds, point)
    rx = rx_arr.item()
    ry = ry_arr.item()

    assert rx == pytest.approx(-u, abs=1e-5, rel=1e-5)

    expected_ry = t * segment_length
    assert ry == pytest.approx(expected_ry, abs=1e-5, rel=1e-5)


@given(
    lengths=st.floats(min_value=0.1, max_value=1000.0),
    # Set rx != 0 to avoid the edge case where equation 5 applies.
    rx=st.one_of(
        st.floats(min_value=0.1, max_value=500.0),
        st.floats(min_value=-500, max_value=-0.1),
    ),
    # ry: longitudinal distance from the start of the segment
    ry=st.floats(min_value=-500.0, max_value=1500.0),
)
def test_segment_weights_integral_match(lengths, rx, ry):
    trace_lengths = np.array([lengths])
    rx_arr = np.array([[rx]])
    ry_arr = np.array([[ry]])

    def integrand(u):
        dist_sq = rx**2 + (u - ry) ** 2
        return 1.0 / dist_sq

    expected_w, _ = sp.integrate.quad(integrand, 0, lengths)

    actual_w_arr = sources.segment_weights(trace_lengths, rx_arr, ry_arr)
    actual_w = actual_w_arr.item()

    assert actual_w == pytest.approx(expected_w, rel=1e-4, abs=1e-8)


@given(
    lengths=st.floats(min_value=0.1, max_value=1000.0),
    ry=st.floats(min_value=-500.0, max_value=1500.0),
)
def test_segment_weights_at_zero(lengths, ry):
    assume(not np.isclose(ry, 0.0) and not np.isclose(ry, lengths))
    trace_lengths = np.array([lengths])
    rx = 0.0
    rx_arr = np.array([[rx]])
    ry_arr = np.array([[ry]])

    expected_w = 1 / (ry - lengths) - (1.0 / ry)

    actual_w_arr = sources.segment_weights(trace_lengths, rx_arr, ry_arr)
    actual_w = actual_w_arr.item()

    assert actual_w == pytest.approx(expected_w, rel=1e-4, abs=1e-8)
