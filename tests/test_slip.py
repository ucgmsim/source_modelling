from collections.abc import Callable

import numpy as np
import pytest
import scipy as sp
from hypothesis import given
from hypothesis import strategies as st

from source_modelling import slip


@given(
    t0=st.floats(0.1, 100), offset=st.floats(1e-6, 100), total_slip=st.floats(1, 1000)
)
def test_box_car_slip(t0: float, offset: float, total_slip: float):
    """Check that the box car slip function is a well-behaved slip function.

    For the box car slip function we must have:

    1. Slip is always non-negative,
    2. Slip function integrates to total slip over the interval [t0, t1].
    """
    t1 = t0 + offset
    t = np.linspace(t0, t1, num=100)
    dt = t[1] - t[0]
    slip_function = slip.box_car_slip(t, t0, t1, total_slip)

    # Slip should be non-negative
    assert np.all(slip_function >= 0)
    total_slip_from_function = sp.integrate.trapezoid(slip_function, dx=dt)
    assert np.allclose(total_slip_from_function, total_slip)


@given(
    t0=st.floats(0.1, 100),
    peak_position=st.floats(0.1, 0.9),
    offset=st.floats(1e-6, 100),
    total_slip=st.floats(1, 1000),
)
def test_triangular_slip(
    t0: float, peak_position: float, offset: float, total_slip: float
):
    """Check that the triangular slip function is a well-behaved slip function.

    For the triangular slip function we must have:

    1. Slip is always non-negative,
    2. Slip is zero at t0 and t1,
    3. Slip function integrates to total slip over the interval [t0, t1].
    """
    t1 = t0 + offset
    peak = t0 + peak_position * offset
    t = np.linspace(t0, t1, num=200)
    dt = t[1] - t[0]
    slip_function = slip.triangular_slip(t, t0, t1, peak, total_slip)

    # Slip should be non-negative and zero at boundaries
    assert np.all(slip_function >= 0)
    assert slip_function[0] == pytest.approx(0)
    assert slip_function[-1] == pytest.approx(0)
    total_slip_from_function = sp.integrate.trapezoid(slip_function, dx=dt)
    assert total_slip_from_function == pytest.approx(total_slip, rel=1e-4)


@given(
    t0=st.floats(0.1, 100),
    offset=st.floats(1e-6, 100),
    total_slip=st.floats(1, 1000),
)
def test_isoceles_triangular_slip(t0: float, offset: float, total_slip: float):
    """Check that the isoceles triangular slip function is a well-behaved slip function.

    For the isoceles slip function we must have:

    1. Slip is always non-negative,
    2. Slip is zero at t0 and t1,
    3. Slip function integrates to total slip over the interval [t0, t1].
    """
    t1 = t0 + offset
    t = np.linspace(t0, t1, num=200)
    dt = t[1] - t[0]
    slip_function = slip.isoceles_triangular_slip(t, t0, t1, total_slip)

    # Slip should be non-negative and zero at boundaries
    assert np.all(slip_function >= 0)
    assert slip_function[0] == pytest.approx(0)
    assert slip_function[-1] == pytest.approx(0)
    total_slip_from_function = sp.integrate.trapezoid(slip_function, dx=dt)
    assert total_slip_from_function == pytest.approx(total_slip, rel=1e-4)


@given(
    t0=st.floats(0.1, 100),
    offset=st.floats(1e-6, 100),
    total_slip=st.floats(1, 1000),
)
def test_cosine_slip(t0: float, offset: float, total_slip: float):
    """Check that the cosine slip function is a well-behaved slip function.

    For the cosine slip function we must have:

    1. Slip is always non-negative,
    2. Slip is zero at t0 and t1,
    3. Slip function integrates to total slip over the interval [t0, t1].
    """
    t1 = t0 + offset
    t = np.linspace(t0, t1, num=200)
    dt = t[1] - t[0]
    slip_function = slip.cosine_slip(t, t0, t1, total_slip)

    # Slip should be non-negative and zero at boundaries
    assert np.all(slip_function >= 0)
    # Need to be a bit more lenient on boundary conditions because of
    # np.cos approximation errors.
    assert slip_function[0] == pytest.approx(0, abs=1e-6)
    assert slip_function[-1] == pytest.approx(0, abs=1e-6)
    total_slip_from_function = sp.integrate.trapezoid(slip_function, dx=dt)
    assert total_slip_from_function == pytest.approx(total_slip, rel=1e-4)


@pytest.mark.parametrize(
    "slip_function",
    [slip.box_car_slip, slip.isoceles_triangular_slip, slip.cosine_slip],
)
@pytest.mark.parametrize("total_slip", [0.0, 50.0])
def test_zero_duration_slip_is_zero(
    slip_function: Callable[..., np.ndarray], total_slip: float
):
    """A zero-duration slip pulse (t1 == t0) gives zero slip rate everywhere."""
    t = np.linspace(0, 5, 6)
    with np.errstate(all="raise"):
        result = slip_function(t, 1.0, 1.0, total_slip)
    assert np.array_equal(result, np.zeros_like(t))


def test_zero_slip_subfault_does_not_contaminate_others():
    """A zero-slip subfault (zero rise time) does not introduce NaNs."""
    t = np.linspace(0, 5, 501)
    moment = np.array([1e16, 0.0])
    rise = slip.rise_time_from_moment(moment)
    t0 = np.array([1.0, 1.0])
    t1 = t0 + rise
    total = np.array([50.0, 0.0])
    for slip_function in [
        slip.box_car_slip,
        slip.isoceles_triangular_slip,
        slip.cosine_slip,
    ]:
        with np.errstate(all="raise"):
            result = slip_function(t[:, None], t0, t1, total)
        assert np.all(np.isfinite(result))
        assert np.all(result[:, 1] == 0)
        assert np.any(result[:, 0] > 0)


def test_triangular_slip_zero_duration():
    """triangular_slip with t0 == peak == t1 is zero everywhere."""
    t = np.linspace(0, 5, 6)
    with np.errstate(all="raise"):
        result = slip.triangular_slip(t, 1.0, 1.0, 1.0, 1.0)
    assert np.array_equal(result, np.zeros_like(t))


@pytest.mark.parametrize("peak", [1.0, 2.0])
def test_triangular_slip_peak_at_endpoint(peak: float):
    """A peak at t0 or t1 gives a right-angled triangle."""
    t0, t1, total_slip = 1.0, 2.0, 3.0
    t = np.linspace(0, 3, 3001)
    with np.errstate(all="raise"):
        result = slip.triangular_slip(t, t0, t1, peak, total_slip)
        at_peak = slip.triangular_slip(peak, t0, t1, peak, total_slip)
        other_end = t1 if peak == t0 else t0
        at_other_end = slip.triangular_slip(other_end, t0, t1, peak, total_slip)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    assert at_peak == pytest.approx(2 * total_slip / (t1 - t0))
    assert at_other_end == pytest.approx(0)
    assert np.all(result[(t < t0) | (t > t1)] == 0)
    inside = (t >= t0) & (t <= t1)
    assert sp.integrate.trapezoid(result[inside], t[inside]) == pytest.approx(
        total_slip, rel=1e-4
    )
