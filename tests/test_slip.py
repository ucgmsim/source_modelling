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
    3. Slip function integrates to total slip over the interval [t0, t1].
    """
    t1 = t0 + offset
    t = np.linspace(t0, t1, num=100)
    dt = t[1] - t[0]
    slip_function = slip.box_car_slip(t, t0, t1, total_slip)

    # Slip should be non-negative
    assert np.all(slip_function >= 0)
    total_slip_from_function = np.trapz(slip_function, dx=dt)
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
    total_slip_from_function = np.trapz(slip_function, dx=dt)
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
    total_slip_from_function = np.trapz(slip_function, dx=dt)
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
    total_slip_from_function = np.trapz(slip_function, dx=dt)
    assert total_slip_from_function == pytest.approx(total_slip, rel=1e-4)
