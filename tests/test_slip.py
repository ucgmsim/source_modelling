import numpy as np
import pytest
import scipy as sp
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from source_modelling import slip


@given(t0=st.floats(0.1, 100), dt=st.floats(1e-6, 100), total_slip=st.floats(1, 1000))
def test_box_car_slip(t0: float, dt: float, total_slip: float):
    """Check that the box car slip function is a well-behaved slip function.

    For the box car slip function we must have:

    1. Slip is always non-negative,
    2. Slip is zero at t0 and t1,
    3. Slip function integrates to total slip over the interval [t0, t1].
    """
    t1 = t0 + dt
    t = np.linspace(t0, t1, num=20)
    slip_function = slip.box_car_slip(t, t0, t1, total_slip)

    # Slip should be non-negative
    assert np.all(slip_function >= 0)
    # Total slip integrated should be close to the total slip given.
    # Using scipy quad because numpy trapz is not good enough
    total_slip_from_function, _ = sp.integrate.quad(
        lambda t: slip.box_car_slip(t, t0, t1, total_slip), t0, t1
    )
    assert np.allclose(total_slip_from_function, total_slip)


@given(
    t0=st.floats(0.1, 100),
    peak_position=st.floats(0.1, 0.9),
    dt=st.floats(1e-6, 100),
    total_slip=st.floats(1, 1000),
)
def test_triangular_slip(t0: float, peak_position: float, dt: float, total_slip: float):
    """Check that the triangular slip function is a well-behaved slip function.

    For the triangular slip function we must have:

    1. Slip is always non-negative,
    2. Slip is zero at t0 and t1,
    3. Slip function integrates to total slip over the interval [t0, t1].
    """
    t1 = t0 + dt
    peak = t0 + peak_position * dt
    t = np.linspace(t0, t1, num=20)
    slip_function = slip.triangular_slip(t, t0, t1, peak, total_slip)

    # Slip should be non-negative and zero at boundaries
    assert np.all(slip_function >= 0)
    assert slip_function[0] == pytest.approx(0)
    assert slip_function[-1] == pytest.approx(0)
    # Total slip integrated should be close to the total slip given.
    # Using scipy quad because numpy trapz is not good enough
    total_slip_from_function, _ = sp.integrate.quad(
        lambda t: slip.triangular_slip(t, t0, t1, peak, total_slip), t0, t1
    )
    assert np.allclose(total_slip_from_function, total_slip)


@given(
    t0=st.floats(0.1, 100),
    dt=st.floats(1e-6, 100),
    total_slip=st.floats(1, 1000),
)
def test_isoceles_triangular_slip(t0: float, dt: float, total_slip: float):
    """Check that the isoceles triangular slip function is a well-behaved slip function.

    For the isoceles slip function we must have:

    1. Slip is always non-negative,
    2. Slip is zero at t0 and t1,
    3. Slip function integrates to total slip over the interval [t0, t1].
    """
    t1 = t0 + dt
    t = np.linspace(t0, t1, num=20)
    slip_function = slip.isoceles_triangular_slip(t, t0, t1, total_slip)

    # Slip should be non-negative and zero at boundaries
    assert np.all(slip_function >= 0)
    assert slip_function[0] == pytest.approx(0)
    assert slip_function[-1] == pytest.approx(0)
    # Total slip integrated should be close to the total slip given.
    # Using scipy quad because numpy trapz is not good enough
    total_slip_from_function, _ = sp.integrate.quad(
        lambda t: slip.isoceles_triangular_slip(t, t0, t1, total_slip), t0, t1
    )
    assert np.allclose(total_slip_from_function, total_slip)


@given(
    t0=st.floats(0.1, 100),
    dt=st.floats(1e-6, 100),
    total_slip=st.floats(1, 1000),
)
def test_cosine_slip(t0: float, dt: float, total_slip: float):
    """Check that the cosine slip function is a well-behaved slip function.

    For the cosine slip function we must have:

    1. Slip is always non-negative,
    2. Slip is zero at t0 and t1,
    3. Slip function integrates to total slip over the interval [t0, t1].
    """
    t1 = t0 + dt
    t = np.linspace(t0, t1, num=20)
    slip_function = slip.cosine_slip(t, t0, t1, total_slip)

    # Slip should be non-negative and zero at boundaries
    assert np.all(slip_function >= 0)
    # Need to be a bit more lenient on boundary conditions because of
    # np.cos approximation errors.
    assert slip_function[0] == pytest.approx(0, abs=1e-6)
    assert slip_function[-1] == pytest.approx(0, abs=1e-6)
    # Total slip integrated should be close to the total slip given.
    # Using scipy quad because numpy trapz is not good enough
    total_slip_from_function, _ = sp.integrate.quad(
        lambda t: slip.cosine_slip(t, t0, t1, total_slip), t0, t1
    )
    # Need to be a bit more lenient for the approximation here.
    assert np.allclose(total_slip_from_function, total_slip, atol=1e-6)
