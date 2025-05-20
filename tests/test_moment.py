from unittest.mock import create_autospec, patch

import numpy as np
import pytest
import scipy as sp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from source_modelling import moment
from source_modelling.sources import Fault, Plane


@given(
    slip_function=nst.arrays(
        float,
        nst.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=20),
        elements=st.floats(0, 1000, allow_nan=False, allow_infinity=False),
    ),
    dt=st.floats(min_value=1e-6, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_moment_rate_from_slip_properties(slip_function: np.ndarray, dt: float):
    """Ensure that the moment rate from slip has basic properties.

    The properties are:
    1. Moment rate is always non-negative,
    2. Moment rate returns the right number of time windows (i.e. == nt),
    3. Cumulative moment is approximately equal to mu * area * average total displacement.
    """
    patch_areas = np.ones((slip_function.shape[0],))
    nt = slip_function.shape[1]
    moment_rate = moment.moment_rate_over_time_from_slip(
        patch_areas, sp.sparse.csr_array(slip_function), dt, nt
    )
    assert (moment_rate["moment_rate"] >= 0).all()
    assert len(moment_rate) == nt
    avg_displacement = (
        np.mean(sp.integrate.trapezoid(slip_function, dx=dt, axis=1)) / 1e6
    )
    cumulative_moment = moment.moment_over_time_from_moment_rate(moment_rate)
    assert moment.MU * np.sum(patch_areas) * avg_displacement == pytest.approx(
        cumulative_moment["moment"].iloc[-1]
    )


def test_moment_to_magnitude():
    """Very simple sanity checks for moment to magnitude.

    Tests are simple because function is simple.
    """
    # From Darfield FSP Atzori et al. 2012
    assert moment.moment_to_magnitude(5.01e19) == pytest.approx(7.10, abs=5e-02)
    # From Fiordland FSP Hayes 2009
    assert moment.moment_to_magnitude(2.82e20) == pytest.approx(7.6, abs=5e-02)
    # Kaikoura FSP Hayes 2017
    assert moment.moment_to_magnitude(8.96e20) == pytest.approx(7.89, abs=5e-02)


# The following test involves some patching to make it feasible to test properly.


@pytest.mark.parametrize(
    "dip_a, dip_b, strike_a, strike_b, distance, strike_delta, expected_connected",
    [
        (30, 32, 90, 92, 1.0, None, True),
        (30, 60, 90, 92, 1.0, None, False),
        (30, 32, 90, 150, 1.0, 10, False),
        (30, 32, 90, 150, 1.0, None, True),
        (30, 32, 90, 92, 3.0, None, False),
    ],
)
def test_find_connected_faults(
    dip_a: float,
    dip_b: float,
    strike_a: float,
    strike_b: float,
    distance: float,
    strike_delta: float | None,
    expected_connected: bool,
):
    with (
        patch(
            "source_modelling.rupture_propagation.distance_between",
            return_value=distance * 1000,
        ),
        patch(
            "source_modelling.sources.closest_points_beneath", return_value=("a", "b")
        ),
    ):
        # Create Plane mocks
        plane1 = create_autospec(Plane, instance=True)
        plane1.strike = strike_a
        plane1.length = 10

        plane2 = create_autospec(Plane, instance=True)
        plane2.strike = strike_b
        plane2.length = 10

        # Create Fault mocks
        fault1 = create_autospec(Fault, instance=True)
        fault1.dip = dip_a
        fault1.planes = [plane1]

        fault2 = create_autospec(Fault, instance=True)
        fault2.dip = dip_b
        fault2.planes = [plane2]

        faults = {"A": fault1, "B": fault2}

        ds = moment.find_connected_faults(faults, strike_delta=strike_delta)
        assert ds.connected("A", "B") == expected_connected
