from unittest.mock import create_autospec, patch

import numpy as np
import pandas as pd
import pytest
import scipy as sp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from source_modelling import moment
from source_modelling.magnitude_scaling import Mw
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


def test_moment_to_magnitude_units():
    # Dusky sound earthquake, nominal magnitude ~ 8.0 but GCMT provides the moment in dyne-cm.
    # Should throw error because the magnitude is too large to be physically plausible.
    mom = 1.44e+28
    with pytest.raises(ValueError, match='Magnitude for moment is unreasonably large'):
        moment.moment_to_magnitude()
         
    
    


# The following test involves some patching to make it feasible to test properly.


@pytest.mark.parametrize(
    "dip_a, dip_b, strike_a, strike_b, distance, expected_connected",
    [
        (30, 32, 90, 92, 1.0, True),
        (30, 60, 90, 92, 1.0, False),
        (30, 32, 90, 150, 1.0, True),
        (30, 32, 90, 92, 3.0, False),
    ],
)
def test_find_connected_faults(
    dip_a: float,
    dip_b: float,
    strike_a: float,
    strike_b: float,
    distance: float,
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
        fault1.bottom_m = (
            10000.0  # closest points beneath is mocked out so it doesn't matter
        )

        fault2 = create_autospec(Fault, instance=True)
        fault2.dip = dip_b
        fault2.planes = [plane2]
        fault2.bottom_m = (
            10000.0  # closest points beneath is mocked out so it doesn't matter
        )

        faults = {"A": fault1, "B": fault2}

        ds = moment.find_connected_faults(
            faults,
        )
        assert ds.connected("A", "B") == expected_connected


def test_point_source_slip_top_depth():
    bad_vm = pd.DataFrame(
        {
            "depth_km": [1.0],
            "Vs": [1.0],
            "rho": [1.0],
        }
    )  # Other qualities not included because point source slip should not use them

    with pytest.raises(ValueError, match="Velocity model does not begin at 0km depth"):
        moment.point_source_slip(1.0, 1.0, bad_vm, 1.0)


def test_point_source_slip_simple():
    """Test point source slip calculation for arbitrary example"""
    simple_vm = pd.DataFrame(
        {
            "depth_km": [0.0],
            "Vs": [1.0],
            "rho": [1.0],
        }
    )  # Other qualities not included because point source slip should not use them

    slip = moment.point_source_slip(1e12, 1.0, simple_vm, 1.0)
    # 1e12 / (1e6 * 1e3 * 1e6) = 1e-3 m slip
    # 1e-3 * 1e2 = 1e-1 cm slip
    assert slip == pytest.approx(0.1)


def test_point_source_slip_middle():
    "Test point source slip calculation in the middle of layers"
    simple_vm = pd.DataFrame(
        {
            "depth_km": [0.0, 1.5],
            "Vs": [1.0, 0.0],
            "rho": [1.0, 0.0],
        }
    )  # Other qualities not included because point source slip should not use them

    # Same calculation as the example, because source depth is 1.0km it should
    # use the top layer (should crash if using the bottom).
    slip = moment.point_source_slip(1e12, 1.0, simple_vm, 1.0)
    assert slip == pytest.approx(0.1)


def test_point_source_slip_boundary():
    "Test point source slip calculation at boundary of layers"
    simple_vm = pd.DataFrame(
        {
            "depth_km": [0.0, 1.0],
            "Vs": [0.0, 1.0],
            "rho": [0.0, 1.0],
        }
    )  # Other qualities not included because point source slip should not use them

    # Same calculation as the example, because source depth is 1.0km it should
    # use the bottom layer not the top (should crash if using the top).
    slip = moment.point_source_slip(1e12, 1.0, simple_vm, 1.0)
    assert slip == pytest.approx(0.1)


def test_point_source_slip_bad_dataframe():
    """Test that point_source_slip crashes with a DataFrame missing required columns."""
    # Create a DataFrame missing the required columns
    bad_velocity_model_df = pd.DataFrame(
        {
            "bad_column1": [0.05, 2, 1038.9999],
            "bad_column2": [0.5, 1.64, 4.6],
            "bad_column3": [1.81, 2.28, 3.33],
        }
    )

    moment_newton_metre = moment.magnitude_to_moment(Mw(5.0))

    # Should raise KeyError when trying to access missing columns
    with pytest.raises(KeyError):
        moment.point_source_slip(
            moment_newton_metre=moment_newton_metre,
            fault_area_km2=100.0,
            velocity_model_df=bad_velocity_model_df,
            source_depth_km=2.0,
        )
