import numpy as np
import pytest
import scipy as sp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from source_modelling import moment


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
    avg_displacement = np.mean(np.trapz(slip_function, dx=dt, axis=1)) / 1e6
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
