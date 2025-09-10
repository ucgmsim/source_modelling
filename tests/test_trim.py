import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from source_modelling import trim


def test_trim_mask_sommervile_basic():
    slip = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    mask = trim.trim_mask_sommervile(slip)
    expected_mask = np.array(
        [[False, False, False], [False, True, False], [False, False, False]]
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_autocorrelation_dimension_nonnegative():
    slip = np.ones((5, 5))
    dx = 1.0
    dim = trim.autocorrelation_dimension(slip, dx)
    assert dim >= 0


def test_trim_array_to_target_length_basic():
    slip = np.array([[0, 0, 0, 1, 1, 0, 0]])
    dx = 1.0
    target_length = 2.0
    left, right = trim.trim_array_to_target_length(slip, dx, target_length, axis=1)
    assert (left, right) == (3, 5)


def test_trim_slip_array_keep_top_false():
    """Check trimming when top is allowed to be trimmed."""
    slip = np.zeros((5, 5))
    slip[2, 2] = 1.0
    dx = dz = 1.0

    mask = trim.trim_mask_thingbaijam(slip, dx, dz, keep_top=False)

    expected_mask = np.zeros_like(slip, dtype=bool)
    expected_mask[2, 2] = True

    np.testing.assert_array_equal(mask, expected_mask)


def test_trim_slip_array_keep_top_true():
    """Check trimming when top is preserved (default behavior)."""
    slip = np.zeros((5, 5))
    slip[2, 2] = 1.0
    dx = dz = 1.0

    mask = trim.trim_mask_thingbaijam(slip, dx, dz, keep_top=True)

    # With keep_top=True, the top edge of the region is kept
    # So the mask will include all rows from top of region down
    # In this tiny example, rows 0-2 will all be True in column 2
    expected_mask = np.zeros_like(slip, dtype=bool)
    expected_mask[0:3, 2] = True
    np.testing.assert_array_equal(mask, expected_mask)


@given(arr=arrays(dtype=float, shape=(5, 5), elements=st.floats(0, 10)))
def test_trim_mask_sommervile_shape(arr):
    mask = trim.trim_mask_sommervile(arr)
    assert mask.shape == arr.shape
    assert mask.dtype == bool


@given(arr=arrays(dtype=float, shape=(5, 5), elements=st.floats(0.01, 10)))
def test_trim_mask_thingbaijam_shape(arr):
    mask = trim.trim_mask_thingbaijam(arr, dx=1.0, dz=1.0)
    assert mask.shape == arr.shape
    assert np.all(np.isin(mask, [True, False]))


@given(arr=arrays(dtype=float, shape=(5, 5), elements=st.floats(0.01, 10)))
def test_autocorrelation_dimension_nonneg(arr):
    dim = trim.autocorrelation_dimension(arr, dx=1.0)
    assert dim >= 0


@given(
    arr=arrays(dtype=float, shape=(5, 5), elements=st.floats(0.01, 10)),
    dx=st.floats(0.1, 2.0),
    target_length=st.floats(0.1, 10.0),
)
def test_trim_array_to_target_length_bounds(arr, dx, target_length):
    try:
        left, right = trim.trim_array_to_target_length(arr, dx, target_length)
        assert 0 <= left < right <= arr.shape[1]
    except ValueError:
        pass
