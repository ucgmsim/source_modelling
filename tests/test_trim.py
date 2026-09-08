import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from source_modelling import trim


def test_trim_mask_somerville_basic() -> None:
    slip = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    mask = trim.trim_mask_somerville(slip)
    expected_mask = np.array(
        [[False, False, False], [False, True, False], [False, False, False]]
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_autocorrelation_dimension_nonnegative() -> None:
    slip = np.ones((5, 5))
    dx = 1.0
    dim = trim.autocorrelation_dimension(slip, dx)
    assert dim >= 0


def test_trim_array_to_target_length_basic() -> None:
    slip = np.array([[0, 0, 0, 1, 1, 0, 0]])
    dx = 1.0
    target_length = 2.0
    left, right = trim.trim_array_to_target_length(slip, dx, target_length, axis=1)
    assert (left, right) == (3, 5)


def test_trim_slip_array_keep_top_false() -> None:
    """Check trimming when top is allowed to be trimmed."""
    slip = np.zeros((5, 5))
    slip[2, 2] = 1.0
    dx = dz = 1.0

    mask = trim.trim_mask_thingbaijam(slip, dx, dz, keep_top=False)

    expected_mask = np.zeros_like(slip, dtype=bool)
    expected_mask[2, 2] = True

    np.testing.assert_array_equal(mask, expected_mask)


def test_trim_slip_array_keep_top_true() -> None:
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
def test_trim_mask_somerville_shape(arr: npt.NDArray[np.floating]) -> None:
    mask = trim.trim_mask_somerville(arr)
    assert mask.shape == arr.shape
    assert mask.dtype == bool


@given(arr=arrays(dtype=float, shape=(5, 5), elements=st.floats(0.01, 10)))
def test_trim_mask_thingbaijam_shape(arr: npt.NDArray[np.floating]) -> None:
    mask = trim.trim_mask_thingbaijam(arr, dx=1.0, dz=1.0)
    assert mask.shape == arr.shape
    assert np.all(np.isin(mask, [True, False]))


@given(arr=arrays(dtype=float, shape=(5, 5), elements=st.floats(0.01, 10)))
def test_autocorrelation_dimension_nonneg(arr: npt.NDArray[np.floating]) -> None:
    dim = trim.autocorrelation_dimension(arr, dx=1.0)
    assert dim >= 0


@given(
    arr=arrays(dtype=float, shape=(5, 5), elements=st.floats(0.01, 10)),
    dx=st.floats(0.1, 2.0),
    target_length=st.floats(0.1, 10.0),
)
def test_trim_array_to_target_length_bounds(
    arr: npt.NDArray[np.floating], dx: float, target_length: float
) -> None:
    try:
        left, right = trim.trim_array_to_target_length(arr, dx, target_length)
        assert 0 <= left < right <= arr.shape[1]
    except ValueError:
        pass


@pytest.mark.parametrize(
    ("slip_function", "target_length", "expected"),
    [
        ([1.0, 1.0, 9.0, 1.0, 1.0], 1.0, (0, 3)),
        ([1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0], 1.0, (1, 4)),
        ([0.0, 0.0, 9.0, 0.0, 0.0], 1.0, (2, 3)),
    ],
)
def test_trim_expansion_does_not_absorb_sub_threshold_cells(
    slip_function: list[float], target_length: float, expected: tuple[int, int]
):
    """The expansion loops must not absorb a neighbour below the keep threshold.

    Regression test: the loops tested ``slip_function[left]`` and
    ``slip_function[right - 1]`` -- cells already inside the window -- instead
    of the candidate cells ``left - 1`` and ``right``. A boundary cell above
    the threshold therefore pulled in its neighbour without that neighbour ever
    being checked, widening the window past the documented
    ``target_length +/- 2 * dx`` tolerance.
    """
    slip_array = np.array(slip_function).reshape(-1, 1)

    left, right = trim.trim_array_to_target_length(
        slip_array, dx=1.0, target_length=target_length
    )

    assert (left, right) == expected
    # the documented tolerance from the docstring
    assert abs((right - left) * 1.0 - target_length) <= 2.0
    # no cell inside the window may be below the keep threshold unless the
    # window is pinned by the tolerance
    keep_threshold = slip_array.max() / 3
    assert slip_array[left:right].max() >= keep_threshold
