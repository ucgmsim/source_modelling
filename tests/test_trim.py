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


# --- trim_dims_somerville -------------------------------------------------


def test_trim_dims_somerville_uniform_array_is_untouched() -> None:
    """A uniform slip distribution has no low-slip edges, so nothing is trimmed."""
    assert trim.trim_dims_somerville(np.ones((3, 4))) == (0, 3, 0, 4)


def test_trim_dims_somerville_zero_array_is_untouched() -> None:
    """With zero average slip no edge is *strictly* below 30% of the mean."""
    assert trim.trim_dims_somerville(np.zeros((3, 4))) == (0, 3, 0, 4)


def test_trim_dims_somerville_trims_low_border_of_large_asperity() -> None:
    """A ring of slip 1 around an asperity of 100 is well below 30% of the mean."""
    slip = np.pad(np.array([[100.0]]), 1, constant_values=1.0)
    assert trim.trim_dims_somerville(slip) == (1, 2, 1, 2)


def test_trim_dims_somerville_removes_smallest_edge_first() -> None:
    """When several edges fall below the threshold, the smallest is trimmed first.

    Initially the left column (mean 1/3) and bottom row (mean 0.75) are both
    below 0.3 * mean = 0.775. Trimming the left column first raises the
    threshold for the bottom row to 1.45, which its remaining value (1.5)
    exceeds, so the bottom row is kept.
    """
    slip = np.array([[1.0, 8.0], [0.0, 5.0], [0.0, 1.5]])
    assert trim.trim_dims_somerville(slip) == (0, 3, 1, 2)


@given(arr=arrays(dtype=float, shape=(4, 5), elements=st.floats(0, 10)))
def test_trim_dims_somerville_retained_edges_meet_threshold(
    arr: npt.NDArray[np.floating],
) -> None:
    """Every edge of the retained window has at least 30% of its mean slip."""
    top, bottom, left, right = trim.trim_dims_somerville(arr)
    assert 0 <= top < bottom <= arr.shape[0]
    assert 0 <= left < right <= arr.shape[1]
    window = arr[top:bottom, left:right]
    threshold = 0.3 * window.mean()
    for edge in (window[0], window[-1], window[:, 0], window[:, -1]):
        assert edge.mean() >= threshold


# --- autocorrelation_dimension --------------------------------------------


def test_autocorrelation_dimension_exact_value_scales_with_dx() -> None:
    """For a constant profile of length n the dimension is (n^2 - 1) / n * dx."""
    slip = np.ones((5, 5))
    assert trim.autocorrelation_dimension(slip, 1.0) == pytest.approx(4.8)
    assert trim.autocorrelation_dimension(slip, 2.0) == pytest.approx(9.6)


def test_autocorrelation_dimension_axis() -> None:
    """The default axis (0) measures the profile along the rows of the array."""
    slip = np.ones((2, 4))
    # axis 0: profile of length 2 -> (4 - 1) / 2
    assert trim.autocorrelation_dimension(slip, 1.0) == pytest.approx(1.5)
    assert trim.autocorrelation_dimension(slip, 1.0, axis=0) == pytest.approx(1.5)
    # axis 1: profile of length 4 -> (16 - 1) / 4
    assert trim.autocorrelation_dimension(slip, 1.0, axis=1) == pytest.approx(3.75)


# --- trim_array_to_target_length ------------------------------------------
# The profiles below are chosen so the expected results do not depend on the
# expansion-loop behaviour questioned in issue #86.


def test_trim_array_to_target_length_default_axis_is_rows() -> None:
    profile = np.array([0, 0, 0, 1, 1, 0, 0], dtype=float)
    slip = np.tile(profile[:, np.newaxis], (1, 2))
    assert trim.trim_array_to_target_length(slip, 1.0, 2.0) == (3, 5)


def test_trim_array_to_target_length_already_within_tolerance() -> None:
    """No trimming happens when the array is within 2*dx of the target length."""
    slip = np.array([[0.5, 0.5, 3, 3, 3, 0.5, 0.5]])
    assert trim.trim_array_to_target_length(slip, 1.0, 5.0, axis=1) == (0, 7)
    # Same physical lengths expressed with a finer grid spacing.
    assert trim.trim_array_to_target_length(slip, 0.5, 2.5, axis=1) == (0, 7)


def test_trim_array_to_target_length_stops_at_tolerance_after_left_trim() -> None:
    """Once trimming starts from the left, the tolerance uses the window length."""
    slip = np.array([[0, 0.5, 0, 3]])
    assert trim.trim_array_to_target_length(slip, 1.0, 1.0, axis=1) == (1, 4)


def test_trim_array_to_target_length_keeps_wide_asperity() -> None:
    """Above-threshold cells are never trimmed, even if the target is shorter."""
    slip = np.array([[0, 0, 3, 3, 3, 3, 3, 3, 0, 0]], dtype=float)
    assert trim.trim_array_to_target_length(slip, 1.0, 1.0, axis=1) == (2, 8)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        # Both ends at/above max / 3 already: nothing to trim.
        ([0.5, 0, 0, 0.5], (0, 4)),
        # Left end exactly at the threshold (3 / 3 == 1) counts as kept.
        ([1, 0, 0, 3], (0, 4)),
        # Right end exactly at the threshold counts as kept.
        ([1, 3, 0, 1], (0, 4)),
        # 1 is below 3.5 / 3, so the left end is trimmed away.
        ([1, 0, 0, 3.5], (3, 4)),
        # Only the right end is above threshold: the left must be trimmed.
        ([0.5, 0, 0, 3], (3, 4)),
    ],
)
def test_trim_array_to_target_length_keep_threshold(
    profile: list[float], expected: tuple[int, int]
) -> None:
    """Trimming stops once both ends reach one third of the maximum slip."""
    slip = np.array([profile], dtype=float)
    assert trim.trim_array_to_target_length(slip, 1.0, 1.0, axis=1) == expected
