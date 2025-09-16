"""Module for trimming and analysing slip arrays.

Provides functions for generating masks for low-slip regions, calculating
autocorrelation dimensions, and trimming slip arrays along specified axes.
"""

import numpy as np
import numpy.typing as npt
import scipy as sp

_SOMERVILLE_SLIP_THRESHOLD = 0.3


def trim_dims_somerville(
    slip_array: npt.NDArray[np.floating],
) -> tuple[int, int, int, int]:
    """Generate masking dimensions for a slip array using the Somerville trimming method [0]_.

    Iteratively removes rows and columns from the edges of the array where
    slip values are significantly lower than the average slip, based on a
    threshold fraction (default 30%) of the array mean.

    Parameters
    ----------
    slip_array : array of floats
        2D array representing slip values.

    Returns
    -------
    tuple of ints
        The (top, bottom, left, right) indices of the trim boundaries.

    References
    ----------
    .. [0] Somerville, P., Irikura, K., Graves, R., Sawada, S., Wald,
           D., Abrahamson, N., ... & Kowada, A. (1999). Characterizing
           crustal earthquake slip models for the prediction of strong ground
           motion. Seismological Research Letters, 70(1), 59-80.
    """
    top = left = 0
    bottom, right = slip_array.shape

    while top < bottom and left < right:
        slip_avg = slip_array[top:bottom, left:right].mean()
        # Find the smallest column of top, left, bottom, right with less than 30% of slip avg
        # Set column or row of mask to False and then shift that pointer

        # Edge means
        top_mean = slip_array[top, left:right].mean()
        bottom_mean = slip_array[bottom - 1, left:right].mean()
        left_mean = slip_array[top:bottom, left].mean()
        right_mean = slip_array[top:bottom, right - 1].mean()

        # pick the smallest edge below threshold
        edges = [
            ("top", top_mean),
            ("bottom", bottom_mean),
            ("left", left_mean),
            ("right", right_mean),
        ]
        edges = [
            (name, val)
            for name, val in edges
            if val < _SOMERVILLE_SLIP_THRESHOLD * slip_avg
        ]

        if not edges:
            break

        edge, _ = min(edges, key=lambda x: x[1])
        if edge == "top":
            top += 1
        elif edge == "bottom":
            bottom -= 1
        elif edge == "left":
            left += 1
        elif edge == "right":
            right -= 1

    return top, bottom, left, right


def trim_mask_somerville(
    slip_array: npt.NDArray[np.floating],
) -> npt.NDArray[np.bool_]:
    """Generate a mask for a slip array using the Somerville trimming method [0]_.

    Iteratively removes rows and columns from the edges of the array where
    slip values are significantly lower than the average slip, based on a
    threshold fraction (default 30%) of the array mean.

    Parameters
    ----------
    slip_array : array of floats
        2D array representing slip values.

    Returns
    -------
    array of bools
        Boolean mask of the same shape as `slip_array`. True indicates
        regions retained, False indicates trimmed regions.

    References
    ----------
    .. [0] Somerville, P., Irikura, K., Graves, R., Sawada, S., Wald,
           D., Abrahamson, N., ... & Kowada, A. (1999). Characterizing
           crustal earthquake slip models for the prediction of strong ground
           motion. Seismological Research Letters, 70(1), 59-80.
    """
    top, bottom, left, right = trim_dims_somerville(slip_array)
    mask = np.zeros_like(slip_array, dtype=np.bool_)
    mask[top:bottom, left:right] = True
    return mask


def autocorrelation_dimension(
    slip_array: npt.NDArray[np.floating], dx: float, axis: int = 0
) -> float:
    """Calculate the autocorrelation dimension of a slip array.

    Parameters
    ----------
    slip_array : array of floats
        The slip array to calculate the autocorrelation dimension of.
    dx : float
        The length of each subfault.
    axis : int, optional
        The axis along which to calculate the autocorrelation dimension, by default 0.

    Returns
    -------
    float
        The autocorrelation dimension of the slip array.
    """
    slip_function = slip_array.sum(axis=1 - axis)

    # Calculate the autocorrelation of the slip array
    autocorrelation = sp.signal.correlate(slip_function, slip_function, mode="full")

    # Calculate the autocorrelation dimension
    return sp.integrate.trapezoid(autocorrelation, dx=dx) / autocorrelation.max()


def trim_array_to_target_length(
    slip_array: npt.NDArray[np.floating],
    dx: float,
    target_length: float,
    axis: int = 0,
    trim_left: bool = True,
) -> tuple[int, int]:
    """Determine indices to trim a slip array to a target spatial length.

    Trims from both ends along the specified axis while avoiding regions with
    slip below a threshold (one third of the maximum slip). Ensures that the
    resulting length is approximately `target_length ± 2*dx`.

    Parameters
    ----------
    slip_array : ndarray
        Array to trim.
    dx : float
        Spatial spacing along the trimming axis.
    target_length : float
        Desired length after trimming.
    axis : int, optional
        Axis along which to trim the array (default is 0).
    trim_left : bool, optional
        Whether to allow trimming from the lower end of the axis (default True).

    Returns
    -------
    left : int
        Starting index of the trimmed region along the axis.
    right : int
        Ending index (exclusive) of the trimmed region along the axis.

    Raises
    ------
    ValueError
        If the array cannot be trimmed to satisfy the target length.
    """

    keep_threshold = slip_array.max() / 3
    slip_function = slip_array.max(axis=1 - axis)
    left = 0
    right = len(slip_function)
    while (
        left < right
        and not (
            slip_function[left] >= keep_threshold
            and slip_function[right - 1] >= keep_threshold
        )
        and abs((right - left) * dx - target_length) > 2 * dx
    ):
        # Here at least one of slip_function[left] < keep_threshold or
        # slip_function[right - 1] < keep_threshold is true. Hence, we
        # just need to check which smallest, as that one *definitely*
        # is below the keep threshold.
        if slip_function[left] < slip_function[right - 1]:
            left += 1
        else:
            right -= 1

    while left > 0 and slip_function[left] >= keep_threshold:
        left -= 1

    while right < len(slip_function) and slip_function[right - 1] >= keep_threshold:
        right += 1

    while slip_function[left] == 0 and left < right:
        left += 1

    while slip_function[right - 1] == 0 and left < right:
        right -= 1

    if not trim_left:
        left = 0

    if left >= right:
        raise ValueError("Cannot trim array to target length")

    return left, right


def trim_dims_thingbaijam(
    slip_array: npt.NDArray[np.floating], dx: float, dz: float, keep_top: bool = True
) -> tuple[int, int, int, int]:
    """Generate trimming dimensions for a slip array, removing regions of low slip.

    Uses the autocorrelation dimension along each axis to determine a
    characteristic length scale, then trims the array along both axes
    based on this scale to exclude low-asperity regions. Uses the
    method of Thingbaijam and Mai [0]_.

    Parameters
    ----------
    slip_array : array of floats
        2D array representing slip values.
    dx : float
        Subfault length along the horizontal axis.
    dz : float
        Subfault width along the vertical axis.
    keep_top : bool, optional
        Whether to prioritise keeping the upper rows when trimming (default True).

    Returns
    -------
    tuple of ints
        The (top, bottom, left, right) indices of the trim boundaries.

    References
    ----------
    .. [0] Thingbaijam, K. K., & Martin Mai, P. (2016). Evidence for
           truncated exponential probability distribution of earthquake slip.
           Bulletin of the Seismological Society of America, 106(4),
           1802-1816.
    """
    autocorrelation_length = autocorrelation_dimension(slip_array, dx, axis=1)

    autocorrelation_width = autocorrelation_dimension(slip_array, dz, axis=0)
    left, right = trim_array_to_target_length(
        slip_array, dx, autocorrelation_length, axis=1
    )
    top, bottom = trim_array_to_target_length(
        slip_array, dz, autocorrelation_width, trim_left=not keep_top
    )
    return top, bottom, left, right


def trim_mask_thingbaijam(
    slip_array: npt.NDArray[np.floating], dx: float, dz: float, keep_top: bool = True
) -> npt.NDArray[np.bool_]:
    """Generate a mask for a slip array, removing regions of low slip.

    Uses the autocorrelation dimension along each axis to determine a
    characteristic length scale, then trims the array along both axes
    based on this scale to exclude low-asperity regions. Uses the
    method of Thingbaijam and Mai [0]_.

    Parameters
    ----------
    slip_array : array of floats
        2D array representing slip values.
    dx : float
        Subfault length along the horizontal axis.
    dz : float
        Subfault width along the vertical axis.
    keep_top : bool, optional
        Whether to prioritise keeping the upper rows when trimming (default True).

    Returns
    -------
    array of bools
        Boolean mask of the same shape as `slip_array`. True indicates
        regions retained, False indicates trimmed regions.

    References
    ----------
    .. [0] Thingbaijam, K. K., & Martin Mai, P. (2016). Evidence for
           truncated exponential probability distribution of earthquake slip.
           Bulletin of the Seismological Society of America, 106(4),
           1802-1816.
    """
    top, bottom, left, right = trim_dims_thingbaijam(slip_array, dx, dz, keep_top)
    mask = np.zeros_like(slip_array, dtype=np.bool_)
    mask[top:bottom, left:right] = True
    return mask
