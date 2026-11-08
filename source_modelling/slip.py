"""Utilities for slip and slip velocity functions."""

import numpy as np
import numpy.typing as npt


def _safe_duration(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Return ``end - start``, replacing non-positive durations with 1.

    The replacement value never reaches a returned result; it only
    prevents division by zero in ``np.where`` branches that are masked
    out.
    """
    duration = end - start
    return np.where(duration > 0, duration, 1.0)


def rise_time_from_moment(moment: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Compute approximate rise time from rupture moment.

    Parameters
    ----------
    moment : array-like
        Moment(s) to convert to rise times.

    Returns
    -------
    ndarray
        Approximate rise time for rupture subfaults.

    References
    ----------
    Baker, J., Bradley, B., & Stafford, P. (2021). Seismic hazard and risk
    analysis (p. 202). Cambridge University Press.
    """
    return 3.12e-7 * np.cbrt(moment)


def box_car_slip(
    t: npt.ArrayLike, t0: npt.ArrayLike, t1: npt.ArrayLike, slip: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Boxcar slip velocity function.

    Parameters
    ----------
    t : array-like
        Time to compute slip for.
    t0 : array-like
        Time slip begins.
    t1 : array-like
        Time slip ends.
    slip : array-like
        Total slip.

    Returns
    -------
    ndarray
        Returns the slip for time t, according to

        f(t) = slip / (t1 - t0) if t0 <= t <= t1
        f(t) = 0                otherwise

        If t1 == t0 (zero duration), f(t) = 0 everywhere.
    """
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    t = np.asarray(t)
    slip = np.asarray(slip)
    return np.where((t >= t0) & (t <= t1) & (t1 > t0), slip / _safe_duration(t0, t1), 0)


def triangular_slip(
    t: npt.ArrayLike,
    t0: npt.ArrayLike,
    t1: npt.ArrayLike,
    peak: npt.ArrayLike,
    slip: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """Assymetric triangular slip function.

    Parameters
    ----------
    t : array-like
        Time to compute slip for.
    t0 : array-like
        Time slip begins.
    t1 : array-like
        Time slip ends.
    peak : array-like
        Time when slip peaks.
    slip : array-like
        Total slip.

    Returns
    -------
    ndarray
        Returns the slip for time t, according to

        f(t) = 2 * slip / (t1 - t0) * (t - t0) / (peak - t0), if t0 <= t <= peak
        f(t) = 2 * slip / (t1 - t0) * (1 - (t - peak) / (t1 - peak))
        f(t) = 0 otherwise

        The peak may coincide with t0 or t1, giving a right-angled
        triangle. If t1 == t0 (zero duration), f(t) = 0 everywhere.
    """
    t = np.asarray(t)
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    peak = np.asarray(peak)
    slip = np.asarray(slip)
    rising = np.where(peak > t0, (t - t0) / _safe_duration(t0, peak), 1)
    falling = np.where(t1 > peak, 1 - (t - peak) / _safe_duration(peak, t1), 1)
    return np.where(
        (t >= t0) & (t <= t1) & (t1 > t0),
        np.minimum(rising, falling) * 2 * slip / _safe_duration(t0, t1),
        0,
    )


def isoceles_triangular_slip(
    t: npt.ArrayLike, t0: npt.ArrayLike, t1: npt.ArrayLike, slip: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Symmetric triangular slip function.

    Parameters
    ----------
    t : array-like
        Time to compute slip for.
    t0 : array-like
        Time slip begins.
    t1 : array-like
        Time slip ends.
    slip : array-like
        Total slip.

    Returns
    -------
    ndarray
        Returns the slip for time t, according to assymetric triangular slip
        with peak = (t0 + t1) / 2.
    """
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    return triangular_slip(t, t0, t1, (t0 + t1) / 2, slip)


def cosine_slip(
    t: npt.ArrayLike, t0: npt.ArrayLike, t1: npt.ArrayLike, slip: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Cosine slip velocity function.

    Parameters
    ----------
    t : array-like
        The time to evaluate the SVF at.
    t0 : array-like
        The time for slip to begin.
    t1 : array-like
        The time for slip to end.
    slip : array-like
        The total slip.

    Returns
    -------
    np.ndarray
        The slip for time t, according to the function

        f(t) = cos(((t - t0) / (t1 - t0) - 1 / 2) * pi)  for t0 <= t <= t1
        f(t) = 0 otherwise

        If t1 == t0 (zero duration), f(t) = 0 everywhere.
    """
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    slip = np.asarray(slip)
    duration = _safe_duration(t0, t1)
    return np.where(
        (t >= t0) & (t <= t1) & (t1 > t0),
        (np.pi * slip / (2 * duration))
        * np.cos((t - t0) / duration * np.pi - np.pi / 2),
        0,
    )
