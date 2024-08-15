"""Utilities for slip and slip velocity functions."""

import numpy as np
import numpy.typing as npt


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
    t: np.floating, t0: npt.ArrayLike, t1: npt.ArrayLike, slip: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Boxcar slip velocity function.

    Parameters
    ----------
    t : float
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
    """
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    slip = np.asarray(slip)
    return np.where((t >= t0) & (t <= t1), slip / (t1 - t0), 0)


def triangular_slip(
    t: np.floating,
    t0: npt.ArrayLike,
    t1: npt.ArrayLike,
    peak: npt.ArrayLike,
    slip: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """Assymetric triangular slip function.

    Parameters
    ----------
    t : float
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
    """
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    peak = np.asarray(peak)
    slip = np.asarray(slip)
    return (
        np.where(
            (t >= t0) & (t <= t1),
            np.minimum((t - t0) / (peak - t0), 1 - (t - peak) / (t1 - peak)),
            0,
        )
        * 2
        * slip
        / (t1 - t0)
    )


def isoceles_triangular_slip(
    t: np.floating, t0: npt.ArrayLike, t1: npt.ArrayLike, slip: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Symmetric triangular slip function.

    Parameters
    ----------
    t : float
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
    t: np.floating, t0: npt.ArrayLike, t1: npt.ArrayLike, slip: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Cosine slip velocity function.

    Parameters
    ----------
    t : float
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
    """
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    slip = np.asarray(slip)
    return (np.pi * slip / (2 * (t1 - t0))) * np.where(
        (t >= t0) & (t <= t1), np.cos((t - t0) / (t1 - t0) * np.pi - np.pi / 2), 0
    )


def yoffe(
    t: np.floating, t0: npt.ArrayLike, t1: npt.ArrayLike, slip: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Yoffe slip function.

    Parameters
    ----------
    t : float
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
        The slip for time t, according to the Yoffe slip velocity function[0].

    References
    ----------
    Yoffe, E. (1951). The moving Griffith crack, Phil. Mag. 42, 739–750.
    """
    t1 = np.asarray(t1)
    t0 = np.asarray(t0)
    slip = np.asarray(slip)
    tau = t1 - t0
    t = t - t0
    # The following is ok because the where clause ensures that values of t <= 0 map to zero anyway.
    with np.errstate(divide="ignore", invalid="ignore"):
        return slip * np.where(
            (t > 0) & (t <= tau), 2 / (np.pi * tau) * np.sqrt((tau - t) / t), 0
        )
