"""Utility functions for working with moment rate and moment."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from scipy.sparse import csr_array

# Shear scaling constant
MU = 3.3e10


def moment_rate_over_time_from_slip(
    area: npt.ArrayLike, slip: csr_array, dt: float, nt: int
) -> pd.DataFrame:
    """Compute a moment rate dataframe from subfaults given with area and slip.

    Parameters
    ----------
    area : array like
        The area of each subfault (in cm^2).
    slip : csr_array
        A sparse containing the slip for each subfault and each time
        window. Has shape (number of points, number of time windows).
    dt : float
        Length of each time window (s).
    nt : int
        The number of time windows.

    Returns
    -------
    pd.DataFrame
        A dataframe with index in time (s) and column 'moment_rate' (Nm/s).
    """
    slip_over_time = (
        np.asarray(sp.sparse.diags(np.asarray(area)).dot(slip).sum(axis=0))[0] / 1e6
    )
    moment_rate = MU * slip_over_time
    time_values = np.arange(nt) * dt

    moment_rate_df = pd.DataFrame(
        {"t": time_values, "moment_rate": moment_rate}
    ).set_index("t")
    return moment_rate_df


def moment_to_magnitude(moment: float) -> float:
    """Convert moment to magnitude.

    NOTE: the qcore mag_scaling module does not have this expression.

    Parameters
    ----------
    moment : float
        The moment of the rupture.

    Returns
    -------
    float
        Rupture magnitude
    """
    return 2 / 3 * np.log10(moment) - 6.03333


def moment_over_time_from_moment_rate(moment_rate_df: pd.DataFrame) -> pd.DataFrame:
    """Integrate a moment rate dataframe into a cumulative moment dataframe.

    Parameters
    ----------
    moment_rate_df : pd.DataFrame
        Dataframe containing time as an index, and 'moment_rate' column.

    Returns
    -------
    pd.DataFrame
        A dataframe the same index and a 'moment' column with cumulative rupture moment.
    """
    integrate_f = np.vectorize(
        lambda i: sp.integrate.trapezoid(
            moment_rate_df["moment_rate"].iloc[: i + 1].to_numpy(),
            moment_rate_df.index.values[: i + 1],
        )
    )
    return pd.DataFrame(
        {
            "t": moment_rate_df.index.values,
            "moment": integrate_f(np.arange(len(moment_rate_df))),
        }
    ).set_index("t")
