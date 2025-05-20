"""Utility functions for working with moment rate and moment."""

import itertools

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from scipy.cluster.hierarchy import DisjointSet
from scipy.sparse import csr_array

from qcore import geo
from source_modelling import rupture_propagation, sources
from source_modelling.sources import Fault, Plane


def find_connected_faults(
    faults: dict[str, Fault | Plane],
    separation_distance: float = 2.0,
    dip_delta: float = 20.0,
    strike_delta: float | None = None,
    min_connected_depth: float = 5.0,
) -> DisjointSet:
    """Identify groups of connected faults based on proximity and dip angle.

    Faults are considered connected if this distance is within a specified
    threshold, their dip angles are similar, and (optionally) their strike
    angles are similar. A DisjointSet data structure is used to group connected
    faults.

    Parameters
    ----------
    faults : dict[str, sources.Fault]
        A dictionary mapping fault names to `sources.Fault` objects.
    separation_distance : float, optional
        The maximum allowable distance (in kilometers) between two faults for
        them to be considered connected. Defaults to 2.0 km. This distance
        is measured between the closest points on the faults below
        `min_connected_depth`.
    dip_delta : float, optional
        The maximum allowable absolute difference in dip angles (in degrees)
        between two faults for them to be considered connected.
        Defaults to 20.0 degrees.
    strike_delta : float, optional
        The maximum allowable absolute difference in the mean strike angles (in degrees)
        between two faults for them to be considered connected. If None, no
        strike comparison is made. Defaults to None.
    min_connected_depth : float, optional
        The minimum depth (in kilometres)
        below which the closest points between faults are determined for the
        distance calculation. Defaults to 5.0.

    Returns
    -------
    DisjointSet
        A `DisjointSet` object where each set represents a group of
        interconnected faults. The elements in the sets are the fault names
        (strings) provided in the input `faults` dictionary.

    """
    fault_names = list(faults)
    fault_components = DisjointSet(fault_names)
    for fault_a_name, fault_b_name in itertools.combinations(fault_names, r=2):
        fault_a = faults[fault_a_name]
        fault_b = faults[fault_b_name]
        if fault_components.connected(fault_a_name, fault_b_name):
            continue

        source_distance = rupture_propagation.distance_between(
            fault_a,
            fault_b,
            *sources.closest_points_beneath(
                fault_a,
                fault_b,
                min(
                    min_connected_depth,
                    0.99 * fault_a.bottom_m / 1000,
                    0.99 * fault_b.bottom_m / 1000,
                ),
            ),
        )
        if hasattr(fault_a, "planes"):
            mean_strike_a = geo.avg_wbearing(
                [(plane.strike, plane.length) for plane in fault_a.planes]
            )
        else:
            mean_strike_a = fault_a.strike
        if hasattr(fault_b, "planes"):
            mean_strike_b = geo.avg_wbearing(
                [(plane.strike, plane.length) for plane in fault_b.planes]
            )
        else:
            mean_strike_b = fault_b.strike

        if (
            source_distance < separation_distance * 1000
            and abs(fault_a.dip - fault_b.dip) < dip_delta
            and (
                strike_delta is None
                or geo.angle_diff(mean_strike_a, mean_strike_b) < strike_delta
            )
        ):
            fault_components.merge(fault_a_name, fault_b_name)

    return fault_components


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
