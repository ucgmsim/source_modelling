"""Utility functions for working with moment rate and moment."""

import itertools

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from scipy.cluster.hierarchy import DisjointSet
from scipy.sparse import csr_array

from source_modelling import rupture_propagation, sources
from source_modelling.sources import Fault, Plane


def find_connected_faults(
    faults: dict[str, Fault | Plane],
    separation_distance: float = 2.0,
    dip_delta: float = 20.0,
    min_connected_depth: float = 5.0,
) -> DisjointSet:
    """Identify groups of connected faults based on proximity and dip angle.

    Faults are considered connected if this distance is within a specified
    threshold and their dip angles are similar. A DisjointSet data structure is
    used to group connected faults.

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
    for fault_a_name, fault_b_name in itertools.product(fault_names, repeat=2):
        if not fault_b_name or fault_components.connected(fault_a_name, fault_b_name):
            continue
        fault_a = faults[fault_a_name]
        fault_b = faults[fault_b_name]

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

        if (
            source_distance < separation_distance * 1000
            and abs(fault_a.dip - fault_b.dip) < dip_delta
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
        The moment of the rupture in Nm.

    Returns
    -------
    float
        Rupture magnitude
    """
    return 2 / 3 * np.log10(moment) - 6.03333


def magnitude_to_moment(magnitude: float) -> float:
    """Convert magnitude to moment.

    Parameters
    ----------
    magnitude : float
        The magnitude of the rupture.

    Returns
    -------
    float
        Rupture moment in Nm.
    """
    return 10 ** ((magnitude + 6.03333) * 3 / 2)


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


def dyne_cm_to_newton_metre(dyne_cm: float) -> float:
    """Convert dyne-cm units to standard SI units Nm.

    Parameters
    ----------
    dyne_cm : float
        Force in dyne-cm.

    Returns
    -------
    float
        Force in Nm.
    """
    _dyne_to_newton = 1e-5
    _cm_to_m = 1e-2
    return dyne_cm * _dyne_to_newton * _cm_to_m


def point_source_slip(
    moment_newton_metre: float,
    fault_area_km2: float,
    velocity_model_df: pd.DataFrame,
    source_depth_km: float,
) -> float:
    """Calculate slip for a point source.

    This calculation is based on the old workflow:
    https://github.com/ucgmsim/Pre-processing/blob/6572ea8be0963da4f7ac6a503ab07dd2519296e5/srf_generation/input_file_generation/realisation_to_srf.py#L321C9-L321C57

    Parameters
    ----------
    moment_newton_metre : float
        The seismic moment in newton-metre.
    fault_area_km2 : float
        The area of the fault in square kilometers.
    velocity_model_df : pd.DataFrame
      columns:
        - "depth_km": The depth in kilometers.
        - "rho_g_per_cm3": The density of the fault in grams per cubic centimeter.
        - "vs_km_per_s": The shear wave velocity in kilometers per second.
    source_depth_km : float
        The depth of the source in kilometers.

    Returns
    -------
    float
        The calculated slip in cm.
    """

    # Find the index of the closest depth in the velocity model
    idx = np.argmin(np.abs(velocity_model_df["depth_km"] - source_depth_km))
    vs_km_per_s = velocity_model_df.iloc[idx]["Vs"]
    rho_g_per_cm3 = velocity_model_df.iloc[idx]["rho"]

    vs_m_per_s = vs_km_per_s * 1e3  # Convert km/s to m/s
    rho_kg_per_m3 = rho_g_per_cm3 * 1e3  # Convert g/cm³ to kg/m³
    fault_area_m2 = fault_area_km2 * 1e6  # Convert km² to m²

    slip_m = moment_newton_metre / (fault_area_m2 * rho_kg_per_m3 * vs_m_per_s**2)

    return slip_m * 1e2  # Convert m to cm
