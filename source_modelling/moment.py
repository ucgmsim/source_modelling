"""Utility functions for working with moment rate and moment."""

import itertools
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from scipy.cluster.hierarchy import DisjointSet
from scipy.sparse import csr_array

from source_modelling import rupture_propagation, sources
from source_modelling.magnitude_scaling import BoldM, Mw
from source_modelling.sources import Fault, Plane

# Moment magnitude scale coefficients for seismic moment in Nm, from
# equations 4 and 7 of Hanks and Kanamori (1979). See the [Hanks1979] reference
# in the `moment_to_magnitude` docstring for the full citation.
EQUATION_4_COEFFICIENT = 6.0666  # `Mw` convention
EQUATION_7_COEFFICIENT = 6.0333  # `BoldM` convention


def find_connected_faults(
    faults: dict[str, Fault | Plane],
    separation_distance: float = 2.0,
    dip_delta: float = 20.0,
    min_connected_depth: float = 5.0,
) -> "DisjointSet[str]":
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
    fault_components: DisjointSet[str] = DisjointSet(fault_names)
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


@typing.overload
def moment_to_magnitude(moment: float, bold_m: typing.Literal[True]) -> BoldM: ...
@typing.overload
def moment_to_magnitude(moment: float, bold_m: typing.Literal[False] = False) -> Mw: ...
def moment_to_magnitude(moment: float, bold_m: bool = True) -> BoldM | Mw:
    """Convert moment to magnitude.

    NOTE: the qcore mag_scaling module does not have this expression.

    Parameters
    ----------
    moment : float
        The moment of the rupture in Nm.
    bold_m : bool, optional
        Set whether Equation 4 or 7 from [Hanks1979]_ is used for the conversion.
        If True, use Equation 7 (`BoldM` convention).
        If False, use Equation 4 (`Mw` convention).

    Returns
    -------
    BoldM | Mw
        Rupture moment magnitude in the convention specified by `bold_m`.

    References
    ----------
    .. [Hanks1979] Hanks, T. C., and H. Kanamori (1979),
           "A moment magnitude scale",
           J. Geophys. Res., 84(B5), 2348-2350,
           doi:10.1029/JB084iB05p02348.
    """

    if bold_m:
        return BoldM(2 / 3 * np.log10(moment) - EQUATION_7_COEFFICIENT)

    if not bold_m:
        return Mw(2 / 3 * np.log10(moment) - EQUATION_4_COEFFICIENT)


@typing.overload
def magnitude_to_moment(magnitude: BoldM, bold_m: typing.Literal[True]) -> float: ...
@typing.overload
def magnitude_to_moment(
    magnitude: Mw, bold_m: typing.Literal[False] = False
) -> float: ...
def magnitude_to_moment(magnitude: BoldM | Mw, bold_m: bool = True) -> float:
    """Convert magnitude to moment.

    Parameters
    ----------
    magnitude : BoldM | Mw
        The magnitude of the rupture, in the convention indicated by `bold_m`.
    bold_m : bool, optional
        Set whether Equation 4 or 7 from [Hanks1979]_ is used for the conversion.
        If True, use Equation 7 (`BoldM` convention).
        If False, use Equation 4 (`Mw` convention).

    Returns
    -------
    float
        Rupture moment in Nm.
    """

    if bold_m:
        return 10 ** ((magnitude + EQUATION_7_COEFFICIENT) * 3 / 2)
    if not bold_m:
        return 10 ** ((magnitude + EQUATION_4_COEFFICIENT) * 3 / 2)


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
        - "depth_km": The *top* depth in kilometers.
        - "rho": The density of the fault in grams per cubic centimeter.
        - "Vs": The shear wave velocity in kilometers per second.
    source_depth_km : float
        The depth of the source in kilometers.

    Returns
    -------
    float
        The calculated slip in cm.
    """

    # While this is not strictly necessary, it does act as a sanity check to
    # ensure that the bug does not reoccur in the future.
    if not np.isclose(velocity_model_df["depth_km"].iloc[0], 0.0):
        raise ValueError(
            "Velocity model does not begin at 0km depth (are you using bottom depth instead of top depth)?"
        )
    # Finds the first index i in the velocity model such that depth[i] <= source depth < depth[i + 1]
    # At a boundary therefore, it returns the bottom-most layer index instead of the top.
    idx = (
        np.searchsorted(
            velocity_model_df["depth_km"].to_numpy(), source_depth_km, side="right"
        )
        - 1
    )
    idx = max(0, idx)
    vs_km_per_s = velocity_model_df.iloc[idx]["Vs"]
    rho_g_per_cm3 = velocity_model_df.iloc[idx]["rho"]

    vs_m_per_s = vs_km_per_s * 1e3  # Convert km/s to m/s
    rho_kg_per_m3 = rho_g_per_cm3 * 1e3  # Convert g/cm³ to kg/m³
    fault_area_m2 = fault_area_km2 * 1e6  # Convert km² to m²

    slip_m = moment_newton_metre / (fault_area_m2 * rho_kg_per_m3 * vs_m_per_s**2)

    return slip_m * 1e2  # Convert m to cm
