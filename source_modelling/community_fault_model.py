"""This module provides classes and functions for the New Zealand Community Fault Model.

The New Zealand Community Fault Model (CFM) is a collection of fault data for New
Zealand. The CFM is stored in a GeoPackage file and can be accessed using the
functions in this module. The function `most_likely_nodal_plane` can be used to
find the most likely nodal plane for a given centroid and two nodal planes, by
querying nearby faults in the CFM.

Examples
--------
>>> from pathlib import Path
>>> model = get_community_fault_model()
>>> centroid = np.array([-45.1929,166.83])
>>> nodal_plane_1 = NodalPlane(strike=20, dip=35, rake=79)
>>> nodal_plane_2 = NodalPlane(strike=213, dip=56, rake=98)
>>> most_likely_plane = most_likely_nodal_plane(model, centroid, nodal_plane_1, nodal_plane_2)
>>> print(most_likely_plane)
NodalPlane(strike=20, dip=35, rake=79)
"""

import re
from dataclasses import dataclass
from enum import Enum, Flag, auto
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import NamedTuple

import fiona
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import shapely

import source_modelling
from qcore import coordinates, geo


class NodalPlane(NamedTuple):
    """Represents a nodal plane with strike, dip, and rake values."""

    strike: float
    """The strike of the nodal plane."""

    dip: float
    """The dip of the nodal plane."""

    rake: float
    """The rake of the nodal plane."""


class CompassDirection(Enum):
    """Enum for compass directions with their corresponding degrees."""

    N = 0
    """North (0 degrees)."""

    NW = 315
    """Northwest (315 degrees)."""

    W = 270
    """West (270 degrees)."""

    SW = 225
    """Southwest (225 degrees)."""

    S = 180
    """South (180 degrees)."""

    SE = 135
    """Southeast (135 degrees)."""

    E = 90
    """East (90 degrees)."""

    NE = 45
    """Northeast (45 degrees)."""


class MovementSense(Flag):
    """Flag for movement sense types."""

    DEXTRAL = auto()
    """Dextral movement."""

    REVERSE = auto()
    """Reverse movement."""

    SINISTRAL = auto()
    """Sinistral movement."""

    NORMAL = auto()
    """Normal movement."""


class NameStatus(Enum):
    """Enum for name status."""

    PUBLISHED = auto()
    """Published status."""

    INFORMAL = auto()
    """Informal status."""


class Lineage(Enum):
    """Enum for lineage status."""

    UNMODIFIED = auto()
    """Unmodified status."""

    MODIFIED = auto()
    """Modified status."""

    NEW = auto()
    """New status."""


class DipMethod(Enum):
    """Enum for dip methods."""

    ELLIS2021 = auto()
    """Ellis 2021 method."""

    OTHER = auto()
    """Other method."""

    DOWN_DIP_INTERSECTION = auto()
    """Down dip intersection method."""


class FaultStatus(Enum):
    """Enum for fault status."""

    ACTIVE_SEISOGENIC = auto()
    """Active seismogenic fault."""

    ACTIVE_NON_SEISOGENIC = auto()
    """Active non-seismogenic fault."""

    INACTIVE = auto()
    """Inactive fault."""


class NeotectonicDomain(Enum):
    """Enum for neotectonic domains with their corresponding IDs."""

    HIKURANGI_SUBDUCTION_FRONT = 10
    """Hikurangi Subduction Front (10)."""
    ALPINE_FAULT = 15
    """Alpine Fault (15)."""
    PUYSEGUR_SUBDUCTION_FRONT = 26
    """Puysegur Subduction Front (26)."""

    NORTH_ISLAND_DEXTRAL_FAULT_BELT = 5
    """North Island Dextral Fault Belt (5)."""
    MARLBOROUGH_FAULT_SYSTEM = 14
    """Marlborough Fault System (14)."""
    PUYSEGUR_RIDGE_BANK = 25
    """Puysegur Ridge Bank (25)."""

    NORTH_WAIKATO_SOUTH_AUCKLAND = 2
    """North Waikato South Auckland (2)."""
    WESTERN_NORTH_ISLAND = 3
    """Western North Island (3)."""
    HAVRE_TROUGH_TAUPO_RIFT = 4
    """Havre Trough Taupo Rift (4)."""
    HIKURANGI_OUTER_RISE = 9
    """Hikurangi Outer Rise (9)."""
    NORTH_MERNOO_FRAC_ZONE = 16
    """North Mernoo Fracture Zone (16)."""
    PUYSEGUR_CASWELL_HIGH_OUTER_RISE = 27
    """Puysegur Caswell High Outer Rise (27)."""

    HIKURANGI_ACC_MARGIN = 7
    """Hikurangi Accretionary Margin (7)."""
    HIKURANGI_ACC_MARGIN_EAST_ZONE = 8
    """Hikurangi Accretionary Margin East Zone (8)."""
    KAPITI_MANAWATU = 12
    """Kapiti Manawatu (12)."""
    NW_SOUTH_ISLAND = 13
    """Northwest South Island (13)."""
    NE_CANTERBURY = 17
    """Northeast Canterbury (17)."""
    CENTRAL_CANTERBURY = 18
    """Central Canterbury (18)."""
    SOUTHERN_ALPS = 19
    """Southern Alps (19)."""
    OTAGO = 20
    """Otago (20)."""
    SOUTHERN_SOUTH_ISLAND = 21
    """Southern South Island (21)."""
    W_FIORDLAND_MARGIN_CASWELL_HIGH = 24
    """West Fiordland Margin Caswell High (24)."""

    EAST_CAPE_BLOCK = 6
    """East Cape Block (6)."""
    FIORDLAND_BLOCK = 23
    """Fiordland Block (23)."""

    NW_ZEALANDIA_PLATFORM = 1
    """Northwest Zealandia Platform (1)."""
    HIKURANGI_PLATEAU = 11
    """Hikurangi Plateau (11)."""
    SE_ZEALANDIA_PLATFORM = 22
    """Southeast Zealandia Platform (22)."""
    TASMAN_SEA_BASIN = 28
    """Tasman Sea Basin (28)."""


class Range(NamedTuple):
    """Represents a range with minimum, preferred, and maximum values."""

    min: float
    """The minimum value of the range."""
    pref: float
    """The preferred value of the range."""
    max: float
    """The maximum value of the range."""


@dataclass
class CommunityFault:
    """Represents a community fault with various attributes."""

    trace: shapely.LineString
    """The trace of the fault."""
    fault_status: FaultStatus
    """The status of the fault."""
    name: str
    """The name of the fault."""
    domain: NeotectonicDomain
    """The neotectonic domain of the fault."""
    dip_range: Range
    """The dip range of the fault."""
    dip_dir: CompassDirection | None
    """The dip direction of the fault."""
    sense: MovementSense
    """The primary movement sense of the fault."""
    secondary_sense: MovementSense | None
    """The secondary movement sense of the fault."""
    rake_range: Range
    """The rake range of the fault."""
    slip_rate: Range
    """The slip rate of the fault."""
    slip_rate_timeframe_pref: int
    """The preferred slip rate timeframe."""
    slip_rate_timeframe_unit: int | None
    """The unit of the slip rate timeframe."""
    down_dip_depth90: float
    """Not sure!"""
    down_dip_depth_dfc: float
    """Not sure!"""
    down_dip_depth90_method: DipMethod
    """Not sure!"""
    down_dip_dfc_method: DipMethod
    """Not sure!"""
    up_dip_depth: Range
    """The up dip depth range."""
    quality_code: int
    """The quality code of the fault."""
    reference: str
    """The reference for the fault data."""
    comments: str | None
    """Additional comments about the fault."""


def load_community_fault_model(
    community_fault_model_shp_ffp: Path | Traversable,
) -> list[CommunityFault]:
    """Load the community fault model from a shapefile.

    Parameters
    ----------
    community_fault_model_shp_ffp : Path | Traversable
        The file path to the shapefile.

    Returns
    -------
    list[CommunityFault]
        A list of CommunityFault objects.
    """
    faults = []
    with fiona.open(community_fault_model_shp_ffp) as fault_model_reader:
        fault_status_map = {
            "A-LS": FaultStatus.ACTIVE_SEISOGENIC,
            "A-US": FaultStatus.ACTIVE_NON_SEISOGENIC,
            "N-PS": FaultStatus.INACTIVE,
        }

        for feature in fault_model_reader:
            faults.append(
                CommunityFault(
                    trace=feature_trace(feature),
                    fault_status=fault_status_map[feature["properties"]["Fault_stat"]],
                    name=feature["properties"]["Name"],
                    domain=NeotectonicDomain(feature["properties"]["Domain_No"]),
                    dip_range=Range(
                        feature["properties"]["Dip_min"],
                        feature["properties"]["Dip_pref"],
                        feature["properties"]["Dip_max"],
                    ),
                    dip_dir=CompassDirection[feature["properties"]["Dip_dir"]]
                    if feature["properties"]["Dip_dir"].lower()
                    != "subvertical and variable"
                    else None,
                    sense=fault_sense_parse(feature["properties"]["Dom_sense"]),
                    secondary_sense=fault_sense_parse(
                        feature["properties"]["Sub_sense"]
                    )
                    if feature["properties"]["Sub_sense"]
                    else None,
                    rake_range=Range(
                        feature["properties"]["Rake_minus"],
                        feature["properties"]["Rake_pref"],
                        feature["properties"]["Rake_plus"],
                    ),
                    slip_rate=Range(
                        feature["properties"]["SR_min"],
                        feature["properties"]["SR_pref"],
                        feature["properties"]["SR_max"],
                    ),
                    slip_rate_timeframe_pref=feature["properties"]["SRT_pref"],
                    slip_rate_timeframe_unit=parse_timeframe_unit(
                        feature["properties"]["SRT_gen"]
                    ),
                    up_dip_depth=Range(
                        feature["properties"]["UpdDth_min"],
                        feature["properties"]["UpdDth_prf"],
                        feature["properties"]["UpdDth_max"],
                    ),
                    down_dip_depth90=feature["properties"]["Depth_D90"],
                    down_dip_depth90_method=DipMethod(
                        feature["properties"]["Method_D90"]
                    ),
                    down_dip_depth_dfc=feature["properties"]["Depth_Dfc"],
                    down_dip_dfc_method=DipMethod(feature["properties"]["Method_Dfc"]),
                    quality_code=feature["properties"]["QualCode"],
                    reference=feature["properties"]["References"],
                    comments=feature["properties"]["Comments"],
                )
            )
    return faults


def get_community_fault_model() -> list[CommunityFault]:
    """Get the community fault model.

    Returns
    -------
    list[CommunityFault]
        A list of CommunityFault objects.
    """
    return load_community_fault_model(
        resources.files(source_modelling) / "NZ_CFM" / "NZ_CFM_v1_0.shp"
    )


def community_fault_model_as_geodataframe() -> gpd.GeoDataFrame:
    """
    Convert the community fault model to a GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the community fault model data, with the
        coordinate reference system set to EPSG:2193 and the geometry column
        set to 'trace'. The index is set to the fault names.
    """
    model = get_community_fault_model()

    # Transform the trace to WGS84
    for fault in model:
        fault.trace = shapely.transform(fault.trace, lambda coord: coord[:, ::-1])
    return gpd.GeoDataFrame(
        [vars(fault) for fault in model], geometry="trace", crs="EPSG:2193"
    ).set_index("name")


def parse_timeframe_unit(unit: str) -> int | None:
    """Parse the timeframe unit from a string.

    Parameters
    ----------
    unit : str
        The unit string.

    Returns
    -------
    int | None
        The parsed unit as an integer, or None if not applicable.
    """
    if not unit:
        return None
    if "Ma" in unit:
        return int(1e6)

    match = re.match(r"(([\d,]+)s? )?yrs", unit)
    if match:
        return int(match.group(2).replace(",", "")) if match.group(2) else 1

    raise ValueError(f"Unknown time unit {unit}")


def fault_sense_parse(sense: str) -> MovementSense:
    """Parse the fault sense from a string.

    Parameters
    ----------
    sense : str
        The sense string.

    Returns
    -------
    MovementSense
        The parsed MovementSense.
    """
    parsed_sense = MovementSense(0)
    if "dextral" in sense:
        parsed_sense |= MovementSense.DEXTRAL
    if "reverse" in sense:
        parsed_sense |= MovementSense.REVERSE
    if "sinistral" in sense:
        parsed_sense |= MovementSense.SINISTRAL
    if "normal" in sense:
        parsed_sense |= MovementSense.NORMAL
    return parsed_sense


def most_likely_nodal_plane(
    faults: list[CommunityFault],
    centroid: npt.NDArray,
    nodal_plane_1: NodalPlane,
    nodal_plane_2: NodalPlane,
    k_neighbours: int = 5,
) -> NodalPlane:
    """Find the most likely nodal plane by a nearest neighbour vote on
    the strike values of the faults near the centroid.

    Parameters
    ----------
    faults : list[CommunityFault]
        A list of CommunityFault objects.
    centroid : npt.NDArray
        The centroid of the fault.
    nodal_plane_1 : NodalPlane
        The first nodal plane.
    nodal_plane_2 : NodalPlane
        The second nodal plane.
    k_neighbours : int, optional
        The number of nearest neighbours to consider (default is 5).

    Returns
    -------
    NodalPlane
        The most likely nodal plane.
    """
    point = shapely.Point(coordinates.wgs_depth_to_nztm(centroid))
    line_segments = {
        shapely.LineString(fault.trace.coords[i : i + 2]): line_segment_strike(
            *fault.trace.coords[i : i + 2]
        )
        for fault in faults
        for i in range(len(fault.trace.coords) - 1)
    }
    closest_segments = sorted(
        line_segments, key=lambda segment: segment.distance(point)
    )[:k_neighbours]
    nodal_plane_1_votes = sum(
        1
        / (
            segment.distance(point)
            * abs(line_segments[segment] - nodal_plane_1.strike + 1e-5)
        )
        for segment in closest_segments
        if abs(line_segments[segment] - nodal_plane_1.strike)
        < abs(line_segments[segment] - nodal_plane_2.strike)
    )
    nodal_plane_2_votes = sum(
        1
        / (
            segment.distance(point)
            * abs(line_segments[segment] - nodal_plane_1.strike + 1e-5)
        )
        for segment in closest_segments
        if abs(line_segments[segment] - nodal_plane_1.strike)
        >= abs(line_segments[segment] - nodal_plane_2.strike)
    )

    if nodal_plane_1_votes >= nodal_plane_2_votes:
        return nodal_plane_1

    return nodal_plane_2


def line_segment_strike(point_a: npt.ArrayLike, point_b: npt.ArrayLike) -> float:
    """Calculate the strike of a line segment defined by two points.

    Parameters
    ----------
    point_a : npt.ArrayLike
        The first point of the line segment.
    point_b : npt.ArrayLike
        The second point of the line segment.

    Returns
    -------
    float
        The strike of the line segment.
    """
    point_a = np.asarray(point_a)
    point_b = np.asarray(point_b)

    return float(
        coordinates.nztm_bearing_to_great_circle_bearing(
            coordinates.nztm_to_wgs_depth(point_a),
            np.linalg.norm(point_b - point_a) / 1000,
            geo.oriented_bearing_wrt_normal(
                np.array([1, 0, 0]),
                np.append(
                    point_b - point_a,
                    0,
                ),
                np.array([0, 0, 1]),
            ),
        )
    )


def feature_trace(feature: fiona.Feature) -> shapely.LineString:
    """Extract the trace of a fault feature as a LineString.

    Parameters
    ----------
    feature : fiona.Feature
        The feature from which to extract the trace.

    Returns
    -------
    shapely.LineString
        The extracted trace as a LineString.
    """
    points = np.array(feature.geometry.coordinates)[:, ::-1]
    strike = line_segment_strike(points[0], points[1])
    try:
        compass_direction = CompassDirection[feature.properties["Dip_dir"]]
        if strike > compass_direction.value:
            points = points[::-1]
    except KeyError:
        pass
    return shapely.LineString(points)
