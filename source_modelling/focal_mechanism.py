"""Nodal plane selection and tectonic type classification for CMT solutions.

A centroid moment tensor (CMT) solution provides two orthogonal nodal
planes. The moment tensor alone cannot distinguish which one is the fault
plane; the choice has to come from external constraints. This module
combines the physical constraints commonly used in New Zealand practice:

1. Consistency with mapped faults in the New Zealand Community Fault Model
   (CFM): agreement in strike, dip direction, dip and rake, and the
   "hanging-wall" test in which the plane projected up-dip from the
   centroid should reach the surface at the mapped fault trace.
2. Consistency with the subduction interface geometry (Slab2, Hayes et
   al. 2018) for the Hikurangi-Kermadec and Puysegur margins: interface
   events rupture the low-angle plane parallel to the slab surface.
3. Andersonian faulting mechanics: optimally oriented normal faults dip
   about 60 degrees, reverse faults about 30 degrees and strike-slip
   faults are near vertical.

These constraints are turned into a feature vector for each nodal plane
and a small random forest (trained on human-picked GeoNet solutions with
``scripts/train_cmt_classifiers.py``) decides which plane is preferred. The
forest operates on the *difference* between the two planes' features, and
is evaluated on both orderings, so swapping the planes exactly flips the
answer.

The tectonic type (crustal, subduction interface or intraslab) is
classified by the NZ NSHM 2022 rule (Rollins et al. 2022): events outside
the slab footprint are crustal; events with an interface-like mechanism
within 10 km of the interface are interface; otherwise events above the
interface are crustal and events below it are intraslab (with normal
faulting inside the tolerance band also taken as intraslab). A second
forest, trained on this rule applied to the GeoNet catalogue under Monte
Carlo depth perturbation, provides class probabilities that reflect the
depth uncertainty.

Examples
--------
>>> from source_modelling import focal_mechanism
>>> from source_modelling.community_fault_model import NodalPlane
>>> classifier = focal_mechanism.CMTClassifier.load()
>>> centroid = np.array([-45.1929, 166.83, 22.0])  # lat, lon, depth (km)
>>> nodal_plane_1 = NodalPlane(strike=20, dip=35, rake=79)
>>> nodal_plane_2 = NodalPlane(strike=213, dip=56, rake=98)
>>> classifier.most_likely_nodal_plane(centroid, nodal_plane_1, nodal_plane_2)
NodalPlane(strike=20, dip=35, rake=79)
>>> classifier.tectonic_type(centroid, nodal_plane_1, nodal_plane_2)
<TectonicType.INTERFACE: 'interface'>
"""

from __future__ import annotations

import functools
import json
import warnings
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

import source_modelling
from qcore import coordinates, geo
from source_modelling.community_fault_model import (
    CommunityFault,
    NodalPlane,
    get_community_fault_model,
)

DEFAULT_CENTROID_DEPTH_KM = 10.0
"""Depth assumed when a centroid is given without a depth."""

INTERFACE_DEPTH_TOLERANCE_KM = 10.0
"""Vertical distance from the interface within which events may be interface events (Rollins et al. 2022)."""

INTERFACE_RAKE_TOLERANCE = 60.0
"""Maximum deviation of rake from pure reverse (90 degrees) for an interface-like plane."""

INTERFACE_MAX_DIP = 75.0
"""Maximum dip of an interface-like plane."""

INTERFACE_MAX_SLAB_ANGLE = 35.0
"""Maximum angle between a plane and the local slab surface for an interface-like plane."""

STRIKE_SLIP_RAKE_TOLERANCE = 30.0
"""A double couple is strike-slip if both planes have rakes within this angle of 0 or 180."""

FAULT_NEIGHBOURS = 8
"""Number of nearest CFM fault segments used for the strike, dip and rake misfit features."""

STRIKE_MATCH_TOLERANCE = 30.0
"""Maximum strike difference for a fault segment to count as a match in the projected trace test."""

MAX_DISTANCE_KM = 100.0
"""Distances larger than this are clipped before being used as features."""

_DATA_DIR = resources.files(source_modelling) / "NZ_CFM"


class TectonicType(Enum):
    """Tectonic type of an earthquake."""

    CRUSTAL = "crustal"
    """Shallow crustal (upper plate) earthquake."""

    INTERFACE = "interface"
    """Subduction interface earthquake."""

    SLAB = "slab"
    """Intraslab (including outer-rise) earthquake."""


TECTONIC_TYPES = [TectonicType.CRUSTAL, TectonicType.INTERFACE, TectonicType.SLAB]
"""Class order used by the tectonic type model."""


def angular_difference(
    a: npt.ArrayLike, b: npt.ArrayLike, period: float = 360.0
) -> npt.NDArray[np.float64]:
    """Smallest absolute difference between two angles.

    Parameters
    ----------
    a : npt.ArrayLike
        First angle(s) in degrees.
    b : npt.ArrayLike
        Second angle(s) in degrees.
    period : float, optional
        Period of the angle: 360 for oriented angles such as strike with a
        known dip direction, 180 for unoriented angles.

    Returns
    -------
    npt.NDArray[np.float64]
        Absolute angular difference in the range [0, period / 2].
    """
    difference = np.mod(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), period)
    return np.minimum(difference, period - difference)


def plane_normal(strike: float, dip: float) -> npt.NDArray[np.float64]:
    """Upward-pointing unit normal to a plane in (north, east, up) coordinates.

    Parameters
    ----------
    strike : float
        Strike in degrees (Aki-Richards convention: dip direction is
        strike + 90).
    dip : float
        Dip in degrees.

    Returns
    -------
    npt.NDArray[np.float64]
        Unit normal vector.
    """
    dip_direction = np.radians(strike + 90.0)
    dip_rad = np.radians(dip)
    return np.array(
        [
            -np.sin(dip_rad) * np.cos(dip_direction),
            -np.sin(dip_rad) * np.sin(dip_direction),
            np.cos(dip_rad),
        ]
    )


def angle_between_planes(
    strike_1: float, dip_1: float, strike_2: float, dip_2: float
) -> float:
    """Angle between two planes (angle between their normals).

    Parameters
    ----------
    strike_1 : float
        Strike of the first plane in degrees.
    dip_1 : float
        Dip of the first plane in degrees.
    strike_2 : float
        Strike of the second plane in degrees.
    dip_2 : float
        Dip of the second plane in degrees.

    Returns
    -------
    float
        Angle between the planes in degrees, in [0, 90].
    """
    cosine = abs(
        float(np.dot(plane_normal(strike_1, dip_1), plane_normal(strike_2, dip_2)))
    )
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def andersonian_dip_misfit(dip: float, rake: float) -> float:
    """Deviation of a plane's dip from the Andersonian optimum for its slip sense.

    Optimally oriented faults with a friction coefficient of about 0.6 dip
    at roughly 60 degrees (normal), 30 degrees (reverse) or 90 degrees
    (strike-slip).

    Parameters
    ----------
    dip : float
        Dip in degrees.
    rake : float
        Rake in degrees (Aki-Richards convention).

    Returns
    -------
    float
        Absolute dip misfit in degrees.
    """
    wrapped_rake = ((rake + 180.0) % 360.0) - 180.0
    if abs(wrapped_rake) <= 30.0 or abs(wrapped_rake) >= 150.0:
        optimal_dip = 90.0
    elif wrapped_rake > 0:
        optimal_dip = 30.0
    else:
        optimal_dip = 60.0
    return abs(dip - optimal_dip)


def _normal_and_slip(
    plane: NodalPlane,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fault normal and slip vector in (north, east, down) coordinates.

    Aki & Richards, Quantitative Seismology, eqs 4.87-4.88.
    """
    strike, dip, rake = (np.radians(x) for x in plane)
    normal = np.array(
        [-np.sin(dip) * np.sin(strike), np.sin(dip) * np.cos(strike), -np.cos(dip)]
    )
    slip = np.array(
        [
            np.cos(rake) * np.cos(strike) + np.cos(dip) * np.sin(rake) * np.sin(strike),
            np.cos(rake) * np.sin(strike) - np.cos(dip) * np.sin(rake) * np.cos(strike),
            -np.sin(rake) * np.sin(dip),
        ]
    )
    return normal, slip


def principal_axis_plunges(plane: NodalPlane) -> tuple[float, float]:
    """Plunges of the T (tension) and P (pressure) axes of a double couple.

    The axes are the same for a nodal plane and its auxiliary plane, so the
    result does not depend on which plane is given.

    Parameters
    ----------
    plane : NodalPlane
        Either nodal plane of the double couple.

    Returns
    -------
    tuple[float, float]
        Plunge of the T axis and of the P axis in degrees, in [0, 90].
    """
    normal, slip = _normal_and_slip(plane)
    t_axis = (normal + slip) / np.sqrt(2.0)
    p_axis = (normal - slip) / np.sqrt(2.0)
    return (
        float(np.degrees(np.arcsin(np.clip(abs(t_axis[2]), 0.0, 1.0)))),
        float(np.degrees(np.arcsin(np.clip(abs(p_axis[2]), 0.0, 1.0)))),
    )


def auxiliary_plane(plane: NodalPlane) -> NodalPlane:
    """Compute the auxiliary (conjugate) nodal plane of a double couple.

    The auxiliary plane has the slip vector of the given plane as its
    normal and the normal of the given plane as its slip vector.

    Parameters
    ----------
    plane : NodalPlane
        Strike, dip and rake in degrees (Aki-Richards convention).

    Returns
    -------
    NodalPlane
        The auxiliary plane with strike in [0, 360), dip in [0, 90] and
        rake in (-180, 180].
    """
    normal, slip = _normal_and_slip(plane)
    aux_normal, aux_slip = slip, normal
    if aux_normal[2] > 0:  # normal must point upwards (negative "down")
        aux_normal, aux_slip = -aux_normal, -aux_slip
    aux_dip = np.arccos(np.clip(-aux_normal[2], -1.0, 1.0))
    aux_strike = np.arctan2(-aux_normal[0], aux_normal[1])
    sin_dip = np.sin(aux_dip)
    if sin_dip < 1e-9:
        # Horizontal plane: strike is undefined, take it from the slip vector.
        aux_strike = np.arctan2(aux_slip[1], aux_slip[0])
        aux_rake = 0.0
    else:
        aux_rake = np.arctan2(
            -aux_slip[2] / sin_dip,
            aux_slip[0] * np.cos(aux_strike) + aux_slip[1] * np.sin(aux_strike),
        )
    return NodalPlane(
        float(np.degrees(aux_strike) % 360.0),
        float(np.degrees(aux_dip)),
        float(np.degrees(aux_rake)),
    )


class SyntheticSolution(NamedTuple):
    """A synthetic CMT solution generated from a CFM fault."""

    fault_name: str
    """Name of the CFM fault the event was generated on."""

    centroid: npt.NDArray[np.float64]
    """Centroid as (lat, lon, depth_km)."""

    fault_plane: NodalPlane
    """The true (noisy) fault plane."""

    auxiliary_plane: NodalPlane
    """The auxiliary plane."""


def synthetic_solutions(
    faults: list[CommunityFault],
    n_solutions: int,
    rng: np.random.Generator,
    angle_noise: float = 10.0,
    location_noise_km: float = 3.0,
    depth_noise_km: float = 3.0,
) -> list[SyntheticSolution]:
    """Generate synthetic CMT solutions with a known fault plane from the CFM.

    A random point on a random fault trace is chosen, a depth is drawn
    within the fault's seismogenic range and the centroid is placed
    down-dip of the trace at that depth. The fault plane is the CFM strike
    (at that point), preferred dip and preferred rake, each perturbed by
    Gaussian noise; the centroid is also perturbed to mimic CMT location
    uncertainty. Only faults with a recorded dip direction and a dip below
    85 degrees are used, since for vertical faults the position of the
    centroid relative to the trace carries no information; faults dipping
    less than 15 degrees are also excluded.

    Parameters
    ----------
    faults : list[CommunityFault]
        Faults of the community fault model.
    n_solutions : int
        Number of solutions to generate.
    rng : np.random.Generator
        Random number generator.
    angle_noise : float, optional
        Standard deviation of the strike, dip and rake noise in degrees.
    location_noise_km : float, optional
        Standard deviation of the horizontal centroid noise in km.
    depth_noise_km : float, optional
        Standard deviation of the centroid depth noise in km.

    Returns
    -------
    list[SyntheticSolution]
        Synthetic solutions.
    """
    candidates = [
        fault
        for fault in faults
        if fault.dip_dir is not None
        and fault.dip_range.pref is not None
        and 15 <= fault.dip_range.pref < 85
        and fault.rake_range.pref is not None
    ]
    segments = FaultSegmentIndex(candidates)
    lengths = np.linalg.norm(segments.end - segments.start, axis=1)
    fault_index = np.repeat(
        np.arange(len(candidates)),
        [len(fault.trace.coords) - 1 for fault in candidates],
    )
    solutions = []
    for segment in rng.choice(
        len(lengths), size=n_solutions, p=lengths / lengths.sum()
    ):
        fault = candidates[fault_index[segment]]
        along = rng.uniform()
        trace_point = segments.start[segment] + along * (
            segments.end[segment] - segments.start[segment]
        )
        strike = segments.strike[segment]
        dip = float(fault.dip_range.pref)
        rake = float(fault.rake_range.pref)
        min_depth = max(float(fault.up_dip_depth.pref or 0.0), 2.0)
        max_depth = max(float(fault.down_dip_depth90 or 15.0), min_depth + 1.0)
        depth = rng.uniform(min_depth, max_depth)

        # Move down-dip from the trace to the centroid at this depth.
        lat, lon = coordinates.nztm_to_wgs_depth(trace_point)[:2]
        horizontal = depth / np.tan(np.radians(dip))
        lat, lon = geo.ll_shift(lat, lon, horizontal, (strike + 90.0) % 360.0)
        # Perturb the location, depth and angles.
        lat, lon = geo.ll_shift(
            lat, lon, abs(rng.normal(0.0, location_noise_km)), rng.uniform(0.0, 360.0)
        )
        depth = float(np.clip(depth + rng.normal(0.0, depth_noise_km), 1.0, 60.0))
        fault_plane = NodalPlane(
            float((strike + rng.normal(0.0, angle_noise)) % 360.0),
            float(np.clip(dip + rng.normal(0.0, angle_noise), 5.0, 89.0)),
            float(((rake + rng.normal(0.0, angle_noise) + 180.0) % 360.0) - 180.0),
        )
        solutions.append(
            SyntheticSolution(
                fault.name,
                np.array([lat, lon, depth]),
                fault_plane,
                auxiliary_plane(fault_plane),
            )
        )
    return solutions


def faulting_style(
    nodal_plane_1: NodalPlane, nodal_plane_2: NodalPlane
) -> tuple[bool, bool]:
    """Classify a double couple as reverse-like, normal-like or strike-slip.

    Both nodal planes of a double couple share the sign of their rake, so
    the reverse/normal sense is unambiguous. The dip-slip/strike-slip
    distinction is taken from the plane with the larger dip-slip component,
    which makes the result independent of the plane ordering (a pair with one
    strike-slip plane and one dip-slip plane is treated as dip-slip).

    Parameters
    ----------
    nodal_plane_1 : NodalPlane
        First nodal plane.
    nodal_plane_2 : NodalPlane
        Second nodal plane.

    Returns
    -------
    tuple[bool, bool]
        ``(reverse_like, normal_like)``; both False for strike-slip.
    """
    rakes = np.array([nodal_plane_1.rake, nodal_plane_2.rake], dtype=float)
    dip_slip = np.abs(np.sin(np.radians(rakes)))
    dominant_rake = rakes[int(np.argmax(dip_slip))]
    if dip_slip.max() < np.sin(np.radians(STRIKE_SLIP_RAKE_TOLERANCE)):
        return False, False
    wrapped = ((dominant_rake + 180.0) % 360.0) - 180.0
    return bool(wrapped > 0), bool(wrapped < 0)


def _great_circle_bearing(
    lat_1: npt.NDArray, lon_1: npt.NDArray, lat_2: npt.NDArray, lon_2: npt.NDArray
) -> npt.NDArray[np.float64]:
    """Initial great-circle bearing from point 1 to point 2 (degrees, vectorised)."""
    lat_1, lon_1, lat_2, lon_2 = (np.radians(x) for x in (lat_1, lon_1, lat_2, lon_2))
    delta_lon = lon_2 - lon_1
    y = np.sin(delta_lon) * np.cos(lat_2)
    x = np.cos(lat_1) * np.sin(lat_2) - np.sin(lat_1) * np.cos(lat_2) * np.cos(
        delta_lon
    )
    return np.degrees(np.arctan2(y, x)) % 360.0


class SlabQuery(NamedTuple):
    """Slab surface properties at a location."""

    depth: float
    """Depth of the slab surface in km (positive down), NaN outside the footprint."""

    dip: float
    """Dip of the slab surface in degrees, NaN outside the footprint."""

    strike: float
    """Strike of the slab surface in degrees, NaN outside the footprint."""

    @property
    def present(self) -> bool:
        """Whether the location is inside a slab footprint."""
        return bool(np.isfinite(self.depth))


class SlabModel:
    """Slab2 subduction interface geometry cropped to New Zealand.

    Contains the Kermadec-Hikurangi ("ker") and Puysegur ("puy") regions of
    Slab2 (Hayes et al. 2018) on their native 0.05 and 0.02 degree grids.
    """

    def __init__(self, grids: dict[str, npt.NDArray]):
        """Create a slab model from named grids.

        Parameters
        ----------
        grids : dict[str, npt.NDArray]
            Mapping with keys ``<region>_lat``, ``<region>_lon``,
            ``<region>_depth``, ``<region>_dip`` and ``<region>_strike``
            for each region.
        """
        self.regions = sorted({key.split("_")[0] for key in grids})
        self._grids = grids

    @classmethod
    @functools.cache
    def load(cls, path: Path | Traversable | None = None) -> SlabModel:
        """Load the packaged Slab2 grids.

        Parameters
        ----------
        path : Path | Traversable | None, optional
            Path to a ``.npz`` file, defaults to the packaged file.

        Returns
        -------
        SlabModel
            The loaded slab model.
        """
        path = path or _DATA_DIR / "slab2_nz.npz"
        with path.open("rb") as handle, np.load(handle) as data:
            grids = {key: np.array(data[key], dtype=float) for key in data}
        return cls(grids)

    def _interpolate(self, region: str, name: str, lat: float, lon: float) -> float:
        lats = self._grids[f"{region}_lat"]
        lons = self._grids[f"{region}_lon"]
        values = self._grids[f"{region}_{name}"]
        if not (lats[0] <= lat <= lats[-1] and lons[0] <= lon <= lons[-1]):
            return float("nan")
        i = int(np.clip(np.searchsorted(lats, lat) - 1, 0, len(lats) - 2))
        j = int(np.clip(np.searchsorted(lons, lon) - 1, 0, len(lons) - 2))
        t = (lat - lats[i]) / (lats[i + 1] - lats[i])
        u = (lon - lons[j]) / (lons[j + 1] - lons[j])
        corners = values[i : i + 2, j : j + 2]
        if name == "strike":
            # Interpolate strike as a unit vector to avoid wraparound.
            angles = np.radians(corners)
            weights = np.array([[(1 - t) * (1 - u), (1 - t) * u], [t * (1 - u), t * u]])
            if np.any(~np.isfinite(corners)):
                return float("nan")
            return float(
                np.degrees(
                    np.arctan2(
                        np.sum(weights * np.sin(angles)),
                        np.sum(weights * np.cos(angles)),
                    )
                )
                % 360.0
            )
        if np.any(~np.isfinite(corners)):
            return float("nan")
        return float(
            (1 - t) * (1 - u) * corners[0, 0]
            + (1 - t) * u * corners[0, 1]
            + t * (1 - u) * corners[1, 0]
            + t * u * corners[1, 1]
        )

    def query(self, lat: float, lon: float) -> SlabQuery:
        """Slab surface depth, dip and strike beneath a location.

        Parameters
        ----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees (either -180..180 or 0..360).

        Returns
        -------
        SlabQuery
            Slab properties, NaN-valued outside the slab footprints.
        """
        for region in self.regions:
            lons = self._grids[f"{region}_lon"]
            query_lon = (
                lon % 360.0 if lons[-1] > 180.0 else ((lon + 180.0) % 360.0) - 180.0
            )
            depth = self._interpolate(region, "depth", lat, query_lon)
            if np.isfinite(depth):
                return SlabQuery(
                    depth,
                    self._interpolate(region, "dip", lat, query_lon),
                    self._interpolate(region, "strike", lat, query_lon),
                )
        return SlabQuery(float("nan"), float("nan"), float("nan"))


class FaultSegmentIndex:
    """Straight-line segments of the CFM fault traces with their attributes.

    Segments are oriented so that the dip direction is strike + 90 (the
    Aki-Richards convention) wherever the CFM records a dip direction.
    """

    def __init__(self, faults: list[CommunityFault]):
        """Build the index from a list of community faults.

        Parameters
        ----------
        faults : list[CommunityFault]
            Faults from the community fault model.
        """
        starts, ends, dip_known, dips, rakes, dip_directions = [], [], [], [], [], []
        for fault in faults:
            coords = np.asarray(fault.trace.coords)[:, :2]
            for i in range(len(coords) - 1):
                starts.append(coords[i])
                ends.append(coords[i + 1])
                dips.append(fault.dip_range.pref)
                rakes.append(fault.rake_range.pref)
                dip_known.append(fault.dip_dir is not None)
                dip_directions.append(
                    fault.dip_dir.value if fault.dip_dir is not None else np.nan
                )
        self.start = np.asarray(starts, dtype=float)
        """Segment start points in NZTM (northing, easting)."""
        self.end = np.asarray(ends, dtype=float)
        """Segment end points in NZTM (northing, easting)."""
        self.dip = np.asarray(dips, dtype=float)
        """Preferred dip of the parent fault (degrees)."""
        self.rake = np.asarray(rakes, dtype=float)
        """Preferred rake of the parent fault (degrees)."""
        self.dip_known = np.asarray(dip_known, dtype=bool)
        """Whether the parent fault has a recorded dip direction."""

        start_wgs = coordinates.nztm_to_wgs_depth(self.start)
        end_wgs = coordinates.nztm_to_wgs_depth(self.end)
        strike = _great_circle_bearing(
            start_wgs[:, 0], start_wgs[:, 1], end_wgs[:, 0], end_wgs[:, 1]
        )
        # Flip segments whose dip direction (strike + 90) disagrees with the CFM.
        dip_directions_arr = np.asarray(dip_directions, dtype=float)
        flip = self.dip_known & (
            angular_difference(strike + 90.0, np.nan_to_num(dip_directions_arr)) > 90.0
        )
        strike[flip] = (strike[flip] + 180.0) % 360.0
        self.start[flip], self.end[flip] = (
            self.end[flip].copy(),
            self.start[flip].copy(),
        )
        self.strike = strike
        """Oriented strike of each segment (degrees)."""

    def distances(self, point: npt.NDArray) -> npt.NDArray[np.float64]:
        """Distance in km from a point to every segment.

        Parameters
        ----------
        point : npt.NDArray
            Point in NZTM (northing, easting).

        Returns
        -------
        npt.NDArray[np.float64]
            Distance to each segment in km.
        """
        direction = self.end - self.start
        offset = point[:2] - self.start
        length_sq = np.maximum(np.einsum("ij,ij->i", direction, direction), 1e-9)
        t = np.clip(np.einsum("ij,ij->i", offset, direction) / length_sq, 0.0, 1.0)
        projection = self.start + t[:, None] * direction
        return np.linalg.norm(point[:2] - projection, axis=1) / 1000.0


PLANE_FEATURE_NAMES = [
    "cfm_strike_unoriented",
    "cfm_strike_oriented",
    "cfm_dip_misfit",
    "cfm_rake_misfit",
    "log_projected_distance_unoriented",
    "log_projected_distance_oriented",
    "andersonian_dip_misfit",
    "slab_angle",
    "slab_rake_misfit",
    "dip",
]
"""Names of the features computed for each nodal plane."""

EVENT_FEATURE_NAMES = [
    "depth",
    "depth_below_slab",
    "slab_present",
    "slab_dip",
    "log_nearest_fault_distance",
    "magnitude",
    "reverse_like",
    "normal_like",
]
"""Names of the features shared by both nodal planes of an event."""

NODAL_PLANE_FEATURE_NAMES = [f"delta_{name}" for name in PLANE_FEATURE_NAMES] + [
    *EVENT_FEATURE_NAMES
]
"""Feature names, in order, used by the nodal plane model."""

FEATURE_DESCRIPTIONS = {
    "cfm_strike_unoriented": "Strike difference from nearby mapped faults, ignoring which way they dip",
    "cfm_strike_oriented": "Strike difference from nearby mapped faults, including their dip direction",
    "cfm_dip_misfit": "Dip difference from nearby mapped faults",
    "cfm_rake_misfit": "Rake (slip sense) difference from nearby mapped faults",
    "log_projected_distance_unoriented": "Distance from the plane's up-dip projection to a similarly striking mapped trace",
    "log_projected_distance_oriented": "Distance from the plane's up-dip projection to a mapped trace that also dips the same way",
    "andersonian_dip_misfit": "Dip difference from the Andersonian optimum for this slip sense",
    "slab_angle": "Angle between the plane and the local subduction interface",
    "slab_rake_misfit": "Rake difference from pure thrust, as expected on the interface",
    "dip": "Dip of the plane",
    "depth": "Centroid depth",
    "depth_below_slab": "Centroid depth below the subduction interface",
    "slab_present": "Whether a subduction interface lies beneath the centroid",
    "slab_dip": "Dip of the local subduction interface",
    "log_nearest_fault_distance": "Distance to the nearest mapped fault trace",
    "magnitude": "Moment magnitude",
    "reverse_like": "Whether the mechanism is reverse faulting",
    "normal_like": "Whether the mechanism is normal faulting",
}
"""Human readable descriptions of the features, for interpretation."""

TECTONIC_FEATURE_NAMES = [
    "depth",
    "depth_below_slab",
    "slab_present",
    "slab_dip",
    "magnitude",
    "reverse_like",
    "normal_like",
    "min_slab_angle",
    "min_dip",
    "interface_like",
]
"""Feature names, in order, used by the tectonic type model."""


@dataclass
class EventFeatures:
    """Physical features of a CMT solution used by the classifiers."""

    plane_features: npt.NDArray[np.float64]
    """Array of shape (2, len(PLANE_FEATURE_NAMES)): per-plane features."""

    event_features: npt.NDArray[np.float64]
    """Array of shape (len(EVENT_FEATURE_NAMES),): event-level features."""

    slab: SlabQuery
    """Slab geometry beneath the centroid."""

    interface_like: bool
    """Whether either plane has an interface-like orientation and slip."""

    def nodal_plane_vector(self, first: int = 0) -> npt.NDArray[np.float64]:
        """Feature vector for the nodal plane model.

        Parameters
        ----------
        first : int, optional
            Index (0 or 1) of the plane treated as "plane 1".

        Returns
        -------
        npt.NDArray[np.float64]
            Feature vector: plane differences followed by event features.
        """
        second = 1 - first
        difference = self.plane_features[first] - self.plane_features[second]
        return np.concatenate([difference, self.event_features])

    def tectonic_vector(self) -> npt.NDArray[np.float64]:
        """Feature vector for the tectonic type model.

        Returns
        -------
        npt.NDArray[np.float64]
            Feature vector in ``TECTONIC_FEATURE_NAMES`` order.
        """
        event = dict(zip(EVENT_FEATURE_NAMES, self.event_features))
        slab_angle = self.plane_features[:, PLANE_FEATURE_NAMES.index("slab_angle")]
        dip = self.plane_features[:, PLANE_FEATURE_NAMES.index("dip")]
        return np.array(
            [
                event["depth"],
                event["depth_below_slab"],
                event["slab_present"],
                event["slab_dip"],
                event["magnitude"],
                event["reverse_like"],
                event["normal_like"],
                float(np.min(slab_angle)),
                float(np.min(dip)),
                float(self.interface_like),
            ]
        )


class FeatureContribution(NamedTuple):
    """How much one feature moved the nodal plane probability."""

    feature: str
    """Name of the feature."""

    description: str
    """Human readable description of the feature."""

    contribution: float
    """Signed change in the probability that plane 1 is the fault plane."""

    plane_1_value: float
    """Value of the feature for plane 1."""

    plane_2_value: float
    """Value of the feature for plane 2, NaN for event-level features."""


@dataclass
class NodalPlaneExplanation:
    """Why the classifier preferred one nodal plane over the other."""

    probability: float
    """Probability that plane 1 is the fault plane."""

    baseline: float
    """Probability before any feature is considered, exactly 0.5."""

    contributions: list[FeatureContribution]
    """Feature contributions, largest absolute contribution first."""


class Criterion(NamedTuple):
    """One step of the tectonic type decision."""

    name: str
    """What is being tested."""

    detail: str
    """The measured values behind the test."""

    satisfied: bool | None
    """Whether the test passed, or None if it is context rather than a test."""


@dataclass
class TectonicExplanation:
    """Why an event received its tectonic type."""

    tectonic_type: TectonicType
    """The type given by the NSHM rule."""

    probabilities: dict[TectonicType, float]
    """Class probabilities under depth uncertainty."""

    criteria: list[Criterion]
    """The rule evaluated step by step."""


def _log_distance(distance_km: float) -> float:
    return float(np.log1p(min(distance_km, MAX_DISTANCE_KM)))


def _parse_centroid(centroid: npt.ArrayLike) -> tuple[float, float, float]:
    centroid = np.asarray(centroid, dtype=float).ravel()
    if centroid.size == 2:
        warnings.warn(
            "Centroid given without depth; assuming "
            f"{DEFAULT_CENTROID_DEPTH_KM} km. Pass (lat, lon, depth_km) for "
            "a reliable classification.",
            stacklevel=3,
        )
        return float(centroid[0]), float(centroid[1]), DEFAULT_CENTROID_DEPTH_KM
    if centroid.size != 3:
        raise ValueError("Centroid must be (lat, lon) or (lat, lon, depth_km).")
    return float(centroid[0]), float(centroid[1]), float(centroid[2])


def compute_features(
    segments: FaultSegmentIndex,
    slab_model: SlabModel,
    centroid: npt.ArrayLike,
    nodal_plane_1: NodalPlane,
    nodal_plane_2: NodalPlane,
    magnitude: float = float("nan"),
) -> EventFeatures:
    """Compute the physical features of a CMT solution.

    Parameters
    ----------
    segments : FaultSegmentIndex
        Indexed CFM fault segments.
    slab_model : SlabModel
        Subduction interface geometry.
    centroid : npt.ArrayLike
        Centroid as (lat, lon, depth_km). If depth is omitted a default is
        assumed with a warning.
    nodal_plane_1 : NodalPlane
        First nodal plane.
    nodal_plane_2 : NodalPlane
        Second nodal plane.
    magnitude : float, optional
        Moment magnitude, NaN if unknown.

    Returns
    -------
    EventFeatures
        Features for both planes and the event.
    """
    lat, lon, depth = _parse_centroid(centroid)
    point = coordinates.wgs_depth_to_nztm(np.array([lat, lon]))
    distances = segments.distances(point)
    nearest = np.argsort(distances)[:FAULT_NEIGHBOURS]
    weights = 1.0 / (distances[nearest] + 2.0)
    weights /= weights.sum()
    slab = slab_model.query(lat, lon)

    plane_rows = []
    interface_like = False
    for plane in (nodal_plane_1, nodal_plane_2):
        oriented = angular_difference(segments.strike[nearest], plane.strike, 360.0)
        unoriented = angular_difference(segments.strike[nearest], plane.strike, 180.0)
        # Faults with unknown dip direction only constrain the unoriented strike.
        oriented = np.where(segments.dip_known[nearest], oriented, unoriented)

        # Hanging-wall test: project the plane up-dip to the surface.
        horizontal_offset = depth / np.tan(np.radians(max(plane.dip, 5.0)))
        up_dip_bearing = (plane.strike - 90.0) % 360.0
        surface_lat, surface_lon = geo.ll_shift(
            lat, lon, horizontal_offset, up_dip_bearing
        )
        surface_point = coordinates.wgs_depth_to_nztm(
            np.array([surface_lat, surface_lon])
        )
        surface_distances = segments.distances(surface_point)
        unoriented_match = (
            angular_difference(segments.strike, plane.strike, 180.0)
            < STRIKE_MATCH_TOLERANCE
        )
        oriented_match = np.where(
            segments.dip_known,
            angular_difference(segments.strike, plane.strike, 360.0)
            < STRIKE_MATCH_TOLERANCE,
            unoriented_match,
        )
        projected_unoriented = (
            surface_distances[unoriented_match].min()
            if unoriented_match.any()
            else MAX_DISTANCE_KM
        )
        projected_oriented = (
            surface_distances[oriented_match].min()
            if oriented_match.any()
            else MAX_DISTANCE_KM
        )

        if slab.present:
            slab_angle = angle_between_planes(
                plane.strike, plane.dip, slab.strike, slab.dip
            )
        else:
            slab_angle = 90.0
        slab_rake_misfit = float(angular_difference(plane.rake, 90.0))
        interface_like |= bool(
            slab.present
            and slab_rake_misfit <= INTERFACE_RAKE_TOLERANCE
            and plane.dip <= INTERFACE_MAX_DIP
            and slab_angle <= INTERFACE_MAX_SLAB_ANGLE
        )

        plane_rows.append(
            [
                float(np.sum(weights * unoriented)),
                float(np.sum(weights * oriented)),
                float(np.sum(weights * np.abs(segments.dip[nearest] - plane.dip))),
                float(
                    np.sum(
                        weights * angular_difference(segments.rake[nearest], plane.rake)
                    )
                ),
                _log_distance(projected_unoriented),
                _log_distance(projected_oriented),
                andersonian_dip_misfit(plane.dip, plane.rake),
                slab_angle,
                slab_rake_misfit,
                float(plane.dip),
            ]
        )

    reverse_like, normal_like = faulting_style(nodal_plane_1, nodal_plane_2)
    event_row = [
        depth,
        depth - slab.depth if slab.present else -MAX_DISTANCE_KM,
        float(slab.present),
        slab.dip if slab.present else 0.0,
        _log_distance(float(distances[nearest[0]])),
        magnitude if np.isfinite(magnitude) else 5.0,
        float(reverse_like),
        float(normal_like),
    ]
    return EventFeatures(
        np.array(plane_rows), np.array(event_row), slab, interface_like
    )


def tectonic_type_rule(features: EventFeatures) -> TectonicType:
    """Classify tectonic type with the NZ NSHM 2022 rule (Rollins et al. 2022).

    Parameters
    ----------
    features : EventFeatures
        Features computed by `compute_features`.

    Returns
    -------
    TectonicType
        Crustal outside the slab footprint. Inside it, interface if a plane
        is interface-like and the centroid is within
        `INTERFACE_DEPTH_TOLERANCE_KM` of the slab surface; otherwise
        crustal above the slab surface and intraslab below it.

        One refinement to Rollins et al. (2022) is made: a normal-faulting
        event within the interface tolerance band is classified as
        intraslab rather than crustal. Normal faulting at interface depth
        reflects bending or down-dip tension in the subducting plate (for
        example the 2007 Gisborne, 2014 Eketahuna and 2016 Te Araroa
        earthquakes), and CMT centroid depths for such events are often
        several kilometres too shallow.
    """
    if not features.slab.present:
        return TectonicType.CRUSTAL
    depth_below_slab = features.event_features[
        EVENT_FEATURE_NAMES.index("depth_below_slab")
    ]
    normal_like = features.event_features[EVENT_FEATURE_NAMES.index("normal_like")]
    if abs(depth_below_slab) <= INTERFACE_DEPTH_TOLERANCE_KM:
        if features.interface_like:
            return TectonicType.INTERFACE
        if normal_like:
            return TectonicType.SLAB
    return TectonicType.CRUSTAL if depth_below_slab <= 0 else TectonicType.SLAB


class Forest:
    """A random forest exported to JSON, evaluated with numpy.

    Each tree is stored with scikit-learn's array layout: ``children_left``,
    ``children_right``, ``feature``, ``threshold`` and per-leaf class
    probabilities ``value``. The forest prediction is the mean of the tree
    probabilities.
    """

    def __init__(self, trees: list[dict], feature_names: list[str], classes: list):
        """Create a forest from decoded JSON.

        Parameters
        ----------
        trees : list[dict]
            Tree dictionaries with the scikit-learn array layout.
        feature_names : list[str]
            Feature names in input order.
        classes : list
            Class labels in probability-column order.
        """
        self.feature_names = feature_names
        self.classes = classes
        self._trees = [
            {
                "children_left": np.asarray(tree["children_left"], dtype=int),
                "children_right": np.asarray(tree["children_right"], dtype=int),
                "feature": np.asarray(tree["feature"], dtype=int),
                "threshold": np.asarray(tree["threshold"], dtype=float),
                "value": _normalise_rows(np.asarray(tree["value"], dtype=float)),
            }
            for tree in trees
        ]

    @classmethod
    def from_json(cls, path: Path | Traversable) -> Forest:
        """Load a forest from a JSON file.

        Parameters
        ----------
        path : Path | Traversable
            Path to the JSON file written by the training script.

        Returns
        -------
        Forest
            The loaded forest.
        """
        with path.open("r") as handle:
            data = json.load(handle)
        return cls(data["trees"], data["feature_names"], data["classes"])

    def _check(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Validate and shape an input array."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {x.shape[1]}."
            )
        return x

    def predict_proba(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities for feature vectors.

        Parameters
        ----------
        x : npt.ArrayLike
            Array of shape (n_samples, n_features) or (n_features,).

        Returns
        -------
        npt.NDArray[np.float64]
            Array of shape (n_samples, n_classes).
        """
        x = self._check(x)
        total = np.zeros((x.shape[0], len(self.classes)))
        for tree in self._trees:
            total += tree["value"][self._leaves(tree, x)]
        return total / len(self._trees)

    def _leaves(self, tree: dict, x: npt.NDArray) -> npt.NDArray[np.int64]:
        """Index of the leaf each sample reaches in a tree."""
        node = np.zeros(x.shape[0], dtype=int)
        sample_index = np.arange(x.shape[0])
        active = tree["children_left"][node] >= 0
        while active.any():
            feature = tree["feature"][node[active]]
            go_left = (
                x[sample_index[active], feature] <= tree["threshold"][node[active]]
            )
            node[active] = np.where(
                go_left,
                tree["children_left"][node[active]],
                tree["children_right"][node[active]],
            )
            active = tree["children_left"][node] >= 0
        return node

    def contributions(
        self, x: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Decompose predictions into per-feature contributions.

        Uses the decision-path decomposition of Saabas: along the path a
        sample takes through a tree, the change in the node's class
        distribution at each split is attributed to the feature split on.
        The decomposition is exact, so ``bias + contributions.sum(axis=1)``
        equals `predict_proba`.

        Parameters
        ----------
        x : npt.ArrayLike
            Array of shape (n_samples, n_features) or (n_features,).

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            The bias (mean root distribution, shape (n_classes,)) and the
            contributions, of shape (n_samples, n_features, n_classes).
        """
        x = self._check(x)
        n_samples, n_features = x.shape
        n_classes = len(self.classes)
        contributions = np.zeros((n_samples, n_features, n_classes))
        bias = np.zeros(n_classes)
        sample_index = np.arange(n_samples)
        for tree in self._trees:
            bias += tree["value"][0]
            node = np.zeros(n_samples, dtype=int)
            active = tree["children_left"][node] >= 0
            while active.any():
                current = node[active]
                feature = tree["feature"][current]
                go_left = x[sample_index[active], feature] <= tree["threshold"][current]
                child = np.where(
                    go_left,
                    tree["children_left"][current],
                    tree["children_right"][current],
                )
                np.add.at(
                    contributions,
                    (sample_index[active], feature),
                    tree["value"][child] - tree["value"][current],
                )
                node[active] = child
                active = tree["children_left"][node] >= 0
        return bias / len(self._trees), contributions / len(self._trees)


def _normalise_rows(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalise the rows of an array to sum to one.

    Leaf distributions are rounded when the model is exported to JSON; this
    restores exact probabilities.
    """
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


@functools.cache
def _load_forest(filename: str) -> Forest | None:
    """Load a packaged forest, or None if it has not been trained."""
    path = _DATA_DIR / filename
    return Forest.from_json(path) if path.is_file() else None


class CMTClassifier:
    """Nodal plane selection and tectonic type classification for CMT solutions."""

    def __init__(
        self,
        segments: FaultSegmentIndex,
        slab_model: SlabModel,
        nodal_plane_model: Forest | None,
        tectonic_type_model: Forest | None,
    ):
        """Create a classifier.

        Parameters
        ----------
        segments : FaultSegmentIndex
            Indexed CFM fault segments.
        slab_model : SlabModel
            Subduction interface geometry.
        nodal_plane_model : Forest | None
            Trained nodal plane forest. If None, `most_likely_nodal_plane`
            raises.
        tectonic_type_model : Forest | None
            Trained tectonic type forest. If None, `tectonic_type` falls
            back to `tectonic_type_rule`.
        """
        self.segments = segments
        self.slab_model = slab_model
        self.nodal_plane_model = nodal_plane_model
        self.tectonic_type_model = tectonic_type_model

    @classmethod
    def from_faults(cls, faults: list[CommunityFault]) -> CMTClassifier:
        """Build the classifier for a list of faults with the packaged models.

        Parameters
        ----------
        faults : list[CommunityFault]
            Faults to index.

        Returns
        -------
        CMTClassifier
            The classifier.
        """
        return cls(
            FaultSegmentIndex(faults),
            SlabModel.load(),
            _load_forest("nodal_plane_model.json"),
            _load_forest("tectonic_type_model.json"),
        )

    @classmethod
    @functools.cache
    def load(cls) -> CMTClassifier:
        """Load the classifier with the packaged CFM, slab grids and models.

        Returns
        -------
        CMTClassifier
            The loaded classifier (cached).
        """
        return cls.from_faults(get_community_fault_model())

    def features(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
        magnitude: float = float("nan"),
    ) -> EventFeatures:
        """Compute features for a CMT solution (see `compute_features`).

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km).
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.
        magnitude : float, optional
            Moment magnitude, NaN if unknown.

        Returns
        -------
        EventFeatures
            The computed features.
        """
        return compute_features(
            self.segments,
            self.slab_model,
            centroid,
            nodal_plane_1,
            nodal_plane_2,
            magnitude,
        )

    def nodal_plane_1_probability(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
        magnitude: float = float("nan"),
    ) -> float:
        """Probability that the first nodal plane is the fault plane.

        The forest is evaluated for both orderings of the planes and the
        results averaged, so the probability for the swapped ordering is
        exactly one minus this value.

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km).
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.
        magnitude : float, optional
            Moment magnitude, NaN if unknown.

        Returns
        -------
        float
            Probability in [0, 1] that `nodal_plane_1` is the fault plane.

        Raises
        ------
        RuntimeError
            If no nodal plane model is loaded.
        """
        if self.nodal_plane_model is None:
            raise RuntimeError("No nodal plane model is loaded.")
        features = self.features(centroid, nodal_plane_1, nodal_plane_2, magnitude)
        return self._nodal_plane_1_probability(features)

    def _nodal_plane_1_probability(self, features: EventFeatures) -> float:
        assert self.nodal_plane_model is not None
        x = np.stack([features.nodal_plane_vector(0), features.nodal_plane_vector(1)])
        probabilities = self.nodal_plane_model.predict_proba(x)
        column = self.nodal_plane_model.classes.index(1)
        return float(0.5 * (probabilities[0, column] + 1.0 - probabilities[1, column]))

    def most_likely_nodal_plane(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
        magnitude: float = float("nan"),
    ) -> NodalPlane:
        """Select the nodal plane most likely to be the fault plane.

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km).
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.
        magnitude : float, optional
            Moment magnitude, NaN if unknown.

        Returns
        -------
        NodalPlane
            The preferred nodal plane.
        """
        probability = self.nodal_plane_1_probability(
            centroid, nodal_plane_1, nodal_plane_2, magnitude
        )
        return nodal_plane_1 if probability >= 0.5 else nodal_plane_2

    def tectonic_type_probabilities(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
        magnitude: float = float("nan"),
    ) -> dict[TectonicType, float]:
        """Tectonic type class probabilities.

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km).
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.
        magnitude : float, optional
            Moment magnitude, NaN if unknown.

        Returns
        -------
        dict[TectonicType, float]
            Probability of each tectonic type. The probabilities come from a
            forest trained on the NSHM rule under Monte Carlo perturbation of
            the depth below the interface (about 7 km standard deviation), so
            they express how robust the rule-based class is to centroid and
            slab depth uncertainty. Without a trained model the rule-based
            class receives probability one.
        """
        features = self.features(centroid, nodal_plane_1, nodal_plane_2, magnitude)
        if self.tectonic_type_model is None:
            rule = tectonic_type_rule(features)
            return {t: float(t == rule) for t in TECTONIC_TYPES}
        probabilities = self.tectonic_type_model.predict_proba(
            features.tectonic_vector()
        )[0]
        return {
            TectonicType(label): float(p)
            for label, p in zip(self.tectonic_type_model.classes, probabilities)
        }

    def tectonic_type(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
        magnitude: float = float("nan"),
    ) -> TectonicType:
        """Tectonic type by the NZ NSHM 2022 rule (see `tectonic_type_rule`).

        This is deterministic given the centroid, planes and slab geometry.
        Use `tectonic_type_probabilities` for class probabilities that account
        for depth uncertainty.

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km).
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.
        magnitude : float, optional
            Moment magnitude, NaN if unknown.

        Returns
        -------
        TectonicType
            The tectonic type.
        """
        return tectonic_type_rule(
            self.features(centroid, nodal_plane_1, nodal_plane_2, magnitude)
        )

    def explain_nodal_plane(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
        magnitude: float = float("nan"),
    ) -> NodalPlaneExplanation:
        """Explain the nodal plane choice as a sum of feature contributions.

        The explanation starts from an even split between the planes and
        attributes the move away from it to individual features, using the
        decision-path decomposition of the tree ensemble (see
        `Forest.contributions`). Because the ensemble is evaluated for both
        plane orderings, the baseline is exactly 0.5 and the contributions
        sum to the probability that plane 1 is the fault plane.

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km).
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.
        magnitude : float, optional
            Moment magnitude, NaN if unknown.

        Returns
        -------
        NodalPlaneExplanation
            The probability and the feature contributions to it.

        Raises
        ------
        RuntimeError
            If no nodal plane model is loaded.
        """
        if self.nodal_plane_model is None:
            raise RuntimeError("No nodal plane model is loaded.")
        features = self.features(centroid, nodal_plane_1, nodal_plane_2, magnitude)
        model = self.nodal_plane_model
        column = model.classes.index(1)
        x = np.stack([features.nodal_plane_vector(0), features.nodal_plane_vector(1)])
        _, contributions = model.contributions(x)
        # Ordering 1 predicts the probability of plane 2, so its contributions
        # are negated; averaging the two makes the explanation antisymmetric.
        combined = 0.5 * (contributions[0, :, column] - contributions[1, :, column])
        n_plane_features = len(PLANE_FEATURE_NAMES)
        rows = []
        for i, name in enumerate(model.feature_names):
            if i < n_plane_features:
                plane_1_value = float(features.plane_features[0, i])
                plane_2_value = float(features.plane_features[1, i])
                key = PLANE_FEATURE_NAMES[i]
            else:
                plane_1_value = float(features.event_features[i - n_plane_features])
                plane_2_value = float("nan")
                key = EVENT_FEATURE_NAMES[i - n_plane_features]
            rows.append(
                FeatureContribution(
                    name,
                    FEATURE_DESCRIPTIONS.get(key, key),
                    float(combined[i]),
                    plane_1_value,
                    plane_2_value,
                )
            )
        rows.sort(key=lambda row: abs(row.contribution), reverse=True)
        return NodalPlaneExplanation(
            self._nodal_plane_1_probability(features), 0.5, rows
        )

    def explain_tectonic_type(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
        magnitude: float = float("nan"),
    ) -> TectonicExplanation:
        """Explain the tectonic type as the NSHM rule evaluated step by step.

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km).
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.
        magnitude : float, optional
            Moment magnitude, NaN if unknown.

        Returns
        -------
        TectonicExplanation
            The type, the probabilities under depth uncertainty and the
            criteria that produced the type.
        """
        features = self.features(centroid, nodal_plane_1, nodal_plane_2, magnitude)
        tectonic_type = tectonic_type_rule(features)
        depth = features.event_features[EVENT_FEATURE_NAMES.index("depth")]
        below = features.event_features[EVENT_FEATURE_NAMES.index("depth_below_slab")]
        reverse_like = bool(
            features.event_features[EVENT_FEATURE_NAMES.index("reverse_like")]
        )
        normal_like = bool(
            features.event_features[EVENT_FEATURE_NAMES.index("normal_like")]
        )
        style = (
            "reverse" if reverse_like else "normal" if normal_like else "strike-slip"
        )
        criteria = [
            Criterion(
                "Subduction interface beneath the centroid",
                f"Slab2 interface at {features.slab.depth:.1f} km, dipping "
                f"{features.slab.dip:.0f} degrees"
                if features.slab.present
                else "No modelled interface at this location, so the event is crustal",
                features.slab.present,
            )
        ]
        if features.slab.present:
            side = "below" if below > 0 else "above"
            criteria.append(
                Criterion(
                    f"Centroid within {INTERFACE_DEPTH_TOLERANCE_KM:.0f} km of the interface",
                    f"Centroid at {depth:.1f} km, {abs(below):.1f} km {side} the interface",
                    bool(abs(below) <= INTERFACE_DEPTH_TOLERANCE_KM),
                )
            )
            for i, plane in enumerate((nodal_plane_1, nodal_plane_2), start=1):
                slab_angle = features.plane_features[
                    i - 1, PLANE_FEATURE_NAMES.index("slab_angle")
                ]
                rake_misfit = features.plane_features[
                    i - 1, PLANE_FEATURE_NAMES.index("slab_rake_misfit")
                ]
                criteria.append(
                    Criterion(
                        f"Nodal plane {i} could be the interface",
                        f"Rake {rake_misfit:.0f} degrees from pure thrust "
                        f"(limit {INTERFACE_RAKE_TOLERANCE:.0f}), dip {plane.dip:.0f} "
                        f"degrees (limit {INTERFACE_MAX_DIP:.0f}), {slab_angle:.0f} "
                        f"degrees from the interface (limit {INTERFACE_MAX_SLAB_ANGLE:.0f})",
                        bool(
                            rake_misfit <= INTERFACE_RAKE_TOLERANCE
                            and plane.dip <= INTERFACE_MAX_DIP
                            and slab_angle <= INTERFACE_MAX_SLAB_ANGLE
                        ),
                    )
                )
            criteria.append(
                Criterion(
                    "Style of faulting",
                    f"{style.capitalize()} faulting"
                    + (
                        "; normal faulting at interface depth is bending of the "
                        "subducting plate, so the event is intraslab"
                        if normal_like and abs(below) <= INTERFACE_DEPTH_TOLERANCE_KM
                        else ""
                    ),
                    None,
                )
            )
        probabilities = self.tectonic_type_probabilities(
            centroid, nodal_plane_1, nodal_plane_2, magnitude
        )
        return TectonicExplanation(tectonic_type, probabilities, criteria)
