"""Nodal plane selection and tectonic type classification for CMT solutions.

The fault plane of a centroid moment tensor (CMT) solution is chosen by
scoring both nodal planes against the nearby mapped faults of the New
Zealand Community Fault Model (CFM): agreement in strike (with and without
dip direction), dip and rake, and the distance from the plane's up-dip
projection to a similarly oriented mapped trace. A logistic regression
without intercept on the difference of the two planes' misfits gives the
probability that plane 1 is the fault plane, so swapping the planes flips
the answer. The weights are fitted with ``scripts/train_cmt_classifiers.py``.

The tectonic type (crustal, subduction interface or intraslab) follows the
NZ NSHM 2022 rule (Rollins et al. 2022) against the Slab2 interface
geometry (Hayes et al. 2018), with probabilities from Gaussian uncertainty
in the depth below the interface. Alternatively, the modified NGA-SUB
(2020) rule used by the NZGMDB classifies from the location and depth
alone, by the event's position relative to the up-dip, seismogenic and
down-dip zones of the interface.

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
>>> classifier.nga_sub_tectonic_type(centroid)
<TectonicType.INTERFACE: 'interface'>
"""

from __future__ import annotations

import functools
import json
import warnings
from enum import Enum
from importlib import resources
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pyproj
import scipy as sp
import shapely

import source_modelling
from qcore import coordinates, geo
from source_modelling.community_fault_model import (
    CommunityFault,
    NodalPlane,
    get_community_fault_model,
)
from source_modelling.magnitude_scaling import RakeType, rake_type

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

DEPTH_BELOW_SLAB_SIGMA_KM = 7.0
"""Standard deviation of the depth below the interface (combined centroid and Slab2 depth uncertainty)."""

SEISMOGENIC_ZONE_DEPTHS_KM = {"ker": (10.0, 47.0), "puy": (11.0, 30.0)}
"""Interface depths bounding the seismogenic zone of each Slab2 region (Hayes et al. 2018)."""

SLAB_ZONE_SEARCH_RADIUS_KM = 10.0
"""Horizontal distance from a slab zone within which an event is assigned to it."""

SEISMOGENIC_CRUSTAL_MAX_DEPTH_KM = 20.0
"""Maximum depth of crustal events above the seismogenic zone (NGA-SUB rule)."""

DOWNDIP_CRUSTAL_MAX_DEPTH_KM = 30.0
"""Depth above which events down-dip of the seismogenic zone are always crustal (NGA-SUB rule)."""

UNDETERMINED_CRUSTAL_MAX_DEPTH_KM = 50.0
"""Depth splitting NGA-SUB "undetermined" events into crustal and intraslab (as in the NZGMDB)."""

INTERFACE_MAX_DEPTH_KM = 60.0
"""Maximum depth of interface events (NGA-SUB rule)."""

CENTROID_DEPTH_SIGMA_KM = 7.0
"""Standard deviation of the centroid depth for the NGA-SUB rule."""

FAULT_NEIGHBOURS = 8
"""Number of nearest CFM fault segments used for the strike, dip and rake misfits."""

STRIKE_MATCH_TOLERANCE = 30.0
"""Maximum strike difference for a fault segment to count as a match in the projected trace test."""

MAX_DISTANCE_KM = 100.0
"""Projected trace distances are clipped to this value."""

NODAL_PLANE_FEATURE_NAMES = [
    "cfm_strike_unoriented",
    "cfm_strike_oriented",
    "cfm_dip_misfit",
    "cfm_rake_misfit",
    "log_projected_distance_unoriented",
    "log_projected_distance_oriented",
]
"""Names, in order, of the per-plane misfits used by the nodal plane model."""

_DATA_DIR = resources.files(source_modelling) / "NZ_CFM"


class TectonicType(Enum):
    """Tectonic type of an earthquake."""

    CRUSTAL = "crustal"
    """Shallow crustal (upper plate) earthquake."""

    INTERFACE = "interface"
    """Subduction interface earthquake."""

    SLAB = "slab"
    """Intraslab (including outer-rise) earthquake."""


class SlabZone(Enum):
    """Part of a subduction interface in the NGA-SUB (2020) classification."""

    UPDIP = "updip"
    """Interface shallower than the seismogenic zone, near the trench (region A)."""

    SEISMOGENIC = "seismogenic"
    """Seismogenic zone of the interface (region B)."""

    DOWNDIP = "downdip"
    """Interface deeper than the seismogenic zone (region C)."""


def _unit_vectors(lat: npt.ArrayLike, lon: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Unit vectors from the Earth's centre to points on a sphere."""
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def _angular_difference(
    a: npt.ArrayLike, b: npt.ArrayLike, period: float = 360.0
) -> npt.NDArray[np.float64]:
    """Smallest absolute difference between angles in degrees with the given period."""
    difference = np.mod(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), period)
    return np.minimum(difference, period - difference)


def _plane_normal(strike: float, dip: float) -> npt.NDArray[np.float64]:
    """Unit normal of a plane in (north, east, down) coordinates (Aki & Richards eq. 4.88)."""
    strike, dip = np.radians(strike), np.radians(dip)
    return np.array(
        [-np.sin(dip) * np.sin(strike), np.sin(dip) * np.cos(strike), -np.cos(dip)]
    )


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


class SlabZoneQuery(NamedTuple):
    """Slab zone near a location."""

    zone: SlabZone | None
    """The slab zone, None away from all zones."""

    depth: float
    """Depth of the slab surface in km at the nearest point of the zone, NaN away from all zones."""


class SlabModel:
    """Slab2 subduction interface geometry cropped to New Zealand.

    Contains the Kermadec-Hikurangi ("ker") and Puysegur ("puy") regions of
    Slab2 (Hayes et al. 2018), interpolated bilinearly. Strike is
    interpolated as a unit vector to avoid wraparound. The grid nodes are
    also split into the NGA-SUB slab zones by their depth relative to the
    region's seismogenic zone.

    Parameters
    ----------
    grids : dict[str, npt.NDArray]
        Mapping with keys ``<region>_lat``, ``<region>_lon``,
        ``<region>_depth``, ``<region>_dip`` and ``<region>_strike``
        for each region.
    seismogenic_zone_depths : dict[str, tuple[float, float]], optional
        Interface depths in km bounding the seismogenic zone of each region.
    """

    def __init__(
        self,
        grids: dict[str, npt.NDArray],
        seismogenic_zone_depths: dict[
            str, tuple[float, float]
        ] = SEISMOGENIC_ZONE_DEPTHS_KM,
    ):
        """Create a slab model from named grids.

        Parameters
        ----------
        grids : dict[str, npt.NDArray]
            Mapping with keys ``<region>_lat``, ``<region>_lon``,
            ``<region>_depth``, ``<region>_dip`` and ``<region>_strike``
            for each region.
        seismogenic_zone_depths : dict[str, tuple[float, float]], optional
            Interface depths in km bounding the seismogenic zone of each region.
        """
        self._regions = []
        zone_points = {zone: [] for zone in SlabZone}
        zone_depths = {zone: [] for zone in SlabZone}
        for region in sorted({key.split("_")[0] for key in grids}):
            lats = grids[f"{region}_lat"]
            lons = grids[f"{region}_lon"]
            depth = grids[f"{region}_depth"]
            strike = np.radians(grids[f"{region}_strike"])
            values = np.stack(
                [depth, grids[f"{region}_dip"], np.sin(strike), np.cos(strike)],
                axis=-1,
            )
            interpolator = sp.interpolate.RegularGridInterpolator(
                (lats, lons), values, bounds_error=False, fill_value=np.nan
            )
            self._regions.append((bool(lons[-1] > 180.0), interpolator))

            points = _unit_vectors(*np.meshgrid(lats, lons, indexing="ij"))
            top, bottom = seismogenic_zone_depths[region]
            # NaN depths (outside the footprint) fall in no zone.
            for zone, in_zone in [
                (SlabZone.UPDIP, depth < top),
                (SlabZone.SEISMOGENIC, (depth >= top) & (depth <= bottom)),
                (SlabZone.DOWNDIP, depth > bottom),
            ]:
                zone_points[zone].append(points[in_zone])
                zone_depths[zone].append(depth[in_zone])
        self._zones = {
            zone: (
                sp.spatial.KDTree(np.concatenate(zone_points[zone])),
                np.concatenate(zone_depths[zone]),
            )
            for zone in SlabZone
        }

    @classmethod
    @functools.cache
    def load(cls) -> SlabModel:
        """Load the packaged Slab2 grids.

        Returns
        -------
        SlabModel
            The loaded slab model (cached).
        """
        with (
            (_DATA_DIR / "slab2_nz.npz").open("rb") as handle,
            np.load(handle) as data,
        ):
            return cls({key: data[key] for key in data})

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
        for positive_lon, interpolator in self._regions:
            query_lon = lon % 360.0 if positive_lon else ((lon + 180.0) % 360.0) - 180.0
            depth, dip, sin_strike, cos_strike = interpolator([lat, query_lon])[0]
            if np.isfinite(depth):
                strike = np.degrees(np.arctan2(sin_strike, cos_strike)) % 360.0
                return SlabQuery(float(depth), float(dip), float(strike))
        return SlabQuery(float("nan"), float("nan"), float("nan"))

    def zone(self, lat: float, lon: float) -> SlabZoneQuery:
        """NGA-SUB slab zone near a location.

        A location is in a zone if it is within `SLAB_ZONE_SEARCH_RADIUS_KM`
        of one of its grid nodes. Near zone boundaries the seismogenic zone
        takes precedence, then the down-dip zone.

        Parameters
        ----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees (either -180..180 or 0..360).

        Returns
        -------
        SlabZoneQuery
            The zone and the slab depth at its nearest node.
        """
        point = _unit_vectors(lat, lon)
        # Chord length on the unit sphere, equal to arc length at this scale.
        radius = SLAB_ZONE_SEARCH_RADIUS_KM / geo.R_EARTH
        for zone in (SlabZone.SEISMOGENIC, SlabZone.DOWNDIP, SlabZone.UPDIP):
            tree, depths = self._zones[zone]
            distance, index = tree.query(point, distance_upper_bound=radius)
            if np.isfinite(distance):
                return SlabZoneQuery(zone, float(depths[index]))
        return SlabZoneQuery(None, float("nan"))


class FaultSegmentIndex:
    """Straight-line segments of the CFM fault traces with their attributes.

    Segments keep the orientation of their trace, which
    `load_community_fault_model` sets so that the dip direction is
    strike + 90 (the Aki-Richards convention).

    Parameters
    ----------
    faults : list[CommunityFault]
        Faults from the community fault model.
    """

    def __init__(self, faults: list[CommunityFault]):
        """Build the index from a list of community faults.

        Parameters
        ----------
        faults : list[CommunityFault]
            Faults from the community fault model.
        """
        coords, fault_index = shapely.get_coordinates(
            [fault.trace for fault in faults], return_index=True
        )
        same_fault = fault_index[:-1] == fault_index[1:]
        segment_fault = fault_index[:-1][same_fault]
        self.start = coords[:-1][same_fault]
        """Segment start points in NZTM (northing, easting)."""
        self.end = coords[1:][same_fault]
        """Segment end points in NZTM (northing, easting)."""
        self.dip = np.array([f.dip_range.pref for f in faults], dtype=float)[
            segment_fault
        ]
        """Preferred dip of the parent fault (degrees)."""
        self.rake = np.array([f.rake_range.pref for f in faults], dtype=float)[
            segment_fault
        ]
        """Preferred rake of the parent fault (degrees)."""
        self.dip_known = np.array([f.dip_dir is not None for f in faults])[
            segment_fault
        ]
        """Whether the parent fault has a recorded dip direction."""

        start_wgs = coordinates.nztm_to_wgs_depth(self.start)
        end_wgs = coordinates.nztm_to_wgs_depth(self.end)
        forward_azimuth, _, _ = pyproj.Geod(ellps="WGS84").inv(
            start_wgs[:, 1], start_wgs[:, 0], end_wgs[:, 1], end_wgs[:, 0]
        )
        self.strike = np.asarray(forward_azimuth) % 360.0
        """Strike of each segment (degrees)."""

    def _distances(self, point: npt.NDArray) -> npt.NDArray[np.float64]:
        """Distance in km from an NZTM point to every segment."""
        direction = self.end - self.start
        offset = point[:2] - self.start
        length_sq = np.maximum(np.einsum("ij,ij->i", direction, direction), 1e-9)
        t = np.clip(np.einsum("ij,ij->i", offset, direction) / length_sq, 0.0, 1.0)
        projection = self.start + t[:, None] * direction
        return np.linalg.norm(point[:2] - projection, axis=1) / 1000.0

    def plane_misfits(
        self, lat: float, lon: float, depth: float, plane: NodalPlane
    ) -> npt.NDArray[np.float64]:
        """Misfits of a nodal plane against the nearby mapped faults.

        Parameters
        ----------
        lat : float
            Centroid latitude in degrees.
        lon : float
            Centroid longitude in degrees.
        depth : float
            Centroid depth in km.
        plane : NodalPlane
            The nodal plane.

        Returns
        -------
        npt.NDArray[np.float64]
            Misfits in `NODAL_PLANE_FEATURE_NAMES` order. Smaller values mean
            a better match.
        """
        distances = self._distances(coordinates.wgs_depth_to_nztm(np.array([lat, lon])))
        nearest = np.argsort(distances)[:FAULT_NEIGHBOURS]
        weights = 1.0 / (distances[nearest] + 2.0)
        weights /= weights.sum()

        unoriented = _angular_difference(self.strike, plane.strike, 180.0)
        # Faults with unknown dip direction only constrain the unoriented strike.
        oriented = np.where(
            self.dip_known, _angular_difference(self.strike, plane.strike), unoriented
        )

        # Project the plane up-dip to the surface and find the nearest
        # similarly striking trace.
        surface_lat, surface_lon = geo.ll_shift(
            lat,
            lon,
            depth / np.tan(np.radians(max(plane.dip, 5.0))),
            (plane.strike - 90.0) % 360.0,
        )
        surface_distances = self._distances(
            coordinates.wgs_depth_to_nztm(np.array([surface_lat, surface_lon]))
        )

        def projected_distance(strike_misfit: npt.NDArray[np.float64]) -> float:
            matches = surface_distances[strike_misfit < STRIKE_MATCH_TOLERANCE]
            return float(np.log1p(min(matches.min(initial=np.inf), MAX_DISTANCE_KM)))

        return np.array(
            [
                weights @ unoriented[nearest],
                weights @ oriented[nearest],
                weights @ np.abs(self.dip[nearest] - plane.dip),
                weights @ _angular_difference(self.rake[nearest], plane.rake),
                projected_distance(unoriented),
                projected_distance(oriented),
            ]
        )


def _parse_centroid(centroid: npt.ArrayLike) -> tuple[float, float, float]:
    """Split a (lat, lon[, depth_km]) centroid, warning if the depth is missing."""
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


def _is_interface_like(plane: NodalPlane, slab: SlabQuery) -> bool:
    """Whether a plane is thrust-like, not too steep and near parallel to the slab surface."""
    if not slab.present:
        return False
    # Angle between the plane and the slab surface (between their normals).
    cosine = abs(
        float(
            _plane_normal(plane.strike, plane.dip)
            @ _plane_normal(slab.strike, slab.dip)
        )
    )
    slab_angle = np.degrees(np.arccos(min(cosine, 1.0)))
    return bool(
        _angular_difference(plane.rake, 90.0) <= INTERFACE_RAKE_TOLERANCE
        and plane.dip <= INTERFACE_MAX_DIP
        and slab_angle <= INTERFACE_MAX_SLAB_ANGLE
    )


def tectonic_type_probabilities(
    depth_below_slab: float,
    interface_like: bool,
    normal_like: bool,
    sigma: float = DEPTH_BELOW_SLAB_SIGMA_KM,
) -> dict[TectonicType, float]:
    """Tectonic type probabilities by the NZ NSHM 2022 rule (Rollins et al. 2022).

    Events outside the slab footprint are crustal. Inside it, events within
    `INTERFACE_DEPTH_TOLERANCE_KM` of the slab surface are interface if a
    plane is interface-like and intraslab if the mechanism is normal
    faulting (bending of the subducting plate, whose CMT centroids are often
    too shallow). Other events are crustal above the slab surface and
    intraslab below it. The rule is applied with the depth below the slab
    normally distributed.

    Parameters
    ----------
    depth_below_slab : float
        Centroid depth below the slab surface in km (negative above it),
        NaN outside the slab footprint.
    interface_like : bool
        Whether either nodal plane is interface-like.
    normal_like : bool
        Whether the mechanism is normal faulting.
    sigma : float, optional
        Standard deviation of the depth below the slab in km. With
        ``sigma=0`` the deterministic rule is applied.

    Returns
    -------
    dict[TectonicType, float]
        Probability of each tectonic type.
    """
    if np.isnan(depth_below_slab):
        return {
            TectonicType.CRUSTAL: 1.0,
            TectonicType.INTERFACE: 0.0,
            TectonicType.SLAB: 0.0,
        }
    tolerance = INTERFACE_DEPTH_TOLERANCE_KM
    # Probability that the depth below the slab is above the tolerance band,
    # above the slab surface, and above the bottom of the band.
    if sigma > 0:
        above_band, above_slab, above_band_bottom = np.asarray(
            sp.stats.norm.cdf([-tolerance, 0.0, tolerance], depth_below_slab, sigma)
        ).tolist()
    else:
        above_band = float(depth_below_slab < -tolerance)
        above_slab = float(depth_below_slab <= 0.0)
        above_band_bottom = float(depth_below_slab <= tolerance)
    band_crustal = above_slab - above_band
    band_slab = above_band_bottom - above_slab
    interface = 0.0
    if interface_like:
        interface, band_crustal, band_slab = band_crustal + band_slab, 0.0, 0.0
    elif normal_like:
        band_crustal, band_slab = 0.0, band_crustal + band_slab
    return {
        TectonicType.CRUSTAL: float(above_band + band_crustal),
        TectonicType.INTERFACE: float(interface),
        TectonicType.SLAB: float(1.0 - above_band_bottom + band_slab),
    }


def _nga_sub_depth_intervals(
    zone: SlabZone | None, slab_depth: float
) -> list[tuple[float, TectonicType]]:
    """Maximum depth of each tectonic type under the NGA-SUB rule, shallowest first."""
    tolerance = INTERFACE_DEPTH_TOLERANCE_KM
    match zone:
        case None:
            return [
                (UNDETERMINED_CRUSTAL_MAX_DEPTH_KM, TectonicType.CRUSTAL),
                (np.inf, TectonicType.SLAB),
            ]
        case SlabZone.UPDIP:
            # Outer-rise events above 60 km, intraslab below.
            return [(np.inf, TectonicType.SLAB)]
        case SlabZone.SEISMOGENIC:
            return [
                (
                    min(slab_depth - tolerance, SEISMOGENIC_CRUSTAL_MAX_DEPTH_KM),
                    TectonicType.CRUSTAL,
                ),
                (
                    min(slab_depth + tolerance, INTERFACE_MAX_DEPTH_KM),
                    TectonicType.INTERFACE,
                ),
                (np.inf, TectonicType.SLAB),
            ]
        case SlabZone.DOWNDIP:
            # Undetermined events between the crustal and slab depths are
            # split at UNDETERMINED_CRUSTAL_MAX_DEPTH_KM.
            crustal_max_depth = max(
                DOWNDIP_CRUSTAL_MAX_DEPTH_KM,
                min(slab_depth - tolerance, UNDETERMINED_CRUSTAL_MAX_DEPTH_KM),
            )
            return [
                (crustal_max_depth, TectonicType.CRUSTAL),
                (np.inf, TectonicType.SLAB),
            ]


def nga_sub_tectonic_type_probabilities(
    depth: float,
    zone: SlabZone | None,
    slab_depth: float,
    sigma: float = CENTROID_DEPTH_SIGMA_KM,
) -> dict[TectonicType, float]:
    """Tectonic type probabilities by the modified NGA-SUB (2020) rule of the NZGMDB.

    The rule depends only on the location and depth, not the mechanism:

    - Up-dip zone: intraslab (outer-rise above 60 km).
    - Seismogenic zone: crustal above both 20 km and 10 km above the slab;
      otherwise interface above both 60 km and 10 km below the slab;
      otherwise intraslab.
    - Down-dip zone: crustal above 30 km, intraslab within 10 km above the
      slab or deeper, and undetermined otherwise.
    - Away from all zones: crustal above 30 km, intraslab below 60 km, and
      undetermined otherwise.

    Undetermined events are crustal above `UNDETERMINED_CRUSTAL_MAX_DEPTH_KM`
    and intraslab below, as in the NZGMDB. The rule is applied with the
    centroid depth normally distributed.

    Parameters
    ----------
    depth : float
        Centroid depth in km.
    zone : SlabZone | None
        Slab zone near the centroid, None away from all zones.
    slab_depth : float
        Slab surface depth in km at the nearest point of the zone (unused
        when `zone` is None).
    sigma : float, optional
        Standard deviation of the centroid depth in km. With ``sigma=0``
        the deterministic rule is applied.

    Returns
    -------
    dict[TectonicType, float]
        Probability of each tectonic type.
    """
    intervals = _nga_sub_depth_intervals(zone, slab_depth)
    bounds = np.array([-np.inf] + [max_depth for max_depth, _ in intervals])
    if sigma > 0:
        cdf = sp.stats.norm.cdf(bounds, depth, sigma)
    else:
        cdf = (depth <= bounds).astype(float)
    probabilities = dict.fromkeys(TectonicType, 0.0)
    for (_, tectonic_type), probability in zip(intervals, np.diff(cdf)):
        probabilities[tectonic_type] += float(probability)
    return probabilities


class CMTClassifier:
    """Nodal plane selection and tectonic type classification for CMT solutions.

    Parameters
    ----------
    segments : FaultSegmentIndex
        Indexed CFM fault segments.
    slab_model : SlabModel
        Subduction interface geometry.
    weights : npt.NDArray[np.float64]
        Logistic regression weights on the plane 1 - plane 2 misfit
        differences, in `NODAL_PLANE_FEATURE_NAMES` order.
    """

    def __init__(
        self,
        segments: FaultSegmentIndex,
        slab_model: SlabModel,
        weights: npt.NDArray[np.float64],
    ):
        """Create a classifier.

        Parameters
        ----------
        segments : FaultSegmentIndex
            Indexed CFM fault segments.
        slab_model : SlabModel
            Subduction interface geometry.
        weights : npt.NDArray[np.float64]
            Logistic regression weights on the plane 1 - plane 2 misfit
            differences, in `NODAL_PLANE_FEATURE_NAMES` order.
        """
        self.segments = segments
        self.slab_model = slab_model
        self.weights = weights

    @classmethod
    def from_faults(cls, faults: list[CommunityFault]) -> CMTClassifier:
        """Build the classifier for a list of faults with the packaged model.

        Parameters
        ----------
        faults : list[CommunityFault]
            Faults to index.

        Returns
        -------
        CMTClassifier
            The classifier.
        """
        with (_DATA_DIR / "nodal_plane_model.json").open("r") as handle:
            model = json.load(handle)
        if model["feature_names"] != NODAL_PLANE_FEATURE_NAMES:
            raise ValueError("Packaged nodal plane model does not match the features.")
        return cls(
            FaultSegmentIndex(faults),
            SlabModel.load(),
            np.asarray(model["weights"], dtype=float),
        )

    @classmethod
    @functools.cache
    def load(cls) -> CMTClassifier:
        """Load the classifier with the packaged CFM, slab grids and model.

        Returns
        -------
        CMTClassifier
            The loaded classifier (cached).
        """
        return cls.from_faults(get_community_fault_model())

    def nodal_plane_contributions(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
    ) -> dict[str, float]:
        """Contribution of each misfit to the log-odds that plane 1 is the fault plane.

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km). If depth is omitted a default
            is assumed with a warning.
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.

        Returns
        -------
        dict[str, float]
            Log-odds contribution of each feature; they sum to the log-odds.
        """
        lat, lon, depth = _parse_centroid(centroid)
        difference = self.segments.plane_misfits(
            lat, lon, depth, nodal_plane_1
        ) - self.segments.plane_misfits(lat, lon, depth, nodal_plane_2)
        return dict(
            zip(NODAL_PLANE_FEATURE_NAMES, (self.weights * difference).tolist())
        )

    def nodal_plane_1_probability(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
    ) -> float:
        """Probability that the first nodal plane is the fault plane.

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km). If depth is omitted a default
            is assumed with a warning.
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.

        Returns
        -------
        float
            Probability in [0, 1] that `nodal_plane_1` is the fault plane.
        """
        contributions = self.nodal_plane_contributions(
            centroid, nodal_plane_1, nodal_plane_2
        )
        return float(sp.special.expit(np.sum(list(contributions.values()))))

    def most_likely_nodal_plane(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
    ) -> NodalPlane:
        """Select the nodal plane most likely to be the fault plane.

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km). If depth is omitted a default
            is assumed with a warning.
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.

        Returns
        -------
        NodalPlane
            The preferred nodal plane.
        """
        probability = self.nodal_plane_1_probability(
            centroid, nodal_plane_1, nodal_plane_2
        )
        return nodal_plane_1 if probability >= 0.5 else nodal_plane_2

    def tectonic_type_probabilities(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
        sigma: float = DEPTH_BELOW_SLAB_SIGMA_KM,
    ) -> dict[TectonicType, float]:
        """Tectonic type probabilities (see the module-level `tectonic_type_probabilities`).

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km). If depth is omitted a default
            is assumed with a warning.
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.
        sigma : float, optional
            Standard deviation of the depth below the slab in km.

        Returns
        -------
        dict[TectonicType, float]
            Probability of each tectonic type.
        """
        lat, lon, depth = _parse_centroid(centroid)
        slab = self.slab_model.query(lat, lon)
        # The planes share the slip sense; the plane with more dip-slip
        # decides whether the mechanism is normal or strike-slip.
        dominant = max(
            nodal_plane_1, nodal_plane_2, key=lambda p: abs(np.sin(np.radians(p.rake)))
        )
        return tectonic_type_probabilities(
            depth - slab.depth,
            _is_interface_like(nodal_plane_1, slab)
            or _is_interface_like(nodal_plane_2, slab),
            rake_type(dominant.rake) in (RakeType.NORMAL, RakeType.NORMAL_OBLIQUE),
            sigma,
        )

    def tectonic_type(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane,
        nodal_plane_2: NodalPlane,
    ) -> TectonicType:
        """Tectonic type by the NZ NSHM 2022 rule (see `tectonic_type_probabilities`).

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km). If depth is omitted a default
            is assumed with a warning.
        nodal_plane_1 : NodalPlane
            First nodal plane.
        nodal_plane_2 : NodalPlane
            Second nodal plane.

        Returns
        -------
        TectonicType
            The tectonic type.
        """
        probabilities = self.tectonic_type_probabilities(
            centroid, nodal_plane_1, nodal_plane_2, sigma=0.0
        )
        return max(probabilities, key=probabilities.__getitem__)

    def nga_sub_tectonic_type_probabilities(
        self, centroid: npt.ArrayLike, sigma: float = CENTROID_DEPTH_SIGMA_KM
    ) -> dict[TectonicType, float]:
        """Tectonic type probabilities by the NGA-SUB rule (see the module-level `nga_sub_tectonic_type_probabilities`).

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km). If depth is omitted a default
            is assumed with a warning.
        sigma : float, optional
            Standard deviation of the centroid depth in km.

        Returns
        -------
        dict[TectonicType, float]
            Probability of each tectonic type.
        """
        lat, lon, depth = _parse_centroid(centroid)
        zone, slab_depth = self.slab_model.zone(lat, lon)
        return nga_sub_tectonic_type_probabilities(depth, zone, slab_depth, sigma)

    def nga_sub_tectonic_type(self, centroid: npt.ArrayLike) -> TectonicType:
        """Tectonic type by the NGA-SUB rule (see `nga_sub_tectonic_type_probabilities`).

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km). If depth is omitted a default
            is assumed with a warning.

        Returns
        -------
        TectonicType
            The tectonic type.
        """
        probabilities = self.nga_sub_tectonic_type_probabilities(centroid, sigma=0.0)
        return max(probabilities, key=probabilities.__getitem__)
