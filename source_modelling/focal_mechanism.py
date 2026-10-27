"""Nodal plane selection and tectonic type classification for CMT solutions.

The fault plane of a centroid moment tensor (CMT) solution is chosen by
scoring both nodal planes against the nearby mapped faults of the New
Zealand Community Fault Model (CFM): agreement in strike (with and without
dip direction), dip and rake, and the distance from the plane's up-dip
projection to a similarly oriented mapped trace. A logistic regression
without intercept on the difference of the two planes' misfits gives the
probability that plane 1 is the fault plane, so swapping the planes flips
the answer. The log-odds are scaled by a temperature that decays with
centroid depth, because the mapped crustal faults say less about deeper
events. The weights and temperature are fitted with
``scripts/train_cmt_classifiers.py``.

The tectonic type (crustal, subduction interface or intraslab) follows the
probabilistic rule of Haynie et al. (2026) against the Slab2 interface
geometry (Hayes et al. 2018). The interface probability is the product of
Gaussian likelihoods in the vertical distance to the interface and in the
Kagan angle between the mechanism and interface slip (pure reverse, or
along the relative plate motion), for events no deeper than 70 km. The
remainder is crustal (upper plate) above the interface and intraslab below
it.

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
<TectonicType.SLAB: 'slab'>
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

DEFAULT_CENTROID_DEPTH_KM = 10.0
"""Depth assumed when a centroid is given without a depth."""

ON_INTERFACE_SIGMA_KM = 10.0
"""Width of the Gaussian on-interface probability in the vertical distance to the interface (Haynie et al. 2026)."""

SUBDUCTION_SLIP_SIGMA = 30.0
"""Width in degrees of the Gaussian subduction slip probability in the Kagan angle (Haynie et al. 2026)."""

SUBDUCTION_MAX_DEPTH_KM = 70.0
"""Maximum centroid depth of subduction interface events (Haynie et al. 2026)."""

PACIFIC_AUSTRALIA_EULER_POLE = (-60.1, -178.3, 1.0744)
"""Latitude, longitude (degrees) and rate (degrees/Myr) of Pacific rotation relative to Australia (NUVEL-1A, DeMets et al. 1994)."""

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


def _slip_vector(strike: float, dip: float, rake: float) -> npt.NDArray[np.float64]:
    """Unit slip vector of the hanging wall in (north, east, down) coordinates (Aki & Richards eq. 4.88)."""
    strike, dip, rake = np.radians(strike), np.radians(dip), np.radians(rake)
    return np.array(
        [
            np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike),
            np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike),
            -np.sin(rake) * np.sin(dip),
        ]
    )


def _principal_axes(plane: NodalPlane) -> npt.NDArray[np.float64]:
    """Rows T, P and B = T x P of a double couple in (north, east, down) coordinates."""
    normal = _plane_normal(plane.strike, plane.dip)
    slip = _slip_vector(plane.strike, plane.dip, plane.rake)
    t_axis = (normal + slip) / np.sqrt(2.0)
    p_axis = (normal - slip) / np.sqrt(2.0)
    return np.stack([t_axis, p_axis, np.cross(t_axis, p_axis)])


def kagan_angle(plane_a: NodalPlane, plane_b: NodalPlane) -> float:
    """Kagan angle between two double-couple focal mechanisms.

    The Kagan angle (Kagan 1991) is the smallest rotation taking one
    mechanism's principal axes onto the other's, allowing for the
    symmetries of the double couple. Either nodal plane of a mechanism gives
    the same angle.

    Parameters
    ----------
    plane_a : NodalPlane
        A nodal plane of the first mechanism.
    plane_b : NodalPlane
        A nodal plane of the second mechanism.

    Returns
    -------
    float
        The Kagan angle in degrees, between 0 and 120.
    """
    # Diagonal of the rotation between the two principal axis frames; the
    # double-couple symmetries flip the signs of two axes at a time.
    m = np.diag(_principal_axes(plane_a) @ _principal_axes(plane_b).T)
    trace = max(
        m[0] + m[1] + m[2], m[0] - m[1] - m[2], -m[0] + m[1] - m[2], -m[0] - m[1] + m[2]
    )
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


def plate_motion_azimuth(
    lat: float,
    lon: float,
    euler_pole: tuple[float, float, float] = PACIFIC_AUSTRALIA_EULER_POLE,
) -> float:
    """Azimuth of relative plate motion at a location.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    euler_pole : tuple[float, float, float], optional
        Latitude, longitude (degrees) and rate of rotation of one plate
        relative to the other. Defaults to Pacific relative to Australia.

    Returns
    -------
    float
        Azimuth in degrees of the rotating plate's motion relative to the
        fixed plate.
    """
    pole_lat, pole_lon, rate = euler_pole
    velocity = np.cross(
        rate * _unit_vectors(pole_lat, pole_lon), _unit_vectors(lat, lon)
    )
    lat, lon = np.radians(lat), np.radians(lon)
    north = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    )
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    return float(np.degrees(np.arctan2(velocity @ east, velocity @ north)) % 360.0)


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
    Slab2 (Hayes et al. 2018), interpolated bilinearly. Strike is
    interpolated as a unit vector to avoid wraparound.

    Parameters
    ----------
    grids : dict[str, npt.NDArray]
        Mapping with keys ``<region>_lat``, ``<region>_lon``,
        ``<region>_depth``, ``<region>_dip`` and ``<region>_strike``
        for each region.
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
        self._regions = []
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


def subduction_slip_planes(
    lat: float, lon: float, slab: SlabQuery
) -> tuple[NodalPlane, NodalPlane]:
    """Mechanisms of slip on the subduction interface at a location.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    slab : SlabQuery
        Slab surface beneath the location.

    Returns
    -------
    tuple[NodalPlane, NodalPlane]
        Slip on the slab surface (a) in pure reverse (rake 90) and (b) in
        oblique reverse with the horizontal slip parallel to the relative
        Pacific-Australia plate motion.
    """
    # Components of the plate motion along strike and down the dip direction.
    relative_azimuth = np.radians(plate_motion_azimuth(lat, lon) - slab.strike)
    along_strike, down_dip = np.cos(relative_azimuth), np.sin(relative_azimuth)
    # Take the sense of the motion that is reverse slip, whichever plate
    # is subducting.
    if down_dip > 0:
        along_strike, down_dip = -along_strike, -down_dip
    # The horizontal slip of a plane is cos(rake) along strike and
    # -sin(rake) cos(dip) down the dip direction.
    rake = np.degrees(
        np.arctan2(-down_dip / np.cos(np.radians(slab.dip)), along_strike)
    )
    return (
        NodalPlane(slab.strike, slab.dip, 90.0),
        NodalPlane(slab.strike, slab.dip, float(rake)),
    )


def tectonic_type_probabilities(
    depth_above_slab: float,
    depth: float,
    subduction_kagan_angle: float | None = None,
    fixed_depth: bool = False,
    on_interface_sigma: float = ON_INTERFACE_SIGMA_KM,
    subduction_slip_sigma: float = SUBDUCTION_SLIP_SIGMA,
) -> dict[TectonicType, float]:
    """Tectonic type probabilities by the rule of Haynie et al. (2026).

    Events outside the slab footprint are crustal (upper plate). Inside it,
    the subduction interface probability is the product of

    - the on-interface probability, ``exp(-dz**2 / (2 * on_interface_sigma**2))``
      for vertical distance ``dz`` to the interface (1 for a fixed depth),
    - the subduction slip probability,
      ``exp(-kagan**2 / (2 * subduction_slip_sigma**2))`` for the Kagan
      angle to interface slip (1 without a focal mechanism), and
    - whether the event is no deeper than `SUBDUCTION_MAX_DEPTH_KM`.

    The remaining probability is upper plate for events above the
    interface and intraslab for events below it.

    Parameters
    ----------
    depth_above_slab : float
        Vertical distance in km from the centroid up to the slab surface
        (negative below it), NaN outside the slab footprint.
    depth : float
        Centroid depth in km.
    subduction_kagan_angle : float | None, optional
        Minimum Kagan angle in degrees between the event's mechanism and
        slip on the interface (see `subduction_slip_planes`), None without
        a focal mechanism.
    fixed_depth : bool, optional
        Whether the centroid depth was fixed rather than solved for, so says
        nothing about the distance to the interface.
    on_interface_sigma : float, optional
        Width in km of the on-interface probability.
    subduction_slip_sigma : float, optional
        Width in degrees of the subduction slip probability.

    Returns
    -------
    dict[TectonicType, float]
        Probability of each tectonic type.
    """
    if np.isnan(depth_above_slab):
        return {
            TectonicType.CRUSTAL: 1.0,
            TectonicType.INTERFACE: 0.0,
            TectonicType.SLAB: 0.0,
        }
    on_interface = (
        1.0
        if fixed_depth
        else np.exp(-(depth_above_slab**2) / (2.0 * on_interface_sigma**2))
    )
    subduction_slip = (
        1.0
        if subduction_kagan_angle is None
        else np.exp(-(subduction_kagan_angle**2) / (2.0 * subduction_slip_sigma**2))
    )
    interface = float(
        on_interface * subduction_slip * (depth <= SUBDUCTION_MAX_DEPTH_KM)
    )
    above_slab = depth_above_slab >= 0
    return {
        TectonicType.CRUSTAL: (1.0 - interface) * above_slab,
        TectonicType.INTERFACE: interface,
        TectonicType.SLAB: (1.0 - interface) * (not above_slab),
    }


def depth_temperature(
    depth: npt.ArrayLike, scale: float, depth_rate: float
) -> npt.NDArray[np.float64]:
    """Multiplier on the nodal plane log-odds, ``scale * exp(-depth_rate * depth)``.

    Parameters
    ----------
    depth : npt.ArrayLike
        Centroid depth in km.
    scale : float
        Multiplier at the surface.
    depth_rate : float
        Decay rate per km of depth; positive values make deeper
        predictions less confident.

    Returns
    -------
    npt.NDArray[np.float64]
        The log-odds multiplier at each depth.
    """
    return scale * np.exp(-depth_rate * np.asarray(depth, dtype=float))


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
    temperature_scale : float, optional
        Log-odds multiplier at the surface (see `depth_temperature`).
    temperature_depth_rate : float, optional
        Decay rate of the log-odds multiplier per km of centroid depth.
    """

    def __init__(
        self,
        segments: FaultSegmentIndex,
        slab_model: SlabModel,
        weights: npt.NDArray[np.float64],
        temperature_scale: float = 1.0,
        temperature_depth_rate: float = 0.0,
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
        temperature_scale : float, optional
            Log-odds multiplier at the surface (see `depth_temperature`).
        temperature_depth_rate : float, optional
            Decay rate of the log-odds multiplier per km of centroid depth.
        """
        self.segments = segments
        self.slab_model = slab_model
        self.weights = weights
        self.temperature_scale = temperature_scale
        self.temperature_depth_rate = temperature_depth_rate

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
            model["temperature"]["scale"],
            model["temperature"]["depth_rate"],
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
            Log-odds contribution of each feature, including the depth
            temperature; they sum to the log-odds.
        """
        lat, lon, depth = _parse_centroid(centroid)
        difference = self.segments.plane_misfits(
            lat, lon, depth, nodal_plane_1
        ) - self.segments.plane_misfits(lat, lon, depth, nodal_plane_2)
        temperature = depth_temperature(
            depth, self.temperature_scale, self.temperature_depth_rate
        )
        return dict(
            zip(
                NODAL_PLANE_FEATURE_NAMES,
                (temperature * self.weights * difference).tolist(),
            )
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
        nodal_plane_1: NodalPlane | None = None,
        nodal_plane_2: NodalPlane | None = None,
        fixed_depth: bool = False,
    ) -> dict[TectonicType, float]:
        """Tectonic type probabilities (see the module-level `tectonic_type_probabilities`).

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km). If depth is omitted a default
            is assumed with a warning and treated as fixed.
        nodal_plane_1 : NodalPlane | None, optional
            First nodal plane, None without a focal mechanism.
        nodal_plane_2 : NodalPlane | None, optional
            Second nodal plane, None without a focal mechanism.
        fixed_depth : bool, optional
            Whether the centroid depth was fixed rather than solved for.

        Returns
        -------
        dict[TectonicType, float]
            Probability of each tectonic type.
        """
        fixed_depth = fixed_depth or np.size(centroid) == 2
        lat, lon, depth = _parse_centroid(centroid)
        slab = self.slab_model.query(lat, lon)
        # Both planes describe the same mechanism; using both keeps the
        # result symmetric when catalogue rounding makes them disagree.
        planes = [p for p in (nodal_plane_1, nodal_plane_2) if p is not None]
        subduction_kagan_angle = None
        if planes and slab.present:
            subduction_kagan_angle = min(
                kagan_angle(plane, subduction_plane)
                for plane in planes
                for subduction_plane in subduction_slip_planes(lat, lon, slab)
            )
        return tectonic_type_probabilities(
            slab.depth - depth, depth, subduction_kagan_angle, fixed_depth
        )

    def tectonic_type(
        self,
        centroid: npt.ArrayLike,
        nodal_plane_1: NodalPlane | None = None,
        nodal_plane_2: NodalPlane | None = None,
        fixed_depth: bool = False,
    ) -> TectonicType:
        """Most probable tectonic type (see `tectonic_type_probabilities`).

        Parameters
        ----------
        centroid : npt.ArrayLike
            Centroid as (lat, lon, depth_km). If depth is omitted a default
            is assumed with a warning and treated as fixed.
        nodal_plane_1 : NodalPlane | None, optional
            First nodal plane, None without a focal mechanism.
        nodal_plane_2 : NodalPlane | None, optional
            Second nodal plane, None without a focal mechanism.
        fixed_depth : bool, optional
            Whether the centroid depth was fixed rather than solved for.

        Returns
        -------
        TectonicType
            The tectonic type.
        """
        probabilities = self.tectonic_type_probabilities(
            centroid, nodal_plane_1, nodal_plane_2, fixed_depth
        )
        return max(probabilities, key=probabilities.__getitem__)
