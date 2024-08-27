"""Module for representing the geometry of seismic sources: point sources, fault planes and faults.

This module provides classes and functions for representing fault planes and
faults, along with methods for calculating various properties such as
dimensions, orientation, and coordinate transformations.

Classes
-------
Point:
    A representation of a point source.

Plane:
    A representation of a single plane of a Fault.

Fault:
    A representation of a fault, consisting of one or more Planes.
"""

import dataclasses
from typing import Optional, Protocol

import numpy as np
import scipy as sp

from qcore import coordinates, geo, grid

_KM_TO_M = 1000


@dataclasses.dataclass
class Point:
    """A representation of point source.

    Attributes
    ----------
    bounds : np.ndarray
        The coordinates (NZTM) of the point source.
    length_m : float
        Length used to approximate the point source as a small planar patch (metres).
    strike : float
        The strike angle of the point source in degrees.
    dip : float
        The dip angle of the point source in degrees.
    dip_dir : float
        The dip direction of the point source in degrees.
    """

    # The bounds of a point source are just the coordinates of the point
    bounds: np.ndarray
    # used to approximate point source as a small planar patch (metres).
    length_m: float
    # The usual strike, dip, dip direction, etc cannot be calculated
    # from a point source and so must be provided by the user.
    strike: float
    dip: float
    dip_dir: float

    @staticmethod
    def from_lat_lon_depth(point_coordinates: np.ndarray, **kwargs) -> "Point":
        """Construct a point source from a lat, lon, depth format.

        Parameters
        ----------
        point_coordinates : np.ndarray
            The coordinates of the point in lat, lon, depth format.
        kwargs
            The remaining point source arguments (see the class-level docstring)

        Returns
        -------
        Point
            The Point source representing this geometry.

        """
        return Point(bounds=coordinates.wgs_depth_to_nztm(point_coordinates), **kwargs)

    @property
    def coordinates(self) -> np.ndarray:
        """np.ndarray: The coordinates of the point in (lat, lon, depth) format. Depth is in metres."""
        return coordinates.nztm_to_wgs_depth(self.bounds)

    @property
    def length(self) -> float:
        """float: The length of the approximating planar patch (in kilometres)."""
        return self.length_m / _KM_TO_M

    @property
    def width_m(self) -> float:
        """float: The width of the approximating planar patch (in metres)."""
        return self.length_m

    @property
    def width(self) -> float:
        """float: The width of the approximating planar patch (in kilometres)."""
        return self.width_m / _KM_TO_M

    @property
    def centroid(self) -> np.ndarray:
        """np.ndarray: The centroid of the point source (which is just the point's coordinates)."""
        return self.coordinates

    def fault_coordinates_to_wgs_depth_coordinates(
        self, fault_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert fault-local coordinates to global (lat, lon, depth) coordinates.

        Parameters
        ----------
        fault_coordinates : np.ndarray
            The local fault coordinates

        Returns
        -------
        np.ndarray
            The global coordinates for these fault-local
            coordinates. Because this is a point-source, the global
            coordinates are just the location of the point source.
        """
        return self.coordinates

    def wgs_depth_coordinates_to_fault_coordinates(
        self, wgs_depth_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert global coordinates into fault-local coordinates.

        Parameters
        ----------
        wgs_depth_coordinates : np.ndarray
            The global coordinates to convert.

        Returns
        -------
        np.ndarray
            The fault-local coordinates. Because this is a
            point-source, the local coordinates are simply (1/2, 1/2)
            near the source point and undefined elsewhere.

        Raises
        ------
        ValueError
            If the point is not near the source point.
        """
        nztm_coordinates = coordinates.wgs_depth_to_nztm(wgs_depth_coordinates)
        distance = np.abs(nztm_coordinates - self.bounds).max() / _KM_TO_M
        if distance < self.length / 2 or np.isclose(distance, self.length / 2):
            return np.array([1 / 2, 1 / 2])  # Point is in the centre of the small patch
        raise ValueError("Given global coordinates out of bounds for point source.")


@dataclasses.dataclass
class Plane:
    """A representation of a single plane of a Fault.

    This class represents a single plane of a fault, providing various
    properties and methods for calculating its dimensions, orientation, and
    converting coordinates between different reference frames.

    Attributes
    ----------
    bounds : np.ndarray
        The corners of the fault plane, in NZTM format. The order of the
        corners is given clockwise from the top left (according to strike
        and dip). See the diagram below.

         0            1
          ┌──────────┐
          │          │
          │          │
          │          │
          │          │
          │          │
          │          │
          │          │
          └──────────┘
         3            2
    """

    # Bounds for plane are just the corners
    bounds: np.ndarray

    @staticmethod
    def from_corners(corners: np.ndarray) -> "Plane":
        """Construct a plane point source from its corners.

        Parameters
        ----------
        corners : np.ndarray
            The corners in lat, lon, depth format. Has shape (4 x 3).

        Returns
        -------
        Plane
            The plane source representing this geometry.
        """
        return Plane(coordinates.wgs_depth_to_nztm(corners))

    @property
    def corners(self) -> np.ndarray:
        """np.ndarray: The corners of the fault plane in (lat, lon, depth) format. The corners are the same as in corners_nztm."""
        return coordinates.nztm_to_wgs_depth(self.bounds)

    @property
    def length_m(self) -> float:
        """float: The length of the fault plane (in metres)."""
        return np.linalg.norm(self.bounds[1] - self.bounds[0])

    @property
    def width_m(self) -> float:
        """float: The width of the fault plane (in metres)."""
        return np.linalg.norm(self.bounds[-1] - self.bounds[0])

    @property
    def bottom_m(self) -> float:
        """float: The bottom depth (in metres)."""
        return self.bounds[-1, -1]

    @property
    def top_m(self) -> float:
        """float: The top depth of the fault."""
        return self.bounds[0, -1]

    @property
    def width(self) -> float:
        """float: The width of the fault plane (in kilometres)."""
        return self.width_m / _KM_TO_M

    @property
    def length(self) -> float:
        """float: The length of the fault plane (in kilometres)."""
        return self.length_m / _KM_TO_M

    @property
    def area(self) -> float:
        """float: The area of the plane (in km^2)."""
        return self.length * self.width

    @property
    def projected_width_m(self) -> float:
        """float: The projected width of the fault plane (in metres)."""
        return self.width_m * np.cos(np.radians(self.dip))

    @property
    def projected_width(self) -> float:
        """float: The projected width of the fault plane (in kilometres)."""
        return self.projected_width_m / _KM_TO_M

    @property
    def strike(self) -> float:
        """float: The bearing of the strike direction of the fault (from north; in degrees)."""
        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        strike_direction = self.bounds[1] - self.bounds[0]
        return geo.oriented_bearing_wrt_normal(
            north_direction, strike_direction, up_direction
        )

    @property
    def dip_dir(self) -> float:
        """float: The bearing of the dip direction (from north; in degrees)."""
        if np.isclose(self.dip, 90):
            return 0  # TODO: Is this right for this case?
        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        dip_direction = self.bounds[-1] - self.bounds[0]
        dip_direction[-1] = 0
        return geo.oriented_bearing_wrt_normal(
            north_direction, dip_direction, up_direction
        )

    @property
    def dip(self) -> float:
        """float: The dip angle of the fault."""
        return np.degrees(np.arcsin(np.abs(self.bottom_m - self.top_m) / self.width_m))

    @staticmethod
    def from_centroid_strike_dip(
        centroid: np.ndarray,
        strike: float,
        dip_dir: Optional[float],
        dtop: float,
        dbottom: float,
        length: float,
        projected_width: float,
    ) -> "Plane":
        """Create a fault plane from the centroid, strike, dip_dir, top, bottom, length, and width.

        This is used for older descriptions of sources. Internally
        converts everything to corners so self.strike ~ strike (but
        not exactly due to rounding errors).

        Parameters
        ----------
        centroid : np.ndarray
            The centre of the fault plane in lat, lon coordinates.
        strike : float
            The strike of the fault (in degrees).
        dip_dir : Optional[float]
            The dip direction of the fault (in degrees). If None this is assumed to be strike + 90 degrees.
        top : float
            The top depth of the plane (in km).
        bottom : float
            The bottom depth of the plane (in km).
        length : float
            The length of the fault plane (in km).
        projected_width : float
            The projected width of the fault plane (in km).

        Returns
        -------
        Plane
            The fault plane with centre at `centroid`, and where the
            parameters strike, dip_dir, top, bottom, length and width
            match what is passed to this function.
        """
        corners = grid.grid_corners(
            centroid,
            strike,
            dip_dir if dip_dir is not None else (strike + 90),
            dtop,
            dbottom,
            length,
            projected_width,
        )
        corners[[2, 3]] = corners[[3, 2]]
        return Plane(coordinates.wgs_depth_to_nztm(np.array(corners)))

    @property
    def centroid(self) -> np.ndarray:
        """np.ndarray: The centre of the fault plane."""
        return self.fault_coordinates_to_wgs_depth_coordinates(np.array([1 / 2, 1 / 2]))

    def fault_coordinates_to_wgs_depth_coordinates(
        self, plane_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert plane coordinates to nztm global coordinates.

        Parameters
        ----------
        plane_coordinates : np.ndarray
            Plane coordinates to convert. Plane coordinates are
            2D coordinates (x, y) given for a fault plane (a plane), where x
            represents displacement along the strike, and y
            displacement along the dip (see diagram below). The
            origin for plane coordinates is the centre of the fault.

                          +x
             0 0   ─────────────────>
                ┌─────────────────────┐ │
                │      < strike >     │ │
                │                 ^   │ │
                │                dip  │ │ +y
                │                 v   │ │
                │                     │ │
                └─────────────────────┘ ∨
                                       1,1

        Returns
        -------
        np.ndarray
            An 3d-vector of (lat, lon, depth) transformed coordinates.
        """
        origin = self.bounds[0]
        top_right = self.bounds[1]
        bottom_left = self.bounds[-1]
        frame = np.vstack((top_right - origin, bottom_left - origin))

        return coordinates.nztm_to_wgs_depth(origin + plane_coordinates @ frame)

    def wgs_depth_coordinates_to_fault_coordinates(
        self,
        global_coordinates: np.ndarray,
    ) -> np.ndarray:
        """Convert coordinates (lat, lon, depth) to plane coordinates (x, y).

        See plane_coordinates_to_global_coordinates for a description of
        plane coordinates.

        Parameters
        ----------
        global_coordinates : np.ndarray
            Global coordinates to convert.

        Returns
        -------
        np.ndarray
            The plane coordinates (x, y) representing the position of
            global_coordinates on the fault plane.

        Raises
        ------
        ValueError
            If the given coordinates do not lie in the fault plane.
        """
        strike_direction = self.bounds[1, :2] - self.bounds[0, :2]
        dip_direction = self.bounds[-1, :2] - self.bounds[0, :2]
        offset = (
            coordinates.wgs_depth_to_nztm(global_coordinates[:2]) - self.bounds[0, :2]
        )
        fault_local_coordinates, _, _, _ = np.linalg.lstsq(
            np.array([strike_direction, dip_direction]).T, offset, rcond=None
        )
        if not np.all(
            (
                (fault_local_coordinates > 0)
                | np.isclose(fault_local_coordinates, 0, atol=1e-6)
            )
            & (
                (fault_local_coordinates < 1)
                | np.isclose(fault_local_coordinates, 1, atol=1e-6)
            )
        ):
            raise ValueError("Specified coordinates do not lie in plane")
        return np.clip(fault_local_coordinates, 0, 1)


@dataclasses.dataclass
class Fault:
    """A representation of a fault, consisting of one or more Planes.

    This class represents a fault, which is composed of one or more Planes.
    It provides methods for computing the area of the fault, getting the widths and
    lengths of all fault planes, retrieving all corners of the fault, converting
    global coordinates to fault coordinates, converting fault coordinates to global
    coordinates, generating a random hypocentre location within the fault, and
    computing the expected fault coordinates.

    Attributes
    ----------
    planes : list[Plane]
        A list containing all the Planes that constitute the fault.
    """

    planes: list[Plane]

    @staticmethod
    def from_corners(fault_corners: np.ndarray) -> "Fault":
        """Construct a plane source geometry from the corners of the plane.

        Parameters
        ----------
        corners : np.ndarray
            The corners of the plane in lat, lon, depth format. Has shape (n x 4 x 3).

        Returns
        -------
        Fault
            The fault object representing this geometry.
        """
        return Fault([Plane.from_corners(corners) for corners in fault_corners])

    def area(self) -> float:
        """Compute the area of a fault.

        Returns
        -------
        float
            The area of the fault.
        """
        return self.width * np.sum(self.lengths)

    @property
    def lengths(self) -> np.ndarray:
        """np.ndarray: A numpy array of each plane length (in km)."""
        return np.array([fault.length for fault in self.planes])

    @property
    def length(self) -> float:
        """float: The total length of each fault plane."""
        return self.lengths.sum()

    @property
    def width(self) -> float:
        """The width of the fault.

        Returns
        -------
        float
            The width of the first fault plane (A fault is assumed to
            have planes of constant width).
        """
        return self.planes[0].width

    @property
    def dip_dir(self) -> float:
        """float: The dip direction of the first fault plane (A fault is assumed to have planes of constant dip direction)."""
        return self.planes[0].dip_dir

    @property
    def corners(self) -> np.ndarray:
        """np.ndarray of shape (4n x 3): The corners in (lat, lon, depth) format of each fault plane in the fault, stacked vertically."""
        return np.vstack([plane.corners for plane in self.planes])

    @property
    def bounds(self) -> np.ndarray:
        """np.ndarray of shape (4n x 3): The corners in NZTM format of each fault plane in the fault, stacked vertically."""
        return np.vstack([plane.bounds for plane in self.planes])

    @property
    def centroid(self) -> np.ndarray:
        """np.ndarray: The centre of the fault."""
        return self.fault_coordinates_to_wgs_depth_coordinates(np.array([1 / 2, 1 / 2]))

    def wgs_depth_coordinates_to_fault_coordinates(
        self, global_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert global coordinates in (lat, lon, depth) format to fault coordinates.

        Fault coordinates are a tuple (s, d) where s is the distance
        from the top left, and d the distance from the top of the
        fault (refer to the diagram). The coordinates are normalised
        such that (0, 0) is the top left and (1, 1) the bottom right.

        (0, 0)
          ┌──────────────────────┬──────┐
          │          |           │      │
          │          |           │      │
          │          | d         │      │
          │          |           │      │
          ├----------*           │      │
          │    s     ^           │      │
          │          |           │      │
          │          |           │      │
          │          |           │      │
          └──────────|───────────┴──────┘
                     +                    (1, 1)
                  point: (s, d)

        Parameters
        ----------
        global_coordinates : np.ndarray of shape (1 x 3)
            The global coordinates to convert.

        Returns
        -------
        np.ndarray
            The fault coordinates.

        Raises
        ------
        ValueError
            If the given point does not lie on the fault.

        """
        # the left edges as a cumulative proportion of the fault length (e.g. [0.1, ..., 0.8])
        left_edges = self.lengths.cumsum() / self.length
        left_edges = np.insert(left_edges, 0, 0)
        for i, plane in enumerate(self.planes):
            try:
                plane_coordinates = plane.wgs_depth_coordinates_to_fault_coordinates(
                    global_coordinates
                )
                return np.array([left_edges[i], 0]) + plane_coordinates * np.array(
                    [left_edges[i + 1] - left_edges[i], 1]
                )
            except ValueError:
                continue
        raise ValueError("Given coordinates are not on fault.")

    def fault_coordinates_to_wgs_depth_coordinates(
        self, fault_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert fault coordinates to global coordinates.

        See global_coordinates_to_fault_coordinates for a description of fault
        coordinates.

        Parameters
        ----------
        fault_coordinates : np.ndarray
            The fault coordinates of the point.

        Returns
        -------
        np.ndarray
            The global coordinates (lat, lon, depth) for this point.
        """
        # the right edges as a cumulative proportion of the fault length (e.g. [0.1, ..., 0.8])
        right_edges = self.lengths.cumsum() / self.length
        for i in range(len(self.planes)):
            if fault_coordinates[0] < right_edges[i] or np.isclose(
                fault_coordinates[0], right_edges[i]
            ):
                break
        fault_segment_index = i
        left_proportion = (
            right_edges[fault_segment_index - 1] if fault_segment_index > 0 else 0
        )
        right_proportion = right_edges[fault_segment_index]
        segment_proportion = (fault_coordinates[0] - left_proportion) / (
            right_proportion - left_proportion
        )

        return self.planes[
            fault_segment_index
        ].fault_coordinates_to_wgs_depth_coordinates(
            np.array([segment_proportion, fault_coordinates[1]])
        )


class IsSource(Protocol):
    """Type definition for a source with local coordinates."""

    bounds: np.ndarray

    def fault_coordinates_to_wgs_depth_coordinates(
        self,
        fault_coordinates: np.ndarray,
    ) -> np.ndarray: ...

    def wgs_depth_coordinates_to_fault_coordinates(
        self,
        fault_coordinates: np.ndarray,
    ) -> np.ndarray: ...


def closest_point_between_sources(
    source_a: IsSource, source_b: IsSource
) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest point between two sources that have local coordinates.

    Parameters
    ----------
    source_a : HasCoordinates
        The first source. Must have a two-dimensional fault coordinate system.
    source_b : HasCoordinates
        The first source. Must have a two-dimensional fault coordinate system.

    Raises
    ------
    ValueError
        Raised when we are unable to converge on the closest points between sources.

    Returns
    -------
    source_a_coordinates : np.ndarray
        The source-local coordinates of the closest point on source a.
    source_b_coordinates : np.ndarray
        The source-local coordinates of the closest point on source b.
    """

    def fault_coordinate_distance(fault_coordinates: np.ndarray) -> float:
        source_a_global_coordinates = (
            source_a.fault_coordinates_to_wgs_depth_coordinates(fault_coordinates[:2])
        )
        source_b_global_coordinates = (
            source_b.fault_coordinates_to_wgs_depth_coordinates(fault_coordinates[2:])
        )
        return coordinates.wgs_depth_to_nztm(
            source_a_global_coordinates
        ) - coordinates.wgs_depth_to_nztm(source_b_global_coordinates)

    res = sp.optimize.least_squares(
        fault_coordinate_distance,
        np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]),
        bounds=([0] * 4, [1] * 4),
        gtol=1e-5,
        ftol=1e-5,
        xtol=1e-5,
    )

    if not res.success and res.status != 0:
        raise ValueError(
            f"Optimisation failed to converge for provided sources: {res.message} with x = {res.x}"
        )
    return res.x[:2], res.x[2:]
