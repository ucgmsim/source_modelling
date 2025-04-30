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

import copy
import dataclasses
import itertools
import json
import warnings
from typing import Optional, Self

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy as sp
import shapely

from qcore import coordinates, geo, grid

_KM_TO_M = 1000


@dataclasses.dataclass
class Point:
    """A representation of a point source.

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

    @classmethod
    def from_lat_lon_depth(cls, point_coordinates: np.ndarray, **kwargs) -> Self:
        """Construct a point source from a lat, lon, depth format.

        Parameters
        ----------
        point_coordinates : np.ndarray
            The coordinates of the point in lat, lon, depth format.
        **kwargs : dict
            The remaining point source arguments (see the class-level docstring).

        Returns
        -------
        Point
            The Point source representing this geometry.

        """
        return cls(bounds=coordinates.wgs_depth_to_nztm(point_coordinates), **kwargs)

    @property
    def coordinates(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: The coordinates of the point in (lat, lon, depth) format. Depth is in metres."""
        return coordinates.nztm_to_wgs_depth(self.bounds)

    @property
    def length(self) -> float:  # numpydoc ignore=RT01
        """float: The length of the approximating planar patch (in kilometres)."""
        return self.length_m / _KM_TO_M

    @property
    def width_m(self) -> float:  # numpydoc ignore=RT01
        """float: The width of the approximating planar patch (in metres)."""
        return self.length_m

    @property
    def width(self) -> float:  # numpydoc ignore=RT01
        """float: The width of the approximating planar patch (in kilometres)."""
        return self.width_m / _KM_TO_M

    @property
    def centroid(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: The centroid of the point source (which is just the point's coordinates)."""
        return self.coordinates

    @property
    def geometry(self) -> shapely.Point:  # numpydoc ignore=RT01
        """shapely.Point: A shapely geometry for the point (projected onto the surface)."""
        return shapely.Point(self.bounds)

    @property
    def geojson(self) -> dict:  # numpydoc ignore=RT01
        """dict: A GeoJSON representation of the fault."""
        return shapely.to_geojson(
            shapely.transform(
                self.geometry,
                lambda coords: coordinates.nztm_to_wgs_depth(coords)[:, ::-1],
            )
        )

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

    def rrup_distance(self, point: np.ndarray) -> float:
        """Compute RRup Distance between a fault and a point.

        Parameters
        ----------
        point : np.ndarray
            The point to compute distance to (in lat, lon, depth format)


        Returns
        -------
        float
            The rrup distance (in metres) between the point and the fault geometry.
        """
        return coordinates.distance_between_wgs_depth_coordinates(
            self.coordinates, point
        )

    def rjb_distance(self, point: np.ndarray) -> float:
        """Return the closest projected distance between the fault and the point.

        Parameters
        ----------
        point : np.ndarray
            The point to compute distance to.


        Returns
        -------
        float
            The Rjb distance (in metres) to the point.
        """
        return self.geometry.distance(
            shapely.Point(coordinates.wgs_depth_to_nztm(point))
        )


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

    bounds: npt.NDArray[np.float32]

    def _check_bounds_form_plane(self):
        """Check that the bounds form a plane.

        The checks performed are:
            1. Two points with a minimum depth.
            2. Two points with a maximum depth.
            3. Four points total, spanning exactly a plane. Not a line or a point.

        Raises
        ------
        ValueError
            If the bounds do not form a plane.
        """
        top_mask = np.isclose(self.bounds[:, 2], self.bounds[:, 2].min())
        top = self.bounds[top_mask]
        bottom = self.bounds[~top_mask]
        if (
            np.linalg.matrix_rank(self.bounds) != 3
            or len(top) != 2
            or len(self.bounds) != 4
            or not np.isclose(bottom[0, 2], bottom[1, 2])
        ):
            raise ValueError("Bounds do not form a plane.")

    def __post_init__(self):
        """Check that the plane bounds are consistent and in the correct order.

        These checks are performed to ensure that the corners of the plane
        actually form a plane, and the order of the points is in the order
        required for the plane coordinates to make sense. The plane coordinates
        require the first two points to be the top left and top right corners
        (with respect to strike), and the last two points to be the bottom right
        and bottom left corners (with respect to strike).

        These checks are non-trivial which is why we implement them here instead
        of requiring the caller to verify this themselves.

        Raises
        ------
        ValueError
            If the bounds do not form a plane.
        """
        self._check_bounds_form_plane()

        top_mask = np.isclose(self.bounds[:, 2], self.bounds[:, 2].min())
        top = self.bounds[top_mask]
        bottom = self.bounds[~top_mask]
        # We want to ensure that the bottom and top are pointing in roughly the
        # same direction. To do this, we compare the dot product of the top and
        # bottom vectors. If the dot product is positive then they are pointing
        # in roughly the same direction, if it is negative then they are
        # pointing in opposite directions.
        if np.dot(top[1] - top[0], bottom[0] - bottom[1]) < 0:
            bottom = bottom[::-1]
        orientation = np.linalg.det(
            np.array([top[1] - top[0], bottom[1] - top[0]])[:, :-1]
        )

        # If the orientation is not close to 0 and is negative, then dip
        # direction is to the left of the strike direction, so we reverse
        # the order of the top and bottom corners.
        if not np.isclose(orientation, 0, atol=1e-3) and orientation < 0:
            top = top[::-1]
            bottom = bottom[::-1]
        self.bounds = np.array([top[0], top[1], bottom[0], bottom[1]])

    @classmethod
    def from_corners(cls, corners: np.ndarray) -> Self:
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
        return cls(coordinates.wgs_depth_to_nztm(corners))

    @property
    def corners(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: The corners of the fault plane in (lat, lon, depth) format. The corners are the same as in corners_nztm."""
        return coordinates.nztm_to_wgs_depth(self.bounds)

    @property
    def length_m(self) -> float:  # numpydoc ignore=RT01
        """float: The length of the fault plane (in metres)."""
        return float(np.linalg.norm(self.bounds[1] - self.bounds[0]))

    @property
    def width_m(self) -> float:  # numpydoc ignore=RT01
        """float: The width of the fault plane (in metres)."""
        return float(np.linalg.norm(self.bounds[-1] - self.bounds[0]))

    @property
    def bottom_m(self) -> float:  # numpydoc ignore=RT01
        """float: The bottom depth (in metres)."""
        return self.bounds[-1, -1]

    @property
    def top_m(self) -> float:  # numpydoc ignore=RT01
        """float: The top depth of the fault."""
        return self.bounds[0, -1]

    @property
    def width(self) -> float:  # numpydoc ignore=RT01
        """float: The width of the fault plane (in kilometres)."""
        return self.width_m / _KM_TO_M

    @property
    def length(self) -> float:  # numpydoc ignore=RT01
        """float: The length of the fault plane (in kilometres)."""
        return self.length_m / _KM_TO_M

    @property
    def area(self) -> float:  # numpydoc ignore=RT01
        """float: The area of the plane (in km^2)."""
        return self.length * self.width

    @property
    def projected_width_m(self) -> float:  # numpydoc ignore=RT01
        """float: The projected width of the fault plane (in metres)."""
        return self.width_m * np.cos(np.radians(self.dip))

    @property
    def projected_width(self) -> float:  # numpydoc ignore=RT01
        """float: The projected width of the fault plane (in kilometres)."""
        return self.projected_width_m / _KM_TO_M

    @property
    def strike(self) -> float:  # numpydoc ignore=RT01
        """float: The WGS84 bearing of the strike direction of the fault (from north; in degrees)."""
        return coordinates.nztm_bearing_to_great_circle_bearing(
            self.corners[0, :2], self.length, self.strike_nztm
        )

    @property
    def strike_nztm(self) -> float:  # numpydoc ignore=RT01
        """float: The bearing of the strike direction of the fault (from north; in degrees)."""
        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        strike_direction = self.bounds[1] - self.bounds[0]
        return geo.oriented_bearing_wrt_normal(
            north_direction, strike_direction, up_direction
        )

    @property
    def dip_dir(self) -> float:  # numpydoc ignore=RT01
        """float: The WGS84 bearing of the dip direction of the fault (from north; in degrees)."""
        if np.isclose(self.dip, 90):
            return 0.0

        return coordinates.nztm_bearing_to_great_circle_bearing(
            self.corners[0, :2], self.width, self.dip_dir_nztm
        )

    @property
    def dip_dir_nztm(self) -> float:  # numpydoc ignore=RT01
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
    def dip(self) -> float:  # numpydoc ignore=RT01
        """float: The dip angle of the fault."""
        return np.degrees(np.arcsin(np.abs(self.bottom_m - self.top_m) / self.width_m))

    @property
    def geometry(self) -> shapely.Polygon | shapely.LineString:  # numpydoc ignore=RT01
        """shapely.Polygon or LineString: A shapely geometry for the plane (projected onto the surface).

        Geometry will be a LineString if `dip = 90`."""
        if self.dip == 90:
            return shapely.LineString(self.bounds[:2])
        return shapely.Polygon(self.bounds)

    @property
    def trace(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: The trace of the fault plane on the surface."""
        return self.bounds[:2]

    @property
    def trace_geometry(self) -> shapely.LineString:  # numpydoc ignore=RT01
        """shapely.LineString: The trace of the fault plane on the surface."""
        return shapely.LineString(self.trace)

    @property
    def geojson(self) -> dict:  # numpydoc ignore=RT01
        """dict: A GeoJSON representation of the fault."""
        return shapely.to_geojson(
            shapely.transform(
                self.geometry,
                lambda coords: coordinates.nztm_to_wgs_depth(coords)[:, ::-1],
            )
        )

    @classmethod
    def from_nztm_trace(
        cls,
        trace_points_nztm: np.ndarray[float],
        dtop: float,
        dbottom: float,
        dip: float,
        dip_dir: Optional[float] = None,
        dip_dir_nztm: Optional[float] = None,
    ) -> Self:
        """Create a fault plane from the surface trace, depth parameters,
        dip and dip direction.

        Note: Strike is defined by the dip direction, not the order of the trace points!

        Parameters
        ----------
        trace_points_nztm : np.ndarray
            The surface trace of the fault in NZTM (y, x) format.
            The order of the points is not important, as
            strike, and therefore the correct order, is determined
            from dip direction.
        dtop : float
            The top depth of the plane (in km).
        dbottom : float
            The bottom depth of the plane (in km).
        dip : float
            The dip of the fault plane (degrees).
        dip_dir : float, optional
            Plane dip direction (degrees).
            One of `dip_dir` or `dip_dir_nztm` must be provided.

            Note: If combining multiple planes into a fault using the great
            circle bearing dip direction will cause issues, as the NZTM dip
            direction across the Planes will not be consistent.
        dip_dir_nztm : float, optional
            Plane NZTM dip direction (degrees).
            One of `dip_dir` or `dip_dir_nztm` must be provided.

        Returns
        -------
        Plane
            The fault plane with the given parameters.
        """
        if dip_dir is not None and dip_dir_nztm is not None:
            raise ValueError("Must supply at most one of dip_dir or dip_dir_nztm.")

        if dip_dir_nztm is None and dip_dir is None:
            raise ValueError("Must supply at least one of dip_dir or dip_dir_nztm.")

        if dip_dir_nztm is None and dip_dir is not None:
            if np.isclose(dip, 90) or dip_dir == 0.0:
                dip_dir_nztm = 0
            else:
                width = (dbottom - dtop) / np.sin(np.deg2rad(dip))
                dip_dir_nztm = coordinates.great_circle_bearing_to_nztm_bearing(
                    coordinates.nztm_to_wgs_depth(trace_points_nztm[0]), width, dip_dir
                )

        if trace_points_nztm.shape != (2, 2):
            raise ValueError("Trace points must be a 2x2 array.")

        if np.isclose(dip, 90) and dip_dir_nztm != 0:
            raise ValueError("Dip direction must be 0 for vertical faults.")

        dtop, dbottom = dtop * _KM_TO_M, dbottom * _KM_TO_M

        # Define the trace corners in NZTM coordinates (y, x, depth)
        corners_top = np.column_stack((trace_points_nztm, np.array([dtop, dtop])))

        # Compute remaining corners
        if np.isclose(dip, 90):
            corners_bottom = np.column_stack(
                (trace_points_nztm, np.array([dbottom, dbottom]))
            )
        else:
            dip_dir_nztm_rad = np.deg2rad(dip_dir_nztm)
            proj_width = (dbottom - dtop) / np.tan(np.deg2rad(dip))

            displacement = np.array(
                [
                    np.cos(dip_dir_nztm_rad) * proj_width,
                    np.sin(dip_dir_nztm_rad) * proj_width,
                    dbottom - dtop,
                ]
            )

            # Apply displacement to trace points to get c3 and c4
            corners_bottom = corners_top + displacement

        # Flip the order of the bottom corners
        corners_bottom = corners_bottom[::-1]

        return cls(np.vstack((corners_top, corners_bottom)))

    @classmethod
    def from_centroid_strike_dip(
        cls,
        centroid: np.ndarray,
        dip: float,
        length: float,
        width: float,
        dtop: Optional[float] = None,
        dbottom: Optional[float] = None,
        strike: Optional[float] = None,
        dip_dir: Optional[float] = None,
        strike_nztm: Optional[float] = None,
        dip_dir_nztm: Optional[float] = None,
    ) -> Self:
        """Create a fault plane from the centroid, strike, dip_dir, top, bottom, length, and width.

        This is used for older descriptions of sources. Internally
        converts everything to corners so self.strike ~ strike (but
        not exactly due to rounding errors).

        Parameters
        ----------
        centroid : np.ndarray
            The centre of the fault plane in lat, lon, and optionally depth (km) coordinates.
        dip : float
            The dip of the fault (in degrees).
        length : float
            The length of the fault plane (in km).
        width : float
            The width of the fault plane (in km).
        dtop : Optional[float]
            The top depth of the plane (in km).
        dbottom : Optional[float]
            The bottom depth of the plane (in km).
        strike : Optional[float]
            The WGS84 strike bearing of the fault (in degrees).
        dip_dir : Optional[float]
            The WGS84 dip direction bearing of the fault (in degrees). If None, this is assumed to be strike + 90 degrees.
        strike_nztm : Optional[float]
            The NZTM strike of the fault (in degrees).
        dip_dir_nztm : Optional[float]
            The NZTM dip direction of the fault (in degrees).

        Returns
        -------
        Plane
            The fault plane with center at `centroid`, and where the
            parameters strike, dip_dir, top, bottom, length and width
            match what is passed to this function.

        Note
        ----
        You must supply at least one of `dtop`, `dbottom` or centroid
        depth (i.e. adding a third component to `centroid` for the
        depth). If you provide more than one, they must be consistent
        with each other with respect to dip. These conditions are
        checked by the method.

        Valid combinations of `strike` and `strike_nztm`:
        - Must supply exactly one of `strike` or `strike_nztm`.

        Valid combinations of `dip_dir` and `dip_dir_nztm`:
        - Must supply at most one of `dip_dir` or `dip_dir_nztm`.
        - If neither is supplied, `dip_dir` is assumed to be `strike + 90`.
        """
        # Check that dtop, dbottom and centroid depth are consistent
        if (
            dtop is not None
            and dbottom is not None
            and not np.isclose(dbottom - dtop, np.sin(dip) * width)
        ):
            raise ValueError(
                "Top and bottom depths are not consistent with dip and width parameters."
            )
        elif (
            dtop is not None
            and dbottom is not None
            and not np.isclose(centroid[2], (dtop + dbottom) / 2)
        ):
            raise ValueError(
                "Top and bottom depths are not consistent with centroid depth."
            )
        elif dtop is None and dbottom is None and len(centroid) == 2:
            raise ValueError(
                "At least one of top, bottom, or centroid depth must be given."
            )
        elif (
            dtop is not None
            and len(centroid) == 3
            and not np.isclose(centroid[2] - dtop, np.sin(np.radians(dip)) * width / 2)
        ):
            raise ValueError("Centroid depth and dtop are inconsistent.")
        elif (
            dbottom is not None
            and len(centroid) == 3
            and not np.isclose(
                dbottom - centroid[2], np.sin(np.radians(dip)) * width / 2
            )
        ):
            raise ValueError("Centroid depth and dbottom are inconsistent.")
        elif not ((strike is None) ^ (strike_nztm is None)):
            raise ValueError("Must supply exactly one of strike or NZTM strike.")
        elif dip_dir is not None and dip_dir_nztm is not None:
            raise ValueError(
                "Must supply at most one of dip direction or NZTM dip direction."
            )

        if strike_nztm is None:
            strike = coordinates.great_circle_bearing_to_nztm_bearing(
                centroid[:2],
                length / 2,
                strike,
            )
        else:
            strike = strike_nztm

        if dip_dir_nztm is None and dip_dir is not None:
            dip_dir = coordinates.great_circle_bearing_to_nztm_bearing(
                centroid[:2],
                width / 2,
                dip_dir,
            )
        elif dip_dir is None and dip_dir_nztm is not None:
            dip_dir = dip_dir_nztm
        else:
            dip_dir = strike + 90

        # Values are definitely consistent, now infer dtop and dbottom
        # based on what we have.
        if dtop is None and dbottom is None:
            dtop = centroid[2] - width / 2 * np.sin(np.radians(dip))
            dbottom = centroid[2] + width / 2 * np.sin(np.radians(dip))
        elif dtop is not None:
            dbottom = dtop + width * np.sin(np.radians(dip))
        elif dbottom is not None:
            dtop = dbottom - width * np.sin(np.radians(dip))

        projected_width = width * np.cos(np.radians(dip))
        corners = grid.grid_corners(
            centroid[:2],
            strike,
            dip_dir,
            dtop,
            dbottom,
            length,
            projected_width,
        )
        return cls(coordinates.wgs_depth_to_nztm(np.array(corners)))

    @property
    def centroid(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: The center of the fault plane."""
        return self.fault_coordinates_to_wgs_depth_coordinates(np.array([1 / 2, 1 / 2]))

    def fault_coordinates_to_wgs_depth_coordinates(
        self, plane_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert plane coordinates to NZTM global coordinates.

        Parameters
        ----------
        plane_coordinates : np.ndarray
            Plane coordinates to convert. Plane coordinates are
            2D coordinates (x, y) given for a fault plane (a plane), where x
            represents displacement along the strike, and y
            displacement along the dip (see diagram below). The
            origin for plane coordinates is the center of the fault.

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
            A 3d-vector of (lat, lon, depth) transformed coordinates.
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

        Notes
        -----
        While not passing depth information is supported, depth information
        *greatly* improves the accuracy of the estimation. No guarantees
        are made about the accuracy of the inversion if you do not pass
        depth information.
        """
        coordinate_length = (
            3 if global_coordinates.shape[-1] == 3 or self.dip == 90 else 2
        )
        strike_direction = (
            self.bounds[1, :coordinate_length] - self.bounds[0, :coordinate_length]
        )
        dip_direction = (
            self.bounds[-1, :coordinate_length] - self.bounds[0, :coordinate_length]
        )
        offset = (
            coordinates.wgs_depth_to_nztm(global_coordinates[:coordinate_length])
            - self.bounds[0, :coordinate_length]
        )
        fault_local_coordinates, _, _, _ = np.linalg.lstsq(
            np.array([strike_direction, dip_direction]).T, offset, rcond=None
        )
        tolerance = 1e-6 if coordinate_length == 3 else 1e-4
        if not np.all(
            (
                (fault_local_coordinates > 0)
                | np.isclose(fault_local_coordinates, 0, atol=tolerance)
            )
            & (
                (fault_local_coordinates < 1)
                | np.isclose(fault_local_coordinates, 1, atol=tolerance)
            )
        ):
            raise ValueError("Specified coordinates do not lie in plane")
        return np.clip(fault_local_coordinates, 0, 1)

    def rrup_distance(self, point: np.ndarray) -> float:
        """Compute RRup Distance between a fault and a point.

        Parameters
        ----------
        point : np.ndarray
            The point to compute distance to (in lat, lon, depth format)

        Returns
        -------
        float
            The rrup distance (in metres) between the point and the fault geometry.
        """
        point_nztm = coordinates.wgs_depth_to_nztm(point)
        frame = np.array(
            [self.bounds[1] - self.bounds[0], self.bounds[-1] - self.bounds[0]]
        )
        local_coords, _, _, _ = np.linalg.lstsq(
            frame.T,
            point_nztm - self.bounds[0],
            rcond=None,
        )
        projected_point = local_coords @ frame + self.bounds[0]
        out_of_plane_distance = np.linalg.norm(point_nztm - projected_point)
        if np.allclose(local_coords, np.clip(local_coords, 0, 1)):
            # solution lies in fault, ergo just return projected distance
            return float(out_of_plane_distance)

        in_plane_distance = min(
            geo.point_to_segment_distance(
                projected_point, self.bounds[i], self.bounds[(i + 1) % 4]
            )
            for i in range(4)
        )

        return np.sqrt(in_plane_distance**2 + out_of_plane_distance**2)

    def rjb_distance(self, point: np.ndarray) -> float:
        """Return the closest projected distance between the fault and the point.

        Parameters
        ----------
        point : np.ndarray
            The point to compute distance to.


        Returns
        -------
        float
            The Rjb distance (in metres) to the point.
        """
        return self.geometry.distance(
            shapely.Point(coordinates.wgs_depth_to_nztm(point))
        )


@dataclasses.dataclass
class Fault:
    """A representation of a fault, consisting of one or more Planes.

    This class represents a fault, which is composed of one or more Planes.
    It provides methods for computing the area of the fault, getting the widths and
    lengths of all fault planes, retrieving all corners of the fault, converting
    global coordinates to fault coordinates, converting fault coordinates to global
    coordinates, generating a random hypocenter location within the fault, and
    computing the expected fault coordinates.

    Attributes
    ----------
    planes : list[Plane]
        A list containing all the Planes that constitute the fault.
    """

    planes: list[Plane]

    def _basic_consistency_checks(self):
        """Check that the planes are consistent in dip and width.

        Raises
        ------
        ValueError
            If the planes are not consistent in dip or width."""
        # Planes can only have one dip, dip direction, and width.
        for plane in self.planes[1:]:
            if not (np.isclose(plane.width, self.planes[0].width)):
                raise ValueError("Fault must have constant width.")
            if not (np.isclose(plane.dip, self.planes[0].dip, atol=0.1)):
                raise ValueError(
                    f"Fault must have a constant dip (plane dip = {plane.dip}, fault dip is {self.planes[0].dip})."
                )

    def _validate_fault_plane_connectivity(self, connection_graph: nx.DiGraph):
        """Validate that the fault planes are connected in a line.

        This function checks that the fault planes form a connected line. It
        ensures that the planes are connected end-to-end along the strike
        direction, with exactly one start node and one end node. This is the
        case if `connection_graph` satisfies the following conditions:

        1. It is connected (weakly, i.e. ignoring the direction of the edges).
        2. There is exactly one node with an in-degree of 0 and out-degree of 1.
           - This is the first node by strike, as nothing points into it.

           plane 0 -> plane 1

        3. There is exactly one node with an in-degree of 1 and out-degree of 0.
           - This is the last node by strike, as nothing points out of it.

           plane (n - 1) -> plane n

        4. Every other node has an in-degree of 1 and out-degree of 1.

           plane (i - 1) -> plane i -> plane (i + 1)


        Parameters
        ----------
        connection_graph : nx.DiGraph[int]
            A directed graph representing the connectivity of the fault planes.

        Raises
        ------
        ValueError
            If the fault planes are not connected in a line.
        """
        if not nx.is_weakly_connected(connection_graph):
            raise ValueError("Fault planes must be connected.")

        start_nodes = 0
        end_nodes = 0
        for node in connection_graph.nodes:
            in_degree = connection_graph.in_degree(node)
            out_degree = connection_graph.out_degree(node)

            if in_degree == 0 and out_degree == 1:
                start_nodes += 1
            elif in_degree == 1 and out_degree == 0:
                end_nodes += 1
            elif in_degree == 1 and out_degree == 1:
                continue
            else:
                raise ValueError("Fault planes must be connected in a line.")

        if not (start_nodes == 1 and end_nodes == 1):
            raise ValueError("Fault planes must be connected in a line.")

    def __post_init__(self) -> None:
        """Ensure that planes are ordered along strike end-to-end.

        The intention of this method is to ensure that the planes given are:

        1. Consistent in dip direction and width. You cannot have a fault with
        differing dip directions or widths because this would imply that the
        fault is not a single fault.

        2. Connected end-to-end along strike. This is to ensure that the fault
        is a single fault and not a series of disconnected faults, or multiple
        faults that splay off from one another.

        The above two checks ensure that the corners of planes line up in such a
        way that the fault can have a fault coordinate system applied to it.
        This is essential for computing the shortest distances between faults.
        """
        # The trivial case where the number of planes is one should be ignored
        if len(self.planes) <= 1:
            return

        self._basic_consistency_checks()
        # We need to check that the plane given is a series of connected planes
        # that meet end-to-end. There are then two cases:
        # 1. The fault splays or is disconnected. We should raise a ValueError in this case.
        # 2. The fault is a line, but just lacks the correct order. Then we should re-order it to be a connected line.
        # The "correct" order for a fault is one that is oriented with each plane end-to-end along strike, that is
        #
        # ">" strike order
        # ┌─────>──┬─────>──┬────>──┐
        # │        │        │       │
        # │        │        │       │
        # │        │        │       │
        # │   0    │   1    │   2   │
        # │        │        │       │
        # │        │        │       │
        # │        │        │       │
        # └────────┴────────┴───────┘
        points_into_relation: dict[int, list[int]] = {
            i: [] for i in range(len(self.planes))
        }

        for i, j in itertools.product(range(len(self.planes)), repeat=2):
            if i == j:
                continue
            # A plane i points into a plane j if the right-edge (by strike)
            # of plane i is close to the left-edge (by strike) of plane j.
            if np.linalg.norm(self.planes[i].bounds[1] - self.planes[j].bounds[0]) < 10:
                points_into_relation[j].append(i)  # Plane i points into plane j

        # This relation can now be used to identify if the list of planes given is a line.
        points_into_graph: nx.DiGraph = nx.from_dict_of_lists(
            points_into_relation, create_using=nx.DiGraph
        )
        try:
            self._validate_fault_plane_connectivity(points_into_graph)
        except ValueError:
            # Sometimes, faults with small segments can confuse the
            # connectivity check, so we can apply a transitive
            # reduction and then do the check again (with a warning).
            warnings.warn(
                "Fault planes are connected, but not in a line."
                " This can occur with very short segments."
                " Trying to safely reduce the connectivity graph."
                " Check the output fault carefully."
            )
            self._validate_fault_plane_connectivity(
                nx.transitive_reduction(points_into_graph)
            )

        # Now that we have established the planes line up correctly, we can
        # obtain the line in question with a topological sort, which will
        # return the correct order of planes
        self.planes = [
            self.planes[i]
            for i in reversed(list(nx.topological_sort(points_into_graph)))
        ]
        # Now check that the plane edges line up.
        for i in range(1, len(self.planes)):
            dip_dir = self.planes[i].bounds[-1] - self.planes[i].bounds[0]
            dip_dir_prev = self.planes[i - 1].bounds[-1] - self.planes[i - 1].bounds[0]
            diff = sp.spatial.distance.cosine(dip_dir, dip_dir_prev)
            if not np.isclose(diff, 0, atol=1e-4):
                raise ValueError(
                    f"Fault must have a constant dip direction, plane {i} has dip direction {dip_dir}, but plane {i - 1} hase dip direction {dip_dir_prev}"
                )

    @property
    def dip(self) -> float:  # numpydoc ignore=RT01
        """float: The dip angle of the fault."""
        return self.planes[0].dip

    @property
    def dip_dir(self) -> float:  # numpydoc ignore=RT01
        """float: The dip direction of the fault."""
        return self.planes[0].dip_dir

    @property
    def dip_dir_nztm(self) -> float:  # numpydoc ignore=RT01
        """float: The dip direction of the fault."""
        return self.planes[0].dip_dir_nztm

    @classmethod
    def from_corners(cls, fault_corners: np.ndarray) -> Self:
        """Construct a plane source geometry from the corners of the plane.

        Parameters
        ----------
        fault_corners : np.ndarray
            The corners of the plane in lat, lon, depth format. Has shape (n x 4 x 3).

        Returns
        -------
        Fault
            The fault object representing this geometry.
        """
        return cls([Plane.from_corners(corners) for corners in fault_corners])

    def area(self) -> float:
        """Compute the area of a fault.

        Returns
        -------
        float
            The area of the fault.
        """
        return self.width * np.sum(self.lengths)

    @property
    def lengths(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: A numpy array of each plane length (in km)."""
        return np.array([fault.length for fault in self.planes])

    @property
    def length(self) -> float:  # numpydoc ignore=RT01
        """float: The total length of each fault plane."""
        return self.lengths.sum()

    @property
    def width(self) -> float:  # numpydoc ignore=RT01
        """The width of the fault.

        Returns
        -------
        float
            The width of the first fault plane (A fault is assumed to
            have planes of constant width).
        """
        return self.planes[0].width

    @property
    def corners(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray of shape (4n x 3): The corners in (lat, lon, depth) format of each fault plane in the fault, stacked vertically."""
        return np.vstack([plane.corners for plane in self.planes])

    @property
    def bounds(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray of shape (4n x 3): The corners in NZTM format of each fault plane in the fault, stacked vertically."""
        return np.vstack([plane.bounds for plane in self.planes])

    @property
    def centroid(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: The center of the fault."""
        return self.fault_coordinates_to_wgs_depth_coordinates(np.array([1 / 2, 1 / 2]))

    @property
    def geometry(self) -> shapely.Polygon | shapely.LineString:  # numpydoc ignore=RT01
        """shapely.Polygon or LineString: A shapely geometry for the fault (projected onto the surface).

        Geometry will be LineString if `dip = 90`.
        """
        return shapely.normalize(
            shapely.union_all([plane.geometry for plane in self.planes])
        )

    @property
    def trace(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: The trace of the fault plane on the surface."""
        return np.vstack([plane.trace for plane in self.planes])

    @property
    def trace_geometry(self) -> shapely.LineString:  # numpydoc ignore=RT01
        """shapely.LineString: The trace of the fault plane on the surface."""
        return shapely.LineString(self.trace)

    def wgs_depth_coordinates_to_fault_coordinates(
        self, global_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert global coordinates in (lat, lon, depth) format to fault coordinates.

        Fault coordinates are a tuple (s, d) where s is the distance
        from the top left, and d the distance from the top of the
        fault (refer to the diagram). The coordinates are normalized
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

    @property
    def geojson(self) -> dict:  # numpydoc ignore=RT01
        """dict: A GeoJSON representation of the fault."""
        return shapely.to_geojson(
            shapely.transform(
                self.geometry,
                lambda coords: coordinates.nztm_to_wgs_depth(coords)[:, ::-1],
            )
        )

    def rrup_distance(self, point: np.ndarray) -> float:
        """Compute RRup Distance between a fault and a point.

        Parameters
        ----------
        point : np.ndarray
            The point to compute distance to (in lat, lon, depth format)

        Returns
        -------
        float
            The rrup distance (in metres) between the point and the fault geometry.
        """

        return min(plane.rrup_distance(point) for plane in self.planes)

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

    def rjb_distance(self, point: np.ndarray) -> float:
        """Return the closest projected distance between the fault and the point.

        Parameters
        ----------
        point : np.ndarray
            The point to compute distance to.


        Returns
        -------
        float
            The Rjb distance (in metres) to the point.
        """
        return self.geometry.distance(
            shapely.Point(coordinates.wgs_depth_to_nztm(point))
        )


IsSource = Plane | Fault | Point


def sources_as_geojson_features(sources: list[IsSource]) -> str:
    """Convert a list of sources to a GeoJSON FeatureCollection.

    Parameters
    ----------
    sources : list[IsSource]
            The sources to convert.

    Returns
    -------
    str
            The GeoJSON FeatureCollection representation of the sources.
    """
    geometries = [json.loads(source.geojson) for source in sources]
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {"id": i},
                }
                for i, geometry in enumerate(geometries)
            ],
        }
    )


def closest_point_between_sources(
    source_a: IsSource, source_b: IsSource
) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest point between two sources that have local coordinates.

    Parameters
    ----------
    source_a : IsSource
        The first source. Must have a two-dimensional fault coordinate system.
    source_b : IsSource
        The second source. Must have a two-dimensional fault coordinate system.

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

    def fault_coordinate_distance(
        fault_coordinates: np.ndarray,
    ) -> float:  # numpydoc ignore=GL08
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


def absorb_planes(plane: Plane, other: Plane) -> Plane:
    """Return a plane containing the left-most and right-most corners of two planes.

    Parameters
    ----------
    plane : Plane
        The first plane.
    other : Plane
        The second plane.

    Returns
    -------
    Plane
        A plane containing the left-most and right-most corners of the two planes.
    """
    return Plane(np.vstack((plane.bounds[[0, -1]], other.bounds[[1, 2]])))


def simplify_fault(fault: Fault, length_tolerance: float) -> Fault:
    """Simplify a fault geometry to remove all segments with length less than a tolerance.

    Parameters
    ----------
    fault : Fault
        The fault to simplify
    length_tolerance : float
        The tolerated length (in km). The returned fault will have no
        segments with length less than this value.


    Returns
    -------
    Fault
        The simplified fault geometry.
    """
    planes = copy.deepcopy(fault.planes)
    if len(planes) == 1:
        return Fault(planes)

    while len(planes) > 1:
        lengths = [plane.length for plane in planes]
        if all(length >= length_tolerance for length in lengths):
            break

        min_length_index = np.argmin(lengths)
        if min_length_index == len(planes) - 1:
            plane = planes.pop()
            other = planes.pop()
            planes.append(absorb_planes(other, plane))
        elif min_length_index == 0:
            other = planes.pop(1)
            plane = planes.pop(0)
            planes.insert(0, absorb_planes(plane, other))
        else:
            left = planes[min_length_index - 1]
            right = planes[min_length_index + 1]
            plane = planes[min_length_index]
            # Test which plane to absorb into by total deviation.
            # Calculated by finding the perpendicular distance between
            # the two new edges.
            if geo.point_to_segment_distance(
                plane.bounds[0], left.bounds[0], right.bounds[0]
            ) < geo.point_to_segment_distance(
                plane.bounds[-1], left.bounds[-1], right.bounds[-1]
            ):
                planes = (
                    planes[: min_length_index - 1]
                    + [absorb_planes(left, plane)]
                    + planes[min_length_index + 1 :]
                )
            else:
                planes = (
                    planes[:min_length_index]
                    + [absorb_planes(plane, right)]
                    + planes[min_length_index + 2 :]
                )

    return Fault(planes)
