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
from typing import NamedTuple, Optional, Self

import networkx as nx
import numpy as np
import numpy.typing as npt
import pyproj
import scipy as sp
import shapely

from qcore import coordinates, geo
from qcore.coordinates import SphericalProjection

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

    centroid: np.ndarray
    # used to approximate point source as a small planar patch (metres).
    length_m: float
    # The usual strike, dip, dip direction, etc cannot be calculated
    # from a point source and so must be provided by the user.
    strike: float
    dip: float
    dip_dir: float

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
    def geometry(self) -> shapely.Point:  # numpydoc ignore=RT01
        """shapely.Point: A shapely geometry for the point (projected onto the surface)."""
        return shapely.Point(self.centroid)

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
        local_coordinates = self._projection(wgs_depth_coordinates)
        distance = np.abs(local_coordinates - self.bounds).max() / _KM_TO_M
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
        point_local = self._projection(*point.T)
        return np.linalg.norm(point_local - self._projection(*self.centroid))

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
        projection = self._projection
        return shapely.transform(self.geometry, lambda xx: projection(*xx.T)).distance(
            shapely.transform(shapely.Point(point), lambda xx: projection(*xx.T))
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

    centroid: npt.NDArray[np.float64]
    strike: float
    dip: float
    dip_dir: Optional[float]
    length_m: float
    width_m: float

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
        # find the centroid of the plane
        geod = pyproj.Geod(ellps="sphere")
        mid_depth = np.mean(corners[:, 2])
        lat0, lon0, _ = corners[0]
        sum_x: float = 0
        sum_y: float = 0
        for lat, lon, _ in corners:
            # Convert to Cartesian offset
            az12, _, dist = geod.inv(lon0, lat0, lon, lat)
            x = dist * np.cos(np.radians(az12))
            y = dist * np.sin(np.radians(az12))
            sum_x += x
            sum_y += y

        avg_x = sum_x / 3
        avg_y = sum_y / 3

        avg_dist = np.hypot(avg_x, avg_y)
        avg_az = np.degrees(np.arctan2(avg_y, avg_x))

        lon_mid, lat_mid, _ = geod.fwd(lon0, lat0, avg_az, avg_dist)
        centroid = np.array([lat_mid, lon_mid, mid_depth])

        # find the strike and dip

        projection = coordinates.SphericalProjection(
            mlon=centroid[1], mlat=centroid[0], mrot=0.0
        )
        corners_local = projection(*corners.T)

        top_mask = corners_local[:, 2] == corners_local[:, 2].min()

        top = corners_local[top_mask]
        bottom = corners_local[~top_mask]
        # We want to ensure that the bottom and top are pointing in roughly the
        # same direction. To do this, we compare the dot product of the top and
        # bottom vectors. If the dot product is positive then they are pointing
        # in roughly the same direction, if it is negative then they are
        # pointing in opposite directions.
        # if np.dot(top[1] - top[0], bottom[0] - bottom[1]) < 0:
        #     bottom = bottom[::-1]
        # orientation = np.linalg.det(
        #     np.array([top[1] - top[0], bottom[1] - top[0]])[:, :-1]
        # )

        # # If the orientation is not close to 0 and is negative, then dip
        # # direction is to the left of the strike direction, so we reverse
        # # the order of the top and bottom corners.
        # if not np.isclose(orientation, 0, atol=1e-3) and orientation < 0:
        #     top = top[::-1]
        #     bottom = bottom[::-1]

        # if (
        #     np.linalg.matrix_rank(corners_local) != 3
        #     or len(top) != 2
        #     or len(corners_local) != 4
        #     or not np.isclose(bottom[0, 2], bottom[1, 2])
        # ):
        #     raise ValueError("Corners do not form a plane.")

        length = np.linalg.norm(top[1] - top[0])
        width = np.linalg.norm(bottom[1] - top[0])

        strike_direction = top[1] - top[0]
        strike = np.degrees(np.arctan2(strike_direction[0], -strike_direction[1])) % 360
        dip_direction = bottom[1] - top[0]
        dip_dir = np.degrees(np.arctan2(dip_direction[0], -dip_direction[1])) % 360

        depth = bottom[0, 2] - top[0, 2]
        dip = np.degrees(np.arcsin(depth / width))

        return cls(
            centroid=centroid,
            strike=strike,
            dip=dip,
            dip_dir=dip_dir,
            length_m=length,
            width_m=width,
        )

    @property
    def corners(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: The corners of the fault plane in (lat, lon, depth) format. The corners are the same as in corners_nztm."""
        return self._projection.inverse(
            self.bounds[:, 0], self.bounds[:, 1], self.bounds[:, 2]
        )

    @property
    def bottom_m(self) -> float:  # numpydoc ignore=RT01
        """float: The bottom depth (in metres)."""
        return self.centroid[2] + np.sin(np.radians(self.dip)) * self.width_m / 2

    @property
    def top_m(self) -> float:  # numpydoc ignore=RT01
        """float: The top depth of the fault."""
        return self.centroid[2] - np.sin(np.radians(self.dip)) * self.width_m / 2

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
    def geometry(self) -> shapely.Polygon | shapely.LineString:  # numpydoc ignore=RT01
        """shapely.Polygon or LineString: A shapely geometry for the plane (projected onto the surface).

        Geometry will be a LineString if `dip = 90`."""
        if self.dip == 90:
            return shapely.LineString(self.corners[:2])
        return shapely.Polygon(self.corners)

    @property
    def geojson(self) -> dict:  # numpydoc ignore=RT01
        """dict: A GeoJSON representation of the fault."""
        return shapely.to_geojson(self.geometry)

    @classmethod
    def from_trace(
        cls,
        trace_points: np.ndarray[float],
        dtop: float,
        dbottom: float,
        dip: float,
        dip_dir: float | None,
    ) -> Self:
        """Create a fault plane from the surface trace, depth parameters,
        dip and dip direction.

        Note: Strike is defined by the dip direction, not the order of the trace points!

        Parameters
        ----------
        trace_points : np.ndarray
            The surface trace of the fault. The order of the points is
            not important, as strike, and therefore the correct order,
            is determined from dip direction.
        dtop : float
            The top depth of the plane (in km).
        dbottom : float
            The bottom depth of the plane (in km).
        dip : float
            The dip of the fault plane (degrees).
        dip_dir : float, optional
            Plane dip direction (degrees). Must be None if dip = 90.

        Returns
        -------
        Plane
            The fault plane with the given parameters.
        """
        if (dip_dir == None) != (dip == 90):
            raise ValueError(
                "Must supply dip direction, or use dip_dir = None if dip = 90."
            )

        width = (dbottom - dtop) / np.sin(np.radians(dip))
        geod = pyproj.Geod(ellps="sphere")
        # The `npts` method returns equally spaced points between two endpoints.
        # Setting the number of points to 1 gives us the midpoint between the two endpoints.
        top_mid_lon, top_mid_lat = geod.npts(
            trace_points[0, 1],
            trace_points[0, 0],
            trace_points[1, 1],
            trace_points[1, 0],
            1,
        )[0]  # npts returns a list, so we take the first element

        if dip_dir is None:
            centroid_lon, centroid_lat = top_mid_lat, top_mid_lon
        else:
            centroid_lon, centroid_lat, _ = geod.fwd(
                trace_points[0, 1], trace_points[0, 0], dip_dir, width / 2
            )
        dtop, dbottom = dtop * _KM_TO_M, dbottom * _KM_TO_M
        centroid = np.array([centroid_lat, centroid_lon, (dbottom + dtop) / 2])
        projection = coordinates.SphericalProjection(
            mlon=centroid_lon, mlat=centroid_lat, mrot=0.0
        )
        # Convert the trace points to the local projection
        trace_points_local = projection(*trace_points.T)

        # Calculate the strike direction
        strike_direction = trace_points_local[1] - trace_points_local[0]
        strike = np.degrees(np.arctan2(strike_direction[0], -strike_direction[1])) % 360
        length_m = np.linalg.norm(trace_points_local[1] - trace_points_local[0])
        return cls(
            centroid=centroid,
            width_m=width * _KM_TO_M,
            length_m=length_m,
            strike=strike,
            dip=dip,
            dip_dir=dip_dir,
        )

    @classmethod
    def from_centroid_strike_dip(
        cls,
        centroid: np.ndarray,
        strike: float,
        dip: float,
        length: float,
        width: float,
        dtop: Optional[float] = None,
        dbottom: Optional[float] = None,
        dip_dir: Optional[float] = None,
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

        if dip_dir is None:
            dip_dir = strike + 90

        if centroid.shape == (2,):
            centroid = np.append(centroid, (dbottom - dtop) / 2 * _KM_TO_M)

        return cls(
            strike=strike,
            dip=dip,
            dip_dir=dip_dir,
            length=length,
            width=width,
            centroid=centroid,
        )

    @property
    def _projection(self) -> SphericalProjection:
        """SphericalProjection: The coordinate projection system for the plane."""
        return SphericalProjection(
            mlon=self.centroid[1], mlat=self.centroid[0], mrot=0.0
        )

    @property
    def bounds(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: The corners of the fault plane in local spherical coordinates."""
        projection = self._projection

        centroid = projection(self.centroid[0], self.centroid[1]).ravel()

        strike_direction = np.array(
            [
                np.sin(np.radians(self.strike)),
                -np.cos(
                    np.radians(self.strike)
                ),  # north is down in the spherical projection
            ]
        )
        dtop = self.top_m
        dbottom = self.bottom_m
        if self.dip_dir is None:
            dip_direction = np.zeros_like(strike_direction)
        else:
            dip_direction = np.array(
                [
                    np.sin(np.radians(self.dip_dir)),
                    -np.cos(np.radians(self.dip_dir)),
                ]
            )

        # Create the frame of reference
        basis = (
            np.array(
                [
                    [-1, -1],  # origin
                    [1, -1],  # x_upper
                    [1, 1],
                    [-1, 1],
                ],
                dtype=np.float64,
            )
            * np.array([self.length_m, self.projected_width_m])
            * 0.5
        )

        corners = centroid + basis @ np.array([strike_direction, dip_direction])
        return np.c_[corners, np.array([dtop, dtop, dbottom, dbottom])]

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
        bounds = self.bounds
        origin = bounds[0]
        top_right = bounds[1]
        bottom_left = bounds[-1]
        frame = np.vstack((top_right - origin, bottom_left - origin))
        spherical_coordinates = origin + plane_coordinates @ frame
        if plane_coordinates.ndim == 1:
            return self._projection.inverse(
                spherical_coordinates[0],
                spherical_coordinates[1],
                spherical_coordinates[2],
            ).reshape((3,))
        return self._projection.inverse(
            spherical_coordinates[:, 0],
            spherical_coordinates[:, 1],
            spherical_coordinates[:, 2],
        ).reshape((-1, 3))

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
        bounds = self.bounds
        coordinate_length = (
            3 if global_coordinates.shape[-1] == 3 or self.dip == 90 else 2
        )
        strike_direction = bounds[1, :coordinate_length] - bounds[0, :coordinate_length]
        dip_direction = bounds[-1, :coordinate_length] - bounds[0, :coordinate_length]
        local_coordinates = self._projection(*global_coordinates[:coordinate_length].T)

        offset = local_coordinates - bounds[0, :coordinate_length]
        frame = np.array([strike_direction, dip_direction])
        fault_local_coordinates, _, _, _ = np.linalg.lstsq(
            frame.T, offset.T, rcond=None
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
        point_local = self._projection(*point)
        bounds = self.bounds
        frame = np.array([bounds[1] - bounds[0], bounds[-1] - bounds[0]])
        local_coords, _, _, _ = np.linalg.lstsq(
            frame.T,
            point_local - bounds[0],
            rcond=None,
        )
        projected_point = local_coords @ frame + bounds[0]
        out_of_plane_distance = np.linalg.norm(point_local - projected_point)
        if np.allclose(local_coords, np.clip(local_coords, 0, 1)):
            # solution lies in fault, ergo just return projected distance
            return float(out_of_plane_distance)

        in_plane_distance = min(
            geo.point_to_segment_distance(
                projected_point, bounds[i], bounds[(i + 1) % 4]
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
        projection = self._projection
        return shapely.transform(self.geometry, lambda xx: projection(*xx.T)).distance(
            shapely.transform(shapely.Point(point), lambda xx: projection(*xx.T))
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

    @classmethod
    def from_trace(cls, trace_points: np.ndarray, **kwargs) -> Self:
        """Create a fault from a trace.

        Parameters
        ----------
        trace_points : np.ndarray
            The surface trace of the fault. The order of the points is
            not important, as strike, and therefore the correct order,
            is determined by the dip direction.
        kwargs : dict
            Additional arguments to pass to `Plane.from_trace`.

        Returns
        -------
        Fault
            The fault object representing this geometry.
        """
        return cls(
            [
                Plane.from_trace(trace_points[i : i + 2], **kwargs)
                for i in range(len(trace_points) - 1)
            ]
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


class SourceBounds(NamedTuple):
    """A named tuple representing the bounds of a source.

    Attributes
    ----------
    x_min : float
        The minimum x-coordinate of the source.
    y_min : float
        The minimum y-coordinate of the source.
    x_max : float
        The maximum x-coordinate of the source.
    y_max : float
        The maximum y-coordinate of the source.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float


def closest_point_between_sources(
    source_a: IsSource,
    source_b: IsSource,
    source_a_bounds: SourceBounds = SourceBounds(x_min=0, x_max=1, y_min=0, y_max=1),
    source_b_bounds: SourceBounds = SourceBounds(x_min=0, x_max=1, y_min=0, y_max=1),
) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest point between two sources that have local coordinates.

    Parameters
    ----------
    source_a : IsSource
        The first source. Must have a two-dimensional fault coordinate system.
    source_b : IsSource
        The second source. Must have a two-dimensional fault coordinate system.
    source_a_bounds : tuple[float, float], optional
        The bounds of the first source in fault coordinates. Used to
        constraint the optimisation process. Default is (0, 1).
    source_b_bounds : tuple[float, float], optional
        The bounds of the second source in fault coordinates. Used to
        constrain the optimisation process. Default is (0, 1).

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
        bounds=(
            [
                source_a_bounds.x_min,
                source_a_bounds.y_min,
                source_b_bounds.x_min,
                source_b_bounds.y_min,
            ],
            [
                source_a_bounds.x_max,
                source_a_bounds.y_max,
                source_b_bounds.x_max,
                source_b_bounds.y_max,
            ],
        ),
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
