"""
Stoch file handling module.

This module provides functionality for reading and working with stochastic slip models,
which are down-sampled slip models used for high frequency inputs in seismic modelling.
It includes classes for representing stochastic planes and associated metadata.
"""

import re
from pathlib import Path
from typing import NamedTuple, TextIO, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from qcore import grid
from source_modelling import parse_utils
from source_modelling.sources import Plane

# Type aliases for NumPy arrays with specific shapes and dtypes
FloatArray2D: TypeAlias = NDArray[np.float32]  # 2D array of float32
LatLonArray: TypeAlias = NDArray[np.float64]  # Array for latitude/longitude
CoordinateArray: TypeAlias = NDArray[np.float64]  # Array for coordinates


class StochHeader(NamedTuple):
    """
    Named tuple containing header information for stochastic plane models.

    Parameters
    ----------
    longitude : float
        Longitude coordinate of the plane's reference point.
    latitude : float
        Latitude coordinate of the plane's reference point.
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    dx : float
        Grid spacing in the x-direction (km).
    dy : float
        Grid spacing in the y-direction (km).
    strike : int
        Strike angle of the fault plane (degrees).
    dip : int
        Dip angle of the fault plane (degrees).
    average_rake : int
        Average rake angle for the slip on the fault plane (degrees).
    dtop : float
        Depth to the top of the fault plane (km).
    shypo : float
        Along-strike distance to the hypocentre (km).
    dhypo : float
        Down-dip distance to the hypocentre (km).
    """

    longitude: float
    latitude: float
    nx: int
    ny: int
    dx: float
    dy: float
    strike: int
    dip: int
    average_rake: int
    dtop: float
    shypo: float
    dhypo: float


class StochPlane(NamedTuple):
    """
    Named tuple representing a stochastic slip plane with its properties.

    Parameters
    ----------
    header : StochHeader
        Metadata for the stochastic plane.
    slip : numpy.ndarray
        2D array of slip values with shape (ny, nx).
    rise : numpy.ndarray
        2D array of rise time values with shape (ny, nx).
    trup : numpy.ndarray
        2D array of rupture time values with shape (ny, nx).
    """

    header: StochHeader
    slip: FloatArray2D
    rise: FloatArray2D
    trup: FloatArray2D


def _read_stoch_header(handle: TextIO) -> StochHeader:
    """
    Read the header information from a stochastic file.

    Parameters
    ----------
    handle : TextIO
        Text file handle positioned at the start of header information.

    Returns
    -------
    StochHeader
        Named tuple containing the parsed header information.
    """
    longitude = parse_utils.read_float(handle, "longitude")
    latitude = parse_utils.read_float(handle, "latitude")
    nx = parse_utils.read_int(handle, "nx")
    ny = parse_utils.read_int(handle, "ny")
    dx = parse_utils.read_float(handle, "dx")
    dy = parse_utils.read_float(handle, "dy")
    strike = parse_utils.read_int(handle, "strike")
    dip = parse_utils.read_int(handle, "dip")
    average_rake = parse_utils.read_int(handle, "average_rake")
    dtop = parse_utils.read_float(handle, "dtop")
    shypo = parse_utils.read_float(handle, "shypo")
    dhypo = parse_utils.read_float(handle, "dhypo")
    return StochHeader(
        longitude,
        latitude,
        nx,
        ny,
        dx,
        dy,
        strike,
        dip,
        average_rake,
        dtop,
        shypo,
        dhypo,
    )


def _read_stoch_plane(handle: TextIO) -> StochPlane:
    """
    Read a complete stochastic plane from a file.

    Parameters
    ----------
    handle : TextIO
        Text file handle positioned at the start of a plane definition.

    Returns
    -------
    StochPlane
        Named tuple containing the plane's header and data arrays (slip, rise, trup).
    """
    header = _read_stoch_header(handle)
    slip_array = np.fromfile(
        handle, dtype=np.float32, sep=" ", count=header.nx * header.ny
    ).reshape((header.ny, header.nx))
    slip = cast(FloatArray2D, slip_array)

    rise_array = np.fromfile(
        handle, dtype=np.float32, sep=" ", count=header.nx * header.ny
    ).reshape((header.ny, header.nx))
    rise = cast(FloatArray2D, rise_array)

    trup_array = np.fromfile(
        handle, dtype=np.float32, sep=" ", count=header.nx * header.ny
    ).reshape((header.ny, header.nx))
    trup = cast(FloatArray2D, trup_array)

    return StochPlane(header, slip, rise, trup)


class StochFile:
    """
    Class for reading and accessing stochastic slip model files.

    This class handles the parsing of stochastic slip model files and provides
    access to the contained planes and their properties.

    Parameters
    ----------
    filename : Path
        Path to the stochastic slip model file.

    Raises
    ------
    ValueError
        If the number of planes specified in the file is not a positive integer.

    Notes
    -----
    Stochastic slip model files contain information about fault planes and their
    properties such as slip, rise time, and rupture time.
    """

    def __init__(self, filename: Path):
        """
        Initialize a StochFile instance by reading data from the specified file.

        Parameters
        ----------
        filename : Path
            Path to the stochastic slip model file.

        Raises
        ------
        ValueError
            If the number of planes specified in the file is not a positive integer.
        """
        self.filename = filename
        with open(filename, "r") as handle:
            n_planes = parse_utils.read_int(handle, "n_planes")
            if n_planes <= 0:
                raise parse_utils.ParseError(
                    f"Expected non-negative integer number of planes, received: {n_planes}."
                )
            planes = [_read_stoch_plane(handle) for _ in range(n_planes)]
            self._planes: list[StochPlane] = planes

    @property
    def planes(self) -> list[Plane]:
        """
        Get a list of Plane objects for the stoch file.

        Returns
        -------
        list[Plane]
            List of Plane objects representing the fault planes in the stochastic model.
        """
        return [
            Plane.from_centroid_strike_dip(
                cast(
                    LatLonArray,
                    np.array([plane.header.latitude, plane.header.longitude]),
                ),
                plane.header.dip,
                plane.header.dx * plane.header.nx,
                plane.header.dy * plane.header.ny,
                strike=plane.header.strike,
                dtop=plane.header.dtop,
            )
            for plane in self._planes
        ]

    @property
    def slip(self) -> list[FloatArray2D]:
        """
        Get a list of slip arrays for all planes.

        Returns
        -------
        list[FloatArray2D]
            List of 2D numpy arrays containing slip values for each plane.
        """
        return [plane.slip for plane in self._planes]

    @property
    def patch_centres(self) -> list[CoordinateArray]:
        """Get a numpy array of the patch centres for each plane.

        Returns
        -------
        list[CoordinateArray]
            List of 2D numpy arrays containing the patch centres for each plane.
        """
        result: list[CoordinateArray] = []
        for plane_description, plane in zip(self._planes, self.planes):
            centers = grid.coordinate_patchgrid(
                plane.corners[0],
                plane.corners[1],
                plane.corners[-1],
                nx=plane_description.header.nx,
                ny=plane_description.header.ny,
            )
            result.append(cast(CoordinateArray, centers))
        return result

    @property
    def rise(self) -> list[FloatArray2D]:
        """
        Get a list of rise time arrays for all planes.

        Returns
        -------
        list[FloatArray2D]
            List of 2D numpy arrays containing rise time values for each plane.
        """
        return [plane.rise for plane in self._planes]

    @property
    def trup(self) -> list[FloatArray2D]:
        """
        Get a list of rupture time arrays for all planes.

        Returns
        -------
        list[FloatArray2D]
            List of 2D numpy arrays containing rupture time values for each plane.
        """
        return [plane.trup for plane in self._planes]
