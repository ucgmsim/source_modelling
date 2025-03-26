"""
Stoch file handling module.

This module provides functionality for reading and working with
stochastic slip models, which are down-sampled slip models used for
high frequency inputs in seismic modelling. It includes classes for
representing stochastic planes and associated metadata. See the
wiki[0]_ for a description of the stoch file format.

References
----------
.. [0] https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-Stochformat
"""

from pathlib import Path
from typing import NamedTuple, TextIO, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

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
    """

    longitude: float
    """Longitude coordinate of the plane's centre point."""
    latitude: float
    """Latitude coordinate of the plane's centre point."""
    nx: int
    """Number of grid points in the x-direction."""
    ny: int
    """Number of grid points in the y-direction."""
    dx: float
    """Grid spacing in the strike-direction (km)."""
    dy: float
    """Grid spacing in the dip-direction (km)."""
    strike: int
    """Strike angle of the fault plane (degrees)."""
    dip: int
    """Dip angle of the fault plane (degrees)."""
    average_rake: int
    """Average rake angle for the slip on the fault plane (degrees)."""
    dtop: float
    """Depth to the top of the fault plane (km)."""
    shypo: float
    """Along-strike distance to the hypocentre (km) from top centre."""
    dhypo: float
    """Down-dip distance to the hypocentre (km)."""


class StochPlane(NamedTuple):
    """
    Named tuple representing a stochastic slip plane with its properties.
    """

    header: StochHeader
    """Metadata for the stochastic plane."""
    slip: FloatArray2D
    """2D array of slip values with shape (ny, nx)."""
    rise: FloatArray2D
    """2D array of rise time values with shape (ny, nx)."""
    trup: FloatArray2D
    """2D array of rupture time values with shape (ny, nx)."""


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

    Attributes
    ----------
    data : list[StochPlane]
        Structured raw data read from the stoch file.

    Raises
    ------
    ValueError
        If the number of planes specified in the file is not a positive integer.

    Notes
    -----
    Stochastic slip model files contain information about fault planes and their
    properties such as slip, rise time, and rupture time.

    Examples
    --------
    >>> # Assuming 'stoch_model.stoch' exists with valid stochastic slip model data
    >>> stoch_file = StochFile('stoch_model.stoch')
    >>> planes = stoch_file.planes
    >>> slips = stoch_file.slip
    >>> rise_times = stoch_file.rise
    >>> rupture_times = stoch_file.trup
    >>> patch_centers = stoch_file.patch_centres
    >>> print(f"Number of planes: {len(planes)}")
    >>> if len(slips) > 0:
    ...     print(f"Slip array shape for the first plane: {slips[0].shape}")
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
        filename = Path(filename)
        with open(filename, "r") as handle:
            n_planes = parse_utils.read_int(handle, "n_planes")
            if n_planes <= 0:
                raise parse_utils.ParseError(
                    f"Expected non-negative integer number of planes, received: {n_planes}."
                )
            planes = [_read_stoch_plane(handle) for _ in range(n_planes)]
            self.data: list[StochPlane] = planes

    @property
    def planes(self) -> list[Plane]:
        """
        Get a list of Plane objects for the stoch file.

        Returns
        -------
        list[Plane]
            List of Plane objects representing the fault planes in the stochastic model.

        Examples
        --------
        >>> stoch_file = StochFile('stoch_model.stoch')
        >>> planes = stoch_file.planes
        >>> print(f"Number of planes: {len(planes)}")
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
            for plane in self.data
        ]

    @property
    def slip(self) -> list[FloatArray2D]:
        """
        Get a list of slip arrays for all planes.

        Returns
        -------
        list[FloatArray2D]
            List of 2D numpy arrays containing slip values for each plane.

        Examples
        --------
        >>> stoch_file = StochFile('stoch_model.stoch')
        >>> slips = stoch_file.slip
        >>> if len(slips) > 0:
        ...     print(f"Slip array shape for the first plane: {slips[0].shape}")
        """
        return [plane.slip for plane in self.data]

    @property
    def patch_centres(self) -> list[CoordinateArray]:
        """Get a numpy array of the patch centres for each plane.

        Returns
        -------
        list[CoordinateArray]
            List of 2D numpy arrays containing the patch centres for each plane.

        Examples
        --------
        >>> stoch_file = StochFile('stoch_model.stoch')
        >>> patch_centers = stoch_file.patch_centres
        >>> if len(patch_centers) > 0:
        ...     print(f"Patch centers array shape for the first plane: {patch_centers[0].shape}")
        """
        result: list[CoordinateArray] = []
        for plane_description, plane in zip(self.data, self.planes):
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

        Examples
        --------
        >>> stoch_file = StochFile('stoch_model.stoch')
        >>> rise_times = stoch_file.rise
        >>> if len(rise_times) > 0:
        ...     print(f"Rise time array shape for the first plane: {rise_times[0].shape}")
        """
        return [plane.rise for plane in self.data]

    @property
    def trup(self) -> list[FloatArray2D]:
        """
        Get a list of rupture time arrays for all planes.

        Returns
        -------
        list[FloatArray2D]
            List of 2D numpy arrays containing rupture time values for each plane.

        Examples
        --------
        >>> stoch_file = StochFile('stoch_model.stoch')
        >>> rupture_times = stoch_file.trup
        >>> if len(rupture_times) > 0:
        ...     print(f"Rupture time array shape for the first plane: {rupture_times[0].shape}")
        """
        return [plane.trup for plane in self.data]
