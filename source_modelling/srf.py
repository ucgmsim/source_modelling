"""Module for handling SRF (Standard Rupture Format) files.

This module provides classes and functions for reading and writing SRF files,
as well as representing their contents.
See https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
for details on the SRF format.

Why Not qcore.srf?
------------------
You might use this module instead of the `qcore.srf` module because:

1. The `qcore.srf` module does not support writing SRF files.

2. Exposing SRF points as a pandas dataframe allows manipulation of
   the points using efficient vectorised operations. We use this in
   rupture propagation to delay ruptures by adding to the `tinit` column.

3. There is better documentation for the new module than the old one.

You should use `qcore.srf` if you do not eventually intend to read all
points of the SRF file (it is memory efficient), or you are working
with code that already uses `qcore.srf`.

Classes
-------
- SrfFile: Representation of an SRF file.

Exceptions
----------
- SrfParseError: Exception raised for errors in parsing SRF files.

Functions
---------
- read_srf: Read an SRF file into memory.
- write_srf: Write an SRF object to a filepath.

Example
-------
>>> srf_file = srf.read_srf('/path/to/srf')
>>> srf_file.points['tinit'].max() # get the last time any point in the SRF ruptures
>>> srf_file.points['tinit'] += 1 # delay all points by one second
>>> coordinates.wgs_depth_to_nztm(srf_file.header[['elat', 'elon']].to_numpy())
#   ^ get the coordinates all the fault plane centres in the rupture in NZTM format
# etc...
>>> srf.write_srf('/path/to/srf', srf_file)
"""

import dataclasses
import functools
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, TextIO

import numpy as np
import pandas as pd
import scipy as sp
import shapely

from qcore import coordinates
from source_modelling import moment, srf_reader

PLANE_COUNT_RE = r"PLANE (\d+)"
POINT_COUNT_RE = r"POINTS (\d+)"


class Segments(Sequence):
    """A read-only view for SRF segments."""

    def __init__(self, header: pd.DataFrame, points: pd.DataFrame):
        self._header = header
        self._points = points

    def __getitem__(self, index: int) -> pd.DataFrame:
        """Get the nth segment in the SRF.

        Parameters
        ----------
        index : int
            The index of the segment.

        Returns
        -------
        int
            The nth segment in the SRF.
        """
        if not isinstance(index, int):
            # NOTE: We are not covering this in test coverage because
            # we intend to support slicing in the future.
            raise TypeError(
                "Segment index must an integer, not slice or tuple"
            )  # pragma: no cover
        points_offset = (self._header["nstk"] * self._header["ndip"]).cumsum()
        if index == 0:
            return self._points.iloc[: points_offset.iloc[index]]
        return self._points.iloc[
            points_offset.iloc[index - 1] : points_offset.iloc[index]
        ]

    def __len__(self) -> int:
        """int: The number of segments in the SRF."""
        return len(self._header)


@dataclasses.dataclass
class SrfFile:
    """
    Representation of an SRF file.

    Attributes
    ----------
    version : str
        The version of this SrfFile
    header : pd.DataFrame
        A list of SrfSegment objects representing the header of the SRF file.
        The columns of the header are:

        - elon: The centre longitude of the plane.
        - elot: The centre latitude of the plane.
        - nstk: The number of patches along strike for the plane.
        - ndip: The number of patches along dip for the plane.
        - len: The length of the plane (in km).
        - wid: The width of the plane (in km).
        - stk: The plane strike.
        - dip: The plane dip.
        - dtop: The top of the plane.
        - shyp: The hypocentre location in strike coordinates.
        - dhyp: The hypocentre location in dip coordinates.


    points : pd.DataFrame
        A list of SrfPoint objects representing the points in the SRF
        file. The columns of the points dataframe are:

        - lon: longitude of the patch.
        - lat: latitude of the patch.
        - dep: depth of the patch (in kilometres).
        - stk: local strike.
        - dip: local dip.
        - area: area of the patch (in cm^2).
        - tinit: initial rupture time for this patch (in seconds).
        - dt: the timestep for all slipt* columns (in seconds).
        - rake: local rake.
        - slip1: total slip in the first component.
        - slip2: total slip in the second component.
        - slip3: total slip in the third component.
        - slip: total slip.

        The final two columns are computed from the SRF and are not saved to
        disk. See the linked documentation on the SRF format for more details.

    slipt{i}_array : csr_array
        A sparse array containing the ith component of slip for each point and at each timestep, where
        slipt{i}_array[i, j] is the slip for the ith patch at time t = j * dt. See also: SRFFile.slip.

    References
    ----------
    SRF File Format Doc: https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
    """

    version: str
    header: pd.DataFrame
    points: pd.DataFrame
    slipt1_array: sp.sparse.csr_array
    slipt2_array: sp.sparse.csr_array
    slipt3_array: sp.sparse.csr_array

    @property
    def slip(self):
        """csr_array: sparse array representing slip in all components."""
        slip_array = self.slipt1_array.power(2)
        if self.slipt2_array:
            slip_array += self.slipt2_array.power(2)
        if self.slipt3_array:
            slip_array += self.slipt3_array.power(2)
        return slip_array.sqrt()

    @property
    def moment_rate(self):
        return moment.moment_rate_over_time_from_slip(
            self.points["area"], self.slip, self.dt, self.nt
        )

    @property
    def moment(self):
        return moment.moment_over_time_from_moment_rate(self.moment_rate)

    @property
    def geometry(self) -> shapely.Geometry:
        """shapely.Geometry: The shapely geometry of all segments in the SRF."""
        polygons = []
        for i, segment in enumerate(self.segments):
            header = self.header.iloc[i]
            nstk = header["nstk"]
            ndip = header["ndip"]
            corners = (
                segment[["lat", "lon"]]
                .iloc[[0, nstk - 1, nstk * (ndip - 1), nstk * ndip - 1]]
                .values
            )
            if header["dip"] == 90:
                polygons.append(
                    shapely.LineString(coordinates.wgs_depth_to_nztm(corners[:2]))
                )
            else:
                polygons.append(
                    shapely.convex_hull(
                        shapely.MultiPoint(coordinates.wgs_depth_to_nztm(corners))
                    )
                )
        return shapely.union_all(polygons).normalize()

    @property
    def nt(self):
        """int: The number of timeslices in the SRF."""
        return self.slipt1_array.shape[1]

    @property
    def dt(self):
        """float: time resolution of SRF."""
        return self.points["dt"].iloc[0]

    @property
    def segments(self) -> Segments:
        """Segments: A sequence of segments in the SRF."""
        return Segments(self.header, self.points)


class SrfParseError(Exception):
    """Exception raised for errors in parsing SRF files."""

    pass


def read_float(srf_file: TextIO, label: Optional[str] = None) -> float:
    """Read a float from an SRF file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to read from.
    label : Optional[str]
        A human friendly label for the floating point (for debugging
        purposes), or None for no label. Defaults to None.

    Raises
    ------
    SrfParseError
        If there is an error reading the float value from the SRF file.

    Returns
    -------
    float
        The float read from the SRF file.
    """
    while (cur := srf_file.read(1)).isspace():
        pass
    float_str = cur
    while not (cur := srf_file.read(1)).isspace():
        float_str += cur
    try:
        return float(float_str)
    except ValueError:
        if label:
            raise SrfParseError(f'Expecting float ({label}), got: "{float_str}"')
        else:
            raise SrfParseError(f'Expecting float, got: "{float_str}"')


def read_int(srf_file: TextIO, label: Optional[str] = None) -> int:
    """Read a int from an SRF file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to read from.

    Raises
    ------
    SrfParseError
        If there is an error reading the int value from the SRF file.

    Returns
    -------
    int
        The int read from the SRF file.
    """
    while (cur := srf_file.read(1)).isspace():
        pass
    int_str = cur
    while not (cur := srf_file.read(1)).isspace():
        int_str += cur
    try:
        return int(int_str)
    except ValueError:
        if label:
            raise SrfParseError(f'Expecting int ({label}), got: "{int_str}"')
        else:
            raise SrfParseError(f'Expecting int, got: "{int_str}"')


def read_srf(srf_ffp: Path) -> SrfFile:
    """Read an SRF file into an SrfFile object.

    Parameters
    ----------
    srf_ffp : Path
        The filepath of the SRF file.

    Returns
    -------
    SrfFile
        The filepath of the SRF file.
    """
    with open(srf_ffp, mode="r", encoding="utf-8") as srf_file_handle:
        version = srf_file_handle.readline().strip()

        plane_count_line = srf_file_handle.readline().strip()
        plane_count_match = re.match(PLANE_COUNT_RE, plane_count_line)
        if not plane_count_match:
            raise SrfParseError(
                f'Expecting PLANE header line, got: "{plane_count_line}"'
            )
        plane_count = int(plane_count_match.group(1))
        segments = []

        for _ in range(plane_count):
            segments.append(
                {
                    "elon": read_float(srf_file_handle),
                    "elat": read_float(srf_file_handle),
                    "nstk": read_int(srf_file_handle),
                    "ndip": read_int(srf_file_handle),
                    "len": read_float(srf_file_handle),
                    "wid": read_float(srf_file_handle),
                    "stk": read_float(srf_file_handle),
                    "dip": read_float(srf_file_handle),
                    "dtop": read_float(srf_file_handle),
                    "shyp": read_float(srf_file_handle),
                    "dhyp": read_float(srf_file_handle),
                }
            )
        headers = pd.DataFrame(segments)
        headers["nstk"] = headers["nstk"].astype(int)
        headers["ndip"] = headers["ndip"].astype(int)

        points_count_line = srf_file_handle.readline().strip()
        points_count_match = re.match(POINT_COUNT_RE, points_count_line)
        if not points_count_match:
            raise SrfParseError(
                f'Expecting POINTS header line, got: "{points_count_line}"'
            )
        point_count = int(points_count_match.group(1))

        points_metadata, slipt1_array, slipt2_array, slipt3_array = (
            srf_reader.read_srf_points(srf_file_handle, point_count)
        )
        points_df = pd.DataFrame(
            data=points_metadata,
            columns=[
                "lon",
                "lat",
                "dep",
                "stk",
                "dip",
                "area",
                "tinit",
                "dt",
                "rake",
                "slip1",
                "slip2",
                "slip3",
            ],
        )

        points_df["slip"] = np.sqrt(
            points_df["slip1"] ** 2 + points_df["slip2"] ** 2 + points_df["slip3"] ** 2
        )
        return SrfFile(
            version, headers, points_df, slipt1_array, slipt2_array, slipt3_array
        )


def write_slip(srf_file: TextIO, slips: np.ndarray) -> None:
    """Write out slip values to an SRF file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to write to.
    slips : np.ndarray
        The slip values to write.
    """
    for i in range(0, len(slips), 6):
        srf_file.write(
            "  "
            + "  ".join(f"{slips[j]:.5E}" for j in range(i, min(len(slips), i + 6)))
            + "\n"
        )


def write_srf_point(srf_file: TextIO, srf: SrfFile, point: pd.Series) -> None:
    """Write out a single SRF point.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to write to.
    srf : SrfFile
        The SRF file object to write.
    point : pd.Series
        The point to write.
    """
    index = int(point["point_index"])
    # We need to get the raw slip values for the slip arrays to write out to the SRF.
    # The slip values for the ith point in the SRF are stored in the ith row of the slip array.
    # The indptr array contains the indices for each row in the slip array, so:
    #
    # - indptr[i] is the start index in the data array for the ith row and
    # - indptr[i + 1] is the start index in the data array for the (i + 1)th row.
    #
    # Hence, data[indptr[i]:indptr[i+1]] collects the slip values for the ith row.
    # Note we cannot use standard multi-dimensional array indexing arr[i, :]
    # because scipy sparse arrays do not support this kind of indexing.
    #
    # Visually:
    #
    # +----+-----+----+---+---+--+
    # |    |     |    |   |   |  | indptr
    # +----+---\-+----+---+-\-+--+
    #           \            \
    #            \            \
    # +----------+\-----------+\----------+-------------+
    # |   ...    |    row i   | row i + 1 |     ...     | data
    # +----------+------------+-----------+-------------+
    #            row_i = data[indptr[i]:indptr[i+1]]
    slipt1 = (
        srf.slipt1_array.data[
            srf.slipt1_array.indptr[index] : srf.slipt1_array.indptr[index + 1]
        ]
        if srf.slipt1_array is not None
        else None
    )
    slipt2 = (
        srf.slipt2_array.data[
            srf.slipt2_array.indptr[index] : srf.slipt2_array.indptr[index + 1]
        ]
        if srf.slipt2_array is not None
        else None
    )
    slipt3 = (
        srf.slipt3_array.data[
            srf.slipt3_array.indptr[index] : srf.slipt3_array.indptr[index + 1]
        ]
        if srf.slipt3_array is not None
        else None
    )
    srf_file.write(
        f"{point['lon']:.6f} {point['lat']:.6f} {point['dep']:g} {point['stk']:g} {point['dip']:g} {point['area']:.4E} {point['tinit']:.4f} {point['dt']:.6E}\n"
    )
    srf_file.write(
        f"{point['rake']:g} {point['slip1']:.4f} {len(slipt1) if slipt1 is not None else 0} {point['slip2']:.4f} {len(slipt2) if slipt2 is not None else 0} {point['slip3']:.4f} {len(slipt3) if slipt3 is not None else 0}\n"
    )
    if slipt1 is not None:
        write_slip(srf_file, slipt1)
    if slipt2 is not None:
        write_slip(srf_file, slipt2)
    if slipt3 is not None:
        write_slip(srf_file, slipt3)


def write_srf(srf_ffp: Path, srf: SrfFile) -> None:
    """Write an SRF object to a filepath.

    Parameters
    ----------
    srf_ffp : Path
        The filepath to write the srf object to.
    srf : SrfFile
        The SRF object.
    """
    with open(srf_ffp, mode="w", encoding="utf-8") as srf_file_handle:
        srf_file_handle.write("1.0\n")
        srf_file_handle.write(f"PLANE {len(srf.header)}\n")
        # Cannot use srf.header.to_string because the newline separating headers is significant!
        # This is ok because the number of headers is typically very small (< 100)
        for _, plane in srf.header.iterrows():
            srf_file_handle.write(
                "\n".join(
                    [
                        f"{plane['elon']:.6f} {plane['elat']:.6f} {int(plane['nstk'])} {int(plane['ndip'])} {plane['len']:.4f} {plane['wid']:.4f}",
                        f"{plane['stk']:.4f} {plane['dip']:.4f} {plane['dtop']:.4f} {plane['shyp']:.4f} {plane['dhyp']:.4f}",
                        "",
                    ]
                )
            )

        srf_file_handle.write(f"POINTS {len(srf.points)}\n")
        srf.points["point_index"] = np.arange(len(srf.points))

        srf.points.apply(
            functools.partial(write_srf_point, srf_file_handle, srf), axis=1
        )
        srf.points = srf.points.drop(columns="point_index")
