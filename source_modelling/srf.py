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

from source_modelling import srf_reader

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
            raise TypeError("Segment index must an integer, not slice or tuple")
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
        - dbottom: The bottom of the plane.
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

    slipt{i}_matrix : csr_matrix
        A sparse matrix containing the ith component of slip for each point and at each timestep, where
        slipt{i}_matrix[i, j] is the slip for the ith patch at time t = j * dt. See also: SRFFile.slip.

    References
    ----------
    SRF File Format Doc: https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
    """

    version: str
    header: pd.DataFrame
    points: pd.DataFrame
    slipt1_matrix: sp.sparse.csr_matrix
    slipt2_matrix: sp.sparse.csr_matrix
    slipt3_matrix: sp.sparse.csr_matrix

    @property
    def slip(self):
        """csr_matrix: sparse matrix representing slip in all components"""
        slip_matrix = self.slipt1_matrix.power(2)
        if self.slipt2_matrix:
            slip_matrix += self.slipt2_matrix.power(2)
        if self.slipt3_matrix:
            slip_matrix += self.slipt3_matrix.power(2)
        return slip_matrix.sqrt()

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


def read_srf_point(srf_file: TextIO) -> dict[str, int | float]:
    """Read a single SRF point from a file handle.

    Parameters
    ----------
    srf_file : TextIO
        The file handle of the SRF file, pointing to the start of the
        line.

    Returns
    -------
    dict[str, int | float]
        A dictionary containing the values for the point.
    """
    # skip optional first spaces
    row = {
        "lon": read_float(srf_file, label="lon"),
        "lat": read_float(srf_file, label="lat"),
        "dep": read_float(srf_file, label="dep"),
        "stk": read_float(srf_file, label="stk"),
        "dip": read_float(srf_file, label="dip"),
        "area": read_float(srf_file, label="area"),
        "tinit": read_float(srf_file, label="tinit"),
        "dt": read_float(srf_file, label="dt"),
        "rake": read_float(srf_file, label="rake"),
        "slip1": read_float(srf_file, label="slip1"),
    }
    nt1 = read_int(srf_file, label="nt1")
    row["slip2"] = read_float(srf_file, label="slip2")
    nt2 = read_int(srf_file, label="nt2")
    row["slip3"] = read_float(srf_file, label="slip3")
    nt3 = read_int(srf_file, label="nt3")
    row["slipt1"] = np.fromiter(
        (read_float(srf_file, label="slipt1") for _ in range(nt1)), float
    )
    row["slipt2"] = np.fromiter(
        (read_float(srf_file, label="slipt2") for _ in range(nt2)), float
    )
    row["slipt3"] = np.fromiter(
        (read_float(srf_file, label="slipt2") for _ in range(nt3)), float
    )
    length = max(len(row["slipt1"]), len(row["slipt2"]), len(row["slipt3"]))
    row["slipt"] = np.sqrt(
        np.pad(row["slipt1"], (0, length - len(row["slipt1"])), mode="constant") ** 2
        + np.pad(row["slipt2"], (0, length - len(row["slipt2"])), mode="constant") ** 2
        + np.pad(row["slipt3"], (0, length - len(row["slipt3"])), mode="constant") ** 2
    )
    return row


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

        points_count_line = srf_file_handle.readline().strip()
        points_count_match = re.match(POINT_COUNT_RE, points_count_line)
        if not points_count_match:
            raise SrfParseError(
                f'Expecting POINTS header line, got: "{points_count_line}"'
            )
        point_count = int(points_count_match.group(1))

        points_metadata, slipt1_matrix, slipt2_matrix, slipt3_matrix = (
            srf_reader.read_srf_points(srf_file_handle, point_count)
        )
        points_df = pd.DataFrame(
            data=points_metadata,
            columns=[
                "lat",
                "lon",
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
            version, headers, points_df, slipt1_matrix, slipt2_matrix, slipt3_matrix
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


def write_srf_point(srf_file: TextIO, point: pd.Series) -> None:
    """Write out a single SRF point.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to write to.
    point : pd.Series
        The point to write.
    """
    srf_file.write(
        f"{point['lon']:.6f} {point['lat']:.6f} {point['dep']:g} {point['stk']:g} {point['dip']:g} {point['area']:.4E} {point['tinit']:.4f} {point['dt']:.6E}\n"
    )
    srf_file.write(
        f"{point['rake']:g} {point['slip1']:.4f} {len(point['slipt1'])} {point['slip2']:.4f} {len(point['slipt2'])} {point['slip3']:.4f} {len(point['slipt3'])}\n"
    )
    if len(point["slipt1"]):
        write_slip(srf_file, point["slipt1"])
    if len(point["slipt2"]):
        write_slip(srf_file, point["slipt2"])
    if len(point["slipt3"]):
        write_slip(srf_file, point["slipt3"])


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
        srf_file_handle.write(
            srf.header.to_string(
                index=False,
                header=None,
                formatters={
                    "elon": lambda elon: f"{elon:.6f}",
                    "elat": lambda elat: f"{elat:.6f}",
                    "len": lambda len: f"{len:.4f}",
                    "wid": lambda wid: f"{wid:.4f}",
                },
            )
            + "\n"
        )
        srf_file_handle.write(f"POINTS {len(srf.points)}\n")
        srf.points.apply(functools.partial(write_srf_point, srf_file_handle), axis=1)
