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
from pathlib import Path
from typing import Optional, TextIO

import numpy as np
import pandas as pd

PLANE_COUNT_RE = r"PLANE (\d+)"
POINT_COUNT_RE = r"POINTS (\d+)"


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
    points : pd.DataFrame
        A list of SrfPoint objects representing the points in the SRF file.
    """

    version: str
    header: pd.DataFrame
    points: pd.DataFrame


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
        version = float(srf_file_handle.readline())

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

        points = pd.DataFrame(
            (read_srf_point(srf_file_handle) for _ in range(point_count))
        )

        return SrfFile(version, headers, points)


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
