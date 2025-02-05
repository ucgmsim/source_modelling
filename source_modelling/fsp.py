"""Module for handling FSP files.

This module provides classes and functions for reading FSP files,
as well as representing their contents.
See http://equake-rc.info/SRCMOD/fileformats/fsp/
for details on the FSP format.

Classes
-------
- FSPFile: Representation of an FSP file.

Exceptions
----------
- FSPParseError: Exception raised for errors in parsing FSP files.

Example
-------
>>> fsp_file = FSPFile.read_from_file(fsp_ffp)
>>> (fsp_file.data['trup'] + fsp_file.data['rise']).max() # Get time of final rise for subfaults.
"""

import dataclasses
import io
import re
from pathlib import Path
from typing import IO, Callable, Optional

import pandas as pd
import parse

HEADER_PATTERN = """EventTAG: {event_tag}
Loc : LAT = {latitude:g} LON = {longitude:g} DEP = {depth:g}
Size : LEN = {length:g} km WID = {width:g} km Mw = {magnitude:g} Mo = {moment:g} Nm
Mech : STRK = {strike:g} DIP = {dip:g} RAKE = {rake:g} Htop = {htop:g} km
Rupt : HypX = {hypx:g} km Hypz = {hypz:g} km avTr = {average_rise_time:g} s avVr = {average_rupture_speed:g} km/s
Invs : Nx = {nx:d} Nz = {nz:d} Fmin = {fmin:g} Hz Fmax = {fmax:g} Hz
Invs : Dx = {dx:g} km Dz = {dz:g} km
Invs : Ntw = {time_window_count:d} Nsg = {segment_count:d} (# of time-windows,# of fault segments)
Invs : LEN = {time_window_length:g} s SHF = {time_shift:g} s (time-window length and time-shift)
SVF : {slip_velocity_function} (type of slip-velocity function used)
"""


def _normalise_value(value: float) -> Optional[float]:
    """Normalise an FSP parameter value.

    Parameters
    ----------
    value : float
        The value to normalise.

    Returns
    -------
    Optional[float]
        None if value is 999 or less than zero.
    """
    return None if value == 999 or value < 0 else value


class FSPParseError(Exception):
    """Exception raised for errors in parsing FSP files."""

    pass


@dataclasses.dataclass
class FSPFile:
    """A representation of the FSP file format used to represent source models in SRCMOD.

    Attributes
    ----------
    event_tag : str
        An identifier for the FSPFile, e.g. "s2004IRIANx01WEIx".
    latitude : float
        Latitude of hypocentre.
    longitude : float
        Longitude of hypocentre.
    depth : float
        Depth of hypocentre (in km).
    hypx : float or None
        The x-coordinate of the hypocentre (along strike, in km).
    hypz : float or None
        The z-coordinate of the hypocentre (along dip, in km).
    length : float
        The length of the rupture (in km).
    width : float
        The width of the rupture (in km).
    strike : float
        Rupture plane strike.
    dip : float
        Rupture plane dip.
    rake : float
        Rupture plane rake.
    htop : float
        Top depth of rupture plane (in km).
    magnitude : float
        Rupture magnitude.
    moment : float
        Rupture moment (in Nm).
    average_rise_time : float or None
        Average rise time of subfaults.
    average_rupture_speed : float or None
        Average speed of rupture front.
    slip_velocity_function : str
        Description of the slip velocity function used
    nx : float or None
        The number of subfaults along strike.
    nz : float or None
        The number of subfaults along dip.
    dx : float
        The length of each subfault.
    dz : float
        The width of each subfault.
    fmin : float or None
        The minimum frequency in the inversion.
    fmax : float or None
        The maximum frequency in the inversion.
    time_window_count : float or None
        The slip (or rake) time windows for each subfault.
    time_window_length : float or None
        The time window length for each subfault.
    time_shift : float or None
        The time shift for each time window.
    segment_count : int
        Number of segments (subfaults).
    data : pd.DataFrame
        The data frame of info on each subfault. Columns vary for this
        dataframe, but each comes with 'segment', 'length', and 'width' which
        are the segement index, length and width respectively. To perform
        operations over multi-segment data, group by 'segment' and apply the
        operation to each group.
    """

    event_tag: str

    # Hypocentre parameters
    latitude: float
    longitude: float
    depth: float
    hypx: Optional[float]
    hypz: Optional[float]

    velocity_model: Optional[pd.DataFrame | float]

    # Fault and parameters
    length: float
    width: float
    strike: float
    dip: float
    rake: float
    htop: float

    # Rupture parameters
    magnitude: float
    moment: float
    average_rise_time: Optional[float]
    average_rupture_speed: Optional[float]
    slip_velocity_function: str

    # Model subfault sizes
    nx: Optional[int]
    nz: Optional[int]
    dx: Optional[float]
    dz: Optional[float]

    # Inversion parameters
    fmin: Optional[float]
    fmax: Optional[float]

    # Time window parameters
    time_window_count: Optional[float]
    time_window_length: Optional[float]
    time_shift: Optional[float]

    # Number of segments (== len(data))
    segment_count: int
    data: pd.DataFrame

    @classmethod
    def read_from_file(cls: Callable, fsp_ffp: Path) -> "FSPFile":
        """Parse an FSPFile.

        Parameters
        ----------
        fsp_ffp : Path
            Path to the FSP file.

        Returns
        -------
        FSPFile
            The parsed FSPFile.
        """
        with open(fsp_ffp, "r") as fsp_file_handle:
            header = ""
            # Collect and normalise all the lines that make up the header
            # of the file.
            for line in fsp_file_handle:
                if line.startswith("% Data"):
                    break
                if (
                    line.startswith("% -")
                    or line.strip() == "%"
                    or line.startswith("% Event :")
                ):
                    continue
                # Strip the leading "% ", and deduplicate the spaces in the line.
                # This is required to normalise the string so that the parse
                # module can handle the rest of the parsing
                header += " ".join(line.lstrip("% ").split()) + "\n"

            # Now we can parse the header according to the header pattern.
            parse_result = parse.parse(
                HEADER_PATTERN.strip(),
                header.strip(),
            )

            if not parse_result:
                raise FSPParseError("Failed to parse FSP file header")
            metadata = parse_result.named

            # now search for the velocity model structure
            for line in fsp_file_handle:
                if line.startswith("% VELOCITY-DENSITY STRUCTURE"):
                    break
            velocity_model = _parse_velocity_density_structure(fsp_file_handle)

            # now sniff ahead for the columns
            data = _parse_segment_slip(fsp_file_handle, metadata["segment_count"])

            fsp_file = cls(data=data, velocity_model=velocity_model, **metadata)

            # A lot of parameters can be 999 or -999 to indicate "no known
            # value", so we can detect that and set to None
            fsp_file.hypx = _normalise_value(fsp_file.hypx)
            fsp_file.hypz = _normalise_value(fsp_file.hypz)

            fsp_file.average_rise_time = _normalise_value(fsp_file.average_rise_time)
            fsp_file.average_rupture_speed = _normalise_value(
                fsp_file.average_rupture_speed
            )

            fsp_file.nx = _normalise_value(fsp_file.nx)
            fsp_file.nz = _normalise_value(fsp_file.nz)

            fsp_file.dx = _normalise_value(fsp_file.dx)
            fsp_file.dz = _normalise_value(fsp_file.dz)

            fsp_file.fmin = _normalise_value(fsp_file.fmin)
            fsp_file.fmax = _normalise_value(fsp_file.fmax)

            fsp_file.time_window_count = _normalise_value(fsp_file.time_window_count)
            fsp_file.time_window_length = _normalise_value(fsp_file.time_window_length)
            fsp_file.time_shift = _normalise_value(fsp_file.time_shift)
            return fsp_file


def _parse_segment_slip(fsp_file_handle: IO[str], segment_count: int) -> pd.DataFrame:
    """
    Parse segment slip data from an FSP file.

    This function extracts slip information for multiple segments from an FSP file.
    Each segment contains a specified number of subfaults, and the function reads
    their respective latitude, longitude, slip values, and other associated data.

    Parameters
    ----------
    fsp_file_handle : IO[str]
        A file-like object containing the FSP file contents.
    segment_count : int
        The number of segments to parse.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing parsed subfault data for all segments.
        The DataFrame includes a "segment" column identifying the segment index.

    Raises
    ------
    FSPParseError
        If the column headers for a segment cannot be found or the number of
        subfaults for a segment is missing.
    """
    segments: list[pd.DataFrame] = []
    for i in range(segment_count):
        subfaults = None
        length = width = None
        for line in fsp_file_handle:
            # hunt for the line % Nsbfs = x, where x is the number of subfaults.
            if match := re.search(r"%\s+Nsbfs\s*=\s*(\d+)", line):
                subfaults = match.group(1)
            # hunt for the line % LEN = x km WID = y km, where x and y are the
            # length and width respectively.
            if dimensions := re.match(
                r"%\s+LEN\s*=\s*(\d+(\.\d+)?)\s+km\s+WID\s*=\s*(\d+(\.\d+)?)\s+km", line
            ):
                length = float(dimensions.group(1))
                width = float(dimensions.group(1))
            if re.match(r"%\s+LAT\s+LON", line):
                break
        else:
            raise FSPParseError(f"Cannot find columns for FSP file on segment {i + 1}.")

        if subfaults is None:
            raise FSPParseError(
                f"Cannot find number of subfaults for FSP file on segment {i + 1}."
            )

        subfaults = int(subfaults)
        columns = line.lower().lstrip("% ").split()
        _ = next(fsp_file_handle)  # Skip header decoration
        # read subfaults lines into StringIO buffer
        lines = io.StringIO(
            "\n".join([next(fsp_file_handle) for _ in range(subfaults)])
        )
        data = pd.read_csv(
            lines,
            delimiter=r"\s+",
            header=None,
            names=columns,
        )
        data = data.rename(
            columns={"x==ew": "x", "y==ns": "y", "x==ns": "x", "y==ew": "y"}
        )
        data["segment"] = i
        if length:
            data["length"] = length
        if width:
            data["width"] = width
        segments.append(data)

    return pd.concat(segments)


def _parse_velocity_density_structure(
    fsp_file_handle: IO[str],
) -> Optional[pd.DataFrame | float]:
    """Parse the velocity-density structure from an FSP file.

    Parameters
    ----------
    fsp_file_handle : IO[str]
        A file-like object containing the FSP file contents.

    Returns
    -------
    Optional[pd.DataFrame | float]
        - If a velocity model table is found, returns a DataFrame with columns:
          ["DEPTH", "P-VEL", "S-VEL", "QP", "QS"].
        - If a shear modulus value is found, returns it as a float.
        - If the structure cannot be determined, returns None.
    """
    # Matches % No. of layers = x, where x is the number of layers in the velocity model.
    number_of_layers_re = re.match(
        r"%\s+No\.\s+of\s+layers\s*=\s*(\d+)", next(fsp_file_handle)
    )
    layer_count = 0
    if number_of_layers_re:
        layer_count = int(number_of_layers_re.group(1))
    else:
        return None

    columns = []
    uses_velocity_model_table = False
    while line := next(fsp_file_handle):
        if re.match(r"%\s+DEPTH\s+P-VEL", line):
            uses_velocity_model_table = True
            columns = [
                column for column in re.split(r"\s+", line.strip("% ")) if column
            ]
            break
        elif "shear modulus" in line.lower():
            # This is the case for an assumed constant mu value for the whole
            # geology.
            break
        elif "crustal model unknown" in line.lower():
            # This case is an edge-case when the crustal model is unknown.
            return None

    if uses_velocity_model_table:
        _ = next(fsp_file_handle)  # Skip units

        lines = io.StringIO(
            "\n".join([next(fsp_file_handle).strip("% ") for _ in range(layer_count)])
        )
        return pd.read_csv(lines, delimiter=r"\s+", header=None, names=columns)

    # Parse out the assumed shear modulus
    # It is split over two lines, a scale which looks like % [10**10 N/m^2]
    # typically...
    scale = re.match(r"%\s+\[(\d+)\*\*(\d+) N/m\^2\]", next(fsp_file_handle))
    if not scale:
        return None
    scale = float(scale.group(1)) ** float(scale.group(2))
    # ...and a significand which is just a floating point on a line of its own.
    # Typically: % 3.30.
    significand = re.match(r"%\s+(\d+(\.\d+)?)", next(fsp_file_handle))
    if not significand:
        return None
    significand = float(significand.group(1))
    return significand * scale
