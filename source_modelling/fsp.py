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
import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import parse

HEADER_PATTERN = """EventTAG: {event_tag}
Loc : LAT = {latitude:f} LON = {longitude:f} DEP = {depth:f}
Size : LEN = {length:f} km WID = {width:f} km Mw = {magnitude:f} Mo = {moment:e} Nm
Mech : STRK = {strike:f} DIP = {dip:f} RAKE = {rake:f} Htop = {htop:f} km
Rupt : HypX = {hypx:f} km Hypz = {hypz:f} km avTr = {average_rise_time:f} s avVr = {average_rupture_speed:f} km/s
Invs : Nx = {nx:d} Nz = {nz:d} Fmin = {fmin:f} Hz Fmax = {fmax:f} Hz
Invs : Dx = {dx:f} km Dz = {dz:f} km
Invs : Ntw = {time_window_count:d} Nsg = {segment_count:d} (# of time-windows,# of fault segments)
Invs : LEN = {time_window_length:f} s SHF = {time_shift:f} s (time-window length and time-shift)
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
        The data frame of info on each subfault.
    """

    event_tag: str

    # Hypocentre parameters
    latitude: float
    longitude: float
    depth: float
    hypx: Optional[float]
    hypz: Optional[float]

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
    dx: float
    dz: float

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
                if line.startswith("% Data :"):
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
                HEADER_PATTERN,
                header,
            )
            if not parse_result:
                raise FSPParseError("Failed to parse FSP file header")
            metadata = parse_result.named

            # now sniff ahead for the columns
            for line in fsp_file_handle:
                if re.match(r"%\s+LAT\s+LON", line):
                    break
            else:
                raise FSPParseError("Cannot find columns for FSP file!")
            columns = line.lower().lstrip("% ").split()

            data = pd.read_csv(
                fsp_file_handle,
                delimiter=r"\s+",
                header=None,
                names=columns,
                comment="%",
            )
            data = data.rename(columns={"x==ew": "x", "y==ns": "y"})
            fsp_file = cls(data=data, **metadata)

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

            fsp_file.fmin = _normalise_value(fsp_file.fmin)
            fsp_file.fmax = _normalise_value(fsp_file.fmax)

            fsp_file.time_window_count = _normalise_value(fsp_file.time_window_count)
            fsp_file.time_window_length = _normalise_value(fsp_file.time_window_length)
            fsp_file.time_shift = _normalise_value(fsp_file.time_shift)
            return fsp_file
