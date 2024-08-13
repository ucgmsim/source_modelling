from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Callable

import pandas as pd
import parse

LOC_PATTERN = "Loc : LAT = {latitude:f} LON = {longitude:f} DEP = {depth:f}"
SIZE_PATTERN = "Size : LEN = {length:f} km WID = {width:f} km Mw = {magnitude:f} Mo = {moment:e} Nm"
MECH_PATTERN = (
    "Mech : STRK = {strike:f} DIP = {dip:f} RAKE = {rake:f} Htop = {htop:f} km"
)
RUPT_PATTERN = "Rupt : HypX = {hypx:f} km Hypz = {hypz:f} km avTr = {average_rise_time:f} s avVr = {average_rupture_speed:f} km/s"
SVF_PATTERN = "SVF : {slip_velocity_function} (type of slip-velocity function used)"
DELTA_PATTERN = "Invs : Dx = {dx:f} km Dz = {dz:f} km"


class FSPParseError(Exception):
    """Exception raised for errors in parsing FSP files."""

    pass


@dataclasses.dataclass
class FSPFile:
    latitude: float
    longitude: float
    depth: float
    length: float
    width: float
    magnitude: float
    moment: float
    strike: float
    dip: float
    rake: float
    htop: float
    hypx: float
    hypz: float
    dx: float
    dz: float
    average_rise_time: float
    average_rupture_speed: float
    data: pd.DataFrame
    slip_velocity_function: str

    @classmethod
    def read_from_file(cls: Callable, fsp_ffp: Path) -> Callable:
        with open(fsp_ffp, "r") as fsp_file_handle:
            metadata: dict[str, float] = {}
            for line in fsp_file_handle:
                if line.strip() == "% SOURCE MODEL PARAMETERS":
                    break
                # Strip the leading "% ", and deduplicate the spaces in the line.
                # This is required to normalise the string so that the parse
                # module can handle the rest of the parsing
                line_dedup_spaces = " ".join(line.lstrip("% ").split())
                parse_result = (
                    parse.parse(LOC_PATTERN, line_dedup_spaces)
                    or parse.parse(SIZE_PATTERN, line_dedup_spaces)
                    or parse.parse(MECH_PATTERN, line_dedup_spaces)
                    or parse.parse(RUPT_PATTERN, line_dedup_spaces)
                    or parse.parse(SVF_PATTERN, line_dedup_spaces)
                    or parse.parse(DELTA_PATTERN, line_dedup_spaces)
                )
                if parse_result:
                    metadata |= parse_result.named
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
            return cls(data=data, **metadata)
