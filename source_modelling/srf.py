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
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Self, overload

import numpy as np
import pandas as pd
import scipy as sp
import shapely
import xarray as xr

from qcore import coordinates
from source_modelling import parse_utils, srf_parser
from source_modelling.sources import Plane

PLANE_COUNT_RE = r"PLANE (\d+)"
POINT_COUNT_RE = r"POINTS (\d+)"


class Segments(Sequence[pd.DataFrame]):
    """A read-only view for SRF segments.

    Parameters
    ----------
    header : pd.DataFrame
        The header of the SRF file.
    points : pd.DataFrame
        The points of the SRF file.
    """

    def __init__(self, header: pd.DataFrame, points: pd.DataFrame) -> None:
        """Initialise the Segments object.

        Parameters
        ----------
        header : pd.DataFrame
            The header of the SRF file.
        points : pd.DataFrame
            The points of the SRF file.
        """
        self._header = header
        self._points = points

    @overload
    def __getitem__(self, index: int) -> pd.DataFrame: ...  # numpydoc ignore=GL08
    @overload
    def __getitem__(
        self, index: slice
    ) -> Sequence[pd.DataFrame]: ...  # numpydoc ignore=GL08

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
        """
        Returns
        -------
        int
            The number of segments in the SRF.
        """
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
        - elat: The centre latitude of the plane.
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
        - dt: the timestep for all slipt columns (in seconds).
        - rake: local rake.
        - slip: total slip.

        The final two columns are computed from the SRF and are not saved to
        disk. See the linked documentation on the SRF format for more details.

    slipt1_array : csr_array
        A sparse array containing the slip for each point and at each timestep, where
        slipt1_array[i, j] is the slip for the ith patch at time t = j * dt.

    References
    ----------
    SRF File Format Doc: https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
    """

    version: str
    header: pd.DataFrame
    points: pd.DataFrame
    slipt1_array: sp.sparse.csr_array

    @classmethod
    def from_file(cls, srf_ffp: Path | str) -> Self:
        """Read an srf file from a filepath.

        Parameters
        ----------
        srf_ffp : Path
            The path to the srf file.

        Returns
        -------
        Self
            The SRFFile instance for this path.
        """
        with open(srf_ffp, mode="r", encoding="utf-8") as srf_file_handle:
            version = srf_file_handle.readline().strip()

            plane_count_line = srf_file_handle.readline().strip()
            plane_count_match = re.match(PLANE_COUNT_RE, plane_count_line)
            if not plane_count_match:
                raise parse_utils.ParseError(
                    f'Expecting PLANE header line, got: "{plane_count_line}"'
                )
            plane_count = int(plane_count_match.group(1))
            segments = []

            for _ in range(plane_count):
                segments.append(
                    {
                        "elon": parse_utils.read_float(srf_file_handle),
                        "elat": parse_utils.read_float(srf_file_handle),
                        "nstk": parse_utils.read_int(srf_file_handle),
                        "ndip": parse_utils.read_int(srf_file_handle),
                        "len": parse_utils.read_float(srf_file_handle),
                        "wid": parse_utils.read_float(srf_file_handle),
                        "stk": parse_utils.read_float(srf_file_handle),
                        "dip": parse_utils.read_float(srf_file_handle),
                        "dtop": parse_utils.read_float(srf_file_handle),
                        "shyp": parse_utils.read_float(srf_file_handle),
                        "dhyp": parse_utils.read_float(srf_file_handle),
                    }
                )
            headers = pd.DataFrame(segments)
            headers["nstk"] = headers["nstk"].astype(int)
            headers["ndip"] = headers["ndip"].astype(int)

            points_count_line = srf_file_handle.readline().strip()
            points_count_match = re.match(POINT_COUNT_RE, points_count_line)
            if not points_count_match:
                raise parse_utils.ParseError(
                    f'Expecting POINTS header line, got: "{points_count_line}"'
                )
            point_count = int(points_count_match.group(1))
            position = srf_file_handle.tell()

        points_metadata, slipt1_array = srf_parser.parse_srf(  # type: ignore
            str(srf_ffp), position, point_count
        )

        points_df = pd.DataFrame(
            points_metadata.reshape((-1, 11)),
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
                "slip",
                "rise",
            ],
        )

        return cls(
            version,
            headers,
            points_df,
            slipt1_array,
        )

    def write_srf(self, srf_ffp: Path) -> None:
        """Write an SRFFile object to a file.

        Parameters
        ----------
        srf_ffp : Path
            The path to the output SRF.

        """

        with open(srf_ffp, mode="w", encoding="utf-8") as srf_file_handle:
            srf_file_handle.write("1.0\n")
            srf_file_handle.write(f"PLANE {len(self.header)}\n")
            # Cannot use self.header.to_string because the newline separating headers is significant!
            # This is ok because the number of headers is typically very small (< 100)
            for _, plane in self.header.iterrows():
                srf_file_handle.write(
                    "\n".join(
                        [
                            f"{plane['elon']:.6f} {plane['elat']:.6f} {int(plane['nstk'])} {int(plane['ndip'])} {plane['len']:.4f} {plane['wid']:.4f}",
                            f"{plane['stk']:.4f} {plane['dip']:.4f} {plane['dtop']:.4f} {plane['shyp']:.4f} {plane['dhyp']:.4f}",
                            "",
                        ]
                    )
                )

            srf_file_handle.write(f"POINTS {len(self.points)}\n")

        srf_parser.write_srf_points(  # type: ignore
            str(srf_ffp),
            self.points.values.astype(np.float32),
            self.slip.indptr,
            self.slip.data,
        )

    def write_hdf5(
        self, hdf5_ffp: Path, include_slip_time_function: bool = True
    ) -> None:
        """Write an SRFFile to disk in an HDF5 format using xarray's to_netcdf.

        Parameters
        ----------
        hdf5_ffp : Path
            The path to the HDF5 file to save to.
        include_slip_time_function : bool
            If True, include the slip time function in the HDF5
            output. Slower and outputs larger files.
        """

        self.to_xarray(include_slip_time_function=include_slip_time_function).to_netcdf(
            hdf5_ffp,
            engine="h5netcdf",
            encoding={
                # Apply compression to the 'data' variable of the sparse array
                "data": {"compression": "zlib", "complevel": 9},
                # Apply compression to the 'indices' variable of the sparse array
                "indices": {"compression": "zlib", "complevel": 9},
            }
            if include_slip_time_function
            else None,
        )

    @classmethod
    def from_hdf5(cls, hdf5_ffp: Path) -> Self:
        """
        Reads an SRFFile object from an HDF5 file.

        Parameters
        ----------
        hdf5_ffp : Path
            The file path to the HDF5 file.

        Returns
        -------
        SrfFile
            An instance of the SrfFile class reconstructed from the HDF5 data.
        """
        ds = xr.open_dataset(hdf5_ffp, engine="h5netcdf")

        header_data = {
            var_name[len("plane_") :]: ds[var_name].values
            for var_name in ds.data_vars
            if isinstance(var_name, str) and var_name.startswith("plane_")
        }
        header_df = pd.DataFrame(header_data)
        header_df[["nstk", "ndip"]] = header_df[["nstk", "ndip"]].astype(int)

        points_data = {
            col: ds[col].values
            for col in ds.data_vars
            if isinstance(col, str)
            and not col.startswith("plane_")
            and col not in {"data", "indices", "indptr"}
        }
        points_df = pd.DataFrame(points_data)

        data = ds["data"].values
        indices = ds["indices"].values
        indptr_saved = ds["indptr"].values
        reconstructed_indptr = np.append(indptr_saved, len(data))

        slipt1_array = sp.sparse.csr_array((data, indices, reconstructed_indptr))

        return cls(
            version=ds.attrs["version"],
            header=header_df,
            points=points_df,
            slipt1_array=slipt1_array,
        )

    def to_xarray(self, include_slip_time_function: bool = True) -> xr.Dataset:
        """Convert an SRFFile into an xarray dataset.

        Parameters
        ----------
        include_slip_time_function : bool, default False
            If True, include the slip time functions as well as the
            slip summaries in the SRF. Slower.

        Returns
        -------
        xr.Dataset
            An xarray dataset containing the information from an SRF
            file.
        """
        # Prepare data variables and coordinates for the header Dataset
        header_data_vars = {
            f"plane_{col}": ("segment", self.header[col].values)
            for col in self.header.columns
        }
        header_coords = {"segment": np.arange(len(self.header))}
        header_ds = xr.Dataset(header_data_vars, coords=header_coords)

        points_data_vars = {
            col: ("patch", self.points[col].values) for col in self.points.columns
        }
        points_coords = {"patch": np.arange(len(self.points))}
        points_ds = xr.Dataset(points_data_vars, coords=points_coords)

        datasets = [header_ds, points_ds]
        if include_slip_time_function:
            n_patches, n_timesteps = self.slipt1_array.shape
            slip_ds = xr.Dataset(
                {
                    "data": (("nz_idx",), self.slipt1_array.data),
                    "indices": (("nz_idx",), self.slipt1_array.indices),
                    "indptr": (
                        ("row",),
                        self.slipt1_array.indptr[:-1],
                    ),  # Apply slicing to the data
                },
                coords={
                    "row": np.arange(n_patches),
                    "col": np.arange(n_timesteps),
                },
                attrs={
                    "sparse_format": "csr",
                    "original_shape": self.slipt1_array.shape,
                    "units": "cm",
                    "description": "Slip for each patch at each timestep",
                },
            )
            datasets.append(slip_ds)
        ds = xr.merge(datasets)
        ds.attrs["version"] = self.version

        return ds

    @property
    def slip(self):  # numpydoc ignore=RT01
        "csr_array: A sparse array containing slip-time functions for each point."
        return self.slipt1_array

    @property
    def geometry(self) -> shapely.Geometry:  # numpydoc ignore=RT01
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
    def nt(self):  # numpydoc ignore=RT01
        """int: The number of timeslices in the SRF."""
        return self.slipt1_array.shape[1]

    @property
    def dt(self):  # numpydoc ignore=RT01
        """float: time resolution of SRF."""
        return self.points["dt"].iloc[0]

    @property
    def segments(self) -> Segments:  # numpydoc ignore=RT01
        """Segments: A sequence of segments in the SRF."""
        return Segments(self.header, self.points)

    @property
    def planes(self) -> list[Plane]:  # numpydoc ignore=RT01
        """list[Plane]: The list of planes in the SRF."""
        # The following method relies as little as possible on the SRF header
        # values. This is because they frequently lie! See the darfield SRF
        # in the test cases for examples of this
        planes = []
        for (_, segment_header), segment in zip(self.header.iterrows(), self.segments):
            nstk = segment_header["nstk"]
            ndip = segment_header["ndip"]
            if nstk == 1 and ndip > 1:
                # If the number of strike points is 1, we have to rely on the segment header for strike.
                centroid = segment_header[["elat", "elon"]]
                strike_nztm = coordinates.great_circle_bearing_to_nztm_bearing(
                    centroid,
                    segment_header["len"],
                    segment_header["stk"],
                )
                strike_direction = (
                    segment_header["len"]
                    * 1000
                    / 2
                    * np.array([np.cos(strike_nztm), np.sin(strike_nztm), 0])
                )
                top = coordinates.wgs_depth_to_nztm(
                    segment[["lat", "lon", "dep"]].iloc[0].values
                    * np.array([1, 1, 1000])
                )
                next = coordinates.wgs_depth_to_nztm(
                    segment[["lat", "lon", "dep"]].iloc[1].values
                    * np.array([1, 1, 1000])
                )
                bottom = coordinates.wgs_depth_to_nztm(
                    segment[["lat", "lon", "dep"]].iloc[-1].values
                    * np.array([1, 1, 1000])
                )
                dip_direction = (next - top) / 2
                planes.append(
                    Plane(
                        np.array(
                            [
                                top - strike_direction - dip_direction,
                                top + strike_direction - dip_direction,
                                bottom - strike_direction + dip_direction,
                                bottom + strike_direction + dip_direction,
                            ]
                        )
                    )
                )
            elif ndip == 1:
                # If the number of dip points is 1, we have to rely on the
                # segment header for dip direction. We will assume that dip
                # direction = strike + 90.
                centroid = segment_header[["elat", "elon"]]
                planes.append(
                    Plane.from_centroid_strike_dip(
                        centroid,
                        segment_header["dip"],
                        segment_header["len"],
                        segment_header["wid"],
                        dtop=segment_header["dtop"],
                        strike=segment_header["stk"],
                    )
                )
            else:
                # These points are the outer-most points and centres of the
                # corner patches in the SRF (* in the diagram below).
                corners = coordinates.wgs_depth_to_nztm(
                    segment[["lat", "lon", "dep"]]
                    .iloc[[0, nstk - 1, nstk * (ndip - 1), nstk * ndip - 1]]
                    .values
                ) * np.array([1, 1, 1000])
                # These points are the next step inside the SRF from the corners
                # (marked . in the diagram below).
                interior = coordinates.wgs_depth_to_nztm(
                    segment[["lat", "lon", "dep"]]
                    .iloc[
                        [
                            nstk + 1,
                            2 * (nstk - 1),
                            (ndip - 2) * nstk + 1,
                            nstk * (ndip - 1) - 2,
                        ]
                    ]
                    .values
                ) * np.array([1, 1, 1000])
                #
                # ┌─────────────────┐
                # │*               *│             * - corner patch centres
                # │                 │             . - interior patch centres
                # │  .           .  │             | - actual geometry
                # │                 │
                # │                 │
                # │                 │
                # │                 │
                # │  .           .  │
                # │                 │
                # │*               *│
                # └─────────────────┘
                # the difference (corners - interior) / 2 is half the distance
                # between patch centres, distance between patch centre and patch corners.
                planes.append(Plane(corners + (corners - interior) / 2))
        return planes


def read_srf(srf_ffp: Path | str) -> SrfFile:
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
    return SrfFile.from_file(srf_ffp)


def write_srf(srf_ffp: Path | str, srf: SrfFile) -> None:
    """Write an SRF object to a filepath.

    Parameters
    ----------
    srf_ffp : Path | str
        The filepath to write the srf object to.
    srf : SrfFile
        The SRF object.
    """
    srf.write_srf(Path(srf_ffp))
