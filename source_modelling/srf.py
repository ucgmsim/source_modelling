"""Module for handling SRF (Standard Rupture Format) files.

This module provides classes and functions for reading and writing SRF files,
as well as representing their contents.
See https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
for details on the SRF format.

**Why not qcore.srf?**

You might use this module instead of the `qcore.srf` module because:

1. The `qcore.srf` module does not support writing SRF files.

2. Exposing SRF points as a pandas dataframe allows manipulation of
   the points using efficient vectorised operations. We use this in
   rupture propagation to delay ruptures by adding to the `tinit` column.

3. There is better documentation for the new module than the old one.

You should use `qcore.srf` if you do not eventually intend to read all
points of the SRF file (it is memory efficient), or you are working
with code that already uses `qcore.srf`.

Classes: ``SrfFile`` (representation of an SRF file).

Functions: ``read_srf`` (read an SRF file into memory), ``write_srf`` (write an SRF
object to a filepath).

Examples
--------
>>> srf_file = srf.read_srf('/path/to/srf')
>>> srf_file.points['tinit'].max() # get the last time any point in the SRF ruptures
>>> srf_file.points['tinit'] += 1 # delay all points by one second
>>> coordinates.wgs_depth_to_nztm(srf_file.header[['elat', 'elon']].to_numpy())
#   ^ get the coordinates all the fault plane centres in the rupture in NZTM format
# etc...
>>> srf.write_srf('/path/to/srf', srf_file)
"""

import dataclasses
import mmap
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import h5py
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

SW4_PLANE_DTYPE = np.dtype(
    [
        ("ELON", "f4"),
        ("ELAT", "f4"),
        ("NSTK", "i4"),
        ("NDIP", "i4"),
        ("LEN", "f4"),
        ("WID", "f4"),
        ("STK", "f4"),
        ("DIP", "f4"),
        ("DTOP", "f4"),
        ("SHYP", "f4"),
        ("DHYP", "f4"),
    ]
)

SW4_POINTS_DTYPE = np.dtype(
    [
        ("LON", "f4"),
        ("LAT", "f4"),
        ("DEP", "f4"),
        ("STK", "f4"),
        ("DIP", "f4"),
        ("AREA", "f4"),
        ("TINIT", "f4"),
        ("DT", "f4"),
        ("VS", "f4"),
        ("DEN", "f4"),
        ("RAKE", "f4"),
        ("SLIP1", "f4"),
        ("NT1", "i4"),
        ("SLIP2", "f4"),
        ("NT2", "i4"),
        ("SLIP3", "f4"),
        ("NT3", "i4"),
    ]
)

_SW4_POINTS_EXTERNAL_FIELDS = {"VS", "DEN", "NT1", "SLIP2", "NT2", "SLIP3", "NT3"}


def _find_point_blocks(srf_ffp: Path | str, start: int) -> list[tuple[int, int]]:
    """Locate every ``POINTS`` block in an SRF file.

    A single-segment (or otherwise concatenated) SRF has one ``POINTS`` block
    holding all subfaults, whereas genslip writes multi-segment ruptures with
    one ``POINTS`` block per plane. This returns each block so the reader can
    consume them all.

    Parameters
    ----------
    srf_ffp : Path | str
        The path to the SRF file.
    start : int
        Byte offset to begin scanning from (i.e. the end of the plane
        headers), so ``POINTS`` tokens inside the leading comment lines are
        never matched.

    Returns
    -------
    list[tuple[int, int]]
        A ``(data_offset, point_count)`` pair for each ``POINTS`` block, in
        file order. ``data_offset`` is the byte offset of the first point line
        (immediately after the ``POINTS N`` header line).
    """
    blocks = []
    with open(srf_ffp, mode="rb") as srf_file_handle:
        srf_bytes = mmap.mmap(srf_file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            pos = srf_bytes.find(b"POINTS", start)
            while pos != -1:
                line_end = srf_bytes.find(b"\n", pos)
                if line_end == -1:
                    line_end = len(srf_bytes)
                # Only treat "POINTS" as a header when it begins a line, so it
                # cannot be confused with point data or comment text.
                at_line_start = pos == 0 or srf_bytes[pos - 1 : pos] in (b"\n", b"\r")
                if at_line_start:
                    header = srf_bytes[pos:line_end].decode("utf-8", "replace").strip()
                    match = re.match(POINT_COUNT_RE, header)
                    if match:
                        blocks.append((line_end + 1, int(match.group(1))))
                pos = srf_bytes.find(b"POINTS", line_end + 1)
        finally:
            srf_bytes.close()
    return blocks


def _combine_slip_blocks(
    slip_blocks: list[tuple["sp.sparse.csr_array | None", int]],
) -> "sp.sparse.csr_array | None":
    """Vertically stack per-block slip-rate arrays into one array.

    Each block's slip-rate array may have a different number of columns
    (timesteps), since floor(tinit / dt) + nt differs between segments. Blocks
    are padded to the widest column count before stacking. Empty blocks (no
    slip samples) contribute all-zero rows so the row count stays aligned with
    the points.

    Parameters
    ----------
    slip_blocks : list[tuple[csr_array | None, int]]
        A ``(slip_array, point_count)`` pair per block, in file order. The slip
        array is ``None`` when the block records no slip samples.

    Returns
    -------
    csr_array | None
        The combined slip-rate array, or ``None`` when no block records any
        slip.
    """
    if all(slip is None for slip, _ in slip_blocks):
        return None
    if len(slip_blocks) == 1:
        return slip_blocks[0][0]
    max_columns = max(
        (slip.shape[1] for slip, _ in slip_blocks if slip is not None), default=0
    )
    padded = []
    for slip, point_count in slip_blocks:
        if slip is None:
            padded.append(sp.sparse.csr_array((point_count, max_columns)))
        else:
            # Re-widen to the common column count; column indices are unchanged.
            padded.append(
                sp.sparse.csr_array(
                    (slip.data, slip.indices, slip.indptr),
                    shape=(slip.shape[0], max_columns),
                )
            )
    return sp.sparse.csr_array(sp.sparse.vstack(padded, format="csr"))


class Segments(Sequence):
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

    # ty: slice overload missing to satisfy Sequence LSP; fix by adding
    # @overload stubs for int and slice once slice support is implemented.
    def __getitem__(self, index: int) -> pd.DataFrame:  # ty: ignore[invalid-method-override]
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
        A dataframe of the points (subfaults) in the SRF file. The columns are:

        - lon: longitude of the patch.
        - lat: latitude of the patch.
        - dep: depth of the patch (in kilometres).
        - stk: local strike.
        - dip: local dip.
        - area: area of the patch (in cm^2).
        - tinit: initial rupture time for this patch (in seconds).
        - dt: the timestep for all slipt columns (in seconds).
        - vs: shear-wave velocity at the patch (in cm/s). Version 2.0 only.
        - den: density at the patch (in g/cm^3). Version 2.0 only.
        - rake: local rake.
        - slip: total slip (in cm).
        - rise: total rise time (in seconds), computed as nt * dt.

        The vs and den columns are only present when version is "2.0". The
        rise column is computed from the SRF and is not written to disk. See
        the linked documentation on the SRF format for more details.

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
            if version not in {"1.0", "2.0"}:
                raise parse_utils.ParseError(f"Unsupported SRF version: {version}")

            plane_count_line = srf_file_handle.readline().strip()
            while plane_count_line.startswith("#"):  # genslip writes comments after the version line
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

            # The points section begins here. A rupture stores its subfaults
            # either in a single POINTS block (all segments concatenated) or,
            # as genslip writes for multi-segment ruptures, in one POINTS block
            # per plane. Both layouts are read by consuming every POINTS block.
            points_section_start = srf_file_handle.tell()

        read_vs_den = version == "2.0"
        point_blocks = _find_point_blocks(srf_ffp, points_section_start)
        if not point_blocks:
            raise parse_utils.ParseError("Expecting POINTS header line, found none")

        metadata_blocks = []
        slip_blocks = []
        for data_offset, point_count in point_blocks:
            points_metadata, slip = srf_parser.parse_srf(  # type: ignore
                str(srf_ffp), data_offset, point_count, read_vs_den
            )
            metadata_blocks.append(points_metadata)
            slip_blocks.append((slip, point_count))

        points_metadata = (
            metadata_blocks[0]
            if len(metadata_blocks) == 1
            else np.concatenate(metadata_blocks)
        )
        slipt1_array = _combine_slip_blocks(slip_blocks)

        columns = ["lon", "lat", "dep", "stk", "dip", "area", "tinit", "dt"]
        if version == "2.0":
            columns += ["vs", "den"]
        columns += ["rake", "slip", "rise"]

        points_df = pd.DataFrame(
            points_metadata.reshape((-1, len(columns))),
            columns=columns,
        )

        return cls(
            version,
            headers,
            points_df,
            slipt1_array,
        )

    def write_srf(self, srf_ffp: str | Path) -> None:
        """Write an SRFFile object to a file.

        Parameters
        ----------
        srf_ffp : Path
            The path to the output SRF.

        """

        with open(srf_ffp, mode="w", encoding="utf-8") as srf_file_handle:
            srf_file_handle.write(f"{self.version}\n")
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

    def write_sw4_hdf5(
        self,
        output_ffp: Path | str,
    ) -> None:
        """Write the SRF file in SW4's SRF-HDF5 format.

        Parameters
        ----------
        output_ffp : Path
            The path to the output HDF5 file.

        References
        ----------
        .. [1] Petersson, N.A. and B. Sjogreen (2017). SW4 v2.0.
           Computational Infrastructure of Geodynamics, Davis, CA.
           DOI: 10.5281/zenodo.1045297.
        .. [2] Petersson, N.A. and B. Sjogreen (2017). User's guide to
           SW4, version 2.0. Technical report LLNL-SM-741439, Lawrence
           Livermore National Laboratory, Livermore, CA.
           https://github.com/geodynamics/sw4/blob/master/doc/SW4_UsersGuide.pdf
        """
        plane_data = np.empty(len(self.header), dtype=SW4_PLANE_DTYPE)
        assert SW4_PLANE_DTYPE.names is not None
        for field in SW4_PLANE_DTYPE.names:
            plane_data[field] = self.header[field.lower()].values.astype(
                SW4_PLANE_DTYPE[field].type  # ty: ignore[invalid-argument-type]
            )  # ty: ignore[invalid-assignment]

        # Build POINTS structured array
        points_data: np.ndarray = np.zeros(len(self.points), dtype=SW4_POINTS_DTYPE)
        assert SW4_POINTS_DTYPE.names is not None
        for field in SW4_POINTS_DTYPE.names:
            if field in _SW4_POINTS_EXTERNAL_FIELDS:
                continue
            points_data[field] = self.points[
                "slip" if field == "SLIP1" else field.lower()
            ].values.astype(SW4_POINTS_DTYPE[field].type)  # ty: ignore

        points_data["NT1"] = np.diff(self.slipt1_array.indptr).astype(np.int32)
        if self.version == "2.0":  # vs/den are mandatory in 2.0; missing columns will fail loudly
            points_data["VS"] = self.points["vs"].to_numpy().astype(np.float32)
            points_data["DEN"] = self.points["den"].to_numpy().astype(np.float32)

        with h5py.File(output_ffp, "w") as h5file:
            h5file.attrs.create("VERSION", np.float32(self.version))
            h5file.attrs.create("PLANE", plane_data)
            h5file.create_dataset("POINTS", data=points_data)
            h5file.create_dataset("SR1", data=self.slipt1_array.data.astype(np.float32))

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
                centroid = segment_header[["elat", "elon"]].to_numpy()
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
                centroid = segment_header[["elat", "elon"]].to_numpy()
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


def write_srf(srf_ffp: str | Path, srf: SrfFile) -> None:
    """Write an SRF object to a filepath.

    Parameters
    ----------
    srf_ffp : Path
        The filepath to write the srf object to.
    srf : SrfFile
        The SRF object.
    """
    srf.write_srf(srf_ffp)
