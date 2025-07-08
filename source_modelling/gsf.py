"""This module provides functions for working with and generating GSF files.

Functions
---------
source_to_gsf_dataframe(gsf_filepath, source, resolution)
    Generates a pandas DataFrame suitable for writing to a GSF file from a source object.
write_gsf(gsf_df, gsf_filepath)
    Writes a pandas DataFrame to a GSF file.
read_gsf(gsf_filepath)
    Parses a GSF file into a pandas DataFrame.

References
----------
The GSF file format is used to define the grid of the source model for a fault.
See https://ucdigitalsms.atlassian.net/wiki/spaces/QuakeCore/pages/3291694768/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-SRFinfoformat
for details on the GSF format.
"""

from pathlib import Path

import pandas as pd

from qcore import grid
from source_modelling import sources


def source_to_gsf_dataframe(
    source: sources.IsSource, resolution: float
) -> pd.DataFrame:
    """Generates a pandas DataFrame suitable for writing to a GSF file from a source object.

    This function discretises the source into a grid of points and returns a DataFrame
    containing the necessary information for a GSF file.  It handles Point, Plane, and
    Fault source types.

    Parameters
    ----------
    source : sources.IsSource
        The source object (Point, Plane, or Fault) to convert.
    resolution : float
        The spacing between grid points in km.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the grid points and their properties, ready to be
        written to a GSF file.  The DataFrame contains the following columns:
        'lat', 'lon', 'dep', 'sub_dx', 'sub_dy', 'loc_stk', 'loc_dip', 'seg_no'.
    """
    segments: list[sources.Plane] = []
    if isinstance(source, sources.Point):
        return pd.DataFrame(
            {
                "lat": [source.centroid[0]],
                "lon": [source.centroid[1]],
                "dep": [source.centroid[2] / 1000],
                "sub_dx": [source.length],
                "sub_dy": [source.width],
                "loc_stk": [source.strike],
                "loc_dip": [source.dip],
                "seg_no": [0],
            }
        )
    elif isinstance(source, sources.Plane):
        segments = [source]
    elif isinstance(source, sources.Fault):
        segments = source.planes
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    meshgrids = [
        grid.coordinate_patchgrid(
            plane.corners[0],
            plane.corners[1],
            plane.corners[-1],
            resolution,
            nx=round(plane.length / resolution),
            ny=round(plane.width / resolution),
        )
        for plane in segments
    ]

    strike_rows: list[pd.DataFrame] = []
    dxs = [plane.length / round(plane.length / resolution) for plane in segments]
    dys = [plane.width / round(plane.width / resolution) for plane in segments]
    n_dip = meshgrids[0].shape[0]

    for dip_index in range(n_dip):
        for plane_index, plane in enumerate(segments):
            points = meshgrids[plane_index][dip_index]
            strike_rows.append(
                pd.DataFrame(
                    {
                        "lat": points[:, 0],
                        "lon": points[:, 1],
                        "dep": points[:, 2] / 1000,
                        "sub_dx": dxs[plane_index],
                        "sub_dy": dys[plane_index],
                        "loc_stk": plane.strike,
                        "loc_dip": plane.dip,
                        "seg_no": plane_index,
                    }
                )
            )
    return pd.concat(strike_rows)


def write_gsf(gsf_df: pd.DataFrame, gsf_filepath: Path):
    """Writes a pandas DataFrame to a GSF file.

    Parameters
    ----------
    gsf_df : pd.DataFrame
        The DataFrame containing the GSF data.  Must contain at least the columns
        'lon', 'lat', 'dep', 'sub_dx', 'sub_dy', 'loc_stk', 'loc_dip', and 'loc_rake'.
    gsf_filepath : Path
        The path to the GSF file to write. Parent directories will be created if they
        do not exist.

    """

    gsf_filepath.parent.mkdir(parents=True, exist_ok=True)

    if "init_time" not in gsf_df:
        gsf_df["init_time"] = -1
    if "slip" not in gsf_df:
        gsf_df["slip"] = -1
    if "loc_rake" not in gsf_df:
        raise ValueError("The DataFrame must have a 'loc_rake' column.")
    with open(gsf_filepath, "w") as gsf_file:
        gsf_file.write(f"{len(gsf_df)}\n")
        gsf_df.to_csv(
            gsf_file,
            sep=" ",
            columns=[
                "lon",
                "lat",
                "dep",
                "sub_dx",
                "sub_dy",
                "loc_stk",
                "loc_dip",
                "loc_rake",
                "slip",
                "init_time",
                "seg_no",
            ],
            header=False,
            index=False,
        )


def read_gsf(gsf_filepath: Path) -> pd.DataFrame:
    """Parses a GSF file into a pandas DataFrame.

    Parameters
    ----------
    gsf_filepath : Path
        The path to the GSF file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the GSF data.  Columns include:
        'lon', 'lat', 'dep', 'sub_dx', 'sub_dy', 'loc_stk', 'loc_dip', 'rake', 'slip',
        'init_time', and 'seg_no'.
    """
    with open(gsf_filepath, mode="r", encoding="utf-8") as gsf_file_handle:
        while next(gsf_file_handle).startswith("#"):
            pass
        # Note this skips one past the last '#' comment in the header, which is
        # ok because that also skips the number of points (which we do not need
        # to read).
        return pd.read_csv(
            gsf_file_handle,
            comment="#",
            sep=r"\s+",
            header=None,
            names=[
                "lon",
                "lat",
                "dep",
                "sub_dx",
                "sub_dy",
                "loc_stk",
                "loc_dip",
                "loc_rake",
                "slip",
                "init_time",
                "seg_no",
            ],
        )
