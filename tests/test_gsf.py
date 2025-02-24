from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy as sp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
import shapely

from qcore import coordinates
from source_modelling import gsf
from source_modelling.sources import Fault, Plane


def test_write_gsf(tmp_path: Path):  # Use tmp_path fixture for temporary files
    df = pd.DataFrame(
        {
            "lon": [1, 2, 3],
            "lat": [4, 5, 6],
            "dep": [7, 8, 9],
            "sub_dx": [10, 11, 12],
            "sub_dy": [13, 14, 15],
            "loc_stk": [16, 17, 18],
            "loc_dip": [19, 20, 21],
            "loc_rake": [22, 23, 24],
            "slip": [-1, -1, -1],
            "init_time": [-1, -1, -1],
            "seg_no": [0, 0, 0],
        }
    )
    filepath = tmp_path / "test.gsf"
    gsf.write_gsf(df, filepath)
    assert filepath.exists()

    # you could read the file back and check that the contents are correct, but that would be testing read_gsf as well.
    read_df = gsf.read_gsf(filepath)
    pd.testing.assert_frame_equal(read_df, df)


def connected_fault(
    lengths: list[float],
    width: float,
    strike: float,
    dip: float,
    start_coordinates: np.ndarray,
) -> Fault:
    """Create a Fault object from connected planes."""
    strike_direction = np.array(
        [np.cos(np.radians(strike)), np.sin(np.radians(strike)), 0]
    )
    dip_rotation = sp.spatial.transform.Rotation.from_rotvec(
        dip * strike_direction, degrees=True
    )
    dip_direction = np.array(
        [np.cos(np.radians(strike + 90)), np.sin(np.radians(strike + 90)), 0]
    )
    dip_direction = width * 1000 * dip_rotation.apply(dip_direction)
    start = np.append(coordinates.wgs_depth_to_nztm(start_coordinates), 0)
    cumulative_lengths = np.cumsum(np.array(lengths) * 1000)
    leading_edges = np.vstack(
        (start, start + np.outer(cumulative_lengths, strike_direction))
    )
    planes = [
        Plane(
            np.array(
                [
                    leading_edges[i],
                    leading_edges[i + 1],
                    leading_edges[i + 1] + dip_direction,
                    leading_edges[i] + dip_direction,
                ]
            )
        )
        for i in range(len(leading_edges) - 1)
    ]
    return Fault(planes)


def coordinate(lat: float, lon: float, depth: Optional[float] = None) -> np.ndarray:
    """Create a coordinate array from latitude, longitude, and optional depth."""
    if depth is not None:
        return np.array([lat, lon, depth])
    return np.array([lat, lon])


FINITE_FAULT = st.builds(
    connected_fault,
    lengths=st.lists(st.floats(0.1, 10), min_size=1, max_size=3),
    width=st.floats(0.1, 10),
    strike=st.floats(0, 179),
    dip=st.floats(0.1, 90),
    start_coordinates=st.builds(
        coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)
    ),
)


@given(fault=FINITE_FAULT)
@settings(deadline=None)
def test_fault_to_gsf(fault: Fault):
    gsf_df = gsf.source_to_gsf_dataframe(fault, 0.1)
    assert (gsf_df["sub_dx"] * gsf_df["sub_dy"]).sum() == pytest.approx(
        fault.area(), abs=0.1**2
    )
    if fault.dip != 90:
        for _, point in gsf_df.iterrows():
            assert fault.geometry.contains(
                shapely.Point(
                    coordinates.wgs_depth_to_nztm(point[["lat", "lon", "dep"]].values)
                )
            )


# Test read_gsf
def test_read_gsf(tmp_path: Path):
    # Create a dummy GSF file for testing
    content = """# Some comment
1 2 3 4 5 6 7 8 -1 -1 0
9 10 11 12 13 14 15 16 -1 -1 0"""
    filepath = tmp_path / "test.gsf"
    filepath.write_text(content)

    df = gsf.read_gsf(filepath)
    assert not df.empty
    assert len(df) == 2
    assert list(df.columns) == [
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
    ]
    np.testing.assert_equal(
        df[
            [
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
            ]
        ].values,
        np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 0],
                [9, 10, 11, 12, 13, 14, 15, 16, -1, -1, 0],
            ]
        ),
    )
