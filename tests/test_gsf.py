from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import scipy as sp
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


@pytest.mark.parametrize(
    "fault",
    [
        Fault(
            planes=[
                Plane(
                    bounds=np.array(
                        [
                            [5.84932589e06, 2.13165408e06, 0.00000000e00],
                            [5.85032569e06, 2.13167328e06, 0.00000000e00],
                            [5.85030458e06, 2.13277306e06, 6.39950449e00],
                            [5.84930477e06, 2.13275386e06, 6.39950449e00],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [5.85032569e06, 2.13167328e06, 0.00000000e00],
                            [5.85082560e06, 2.13168288e06, 0.00000000e00],
                            [5.85080449e06, 2.13278266e06, 6.39950449e00],
                            [5.85030458e06, 2.13277306e06, 6.39950449e00],
                        ]
                    )
                ),
            ]
        ),
        Fault(
            planes=[
                Plane(
                    bounds=np.array(
                        [
                            [6.39588868e06, 4.62893158e05, 0.00000000e00],
                            [6.39478884e06, 4.62912355e05, 0.00000000e00],
                            [6.39478458e06, 4.62668397e05, 2.27106518e02],
                            [6.39588442e06, 4.62649199e05, 2.27106518e02],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [6.39478884e06, 4.62912355e05, 0.00000000e00],
                            [6.39445556e06, 4.62918173e05, 0.00000000e00],
                            [6.39445130e06, 4.62674214e05, 2.27106518e02],
                            [6.39478458e06, 4.62668397e05, 2.27106518e02],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [6.39445556e06, 4.62918173e05, 0.00000000e00],
                            [6.39435558e06, 4.62919918e05, 0.00000000e00],
                            [6.39435132e06, 4.62675960e05, 2.27106518e02],
                            [6.39445130e06, 4.62674214e05, 2.27106518e02],
                        ]
                    )
                ),
            ]
        ),
        Fault(
            planes=[
                Plane(
                    bounds=np.array(
                        [
                            [5.46622207e06, 2.19121633e06, 0.00000000e00],
                            [5.46632207e06, 2.19121633e06, 0.00000000e00],
                            [5.46632207e06, 2.19129722e06, 3.23371399e02],
                            [5.46622207e06, 2.19129722e06, 3.23371399e02],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [5.46632207e06, 2.19121633e06, 0.00000000e00],
                            [5.46742207e06, 2.19121633e06, 0.00000000e00],
                            [5.46742207e06, 2.19129722e06, 3.23371399e02],
                            [5.46632207e06, 2.19129722e06, 3.23371399e02],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [5.46742207e06, 2.19121633e06, 0.00000000e00],
                            [5.46852207e06, 2.19121633e06, 0.00000000e00],
                            [5.46852207e06, 2.19129722e06, 3.23371399e02],
                            [5.46742207e06, 2.19129722e06, 3.23371399e02],
                        ]
                    )
                ),
            ]
        ),
        Fault(
            planes=[
                Plane(
                    bounds=np.array(
                        [
                            [4.43785023e06, 2.10145126e06, 0.00000000e00],
                            [4.43895023e06, 2.10145126e06, 0.00000000e00],
                            [4.43895023e06, 2.10145126e06, 2.00001000e03],
                            [4.43785023e06, 2.10145126e06, 2.00001000e03],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [4.43895023e06, 2.10145126e06, 0.00000000e00],
                            [4.44205444e06, 2.10145126e06, 0.00000000e00],
                            [4.44205444e06, 2.10145126e06, 2.00001000e03],
                            [4.43895023e06, 2.10145126e06, 2.00001000e03],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [4.44205444e06, 2.10145126e06, 0.00000000e00],
                            [4.44305443e06, 2.10145126e06, 0.00000000e00],
                            [4.44305443e06, 2.10145126e06, 2.00001000e03],
                            [4.44205444e06, 2.10145126e06, 2.00001000e03],
                        ]
                    )
                ),
            ]
        ),
        Fault(
            planes=[
                Plane(
                    bounds=np.array(
                        [
                            [4.55517696e06, 2.03874311e06, 0.00000000e00],
                            [4.56517087e06, 2.03909211e06, 0.00000000e00],
                            [4.56517087e06, 2.03909211e06, 9.99990000e02],
                            [4.55517696e06, 2.03874311e06, 9.99990000e02],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [4.56517087e06, 2.03909211e06, 0.00000000e00],
                            [4.56567056e06, 2.03910956e06, 0.00000000e00],
                            [4.56567056e06, 2.03910956e06, 9.99990000e02],
                            [4.56517087e06, 2.03909211e06, 9.99990000e02],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [4.56567056e06, 2.03910956e06, 0.00000000e00],
                            [4.57566447e06, 2.03945855e06, 0.00000000e00],
                            [4.57566447e06, 2.03945855e06, 9.99990000e02],
                            [4.56567056e06, 2.03910956e06, 9.99990000e02],
                        ]
                    )
                ),
            ]
        ),
        Fault(
            planes=[
                Plane(
                    bounds=np.array(
                        [
                            [6549282.98796971, 2269047.98669433, 0.0],
                            [6548496.56209932, 2269665.65517376, 0.0],
                            [6546348.99366583, 2266931.33501533, 9376.11023265],
                            [6547135.41953622, 2266313.6665359, 9376.11023265],
                        ]
                    )
                ),
                Plane(
                    bounds=np.array(
                        [
                            [6548496.56209932, 2269665.65517376, 0.0],
                            [6547002.33800333, 2270839.23702049, 0.0],
                            [6544854.76956984, 2268104.91686207, 9376.11023265],
                            [6546348.99366583, 2266931.33501533, 9376.11023265],
                        ]
                    )
                ),
            ]
        ),
    ],
)
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
2
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
