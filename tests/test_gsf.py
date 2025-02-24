from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely

from qcore import coordinates
from source_modelling import gsf
from source_modelling.sources import Fault, Plane, Point


def test_point_gsf():  # Use tmp_path fixture
    point_source = Point.from_lat_lon_depth(
        np.array([-43.0, 172.0, 500.0]), strike=45, dip=90, dip_dir=25, length_m=1e4
    )
    gsf_df = gsf.source_to_gsf_dataframe(point_source, 1.0)
    assert len(gsf_df) == 1
    assert gsf_df["lat"][0] == pytest.approx(-43.0)
    assert gsf_df["lon"][0] == pytest.approx(172.0)
    assert gsf_df["dep"][0] == pytest.approx(0.5)
    assert gsf_df["loc_stk"][0] == pytest.approx(45.0)
    assert gsf_df["loc_dip"][0] == pytest.approx(90.0)
    assert gsf_df["sub_dx"][0] == pytest.approx(10.0)
    assert gsf_df["sub_dy"][0] == pytest.approx(10.0)
    assert gsf_df["seg_no"][0] == 0


def test_plane_gsf():  # Use tmp_path fixture
    plane = Plane.from_centroid_strike_dip(
        np.array([-43.0, 172.0, 15e4]), 45, 5, 10, strike=90
    )
    gsf_df = gsf.source_to_gsf_dataframe(plane, 1.0)
    assert (
        len(gsf_df) == 5 * 10
    )  # Correct number of points for this setup. Adjust if corners are changed.
    assert gsf_df["loc_stk"].unique()[0] == pytest.approx(90, abs=0.1)
    assert gsf_df["loc_dip"].unique()[0] == pytest.approx(45)
    assert gsf_df["sub_dx"].unique()[0] == pytest.approx(1.0)
    assert gsf_df["sub_dy"].unique()[0] == pytest.approx(1.0)
    assert gsf_df["seg_no"].unique()[0] == 0

    for _, point in gsf_df.iterrows():
        assert plane.geometry.contains(
            shapely.Point(
                coordinates.wgs_depth_to_nztm(point[["lat", "lon", "dep"]].values)
            )
        )


def test_bad_gsf_type():
    with pytest.raises(TypeError):
        gsf.source_to_gsf_dataframe(1, 0.1)


def test_write_gsf(tmp_path: Path):
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

    read_df = gsf.read_gsf(filepath)
    pd.testing.assert_frame_equal(read_df, df)


def test_write_bad_df():

    df = pd.DataFrame(
        {
            "lon": [1, 2, 3],
            "lat": [4, 5, 6],
            "dep": [7, 8, 9],
            "sub_dx": [10, 11, 12],
            "sub_dy": [13, 14, 15],
            "loc_stk": [16, 17, 18],
            "loc_dip": [19, 20, 21],
            "slip": [-1, -1, -1],
            "init_time": [-1, -1, -1],
            "seg_no": [0, 0, 0],
        }
    )
    filepath = "test.gsf"
    with pytest.raises(
        ValueError, match="The DataFrame must have a 'loc_rake' column."
    ):
        gsf.write_gsf(df, filepath)


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


def test_read_gsf(tmp_path: Path):
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
