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
        np.array([-43.0, 172.0, 500.0]),
        strike=45,
        dip=90,
        dip_dir=25,
        length_m=1e4,
        width_m=1e4,
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
        np.array([-43.0, 172.0, 15]), 45, 5, 10, strike=90
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
    assert gsf_df["dep"].min() >= 10 and gsf_df["dep"].max() <= 20

    for _, point in gsf_df.iterrows():
        assert shapely.contains(
            plane.geometry,
            shapely.Point(
                coordinates.wgs_depth_to_nztm(point[["lat", "lon", "dep"]].to_numpy())
            ),
        )


def test_bad_gsf_type():
    with pytest.raises(TypeError):
        gsf.source_to_gsf_dataframe(1, 0.1)  # ty: ignore[invalid-argument-type]


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
    filepath = Path("test.gsf")
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
            assert shapely.contains(
                fault.geometry,
                shapely.Point(
                    coordinates.wgs_depth_to_nztm(
                        point[["lat", "lon", "dep"]].to_numpy()
                    )
                ),
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


def test_write_gsf_does_not_mutate_input(tmp_path: Path):
    """write_gsf must serialise its input without modifying it.

    Regression test: the ``init_time`` and ``slip`` defaults were assigned
    directly into the caller's DataFrame, so a frame handed to ``write_gsf``
    came back carrying two columns of ``-1`` sentinels. The assignments also
    sat above the ``loc_rake`` validation, so the mutation happened even when
    ``write_gsf`` went on to raise and write nothing.
    """
    gsf_df = pd.DataFrame(
        {
            "lon": [172.6],
            "lat": [-43.5],
            "dep": [1.0],
            "sub_dx": [1.0],
            "sub_dy": [1.0],
            "loc_stk": [0.0],
            "loc_dip": [0.0],
            "loc_rake": [0.0],
            "seg_no": [0],
        }
    )
    expected_columns = list(gsf_df.columns)

    gsf.write_gsf(gsf_df, tmp_path / "out.gsf")

    assert list(gsf_df.columns) == expected_columns

    # the failure path must not mutate either
    missing_rake = gsf_df.drop(columns=["loc_rake"])
    expected_columns = list(missing_rake.columns)
    with pytest.raises(ValueError, match="loc_rake"):
        gsf.write_gsf(missing_rake, tmp_path / "out2.gsf")
    assert list(missing_rake.columns) == expected_columns


def _expected_patch_centres(bounds: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Patch centres of a plane in NZTM, as an (ny, nx, 3) array.

    Computed independently of ``qcore.grid``: the centre of patch (j, i) is
    the top-left corner plus ``(i + 1/2) / nx`` of the strike edge and
    ``(j + 1/2) / ny`` of the dip edge.
    """
    strike_edge = bounds[1] - bounds[0]
    dip_edge = bounds[-1] - bounds[0]
    alpha = (np.arange(nx) + 0.5) / nx
    beta = (np.arange(ny) + 0.5) / ny
    return (
        bounds[0]
        + alpha[np.newaxis, :, np.newaxis] * strike_edge
        + beta[:, np.newaxis, np.newaxis] * dip_edge
    )


def test_fault_gsf_exact_patch_centres():
    """Every row of the GSF frame is the centre of a patch, in a fixed order.

    The frame is ordered by dip row, then by segment, then along strike, and
    depths are reported in kilometres.
    """
    north, east = 5_200_000.0, 1_570_000.0
    # Two adjoining planes striking north and dipping east: widths are
    # 5 km (3 km east, 4 km down), lengths 4 km and 2 km.
    plane_a_bounds = np.array(
        [
            [north, east, 0.0],
            [north + 4000.0, east, 0.0],
            [north + 4000.0, east + 3000.0, 4000.0],
            [north, east + 3000.0, 4000.0],
        ]
    )
    plane_b_bounds = np.array(
        [
            [north + 4000.0, east, 0.0],
            [north + 6000.0, east, 0.0],
            [north + 6000.0, east + 3000.0, 4000.0],
            [north + 4000.0, east + 3000.0, 4000.0],
        ]
    )
    fault = Fault(planes=[Plane(bounds=plane_a_bounds), Plane(bounds=plane_b_bounds)])

    gsf_df = gsf.source_to_gsf_dataframe(fault, 1.0)

    centres_a = _expected_patch_centres(plane_a_bounds, nx=4, ny=5)
    centres_b = _expected_patch_centres(plane_b_bounds, nx=2, ny=5)
    expected_nztm = np.vstack(
        [np.vstack([centres_a[j], centres_b[j]]) for j in range(5)]
    )
    expected_seg_no = np.tile([0] * 4 + [1] * 2, 5)
    expected_wgs = coordinates.nztm_to_wgs_depth(expected_nztm)

    assert len(gsf_df) == 30
    np.testing.assert_allclose(gsf_df["lat"].to_numpy(), expected_wgs[:, 0], atol=1e-6)
    np.testing.assert_allclose(gsf_df["lon"].to_numpy(), expected_wgs[:, 1], atol=1e-6)
    # Depths go from metres to kilometres: 0.4 km for the top row of patches
    # down to 3.6 km for the bottom row.
    np.testing.assert_allclose(
        gsf_df["dep"].to_numpy(), expected_nztm[:, 2] / 1000, atol=1e-6
    )
    assert sorted(gsf_df["dep"].unique()) == pytest.approx([0.4, 1.2, 2.0, 2.8, 3.6])
    np.testing.assert_array_equal(gsf_df["seg_no"].to_numpy(), expected_seg_no)
    np.testing.assert_allclose(gsf_df["sub_dx"].to_numpy(), 1.0)
    np.testing.assert_allclose(gsf_df["sub_dy"].to_numpy(), 1.0)


def test_bad_gsf_type_message():
    """The TypeError names the type that was rejected."""
    with pytest.raises(TypeError, match=r"^Unsupported source type: <class 'str'>$"):
        gsf.source_to_gsf_dataframe("not a source", 1.0)  # ty: ignore[invalid-argument-type]


def test_write_bad_df_exact_message(tmp_path: Path):
    """The missing loc_rake error message is exactly as documented."""
    df = pd.DataFrame({"lon": [1.0], "lat": [2.0], "dep": [3.0]})
    with pytest.raises(
        ValueError, match=r"^The DataFrame must have a 'loc_rake' column\.$"
    ):
        gsf.write_gsf(df, tmp_path / "out.gsf")


def _base_gsf_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lon": [172.5, 172.25],
            "lat": [-43.5, -43.75],
            "dep": [1.5, 2.5],
            "sub_dx": [0.5, 0.25],
            "sub_dy": [0.75, 0.125],
            "loc_stk": [10.0, 20.0],
            "loc_dip": [30.0, 40.0],
            "loc_rake": [90.0, 180.0],
            "seg_no": [0, 1],
        }
    )


def test_write_gsf_exact_file_contents(tmp_path: Path):
    """write_gsf writes the point count, then one line per row without header
    or index, in the fixed GSF column order regardless of the frame's column
    order or any extra columns. Missing slip and init_time default to -1."""
    df = _base_gsf_frame()
    # Shuffle the columns and add one that GSF does not know about.
    df = df[list(reversed(df.columns))].assign(extra=[99, 98])
    filepath = tmp_path / "out.gsf"

    gsf.write_gsf(df, filepath)

    assert filepath.read_text().splitlines() == [
        "2",
        "172.5 -43.5 1.5 0.5 0.75 10.0 30.0 90.0 -1 -1 0",
        "172.25 -43.75 2.5 0.25 0.125 20.0 40.0 180.0 -1 -1 1",
    ]


def test_write_gsf_keeps_given_slip_and_init_time(tmp_path: Path):
    """Existing slip and init_time columns are written, not replaced by -1."""
    df = _base_gsf_frame().assign(slip=[2.5, 3.5], init_time=[0.0, 1.25])
    filepath = tmp_path / "out.gsf"

    gsf.write_gsf(df, filepath)

    read_df = gsf.read_gsf(filepath)
    np.testing.assert_array_equal(read_df["slip"].to_numpy(), [2.5, 3.5])
    np.testing.assert_array_equal(read_df["init_time"].to_numpy(), [0.0, 1.25])
    # The input frame is left untouched.
    assert list(df["slip"]) == [2.5, 3.5]
    assert list(df["init_time"]) == [0.0, 1.25]


@pytest.mark.parametrize("missing", ["slip", "init_time"])
def test_write_gsf_defaults_single_missing_column(tmp_path: Path, missing: str):
    """Only the missing column is defaulted to -1; the other is preserved."""
    df = _base_gsf_frame().assign(slip=[2.5, 3.5], init_time=[0.5, 1.25])
    df = df.drop(columns=[missing])
    present = "init_time" if missing == "slip" else "slip"
    filepath = tmp_path / "out.gsf"

    gsf.write_gsf(df, filepath)

    read_df = gsf.read_gsf(filepath)
    np.testing.assert_array_equal(read_df[missing].to_numpy(), [-1, -1])
    np.testing.assert_array_equal(read_df[present].to_numpy(), df[present].to_numpy())


def test_fault_gsf_sub_resolution_segment():
    """A plane shorter than resolution / 2 is discretised as a single patch."""
    trace = np.array(
        [[5180000.0, 1570000.0], [5180000.0, 1580000.0], [5180000.0, 1580400.0]]
    )
    fault = Fault(
        [
            Plane.from_nztm_trace(trace[i : i + 2], 0.0, 10.0, 60.0, 0.0)
            for i in range(2)
        ]
    )
    short_index = int(np.argmin([plane.length for plane in fault.planes]))
    short_plane = fault.planes[short_index]
    assert short_plane.length < 0.5
    n_dip = round(short_plane.width)

    gsf_df = gsf.source_to_gsf_dataframe(fault, 1.0)

    short_patches = gsf_df[gsf_df["seg_no"] == short_index]
    assert len(short_patches) == n_dip
    assert short_patches["sub_dx"].to_numpy() == pytest.approx(short_plane.length)
    assert short_patches["sub_dy"].to_numpy() == pytest.approx(
        short_plane.width / n_dip
    )
    assert len(gsf_df[gsf_df["seg_no"] == 1 - short_index]) == 10 * n_dip
