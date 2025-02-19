from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from source_modelling import gsf


@pytest.fixture
def sample_gsf_df():
    data = {
        "length": [10.0, 15.0],
        "width": [7.0, 10.0],
        "strike": [90.0, 120.0],
        "dip": [30.0, 45.0],
        "rake": [0.0, 10.0],
        "meshgrid": [
            np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [[0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]]),
            np.array([[[1.0, 1.0, 1000.0], [1.5, 1.0, 1000.0]], [[1.0, 1.5, 1000.0], [1.5, 1.5, 1000.0]]])
        ]
    }
    return pd.DataFrame(data)

def test_write_fault_to_gsf_file(tmp_path: Path, sample_gsf_df: pd.DataFrame):
    gsf_filepath = tmp_path / "test.gsf"
    gsf.write_fault_to_gsf_file(gsf_filepath, sample_gsf_df)
    assert gsf_filepath.exists()

def test_read_gsf(tmp_path: Path, sample_gsf_df: pd.DataFrame):
    gsf_filepath = tmp_path / "test.gsf"
    gsf.write_fault_to_gsf_file(gsf_filepath, sample_gsf_df)
    df = gsf.read_gsf(gsf_filepath)
    assert not df.empty
    assert list(df.columns) == ["lon", "lat", "depth", "sub_dx", "sub_dy", "strike", "dip", "rake", "slip", "init_time", "seg_no"]

    expected_points = [
        (0.0, 0.0, 0.0, 5.0, 3.5, 90.0, 30.0, 0.0, -1.0, -1.0, 0),
        (0.5, 0.0, 0.0, 5.0, 3.5, 90.0, 30.0, 0.0, -1.0, -1.0, 0),
        (1.0, 1.0, 1.0, 7.5, 5.0, 120.0, 45.0, 10.0, -1.0, -1.0, 1),
        (1.5, 1.0, 1.0, 7.5, 5.0, 120.0, 45.0, 10.0, -1.0, -1.0, 1),
        (0.0, 0.5, 0.0, 5.0, 3.5, 90.0, 30.0, 0.0, -1.0, -1.0, 0),
        (0.5, 0.5, 0.0, 5.0, 3.5, 90.0, 30.0, 0.0, -1.0, -1.0, 0),
        (1.0, 1.5, 1.0, 7.5, 5.0, 120.0, 45.0, 10.0, -1.0, -1.0, 1),
        (1.5, 1.5, 1.0, 7.5, 5.0, 120.0, 45.0, 10.0, -1.0, -1.0, 1)
    ]
    for i, point in enumerate(expected_points):
        assert df.iloc[i]["lon"] == point[1]
        assert df.iloc[i]["lat"] == point[0]
        assert df.iloc[i]["depth"] == point[2]
        assert df.iloc[i]["sub_dx"] == point[3]
        assert df.iloc[i]["sub_dy"] == point[4]
        assert df.iloc[i]["strike"] == point[5]
        assert df.iloc[i]["dip"] == point[6]
        assert df.iloc[i]["rake"] == point[7]
        assert df.iloc[i]["slip"] == point[8]
        assert df.iloc[i]["init_time"] == point[9]
        assert df.iloc[i]["seg_no"] == point[10]


def test_gsf_bigger(tmp_path: Path):
    data = {
        "length": [20.0, 25.0],
        "width": [14.0, 20.0],
        "strike": [100.0, 130.0],
        "dip": [35.0, 50.0],
        "rake": [5.0, 15.0],
        "meshgrid": [
            np.array([[[2.0, 2.0, 2000.0], [2.5, 2.0, 2000.0]], [[2.0, 2.5, 2000.0], [2.5, 2.5, 2000.0]]]),
            np.array([[[3.0, 3.0, 3000.0], [3.5, 3.0, 3000.0]], [[3.0, 3.5, 3000.0], [3.5, 3.5, 3000.0]]])
        ]
    }
    sample_gsf_df = pd.DataFrame(data)
    gsf_filepath = tmp_path / "test2.gsf"
    gsf.write_fault_to_gsf_file(gsf_filepath, sample_gsf_df)
    assert gsf_filepath.exists()

    df = gsf.read_gsf(gsf_filepath)
    assert not df.empty
    assert list(df.columns) == ["lon", "lat", "depth", "sub_dx", "sub_dy", "strike", "dip", "rake", "slip", "init_time", "seg_no"]

    expected_points = [
        (2.0, 2.0, 2.0, 10.0, 7.0, 100.0, 35.0, 5.0, -1.0, -1.0, 0),
        (2.5, 2.0, 2.0, 10.0, 7.0, 100.0, 35.0, 5.0, -1.0, -1.0, 0),
        (3.0, 3.0, 3.0, 12.5, 10.0, 130.0, 50.0, 15.0, -1.0, -1.0, 1),
        (3.5, 3.0, 3.0, 12.5, 10.0, 130.0, 50.0, 15.0, -1.0, -1.0, 1),
        (2.0, 2.5, 2.0, 10.0, 7.0, 100.0, 35.0, 5.0, -1.0, -1.0, 0),
        (2.5, 2.5, 2.0, 10.0, 7.0, 100.0, 35.0, 5.0, -1.0, -1.0, 0),
        (3.0, 3.5, 3.0, 12.5, 10.0, 130.0, 50.0, 15.0, -1.0, -1.0, 1),
        (3.5, 3.5, 3.0, 12.5, 10.0, 130.0, 50.0, 15.0, -1.0, -1.0, 1)
    ]
    for i, point in enumerate(expected_points):
        assert df.iloc[i]["lon"] == point[1]
        assert df.iloc[i]["lat"] == point[0]
        assert df.iloc[i]["depth"] == point[2]
        assert df.iloc[i]["sub_dx"] == point[3]
        assert df.iloc[i]["sub_dy"] == point[4]
        assert df.iloc[i]["strike"] == point[5]
        assert df.iloc[i]["dip"] == point[6]
        assert df.iloc[i]["rake"] == point[7]
        assert df.iloc[i]["slip"] == point[8]
        assert df.iloc[i]["init_time"] == point[9]
        assert df.iloc[i]["seg_no"] == point[10]


def test_complicated_gsf_file(tmp_path: Path):
    data = {
        "length": [30.0, 35.0, 40.0],
        "width": [21.0, 25.0, 30.0],
        "strike": [110.0, 140.0, 160.0],
        "dip": [40.0, 55.0, 70.0],
        "rake": [10.0, 20.0, 30.0],
        "meshgrid": [
            np.array([
                [[4.0, 4.0, 4000.0], [4.5, 4.0, 4000.0], [5.0, 4.0, 4000.0]],
                [[4.0, 4.5, 4000.0], [4.5, 4.5, 4000.0], [5.0, 4.5, 4000.0]],
                [[4.0, 5.0, 4000.0], [4.5, 5.0, 4000.0], [5.0, 5.0, 4000.0]]
            ]),
            np.array([
                [[5.0, 5.0, 5000.0], [5.5, 5.0, 5000.0], [6.0, 5.0, 5000.0]],
                [[5.0, 5.5, 5000.0], [5.5, 5.5, 5000.0], [6.0, 5.5, 5000.0]],
                [[5.0, 6.0, 5000.0], [5.5, 6.0, 5000.0], [6.0, 6.0, 5000.0]]
            ]),
            np.array([
                [[6.0, 6.0, 6000.0], [6.5, 6.0, 6000.0], [7.0, 6.0, 6000.0]],
                [[6.0, 6.5, 6000.0], [6.5, 6.5, 6000.0], [7.0, 6.5, 6000.0]],
                [[6.0, 7.0, 6000.0], [6.5, 7.0, 6000.0], [7.0, 7.0, 6000.0]]
            ])
        ]
    }
    sample_gsf_df = pd.DataFrame(data)
    gsf_filepath = tmp_path / "test3.gsf"
    gsf.write_fault_to_gsf_file(gsf_filepath, sample_gsf_df)
    assert gsf_filepath.exists()

    df = gsf.read_gsf(gsf_filepath)
    assert not df.empty
    assert list(df.columns) == ["lon", "lat", "depth", "sub_dx", "sub_dy", "strike", "dip", "rake", "slip", "init_time", "seg_no"]

    expected_points = [
        (4.0, 4.0, 4.0, 10.0, 7.0, 110.0, 40.0, 10.0, -1.0, -1.0, 0),
        (4.5, 4.0, 4.0, 10.0, 7.0, 110.0, 40.0, 10.0, -1.0, -1.0, 0),
        (5.0, 4.0, 4.0, 10.0, 7.0, 110.0, 40.0, 10.0, -1.0, -1.0, 0),
        (5.0, 5.0, 5.0, 35/3, 25/3, 140.0, 55.0, 20.0, -1.0, -1.0, 1),
        (5.5, 5.0, 5.0, 35/3, 25/3, 140.0, 55.0, 20.0, -1.0, -1.0, 1),
        (6.0, 5.0, 5.0, 35/3, 25/3, 140.0, 55.0, 20.0, -1.0, -1.0, 1),
        (6.0, 6.0, 6.0, 40/3, 10.0, 160.0, 70.0, 30.0, -1.0, -1.0, 2),
        (6.5, 6.0, 6.0, 40/3, 10.0, 160.0, 70.0, 30.0, -1.0, -1.0, 2),
        (7.0, 6.0, 6.0, 40/3, 10.0, 160.0, 70.0, 30.0, -1.0, -1.0, 2),
        (4.0, 4.5, 4.0, 10.0, 7.0, 110.0, 40.0, 10.0, -1.0, -1.0, 0),
        (4.5, 4.5, 4.0, 10.0, 7.0, 110.0, 40.0, 10.0, -1.0, -1.0, 0),
        (5.0, 4.5, 4.0, 10.0, 7.0, 110.0, 40.0, 10.0, -1.0, -1.0, 0),
        (5.0, 5.5, 5.0, 35/3, 25/3, 140.0, 55.0, 20.0, -1.0, -1.0, 1),
        (5.5, 5.5, 5.0, 35/3, 25/3, 140.0, 55.0, 20.0, -1.0, -1.0, 1),
        (6.0, 5.5, 5.0, 35/3, 25/3, 140.0, 55.0, 20.0, -1.0, -1.0, 1),
        (6.0, 6.5, 6.0, 40/3, 10.0, 160.0, 70.0, 30.0, -1.0, -1.0, 2),
        (6.5, 6.5, 6.0, 40/3, 10.0, 160.0, 70.0, 30.0, -1.0, -1.0, 2),
        (7.0, 6.5, 6.0, 40/3, 10.0, 160.0, 70.0, 30.0, -1.0, -1.0, 2),
        (4.0, 5.0, 4.0, 10.0, 7.0, 110.0, 40.0, 10.0, -1.0, -1.0, 0),
        (4.5, 5.0, 4.0, 10.0, 7.0, 110.0, 40.0, 10.0, -1.0, -1.0, 0),
        (5.0, 5.0, 4.0, 10.0, 7.0, 110.0, 40.0, 10.0, -1.0, -1.0, 0),
        (5.0, 6.0, 5.0, 35/3, 25/3, 140.0, 55.0, 20.0, -1.0, -1.0, 1),
        (5.5, 6.0, 5.0, 35/3, 25/3, 140.0, 55.0, 20.0, -1.0, -1.0, 1),
        (6.0, 6.0, 5.0, 35/3, 25/3, 140.0, 55.0, 20.0, -1.0, -1.0, 1),
        (6.0, 7.0, 6.0, 40/3, 10.0, 160.0, 70.0, 30.0, -1.0, -1.0, 2),
        (6.5, 7.0, 6.0, 40/3, 10.0, 160.0, 70.0, 30.0, -1.0, -1.0, 2),
        (7.0, 7.0, 6.0, 40/3, 10.0, 160.0, 70.0, 30.0, -1.0, -1.0, 2)
    ]
    for i, point in enumerate(expected_points):
        assert df.iloc[i]["lon"] == pytest.approx(point[1], rel=1e-4), f"Longitude mismatch at index {i}: expected {point[1]}, got {df.iloc[i]['lon']}"
        assert df.iloc[i]["lat"] == pytest.approx(point[0], rel=1e-4), f"Latitude mismatch at index {i}: expected {point[0]}, got {df.iloc[i]['lat']}"
        assert df.iloc[i]["depth"] == pytest.approx(point[2], rel=1e-4), f"Depth mismatch at index {i}: expected {point[2]}, got {df.iloc[i]['depth']}"
        assert df.iloc[i]["sub_dx"] == pytest.approx(point[3], rel=1e-4), f"sub_dx mismatch at index {i}: expected {point[3]}, got {df.iloc[i]['sub_dx']}"
        assert df.iloc[i]["sub_dy"] == pytest.approx(point[4], rel=1e-4), f"sub_dy mismatch at index {i}: expected {point[4]}, got {df.iloc[i]['sub_dy']}"
        assert df.iloc[i]["strike"] == pytest.approx(point[5], rel=1e-4), f"Strike mismatch at index {i}: expected {point[5]}, got {df.iloc[i]['strike']}"
        assert df.iloc[i]["dip"] == pytest.approx(point[6], rel=1e-4), f"Dip mismatch at index {i}: expected {point[6]}, got {df.iloc[i]['dip']}"
        assert df.iloc[i]["rake"] == pytest.approx(point[7], rel=1e-4), f"Rake mismatch at index {i}: expected {point[7]}, got {df.iloc[i]['rake']}"
        assert df.iloc[i]["slip"] == pytest.approx(point[8], rel=1e-4), f"Slip mismatch at index {i}: expected {point[8]}, got {df.iloc[i]['slip']}"
        assert df.iloc[i]["init_time"] == pytest.approx(point[9], rel=1e-4), f"Init_time mismatch at index {i}: expected {point[9]}, got {df.iloc[i]['init_time']}"
        assert df.iloc[i]["seg_no"] == pytest.approx(point[10], rel=1e-4), f"Seg_no mismatch at index {i}: expected {point[10]}, got {df.iloc[i]['seg_no']}"
