from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from source_modelling import fsp

FSP_PATH = Path(__file__).parent / "fsps"


def test_darfield_fsp():
    """Check that the darfield FSP file is parsed with the correct values extracted."""
    fsp_file = fsp.FSPFile.read_from_file(FSP_PATH / "s2010DARFIE01ATZO.fsp")
    assert fsp_file.event_tag == "s2010DARFIE01ATZO"
    assert fsp_file.latitude == -43.55
    assert fsp_file.longitude == 172.2
    assert fsp_file.depth == 10.0
    assert fsp_file.hypx == 4.15
    assert fsp_file.hypz == 0.52
    assert fsp_file.length == 75.0
    assert fsp_file.width == 94.0
    assert fsp_file.strike == 218.81354401805797
    assert fsp_file.dip == 73.16094808126424
    assert fsp_file.rake == 163.33732382635753
    assert fsp_file.htop == 0.0
    assert fsp_file.magnitude == 7.10
    assert fsp_file.moment == 5.01e19
    assert fsp_file.average_rise_time is None
    assert fsp_file.average_rupture_speed is None
    assert fsp_file.slip_velocity_function == "not-applicable"
    assert fsp_file.nx == 9
    assert fsp_file.nz == 11
    assert fsp_file.dx == 1
    assert fsp_file.dz == 1
    assert fsp_file.fmin is None
    assert fsp_file.fmax is None
    assert fsp_file.time_window_count == 0
    assert fsp_file.time_window_length == 0
    assert fsp_file.time_shift == 0
    assert fsp_file.segment_count == 8
    assert len(fsp_file.data) == 886
    assert fsp_file.data.iloc[0].to_dict() == {
        "segment": 0,
        "lat": -43.5965,
        "lon": 172.0673,
        "x": -10.6899,
        "y": -5.1682,
        "z": 0.0000,
        "slip": 4.8065,
    }
    assert fsp_file.segments[0].strike == pytest.approx(307.0)
    assert fsp_file.segments[0].dip == pytest.approx(86.2)
    assert fsp_file.segments[0].length == pytest.approx(10.0)
    assert fsp_file.segments[0].width == pytest.approx(12.0)
    assert fsp_file.segments[0].dx == pytest.approx(1.0)
    assert fsp_file.segments[0].dz == pytest.approx(1.0)
    assert fsp_file.segments[0].dtop == pytest.approx(0.0)
    assert fsp_file.segments[0].top_centre == pytest.approx(
        np.array([-43.5721, 172.0226])
    )
    assert fsp_file.segments[0].hypocentre == pytest.approx(
        np.array([4.151557898146846, 0.5150635970119828])
    )
    assert fsp_file.segments[0].subfaults == 120
    assert fsp_file.velocity_model is None

    plane = fsp_file.segments[0].as_plane()
    assert plane.strike == pytest.approx(307.0, abs=1)
    assert plane.dip == pytest.approx(86.2, abs=1)
    assert plane.length == pytest.approx(10.0)
    assert plane.width == pytest.approx(12.0)
    assert plane.bounds[0, -1] == pytest.approx(0.0)
    assert plane.wgs_depth_coordinates_to_fault_coordinates(
        np.array([-43.5721, 172.0226])
    ) == pytest.approx(np.array([0.5, 0]))


def test_kanto_fsp():
    """Check that the Kanto FSP file is parsed with the correct values extracted."""
    fsp_file = fsp.FSPFile.read_from_file(FSP_PATH / "s1923KANTOJkoba.fsp")
    assert fsp_file.event_tag == "s1923KANTOJkoba"
    assert fsp_file.latitude == 35.400
    assert fsp_file.longitude == 139.200
    assert fsp_file.depth == 14.60
    assert fsp_file.hypx == 110.50
    assert fsp_file.hypz == 35.00
    assert fsp_file.length == 130.00
    assert fsp_file.width == 70.00
    assert fsp_file.strike == 290
    assert fsp_file.dip == 25
    assert fsp_file.rake == 140
    assert fsp_file.htop == 2.00
    assert fsp_file.magnitude == 8.08
    assert fsp_file.moment == 1.46e21
    assert fsp_file.average_rise_time == 10.5
    assert fsp_file.average_rupture_speed == 2.6
    assert fsp_file.slip_velocity_function == "triang"
    assert fsp_file.nx == 10
    assert fsp_file.nz == 7
    assert fsp_file.dx == 13.00
    assert fsp_file.dz == 10.00
    assert fsp_file.fmin == 0.02
    assert fsp_file.fmax == 0.4
    assert fsp_file.time_window_count == 1
    assert fsp_file.time_window_length == 1.50
    assert fsp_file.time_shift == 0.00
    assert fsp_file.segment_count == 1
    assert len(fsp_file.data) == 70
    assert fsp_file.data.iloc[0].to_dict() == {
        "segment": 0,
        "lat": 34.812,
        "lon": 140.159,
        "x": 86.879,
        "y": -65.378,
        "z": 2.000,
        "slip": 2.585,
    }
    expected_velocity_model = pd.DataFrame(
        [
            {"depth": 0.00, "Vp": 5.60, "Vs": 2.90, "density": 2.50},
            {"depth": 6.10, "Vp": 6.00, "Vs": 3.40, "density": 2.60},
            {"depth": 19.00, "Vp": 6.80, "Vs": 4.00, "density": 3.00},
        ]
    )
    pd.testing.assert_frame_equal(fsp_file.velocity_model, expected_velocity_model)


def test_nankaido_fsp():
    """Check that the Nankaido FSP file is parsed with the correct values extracted."""
    fsp_file = fsp.FSPFile.read_from_file(FSP_PATH / "s1946NANKAItani.fsp")
    assert fsp_file.event_tag == "s1946NANKAItani"
    assert fsp_file.latitude == 33.030
    assert fsp_file.longitude == 135.620
    assert fsp_file.depth == 30.00
    assert fsp_file.hypx == 90.00
    assert fsp_file.hypz == 45.00
    assert fsp_file.length == 360.00
    assert fsp_file.width == 180.00
    assert fsp_file.strike == 250
    assert fsp_file.dip == 13
    assert fsp_file.rake == 120
    assert fsp_file.htop == 1.00
    assert fsp_file.magnitude == 8.40
    assert fsp_file.moment == 4.51e21
    assert fsp_file.average_rise_time == 0.0
    assert fsp_file.average_rupture_speed == 0.0
    assert fsp_file.slip_velocity_function == "tsunami modeling"
    assert fsp_file.nx == 8
    assert fsp_file.nz == 4
    assert fsp_file.dx == 45.00
    assert fsp_file.dz == 45.00
    assert fsp_file.fmin == 0.00
    assert fsp_file.fmax == 0.0
    assert fsp_file.time_window_count == 0
    assert fsp_file.time_window_length == 0.00
    assert fsp_file.time_shift == 0.00
    assert fsp_file.segment_count == 1
    assert len(fsp_file.data) == 32
    assert fsp_file.data.iloc[0].to_dict() == {
        "segment": 0,
        "lat": 32.867,
        "lon": 136.462,
        "x": 78.426,
        "y": -18.116,
        "z": 1.000,
        "slip": 0.850,
    }
    assert fsp_file.velocity_model == 3.3e10


FSPS = list(FSP_PATH.glob("*.fsp"))
BAD_FSPS = list((FSP_PATH / "bad").glob("*.fsp"))


@pytest.mark.parametrize("fsp_path", FSPS, ids=[fsp_path.stem for fsp_path in FSPS])
def test_loads_srcmod_fsp(fsp_path: Path):
    """Check that the FSP files are parsed correctly."""
    fsp_file = fsp.FSPFile.read_from_file(fsp_path)

    # Check that the required attributes are present and have correct types
    assert isinstance(
        fsp_file.event_tag, str
    ), f"event_tag should be a string, got {type(fsp_file.event_tag)}"

    # Latitude and Longitude bounds
    assert (
        -90 <= fsp_file.latitude <= 90
    ), f"latitude should be between -90 and 90, got {fsp_file.latitude}"
    assert (
        -180 <= fsp_file.longitude <= 180
    ), f"longitude should be between -180 and 180, got {fsp_file.longitude}"

    # Depth bounds (depth is usually positive for depth in km)
    assert (
        fsp_file.depth >= 0
    ), f"depth should be a positive number, got {fsp_file.depth}"

    # Hypocenter coordinates bounds (can be None)
    if fsp_file.hypx is not None:
        assert (
            fsp_file.hypx >= 0
        ), f"hypx should be a non-negative number, got {fsp_file.hypx}"
    if fsp_file.hypz is not None:
        assert (
            fsp_file.hypz >= 0
        ), f"hypz should be a non-negative number, got {fsp_file.hypz}"

    # Fault and rupture parameters bounds
    assert fsp_file.length > 0, f"length should be positive, got {fsp_file.length}"
    assert fsp_file.width > 0, f"width should be positive, got {fsp_file.width}"
    assert (
        0 <= fsp_file.strike <= 360
    ), f"strike should be between 0 and 360, got {fsp_file.strike}"
    assert (
        0 <= fsp_file.dip <= 90
    ), f"dip should be between 0 and 90, got {fsp_file.dip}"
    assert isinstance(
        fsp_file.rake, float
    ), f"rake should be a float, got {type(fsp_file.rake)}"
    assert fsp_file.htop >= 0, f"htop should be non-negative, got {fsp_file.htop}"

    # Magnitude and Moment bounds
    assert (
        fsp_file.magnitude > 0
    ), f"magnitude should be positive, got {fsp_file.magnitude}"
    assert fsp_file.moment > 0, f"moment should be positive, got {fsp_file.moment}"

    # Average rise time and rupture speed bounds (if present)
    if fsp_file.average_rise_time is not None:
        assert (
            fsp_file.average_rise_time >= 0
        ), f"average_rise_time should be positive, got {fsp_file.average_rise_time}"
    if fsp_file.average_rupture_speed is not None:
        assert (
            fsp_file.average_rupture_speed >= 0
        ), f"average_rupture_speed should be positive, got {fsp_file.average_rupture_speed}"

    # Slip velocity function should be a non-empty string
    assert (
        isinstance(fsp_file.slip_velocity_function, str)
        and fsp_file.slip_velocity_function.strip()
    ), f"slip_velocity_function should be a non-empty string, got {fsp_file.slip_velocity_function}"

    # Model subfault sizes bounds
    if fsp_file.nx is not None:
        assert fsp_file.nx > 0, f"nx should be positive, got {fsp_file.nx}"
    if fsp_file.nz is not None:
        assert fsp_file.nz > 0, f"nz should be positive, got {fsp_file.nz}"

    if fsp_file.dx is not None:
        assert fsp_file.dx > 0, f"dx should be positive, got {fsp_file.dx}"
    if fsp_file.dz is not None:
        assert fsp_file.dz > 0, f"dz should be positive, got {fsp_file.dz}"

    # Inversion parameters bounds
    if fsp_file.fmin is not None:
        assert fsp_file.fmin >= 0, f"fmin should be positive, got {fsp_file.fmin}"
    if fsp_file.fmax is not None:
        assert fsp_file.fmax >= 0, f"fmax should be positive, got {fsp_file.fmax}"
        assert (
            fsp_file.fmin <= fsp_file.fmax
        ), f"fmin should be no more than fmax, got fmin={fsp_file.fmin} and fmax={fsp_file.fmax}"

    # Time window parameters bounds
    if fsp_file.time_window_count is not None:
        assert (
            fsp_file.time_window_count >= 0
        ), f"time_window_count should be non-negative, got {fsp_file.time_window_count}"
    if fsp_file.time_window_length is not None:
        assert (
            fsp_file.time_window_length >= 0
        ), f"time_window_length should be non-negative, got {fsp_file.time_window_length}"
    if fsp_file.time_shift is not None:
        assert (
            fsp_file.time_shift >= 0
        ), f"time_shift should be non-negative, got {fsp_file.time_shift}"

    # Segment count should be positive
    assert (
        fsp_file.segment_count > 0
    ), f"segment_count should be positive, got {fsp_file.segment_count}"

    # Check that the data is a pandas DataFrame
    assert isinstance(
        fsp_file.data, pd.DataFrame
    ), f"data should be a pandas DataFrame, got {type(fsp_file.data)}"
    assert not fsp_file.data.empty, "data DataFrame should not be empty"

    # Check that there is at least one row in the data
    assert fsp_file.data.shape[0] > 0, "data DataFrame should have at least one row"
    # Check the number of segments matches the reported value in the header.
    assert fsp_file.data["segment"].max() == fsp_file.segment_count - 1


@pytest.mark.parametrize(
    "fsp_path", BAD_FSPS, ids=[fsp_path.stem for fsp_path in BAD_FSPS]
)
def test_bad_fsp(fsp_path: Path):
    """Test malformed FSP files that would throw an FSPParseError."""
    with pytest.raises(fsp.FSPParseError):
        fsp.FSPFile.read_from_file(fsp_path)
