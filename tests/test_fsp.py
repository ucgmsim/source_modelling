from pathlib import Path

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
        "lat": -43.5965,
        "lon": 172.0673,
        "x": -10.6899,
        "y": -5.1682,
        "z": 0.0000,
        "slip": 4.8065,
    }


def test_bad_fsp():
    """Test malformed FSP files that would throw an FSPParseError."""
    with pytest.raises(fsp.FSPParseError):
        fsp.FSPFile.read_from_file(FSP_PATH / "bad_header.fsp")
    with pytest.raises(fsp.FSPParseError):
        fsp.FSPFile.read_from_file(FSP_PATH / "no_columns.fsp")
