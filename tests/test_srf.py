import tempfile
from pathlib import Path

import pytest

from source_modelling import srf

SRF_DIR = Path(__file__).parent / "srfs"


def test_christchurch_srf():
    """Test that the SRF reader can parse the Christchurch SRF and validate basic properties."""
    christchurch_srf = srf.read_srf(SRF_DIR / "3468575.srf")
    assert christchurch_srf.version == "1.0"
    assert len(christchurch_srf.header) == 1
    assert len(christchurch_srf.points) == 14400
    assert christchurch_srf.header.iloc[0].to_dict() == {
        "elon": 172.6966,
        "elat": -43.5446,
        "nstk": 160,
        "ndip": 90,
        "len": 16.00,
        "wid": 9.00,
        "stk": 59,
        "dip": 69,
        "dtop": 0.63,
        "shyp": -2.00,
        "dhyp": 6.00,
    }
    # local strike and dip should match the header
    assert (christchurch_srf.points["dip"] == 69).all()
    assert (christchurch_srf.points["stk"] == 59).all()
    assert christchurch_srf.points["tinit"].min() == 0.0
    # For the Christchurch event, the slip is only defined in the t1 component.
    assert christchurch_srf.slipt1_array.shape[0] == len(christchurch_srf.points)
    assert christchurch_srf.slipt2_array is None
    assert christchurch_srf.slipt3_array is None
    assert (christchurch_srf.points["slip"] == christchurch_srf.points["slip1"]).all()
    # This test asserts that the slip and slipt1 values are not different (nnz counts the number of different entries).
    assert (christchurch_srf.slip != christchurch_srf.slipt1_array).nnz == 0
    # dt is constant for SRF
    assert christchurch_srf.dt == 2.5e-02
    # Check that the segments code correctly identifies one segment
    assert len(christchurch_srf.segments) == 1
    assert len(christchurch_srf.segments[0]) == len(christchurch_srf.points)
    # NOTE: This value is not validated in any way, it's more of a
    # regression test for future parsing changes.
    assert christchurch_srf.nt == 361

    assert christchurch_srf.points.iloc[0].to_dict() == {
        "lon": 172.6127,
        "lat": -43.5821,
        "dep": 0.6767,
        "stk": 59,
        "dip": 69,
        "area": 1.0e08,
        "tinit": 5.7029,
        "dt": 2.5e-02,
        "rake": 102,
        "slip1": 17.49,
        "slip2": 0.0,
        "slip3": 0.0,
        "slip": 17.49,
    }
    tinit_index = int(christchurch_srf.points["tinit"].iloc[0] // christchurch_srf.dt)
    # have to manually slice because the sparse arrays do not support slicing
    slip_window = [
        christchurch_srf.slipt1_array[0, t]
        for t in range(tinit_index, tinit_index + 12)
    ]
    assert slip_window == [
        0.00000e00,
        2.07568e02,
        2.42313e02,
        5.90245e01,
        4.89368e01,
        4.26333e01,
        3.50411e01,
        2.68253e01,
        1.87057e01,
        1.13937e01,
        5.52983e00,
        1.62786e00,
    ]
    for (_, header), plane in zip(
        christchurch_srf.header.iterrows(), christchurch_srf.planes
    ):
        assert header[["elat", "elon"]].values == pytest.approx(
            plane.centroid[:2], abs=0.1
        )
        assert header["wid"] == pytest.approx(plane.width, abs=0.01)
        assert header["len"] == pytest.approx(plane.length, abs=0.01)
        assert header["dip"] == pytest.approx(plane.dip, abs=0.1)
        assert header["stk"] == pytest.approx(plane.strike, abs=0.1)
        assert header["dtop"] == pytest.approx(plane.corners[0, -1] / 1000, abs=0.1)


def test_darfield_srf():
    """Test that the SRF reader can parse the Darfield SRF and validate basic properties."""
    darfield_srf = srf.read_srf(SRF_DIR / "3366146.srf")
    assert darfield_srf.header.to_dict(orient="records") == [
        {
            "elon": 172.133408,
            "elat": -43.550999,
            "nstk": 50,
            "ndip": 90,
            "len": 10.0000,
            "wid": 18.0000,
            "stk": 40,
            "dip": 75,
            "dtop": 1.0000,
            "shyp": 1.0000,
            "dhyp": 10.0000,
        },
        {
            "elon": 172.003906,
            "elat": -43.568298,
            "nstk": 60,
            "ndip": 90,
            "len": 12.0000,
            "wid": 18.0000,
            "stk": 121,
            "dip": 105,
            "dtop": 0.0000,
            "shyp": 6.0000,
            "dhyp": 6.0000,
        },
        {
            "elon": 172.194901,
            "elat": -43.588299,
            "nstk": 100,
            "ndip": 90,
            "len": 20.0000,
            "wid": 18.0000,
            "stk": 87,
            "dip": 85,
            "dtop": 0.0000,
            "shyp": -10.0000,
            "dhyp": 6.0000,
        },
        {
            "elon": 172.379898,
            "elat": -43.571301,
            "nstk": 70,
            "ndip": 90,
            "len": 14.0000,
            "wid": 18.0000,
            "stk": 87,
            "dip": 85,
            "dtop": 0.0000,
            "shyp": -7.0000,
            "dhyp": 6.0000,
        },
        {
            "elon": 171.944305,
            "elat": -43.578400,
            "nstk": 35,
            "ndip": 90,
            "len": 7.0000,
            "wid": 18.0000,
            "stk": 216,
            "dip": 50,
            "dtop": 0.0000,
            "shyp": 3.5000,
            "dhyp": 6.0000,
        },
        {
            "elon": 172.309799,
            "elat": -43.549900,
            "nstk": 55,
            "ndip": 90,
            "len": 11.0000,
            "wid": 18.0000,
            "stk": 40,
            "dip": 80,
            "dtop": 0.0000,
            "shyp": -5.5000,
            "dhyp": 6.0000,
        },
        {
            "elon": 172.182205,
            "elat": -43.508999,
            "nstk": 40,
            "ndip": 90,
            "len": 8.0000,
            "wid": 18.0000,
            "stk": 150,
            "dip": 54,
            "dtop": 0.0000,
            "shyp": 4.0000,
            "dhyp": 6.0000,
        },
    ]
    # Will not test the basic properties again because that is tested
    # in the Christchurch case pretty thoroughly. Will, however, test
    # the segment iteration thoroughly
    assert len(darfield_srf.segments) == len(darfield_srf.header)
    for i, segment in enumerate(darfield_srf.segments):
        segment_header = darfield_srf.header.iloc[i]
        segment = darfield_srf.segments[i]
        assert len(segment) == segment_header["nstk"] * segment_header["ndip"]
        assert (segment["dip"] == segment_header["dip"]).all()
        assert (segment["stk"] == segment_header["stk"]).all()
    for (_, header), plane in zip(darfield_srf.header.iterrows(), darfield_srf.planes):
        assert header[["elat", "elon"]].values == pytest.approx(
            plane.centroid[:2], abs=0.1
        )
        assert header["wid"] == pytest.approx(plane.width, abs=0.01)
        assert header["len"] == pytest.approx(plane.length, abs=0.025)
        # assert header["dip"] == pytest.approx(plane.dip, abs=0.1)
        # assert header["stk"] == pytest.approx(plane.strike, abs=0.1)
        assert header["dtop"] == pytest.approx(plane.corners[0, -1] / 1000, abs=0.1)


def test_junk_srfs():
    """Test that malformed SRFs raise srf parsing errors."""
    with pytest.raises(srf.SrfParseError):
        srf.read_srf(SRF_DIR / "empty.srf")

    with pytest.raises(srf.SrfParseError):
        srf.read_srf(SRF_DIR / "bad_int.srf")

    with pytest.raises(srf.SrfParseError):
        srf.read_srf(SRF_DIR / "bad_float.srf")

    with pytest.raises(srf.SrfParseError):
        srf.read_srf(SRF_DIR / "no_points.srf")


def test_writing_christchurch():
    """Check that writing a copy an SRF produces an SRF with the same values."""
    christchurch_srf = srf.read_srf(SRF_DIR / "3468575.srf")
    with tempfile.NamedTemporaryFile() as tmp_christchurch_srf_handle:
        srf.write_srf(tmp_christchurch_srf_handle.name, christchurch_srf)
        christchurch_srf_tmp = srf.read_srf(tmp_christchurch_srf_handle.name)
        assert christchurch_srf.header.equals(christchurch_srf_tmp.header)
        assert christchurch_srf.points.equals(christchurch_srf_tmp.points)
        assert (christchurch_srf.slip != christchurch_srf_tmp.slip).nnz == 0
