import itertools
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
import scipy as sp
import shapely
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from qcore import coordinates, geo
from source_modelling import sources
from source_modelling.sources import Fault, Plane

DATA_PATH = Path("tests") / "data"
np.random.seed(0)


def coordinate(lat: float, lon: float, depth: float | None = None) -> np.ndarray:
    """Create a coordinate array from latitude, longitude, and optional depth."""
    if depth is not None:
        return np.array([lat, lon, depth])
    return np.array([lat, lon])


def valid_coordinates(point_coordinates: np.ndarray) -> bool:
    """Check if the given coordinates are valid."""
    return bool(np.all(np.isfinite(coordinates.wgs_depth_to_nztm(point_coordinates))))


@given(
    point_coordinates=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(0, 100),
    ),
    length_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    width_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    strike=st.floats(0, 360),
    dip=st.floats(0, 180),
    dip_dir=st.floats(0, 360),
)
def test_point_construction(
    point_coordinates: np.ndarray,
    length_m: float,
    width_m: float,
    strike: float,
    dip: float,
    dip_dir: float,
):
    """Test the construction of a Point object from latitude, longitude, and depth."""
    assume(valid_coordinates(point_coordinates))
    point = sources.Point.from_lat_lon_depth(
        point_coordinates,
        length_m=length_m,
        width_m=width_m,
        strike=strike,
        dip=dip,
        dip_dir=dip_dir,
    )
    assert np.allclose(point.coordinates, point_coordinates)
    assert point.length == length_m / 1000
    assert point.width == width_m / 1000
    assert np.allclose(
        shapely.get_coordinates(point.geometry, include_z=True),
        np.atleast_2d(point.bounds),
    )
    assert np.isclose(point.width_m, point.width * 1000)
    assert np.allclose(point.centroid, point_coordinates)


@given(
    point_coordinates=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(0, 100),
    ),
    length_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    width_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    strike=st.floats(0, 360),
    dip=st.floats(0, 180),
    dip_dir=st.floats(0, 360),
    local_coordinates=nst.arrays(
        float, (2,), elements={"min_value": 0, "max_value": 1}
    ),
)
def test_point_coordinate_system(
    point_coordinates: np.ndarray,
    length_m: float,
    width_m: float,
    strike: float,
    dip: float,
    dip_dir: float,
    local_coordinates: np.ndarray,
):
    """Test the coordinate system transformation for a Point object."""
    assume(valid_coordinates(point_coordinates))

    point = sources.Point.from_lat_lon_depth(
        point_coordinates,
        length_m=length_m,
        width_m=width_m,
        strike=strike,
        dip=dip,
        dip_dir=dip_dir,
    )
    # NOTE: cannot assert invertibility coordinate mapping is not bijective
    assert np.allclose(
        point.fault_coordinates_to_wgs_depth_coordinates(local_coordinates),
        point_coordinates,
    )


@given(
    point_coordinates=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(0, 100),
    ),
    length_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    width_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    strike=st.floats(0, 360),
    dip=st.floats(0, 180),
    dip_dir=st.floats(0, 360),
    local_coordinates=nst.arrays(
        float, (2,), elements={"min_value": 0, "max_value": 1}
    ),
)
def test_point_coordinate_inversion(
    point_coordinates: np.ndarray,
    length_m: float,
    width_m: float,
    strike: float,
    dip: float,
    dip_dir: float,
    local_coordinates: np.ndarray,
):
    """Test the inversion of coordinate transformations for a Point object."""
    assume(valid_coordinates(point_coordinates))

    point = sources.Point.from_lat_lon_depth(
        point_coordinates,
        length_m=length_m,
        width_m=width_m,
        strike=strike,
        dip=dip,
        dip_dir=dip_dir,
    )
    # NOTE: cannot assert invertibility coordinate mapping is not bijective
    assert np.allclose(
        point.wgs_depth_coordinates_to_fault_coordinates(
            point.fault_coordinates_to_wgs_depth_coordinates(local_coordinates)
        ),
        np.array([1 / 2, 1 / 2]),
    )


@given(
    point_coordinates=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(0, 100),
    ),
    length_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    width_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    strike=st.floats(0, 360),
    dip=st.floats(0, 180),
    dip_dir=st.floats(0, 360),
    other_point=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(0, 100),
    ),
)
def test_point_rrup(
    point_coordinates: np.ndarray,
    length_m: float,
    width_m: float,
    strike: float,
    dip: float,
    dip_dir: float,
    other_point: np.ndarray,
):
    assume(valid_coordinates(point_coordinates))

    point = sources.Point.from_lat_lon_depth(
        point_coordinates,
        length_m=length_m,
        width_m=width_m,
        strike=strike,
        dip=dip,
        dip_dir=dip_dir,
    )
    assert np.isclose(
        point.rrup_distance(other_point),
        coordinates.distance_between_wgs_depth_coordinates(
            point_coordinates, other_point
        ),
    )


@given(
    point_coordinates=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(0, 100),
    ),
    length_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    width_m=st.floats(1e-16, allow_nan=False, allow_infinity=False),
    strike=st.floats(0, 360),
    dip=st.floats(0, 180),
    dip_dir=st.floats(0, 360),
    distance=st.floats(1, 1000),
)
def test_point_rjb(
    point_coordinates: np.ndarray,
    length_m: float,
    width_m: float,
    strike: float,
    dip: float,
    dip_dir: float,
    distance: float,
):
    assume(valid_coordinates(point_coordinates))

    point = sources.Point.from_lat_lon_depth(
        point_coordinates,
        length_m=length_m,
        width_m=width_m,
        strike=strike,
        dip=dip,
        dip_dir=dip_dir,
    )
    buffer = shapely.buffer(point.geometry, distance * 1000)
    for other_point in coordinates.nztm_to_wgs_depth(np.array(buffer.exterior.coords)):
        assert np.isclose(
            point.rjb_distance(other_point),
            distance * 1000,
            atol=1e-4,
        )


@given(
    length=st.floats(0.1, 1000),
    width=st.floats(0.1, 1000),
    strike=st.floats(0, 179),
    dip_dir=st.floats(0, 179),
    dip=st.floats(0.1, 90),
    centroid=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(1, 10),
    ),
)
def test_plane_construction(
    length: float,
    width: float,
    strike: float,
    dip: float,
    dip_dir: float,
    centroid: np.ndarray,
):
    """Test the construction of a Plane object from centroid, strike, and dip."""
    assume(valid_coordinates(centroid))
    assume(dip_dir > strike + 1)
    plane = Plane.from_centroid_strike_dip(
        centroid,
        dip,
        length,
        width,
        strike_nztm=strike,
        dip_dir_nztm=dip_dir,
    )
    assert np.isclose(plane.length, length, atol=1e-6)
    assert np.isclose(plane.width, width, atol=1e-6)
    assert np.isclose(
        plane.projected_width, plane.width * np.cos(np.radians(plane.dip)), atol=1e-6
    )
    if plane.dip == 90:
        assert shapely.get_coordinates(plane.geometry, include_z=True) == pytest.approx(
            plane.bounds[:2]
        )
    else:
        assert shapely.get_coordinates(plane.geometry, include_z=True)[
            :-1
        ] == pytest.approx(plane.bounds)

    assert np.isclose(plane.strike_nztm, strike, atol=1e-6)

    # Check that the plane bounds orientation makes sense.
    assert (
        np.dot(plane.bounds[1] - plane.bounds[0], plane.bounds[2] - plane.bounds[3]) > 0
    )
    assert np.isclose(plane.strike_nztm, strike, atol=1e-6)
    if plane.dip != 90:
        assert np.isclose(plane.dip_dir_nztm, dip_dir, atol=1e-6)
        # The constructor should not care about plane bound orientation
        assert np.allclose(Plane(plane.bounds[::-1]).bounds, plane.bounds)
    assert np.allclose(plane.centroid, centroid * np.array([1, 1, 1000]), atol=1e-6)


# Test 1: Less than 4 points
def test_less_than_four_points():
    """Test that constructing a Plane with less than 4 points raises a ValueError."""
    bounds = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])  # Only 3 points
    with pytest.raises(ValueError, match="Bounds do not form a plane."):
        Plane(bounds)


# Test 2: More than 4 points
def test_more_than_four_points():
    """Test that constructing a Plane with more than 4 points raises a ValueError."""
    bounds = np.array(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
    )  # 5 points
    with pytest.raises(ValueError, match="Bounds do not form a plane."):
        Plane(bounds)


# Test 3: Matrix rank not equal to 3
def test_matrix_rank_not_three():
    """Test that constructing a Plane with matrix rank not equal to 3 raises a ValueError."""
    bounds = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])  # Rank < 3
    with pytest.raises(ValueError, match="Bounds do not form a plane."):
        Plane(bounds)


# Test 4: Top points do not equal 2
def test_top_points_not_two():
    """Test that constructing a Plane with top points not equal to 2 raises a ValueError."""
    bounds = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    )  # All points lie on the same top plane
    with pytest.raises(ValueError, match="Bounds do not form a plane."):
        Plane(bounds)


# Test 5: General invalid input (not forming a valid plane)
def test_general_invalid_input():
    """Test that constructing a Plane with invalid input raises a ValueError."""
    bounds = np.array(
        [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 2]]
    )  # Points do not form a plane
    with pytest.raises(ValueError, match="Bounds do not form a plane."):
        Plane(bounds)


def trace(
    start_trace_nztm: npt.NDArray[float], length: float, strike: float
) -> np.ndarray:
    # Do this in NZTM to prevent any issues with the coordinate system conversions
    strike_vec = np.array([np.cos(np.radians(strike)), np.sin(np.radians(strike))])
    end_trace_nztm = start_trace_nztm + strike_vec * length

    trace_nztm = np.stack((start_trace_nztm, end_trace_nztm), axis=0)
    return coordinates.nztm_to_wgs_depth(trace_nztm)


@st.composite
def valid_trace_definition(draw: st.DrawFn):
    trace_point_1 = draw(
        st.builds(
            coordinate,
            lat=st.floats(-50, -31),
            lon=st.floats(160, 180),
        )
    )
    assume(valid_coordinates(trace_point_1))
    trace_point_1_nztm = coordinates.wgs_depth_to_nztm(trace_point_1)

    strike_nztm = draw(st.floats(0, 359))
    strike_vec = np.array(
        [np.cos(np.radians(strike_nztm)), np.sin(np.radians(strike_nztm))]
    )
    length = draw(st.floats(0.1, 100)) * 1000
    trace_point_2_nztm = trace_point_1_nztm + strike_vec * length

    trace_points_nztm = np.stack((trace_point_1_nztm, trace_point_2_nztm), axis=0)
    assume(np.linalg.matrix_rank(trace_points_nztm) == 2)

    dtop = draw(st.floats(0, 100))
    depth = draw(st.floats(1, 100))
    dip = draw(st.floats(1, 90))
    width = depth / np.sin(np.radians(dip))
    assume(width < 100)

    dip_dir_nztm = (strike_nztm + draw(st.floats(1, 179))) % 360
    assume(dip_dir_nztm <= 359)
    assume(dip_dir_nztm >= 1)
    if np.isclose(dip, 90):
        dip_dir_nztm, dip_dir = 0, 0
    else:
        dip_dir = coordinates.nztm_bearing_to_great_circle_bearing(
            trace_point_1, width, dip_dir_nztm
        )

    return (
        trace_points_nztm,
        dtop,
        depth,
        dip,
        dip_dir,
        dip_dir_nztm,
        strike_nztm,
        length,
        width,
    )


@given(valid_trace_definition())
def test_plane_from_trace(data: tuple):
    (
        trace_points_nztm,
        dtop,
        depth,
        dip,
        dip_dir,
        dip_dir_nztm,
        strike_nztm,
        length,
        width,
    ) = data

    plane = Plane.from_nztm_trace(
        trace_points_nztm, dtop, dtop + depth, dip, dip_dir_nztm=dip_dir_nztm
    )
    assert pytest.approx(plane.top_m, abs=1e-3) == dtop * 1000
    assert pytest.approx(plane.bottom_m, abs=1e-3) == (dtop + depth) * 1000
    assert pytest.approx(plane.dip, abs=1e-6) == dip
    assert pytest.approx(plane.dip_dir, abs=1e-1) == dip_dir
    assert pytest.approx(plane.dip_dir_nztm, abs=1e-3) == dip_dir_nztm
    assert pytest.approx(plane.strike_nztm, abs=1e-6) == strike_nztm
    assert pytest.approx(plane.width, abs=1e-3) == width
    assert pytest.approx(plane.length_m, abs=1e-3) == length
    assert pytest.approx(plane.projected_width, abs=1e-3) == width * np.cos(
        np.radians(dip)
    )
    if plane.dip == 90:
        assert shapely.get_coordinates(plane.geometry, include_z=True) == pytest.approx(
            plane.bounds[:2]
        )
    else:
        assert shapely.get_coordinates(plane.geometry, include_z=True)[
            :-1
        ] == pytest.approx(plane.bounds)

    # Generate plane using dip_dir
    plane = Plane.from_nztm_trace(
        trace_points_nztm, dtop, dtop + depth, dip, dip_dir=dip_dir
    )
    assert plane.top_m == pytest.approx(dtop * 1000, abs=1e-3)
    assert plane.bottom_m == pytest.approx((dtop + depth) * 1000, abs=1e-3)
    assert plane.dip == pytest.approx(dip, abs=1e-6)
    assert plane.dip_dir == pytest.approx(dip_dir, abs=10)
    assert plane.dip_dir_nztm == pytest.approx(dip_dir_nztm, abs=1)
    assert plane.strike_nztm == pytest.approx(strike_nztm, abs=1e-6)
    assert plane.width == pytest.approx(
        plane.projected_width / np.cos(np.radians(plane.dip)), abs=1e-6
    )
    assert pytest.approx(plane.length_m, abs=1e-3) == length
    assert shapely.get_coordinates(
        plane.trace_geometry, include_z=True
    ) == pytest.approx(plane.bounds[:2])

    if plane.dip == 90:
        assert shapely.get_coordinates(plane.geometry, include_z=True) == pytest.approx(
            plane.bounds[:2]
        )
    else:
        assert shapely.get_coordinates(plane.geometry, include_z=True)[
            :-1
        ] == pytest.approx(plane.bounds)


def test_invalid_trace_points():
    """Test that constructing a Plane with invalid trace points raises a ValueError."""
    trace_points = np.array([[0, 0], [1, 1], [2, 2]])  # 3 points
    with pytest.raises(ValueError, match="Trace points must be a 2x2 array."):
        Plane.from_nztm_trace(trace_points, 0, 1, 45, 45)


def test_invalid_dip_dir_90_dip():
    """Test that constructing a Plane with an invalid dip direction raises a ValueError."""
    start_trace_point = np.asarray([-43.5321, 172.6362])
    end_trace_point = geo.ll_shift(start_trace_point[0], start_trace_point[1], 5, 0)
    trace_points_nztm = coordinates.wgs_depth_to_nztm(
        np.stack((start_trace_point, end_trace_point), axis=0)
    )

    with pytest.raises(
        ValueError,
        match="Dip direction must be 0 for vertical faults.",
    ):
        Plane.from_nztm_trace(trace_points_nztm, 0, 1, 90, dip_dir_nztm=20)


def test_missing_dip_dir():
    """Test that constructing a Plane without dip_dir or dip_dir_nztm raises a ValueError."""
    trace_points = np.array([[0, 0], [1, 1]])
    with pytest.raises(
        ValueError, match="Must supply at least one of dip_dir or dip_dir_nztm."
    ):
        Plane.from_nztm_trace(trace_points, 0, 1, 45)


def test_both_dip_dir_provided():
    """Test that constructing a Plane with both dip_dir and dip_dir_nztm raises a ValueError."""
    trace_points = np.array([[0, 0], [1, 1]])
    with pytest.raises(
        ValueError, match="Must supply at most one of dip_dir or dip_dir_nztm."
    ):
        Plane.from_nztm_trace(trace_points, 0, 1, 45, dip_dir=90, dip_dir_nztm=90)


def test_fault_with_short_segments():
    """Check that a fault containing a less than 10m segment between two other segments does not raise an error."""
    trace_points_offset = np.array(
        [
            [0, 0],
            [100, 0],
            [2, 0],
            [300, 0],
        ]
    ).cumsum(axis=0)
    trace_points_nztm = (
        coordinates.wgs_depth_to_nztm(np.array([-43.0, 172.0])) + trace_points_offset
    )
    planes = [
        Plane.from_nztm_trace(
            trace_points_nztm[i : i + 2], dip=90, dtop=0, dbottom=10, dip_dir=90
        )
        for i in range(len(trace_points_nztm) - 1)
    ]
    with pytest.warns(UserWarning):
        _ = Fault(planes)


@pytest.mark.parametrize(
    "centroid, strike, dip, dip_dir, length, width, dtop, dbottom",
    [
        # Case where top and bottom depths are not consistent with dip and width
        (
            np.array([0, 0, 0]),
            45,
            30,
            None,
            10,
            5,
            1,
            2,
        ),
        # Case where top and bottom depths are not consistent with centroid depth
        (
            np.array([0, 0, 5]),
            45,
            30,
            None,
            10,
            5,
            1,
            10,
        ),
        # Case where neither top, bottom, nor centroid depth is given
        (
            np.array([0, 0]),
            45,
            30,
            None,
            10,
            5,
            None,
            None,
        ),
        # Case where centroid depth and dtop are inconsistent
        (
            np.array([0, 0, 5]),
            45,
            30,
            None,
            10,
            5,
            1,
            None,
        ),
        # Case where centroid depth and dbottom are inconsistent
        (
            np.array([0, 0, 5]),
            45,
            30,
            None,
            10,
            5,
            None,
            9,
        ),
    ],
)
def test_from_centroid_strike_dip_failure_cases(
    centroid: np.ndarray,
    strike: float,
    dip: float,
    dip_dir: float | None,
    length: float,
    width: float,
    dtop: float | None,
    dbottom: float | None,
):
    with pytest.raises(ValueError):
        Plane.from_centroid_strike_dip(
            centroid,
            dip,
            length,
            width,
            dtop=dtop,
            dbottom=dbottom,
            strike_nztm=strike,
            dip_dir_nztm=dip_dir,
        )


@pytest.mark.parametrize(
    "centroid, strike, dip, dip_dir, length, width, dtop, dbottom, expected_dtop, expected_dbottom",
    [
        # Case where both dtop and dbottom are None
        (np.array([0, 0, 5]), 45, 30, None, 10, 5, None, None, 5 - 5 / 4, 5 + 5 / 4),
        # Case where dtop is None and dbottom is provided
        (np.array([0, 0]), 45, 30, None, 10, 5, None, 10, 10 - 5 / 2, 10),
        # Case where dbottom is None and dtop is provided
        (np.array([0, 0]), 45, 30, None, 10, 5, 0, None, 0, 5 / 2),
    ],
)
def test_from_centroid_strike_dip_dtop_dbottom_derivation(
    centroid: np.ndarray,
    strike: float,
    dip: float,
    dip_dir: float | None,
    length: float,
    width: float,
    dtop: float | None,
    dbottom: float | None,
    expected_dtop: float,
    expected_dbottom: float,
):
    plane = Plane.from_centroid_strike_dip(
        centroid,
        dip,
        length,
        width,
        dtop=dtop,
        dbottom=dbottom,
        strike_nztm=strike,
        dip_dir_nztm=dip_dir,
    )
    assert np.isclose(plane.corners[0, -1], expected_dtop * 1000)
    assert np.isclose(plane.corners[-1, -1], expected_dbottom * 1000)


fault_plane = st.builds(
    Plane.from_centroid_strike_dip,
    centroid=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(1, 10),
    ),
    length=st.floats(0.1, 1000),
    width=st.floats(0.1, 1000),
    strike_nztm=st.floats(0, 179),
    dip_dir_nztm=st.floats(5, 179),
    dip=st.floats(0.1, 90),
)


@given(
    plane=fault_plane,
    point=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(0, 100),
    ),
)
@settings(deadline=None)
def test_plane_rrup(plane: Plane, point: np.ndarray):
    assume(plane.dip_dir >= plane.strike + 5)
    point = coordinates.wgs_depth_to_nztm(point)

    def fault_coordinate_distance(fault_coordinates: np.ndarray) -> float:
        fault_point = coordinates.wgs_depth_to_nztm(
            plane.fault_coordinates_to_wgs_depth_coordinates(fault_coordinates)
        )
        return point - fault_point

    res = sp.optimize.least_squares(
        fault_coordinate_distance,
        np.array([1 / 2, 1 / 2]),
        bounds=([0] * 2, [1] * 2),
        gtol=1e-5,
        ftol=1e-5,
    )
    optimized_res = np.linalg.norm(res.fun)
    assert np.isclose(
        plane.rrup_distance(coordinates.nztm_to_wgs_depth(point)),
        optimized_res,
        atol=1e-3,
    )


@given(
    plane=fault_plane,
    local_coordinates=nst.arrays(
        float, (2,), elements={"min_value": 0, "max_value": 1}
    ),
)
def test_plane_rrup_in_plane(plane: Plane, local_coordinates: np.ndarray):
    assume(plane.dip_dir >= plane.strike + 5)
    assert np.isclose(
        plane.rrup_distance(
            plane.fault_coordinates_to_wgs_depth_coordinates(local_coordinates)
        ),
        0,
    )


@given(
    plane=fault_plane,
    distance=st.floats(1, 1000),
)
def test_plane_rjb(plane: Plane, distance: float):
    # if dip dir is too close to strike it will create a degenerate geometry that rjb distance isn't designed for anyway.
    assume(plane.dip_dir_nztm >= plane.strike_nztm + 1)
    assume(plane.dip != 90)
    buffer = shapely.buffer(plane.geometry, distance * 1000)
    for point in coordinates.nztm_to_wgs_depth(np.array(buffer.exterior.coords)):
        assert np.isclose(
            plane.rjb_distance(point),
            distance * 1000,
            atol=1e-4,
        )


@given(
    plane=fault_plane,
    local_coordinates=nst.arrays(
        float, (2,), elements={"min_value": 0, "max_value": 1}
    ),
)
@seed(1)
def test_plane_coordinate_inversion(plane: Plane, local_coordinates: np.ndarray):
    """Test the inversion of coordinate transformations for a Plane object."""
    assume(not np.isclose(plane.dip_dir, plane.strike))
    assert np.allclose(
        plane.wgs_depth_coordinates_to_fault_coordinates(
            plane.fault_coordinates_to_wgs_depth_coordinates(local_coordinates)
        ),
        local_coordinates,
        atol=1e-6,
    )


@given(
    plane=st.builds(
        Plane.from_centroid_strike_dip,
        centroid=st.builds(
            coordinate,
            lat=st.floats(-50, -31),
            lon=st.floats(160, 180),
            depth=st.floats(1, 10),
        ),
        length=st.floats(0.1, 1000),
        width=st.floats(0.1, 1000),
        strike_nztm=st.floats(0, 179),
        dip=st.just(90),
    ),
    local_coordinates=nst.arrays(
        float, (2,), elements={"min_value": 0, "max_value": 1}
    ),
)
@seed(1)
def test_vertical_plane_coordinate_inversion(
    plane: Plane, local_coordinates: np.ndarray
):
    """Test the inversion of coordinate transformations for a Plane object."""
    assert np.allclose(
        plane.wgs_depth_coordinates_to_fault_coordinates(
            plane.fault_coordinates_to_wgs_depth_coordinates(local_coordinates)
        ),
        local_coordinates,
        atol=1e-6,
    )


def connected_fault(
    lengths: list[float], width: float, strike: float, start_coordinates: np.ndarray
) -> Fault:
    """Create a Fault object from connected planes."""
    strike_direction = np.array(
        [np.cos(np.radians(strike)), np.sin(np.radians(strike)), 0]
    )
    dip = 45  # Fixed dip for now
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


@given(
    fault=st.builds(
        connected_fault,
        lengths=st.lists(st.floats(0.1, 100), min_size=1, max_size=5),
        width=st.floats(0.1, 100),
        strike=st.floats(0, 179),
        start_coordinates=st.builds(
            coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)
        ),
    )
)
@settings(deadline=None)
def test_fault_reordering(fault: Fault):
    """Ensure that the plane order in faults is completely determined by the planes."""
    for order in itertools.permutations(range(len(fault.planes))):
        planes = [fault.planes[i] for i in order]
        fault_reorder = Fault(planes)
        assert np.allclose(fault_reorder.corners, fault.corners)


@given(
    fault=st.builds(
        connected_fault,
        lengths=st.lists(st.floats(0.1, 100), min_size=1, max_size=10),
        width=st.floats(0.1, 100),
        strike=st.floats(0, 179),
        start_coordinates=st.builds(
            coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)
        ),
    ),
    distance=st.floats(1, 1000),
)
def test_fault_rjb(fault: Fault, distance: float):
    # if dip dir is too close to strike it will create a degenerate geometry that rjb distance isn't designed for anyway.
    buffer = shapely.buffer(fault.geometry, distance * 1000)
    for point in coordinates.nztm_to_wgs_depth(np.array(buffer.exterior.coords)):
        assert np.isclose(
            fault.rjb_distance(point),
            distance * 1000,
            atol=1e-4,
        )


@given(
    fault=st.builds(
        connected_fault,
        lengths=st.lists(st.floats(0.1, 100), min_size=1, max_size=10),
        width=st.floats(0.1, 100),
        strike=st.floats(0, 179),
        start_coordinates=st.builds(
            coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)
        ),
    ),
    point=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
        depth=st.floats(0, 100),
    ),
)
def test_fault_rrup(fault: Fault, point: np.ndarray):
    # The fault rrup should be equal to the smallest rrup among the planes in the fault.
    fault_rrup = fault.rrup_distance(point)
    assert np.isclose(
        min(plane.rrup_distance(point) for plane in fault.planes), fault_rrup, atol=1e-3
    )


@given(
    fault=st.builds(
        connected_fault,
        lengths=st.lists(st.floats(0.1, 100), min_size=1, max_size=10),
        width=st.floats(0.1, 100),
        strike=st.floats(0, 179),
        start_coordinates=st.builds(
            coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)
        ),
    )
)
def test_fault_construction(fault: Fault):
    """Test the construction of a Fault object from connected planes."""
    assert fault.width == fault.planes[0].width
    assert fault.dip_dir == fault.planes[0].dip_dir
    assert fault.dip_dir_nztm == fault.planes[0].dip_dir_nztm
    assert fault.dip == fault.planes[0].dip
    assert fault.corners.shape == (4 * len(fault.planes), 3)
    assert np.isclose(fault.area(), np.sum([plane.area for plane in fault.planes]))
    assert fault.geometry.equals(
        shapely.union_all([plane.geometry for plane in fault.planes])
    )
    assert fault.trace_geometry.equals(
        shapely.union_all([plane.trace_geometry for plane in fault.planes])
    )
    assert np.allclose(
        fault.wgs_depth_coordinates_to_fault_coordinates(fault.centroid),
        np.array([1 / 2, 1 / 2]),
    )


@given(
    fault=st.builds(
        connected_fault,
        lengths=st.lists(st.floats(0.1, 100), min_size=1, max_size=10),
        strike=st.just(0),
        width=st.floats(0.1, 10),
        start_coordinates=st.builds(
            coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)
        ),
    ),
    other_fault=st.builds(
        connected_fault,
        lengths=st.lists(st.floats(0.1, 100), min_size=1, max_size=10),
        strike=st.just(0),
        width=st.floats(0.1, 10),
        start_coordinates=st.builds(
            coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)
        ),
    ),
)
@settings(deadline=1000)
@seed(1)
def test_fault_closest_point_comparison(fault: Fault, other_fault: Fault):
    """Test the closest point comparison between two Fault objects."""
    pairwise_distance = sp.spatial.distance.cdist(fault.bounds, other_fault.bounds)
    assume(pairwise_distance.min() > 1)
    point_a, point_b = sources.closest_point_between_sources(fault, other_fault)
    x, y = np.meshgrid(np.linspace(0, 1, num=10), np.linspace(0, 1, num=10))
    local_coords = np.c_[x.ravel(), y.ravel()]
    points_on_a = np.array(
        [
            fault.fault_coordinates_to_wgs_depth_coordinates(coord)
            for coord in local_coords
        ]
    )
    points_on_b = np.array(
        [
            other_fault.fault_coordinates_to_wgs_depth_coordinates(coord)
            for coord in local_coords
        ]
    )
    min_distance = np.min(
        coordinates.distance_between_wgs_depth_coordinates(points_on_a, points_on_b)
    )
    assume(min_distance > 0)
    computed_distance = coordinates.distance_between_wgs_depth_coordinates(
        fault.fault_coordinates_to_wgs_depth_coordinates(point_a),
        other_fault.fault_coordinates_to_wgs_depth_coordinates(point_b),
    )
    assert computed_distance < min_distance or np.isclose(
        computed_distance, min_distance, atol=1e-1
    )


@pytest.mark.parametrize(
    "planes, expected_message",
    [
        # Inconsistent dip directions
        (
            [
                Plane.from_centroid_strike_dip(
                    np.array([-41.2865, 174.7762, 50]),
                    30,
                    4,
                    4,
                    strike_nztm=0,
                    dip_dir_nztm=30,
                ),
                Plane.from_centroid_strike_dip(
                    coordinates.nztm_to_wgs_depth(
                        coordinates.wgs_depth_to_nztm(
                            np.array([-41.2865, 174.7762, 50])
                        )
                        + np.array([4000, 0, 0], dtype=float)
                        + np.array([-15.3426846, 26.04670979, 0.0])
                    ),
                    30,
                    4,
                    4,
                    strike_nztm=0,
                    dip_dir_nztm=31,
                ),
            ],
            "Fault must have a constant dip direction",
        ),
        # Inconsistent dip angles
        (
            [
                Plane.from_centroid_strike_dip(
                    np.array([-41.2865, 174.7762, 5000]),
                    30,
                    4,
                    4,
                    strike_nztm=0,
                    dip_dir_nztm=30,
                ),
                Plane.from_centroid_strike_dip(
                    np.array(
                        [-41.2865 + 0.036, 174.7862, 5000]
                    ),  # 1 degree of latitude is approximately 111 km
                    31,
                    4,
                    4,
                    strike_nztm=0,
                    dip_dir_nztm=30,
                ),
            ],
            "Fault must have a constant dip",
        ),
        # Inconsistent widths
        (
            [
                Plane.from_centroid_strike_dip(
                    np.array([-41.2865, 174.7762, 5000]),
                    30,
                    4,
                    4.01,
                    strike_nztm=0,
                    dip_dir_nztm=30,
                ),
                Plane.from_centroid_strike_dip(
                    np.array(
                        [-41.2865 + 0.036, 174.7862, 5000]
                    ),  # 1 degree of latitude is approximately 111 km
                    30,
                    4,
                    4,
                    strike_nztm=0,
                    dip_dir_nztm=30,
                ),
            ],
            "Fault must have constant width",
        ),
        # Not connected end-to-end
        (
            [
                Plane.from_centroid_strike_dip(
                    np.array([-41.2865, 174.7762, 5000]),
                    30,
                    4,
                    4,
                    strike_nztm=0,
                    dip_dir_nztm=30,
                ),
                Plane.from_centroid_strike_dip(
                    np.array(
                        [-41.2865 + 0.05, 174.7862, 5000]
                    ),  # 1 degree of latitude is approximately 111 km
                    30,
                    4,
                    4,
                    strike_nztm=0,
                    dip_dir_nztm=30,
                ),
            ],
            "Fault planes must be connected",
        ),
    ],
)
def test_fault_construction_failures(planes: list[Plane], expected_message: str):
    with pytest.raises(ValueError) as excinfo:
        Fault(planes=planes)
    assert expected_message in str(excinfo.value)


@pytest.mark.parametrize(
    ("fault_name"),
    [
        "Alpine: Caswell",
        "Alpine: Caswell - South George",
        "Alpine: George landward",
        "Alpine: George to Jacksons",
        "Alpine: Jacksons to Kaniere",
        "Alpine: Resolution - Charles",
        "Alpine: Resolution - Dagg",
        "Alpine: Resolution - Five Fingers",
        "Hope: Hanmer NW",
        "Hope: Hurunui",
        "Hope: Kakapo-2-Hamner",
        "Kakapo",
        "Kelly",
    ],
)
def test_load_fault(fault_name: str):
    with open(DATA_PATH / "alpine_faults.json") as f:
        fault_json = json.load(f)[fault_name]

    corners = np.array(
        [
            [corner["latitude"], corner["longitude"], corner["depth"]]
            for corner in fault_json["corners"]
        ]
    ).reshape((-1, 4, 3))
    dip = fault_json["dip"]
    dip_dir = fault_json["dip_dir"]
    fault = Fault(planes=[Plane.from_corners(c) for c in corners])
    assert fault.dip == pytest.approx(dip, abs=0.5)
    assert fault.dip_dir == pytest.approx(dip_dir, abs=1)


def generate_sheared_fault_with_lengths(
    lengths: np.ndarray,
    shears: np.ndarray,
    width: float,
    dip: float,
    strike: float,
    start_coordinates: np.ndarray,
) -> Fault:
    dip_rotvec = sp.spatial.transform.Rotation.from_rotvec(
        np.array([dip, 0, 0]), degrees=True
    )
    strike_rotvec = sp.spatial.transform.Rotation.from_rotvec(
        np.array([0, 0, strike]), degrees=True
    )
    trace_lengths = np.sqrt(lengths**2 - shears**2)
    trace = np.zeros((len(lengths) + 1, 3))
    trace[1:, 0] = np.cumsum(trace_lengths)

    bottom_trace = trace.copy()
    bottom_trace[:, 1] = width

    trace = dip_rotvec.apply(trace)
    bottom_trace = dip_rotvec.apply(bottom_trace)

    trace[1:, 1] += shears.cumsum()
    bottom_trace[1:, 1] += shears.cumsum()

    trace = strike_rotvec.apply(trace)
    bottom_trace = strike_rotvec.apply(bottom_trace)

    trace += start_coordinates
    bottom_trace += start_coordinates

    return Fault(
        [
            Plane(
                np.array(
                    [
                        trace[i],
                        trace[i + 1],
                        bottom_trace[i + 1],
                        bottom_trace[i],
                    ]
                )
            )
            for i in range(len(lengths))
        ]
    )


@st.composite
def fault(
    draw: st.DrawFn,
    min_segments: int = 1,
    max_segments: int = 10,
    min_length: float = 0.1,
    max_length: float = 100,
    max_dip: float = 90,
    max_width: float = 100,
) -> Fault:
    lengths = draw(
        st.lists(
            st.floats(min_length, max_length),
            min_size=min_segments,
            max_size=max_segments,
        )
    )
    shears = [draw(st.floats(0.95 * -length, 0.95 * length)) for length in lengths]
    width = draw(st.floats(0.1, max_width)) * 1000
    dip = draw(st.floats(1, max_dip))
    strike = draw(st.floats(0, 360))
    start_coordinates = coordinates.wgs_depth_to_nztm(
        draw(
            st.builds(
                coordinate,
                lat=st.floats(-47, -34),
                lon=st.floats(165, 178),
                depth=st.just(0),
            )
        )
    )
    return generate_sheared_fault_with_lengths(
        np.array(lengths) * 1000,
        np.array(shears) * 1000,
        width,
        dip,
        strike,
        start_coordinates,
    )


@given(fault(min_segments=2, max_segments=10, min_length=0.4, max_length=100))
def test_simplify_fault(fault: Fault):
    tolerance = 0.4
    simplified_fault = sources.simplify_fault(fault, tolerance)
    for plane in simplified_fault.planes:
        assert plane.length > tolerance or plane.length == pytest.approx(tolerance)

    consecutive_small_planes = False
    for plane, next_plane in itertools.pairwise(fault.planes):
        if plane.length < tolerance and next_plane.length < tolerance:
            consecutive_small_planes = True
            break

    if not consecutive_small_planes:
        assert len(simplified_fault.planes) == sum(
            1
            for plane in fault.planes
            if plane.length > tolerance or np.isclose(plane.length, tolerance)
        )

    # Check that the simplified fault is similar to the original fault
    assert simplified_fault.area() / fault.area() == pytest.approx(1, abs=0.2)


@given(
    st.lists(
        fault(min_segments=2, max_segments=10, min_length=0.4, max_length=100),
        min_size=2,
        max_size=10,
    ),
)
def test_sources_as_geojson(faults: list[Fault]):
    geojson = json.loads(sources.sources_as_geojson_features(faults))
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == len(faults)
    for feature in geojson["features"]:
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] in {
            "Polygon",
            "MultiPolygon",
            "LineString",
            "MultiLineString",
        }


@given(
    fault=st.builds(
        connected_fault,
        lengths=st.lists(st.floats(0.1, 100), min_size=1, max_size=5),
        width=st.floats(0.1, 100),
        strike=st.floats(0, 179),
        start_coordinates=st.builds(
            coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)
        ),
    )
)
def test_fault_from_trace(fault: Fault):
    trace = coordinates.nztm_to_wgs_depth(fault.trace[:, :2])

    # Remove duplicated points
    if len(fault.planes) > 1:
        trace = np.concatenate(
            (trace[0, None], trace[1:-1, :][::2], trace[-1, None]), axis=0
        )

    new_fault = Fault.from_trace_points(
        trace,
        min(fault.bounds[:, 2]) / 1000,
        max(fault.bounds[:, 2]) / 1000,
        fault.dip,
        dip_dir_nztm=fault.dip_dir_nztm,
    )

    assert pytest.approx(new_fault.dip, abs=1e-6) == fault.dip
    assert pytest.approx(new_fault.dip_dir, abs=1e-3) == fault.dip_dir
    assert np.allclose(new_fault.trace, fault.trace, atol=1e-3)
    assert np.allclose(new_fault.bounds, fault.bounds, atol=1e-3)


@pytest.mark.parametrize(
    "top_a, bottom_a, top_b, bottom_b, min_depth_km, expect_error, expected_dip_a, expected_dip_b",
    [
        (0, 10000, 0, 12000, 5, False, 0.5, 0.416666666667),
        (0, 10000, 0, 10000, 10, True, None, None),
        (0, 10000, 0, 10000, 15, True, None, None),
        (0, 100, 0, 100, 0.01, False, 0.1, 0.1),
    ],
)
def test_closest_points_beneath(
    top_a: float,
    bottom_a: float,
    top_b: float,
    bottom_b: float,
    min_depth_km: float,
    expect_error: bool,
    expected_dip_a: float,
    expected_dip_b: float,
):
    source_a = MagicMock()
    source_a.top_m = top_a
    source_a.bottom_m = bottom_a

    source_b = MagicMock()
    source_b.top_m = top_b
    source_b.bottom_m = bottom_b

    with patch("source_modelling.sources.closest_point_between_sources") as mock_cpbs:
        mock_cpbs.return_value = ("mock_a", "mock_b")

        if expect_error:
            with pytest.raises(ValueError):
                sources.closest_points_beneath(source_a, source_b, min_depth_km)
            return

        result = sources.closest_points_beneath(source_a, source_b, min_depth_km)

        assert result == ("mock_a", "mock_b")

        args = mock_cpbs.call_args[0]
        assert args[0] is source_a
        assert args[1] is source_b

        a_bounds, b_bounds = args[2], args[3]

        assert a_bounds.min_strike == 0
        assert a_bounds.max_strike == 1
        assert b_bounds.min_strike == 0
        assert b_bounds.max_strike == 1

        assert float(a_bounds.min_dip) == pytest.approx(expected_dip_a)
        assert a_bounds.max_dip == 1
        assert float(b_bounds.min_dip) == pytest.approx(expected_dip_b)
        assert b_bounds.max_dip == 1
