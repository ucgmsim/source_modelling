import itertools
from typing import Optional

import numpy as np
import pytest
import scipy as sp
import shapely
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from qcore import coordinates, geo
from source_modelling import sources
from source_modelling.sources import Fault, Plane


def coordinate(lat: float, lon: float, depth: Optional[float] = None) -> np.ndarray:
    """Create a coordinate array from latitude, longitude, and optional depth."""
    if depth is not None:
        return np.array([lat, lon, depth])
    return np.array([lat, lon])


def nztm_coordinate(y: float, x: float, depth: Optional[float] = None) -> np.ndarray:
    """Create a coordinate array from NZTM coordinates and optional depth."""
    if depth is not None:
        return np.array([y, x, depth])
    return np.array([y, x])


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
    strike=st.floats(0, 360),
    dip=st.floats(0, 180),
    dip_dir=st.floats(0, 360),
)
def test_point_construction(
    point_coordinates: np.ndarray,
    length_m: float,
    strike: float,
    dip: float,
    dip_dir: float,
):
    """Test the construction of a Point object from latitude, longitude, and depth."""
    assume(valid_coordinates(point_coordinates))
    point = sources.Point.from_lat_lon_depth(
        point_coordinates, length_m=length_m, strike=strike, dip=dip, dip_dir=dip_dir
    )
    assert np.allclose(point.coordinates, point_coordinates)
    assert point.length == point.width == length_m / 1000
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
    strike: float,
    dip: float,
    dip_dir: float,
    local_coordinates: np.ndarray,
):
    """Test the coordinate system transformation for a Point object."""
    assume(valid_coordinates(point_coordinates))

    point = sources.Point.from_lat_lon_depth(
        point_coordinates, length_m=length_m, strike=strike, dip=dip, dip_dir=dip_dir
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
    strike: float,
    dip: float,
    dip_dir: float,
    local_coordinates: np.ndarray,
):
    """Test the inversion of coordinate transformations for a Point object."""
    assume(valid_coordinates(point_coordinates))

    point = sources.Point.from_lat_lon_depth(
        point_coordinates, length_m=length_m, strike=strike, dip=dip, dip_dir=dip_dir
    )
    # NOTE: cannot assert invertibility coordinate mapping is not bijective
    assert np.allclose(
        point.wgs_depth_coordinates_to_fault_coordinates(
            point.fault_coordinates_to_wgs_depth_coordinates(local_coordinates)
        ),
        np.array([1 / 2, 1 / 2]),
    )


@given(
    length=st.floats(0.1, 1000),
    projected_width=st.floats(0.1, 1000),
    strike=st.floats(0, 179),
    dip_dir=st.floats(0, 179),
    top=st.floats(0, 100),
    depth=st.floats(0.1, 100),
    centroid=st.builds(coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)),
)
def test_plane_construction(
    length: float,
    projected_width: float,
    strike: float,
    dip_dir: float,
    top: float,
    depth: float,
    centroid: np.ndarray,
):
    """Test the construction of a Plane object from centroid, strike, and dip."""
    assume(valid_coordinates(centroid))
    assume(dip_dir > strike + 1)
    plane = Plane.from_centroid_strike_dip(
        centroid, strike, dip_dir, top, top + depth, length, projected_width
    )
    assert np.isclose(plane.length, length, atol=1e-6)
    assert np.isclose(
        plane.width, plane.projected_width / np.cos(np.radians(plane.dip)), atol=1e-6
    )
    assert np.allclose(
        shapely.get_coordinates(plane.geometry, include_z=True)[:-1], plane.bounds
    )
    assert np.isclose(plane.projected_width, projected_width, atol=1e-6)
    assert np.isclose(plane.strike, strike, atol=1e-6)
    assert np.isclose(plane.dip_dir, dip_dir, atol=1e-6)
    assert np.allclose(plane.centroid[:2], centroid, atol=1e-6)
    # The constructor should not care about plane bound orientation
    assert np.allclose(Plane(plane.bounds[::-1]).bounds, plane.bounds)


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
    start_trace_nztm: np.ndarray[float], length: float, strike: float
) -> np.ndarray:
    # Do this in NZTM to prevent any issues with the coordinate system conversions
    strike_vec = np.array([np.cos(np.radians(strike)), np.sin(np.radians(strike))])
    end_trace_nztm = start_trace_nztm + strike_vec * length

    trace_nztm = np.stack((start_trace_nztm, end_trace_nztm), axis=0)
    return coordinates.nztm_to_wgs_depth(trace_nztm)


@st.composite
def valid_trace_definition(draw):
    strike = draw(st.floats(1, 179))

    start_trace_nztm = draw(
        st.builds(
            nztm_coordinate,
            y=st.floats(4715500, 6221500),
            x=st.floats(1073000, 2154000),
        )
    )

    length = draw(st.floats(100, 1000_000))
    trace_points = trace(
        start_trace_nztm=start_trace_nztm,
        length=length,
        strike=strike,
    )

    dtop = draw(st.floats(0, 100))
    depth = draw(st.floats(1, 100))
    dip = draw(st.floats(1, 90))

    if np.isclose(dip, 90):
        dip_dir_nztm, dip_dir = 0, 0
    else:
        dip_dir_nztm = (strike + draw(st.floats(1, 179))) % 360
        dip_dir = coordinates.nztm_bearing_to_great_circle_bearing(
            trace_points[0], 1, dip_dir_nztm
        )

    return (
        trace_points,
        dtop,
        depth,
        dip,
        dip_dir,
        dip_dir_nztm,
        strike,
        length,
        start_trace_nztm,
    )


@given(valid_trace_definition())
def test_plane_from_trace(data):
    (
        trace_points,
        dtop,
        depth,
        dip,
        dip_dir,
        dip_dir_nztm,
        strike,
        length,
        start_trace_nztm,
    ) = data

    assume(valid_coordinates(trace_points[0]))
    assume(valid_coordinates(trace_points[1]))

    plane = Plane.from_trace(trace_points, dtop, dtop + depth, dip, dip_dir)
    assert np.isclose(plane.top_m, dtop * 1000, atol=1e-3)
    assert np.isclose(plane.bottom_m, (dtop + depth) * 1000, atol=1e-3)
    assert np.isclose(plane.dip, dip, atol=1e-6)
    assert np.isclose(plane.dip_dir, dip_dir_nztm, atol=1e-3)
    assert np.isclose(plane.strike, strike, atol=1e-6)
    assert np.isclose(plane.length_m, length, atol=1e-3)
    assert np.isclose(
        plane.width, plane.projected_width / np.cos(np.radians(plane.dip)), atol=1e-6
    )
    assert np.allclose(
        shapely.get_coordinates(plane.geometry, include_z=True)[:-1], plane.bounds
    )


def test_invalid_trace_points():
    """Test that constructing a Plane with invalid trace points raises a ValueError."""
    trace_points = np.array([[0, 0], [1, 1], [2, 2]])  # 3 points
    with pytest.raises(ValueError, match="Trace points must be a 2x2 array."):
        Plane.from_trace(trace_points, 0, 1, 45, 45)


def test_invalid_dip_dir():
    """Test that constructing a Plane with an invalid dip direction raises a ValueError."""
    start_trace_point = np.asarray([-43.5321, 172.6362])
    end_trace_point = geo.ll_shift(start_trace_point[0], start_trace_point[1], 5, 0)
    trace_points = np.stack((start_trace_point, end_trace_point), axis=0)

    with pytest.raises(
        ValueError,
        match="Dip direction is inconsistent with the strike defined by the trace points.",
    ):
        Plane.from_trace(trace_points, 0, 1, 45, 260)


def test_invalid_dip_dir_90_dip():
    """Test that constructing a Plane with an invalid dip direction raises a ValueError."""
    start_trace_point = np.asarray([-43.5321, 172.6362])
    end_trace_point = geo.ll_shift(start_trace_point[0], start_trace_point[1], 5, 0)
    trace_points = np.stack((start_trace_point, end_trace_point), axis=0)

    with pytest.raises(
        ValueError,
        match="Dip direction must be 0 for vertical faults.",
    ):
        Plane.from_trace(trace_points, 0, 1, 90, 20)


def fault_plane(
    length: float,
    projected_width: float,
    strike: float,
    dip_dir: float,
    top: float,
    depth: float,
    centroid: np.ndarray,
) -> Plane:
    """Create a Plane object from fault parameters."""
    return Plane.from_centroid_strike_dip(
        centroid, strike, dip_dir, top, top + depth, length, projected_width
    )


@given(
    plane=st.builds(
        fault_plane,
        length=st.floats(0.1, 1000),
        projected_width=st.floats(0.1, 1000),
        strike=st.floats(0, 179),
        dip_dir=st.floats(1, 179),
        top=st.floats(0, 100),
        depth=st.floats(0.1, 100),
        centroid=st.builds(
            coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180)
        ),
    ),
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
    )
)
def test_fault_construction(fault: Fault):
    """Test the construction of a Fault object from connected planes."""
    assert fault.width == fault.planes[0].width
    assert np.isclose(fault.dip_dir, fault.planes[0].strike + 90)
    assert fault.corners.shape == (4 * len(fault.planes), 3)
    assert np.isclose(fault.area(), np.sum([plane.area for plane in fault.planes]))
    assert fault.geometry.equals(
        shapely.union_all([plane.geometry for plane in fault.planes])
    )
    assert np.allclose(
        fault.wgs_depth_coordinates_to_fault_coordinates(fault.centroid),
        np.array([1 / 2, 1 / 2]),
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
    local_coordinates=nst.arrays(
        float, (2,), elements={"min_value": 0, "max_value": 1}
    ),
)
def test_fault_coordinate_inversion(fault: Fault, local_coordinates: np.ndarray):
    """Test the inversion of coordinate transformations for a Fault object."""
    assert np.allclose(
        fault.wgs_depth_coordinates_to_fault_coordinates(
            fault.fault_coordinates_to_wgs_depth_coordinates(local_coordinates)
        ),
        local_coordinates,
        atol=1e-6,
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


if __name__ == "__main__":
    data = (
        np.array([[-47.49983978, 166.00106361], [-47.49894825, 166.00120587]]),
        0.0,
        1.0,
        1.0,
        7.149842834075444,
        2.0,
        1.0,
        100.0,
        nztm_coordinate(y=4715500.0, x=1073000.0),
    )
    test_plane_from_trace(data)
    print(f"wtf")
