from typing import Optional

import numpy as np
import scipy as sp
import shapely
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from qcore import coordinates
from source_modelling import sources
from source_modelling.sources import Fault, Plane


def coordinate(lat: float, lon: float, depth: Optional[float] = None) -> np.ndarray:
    if depth is not None:
        return np.array([lat, lon, depth])
    return np.array([lat, lon])


def valid_coordinates(point_coordinates: np.ndarray) -> bool:
    return np.all(np.isfinite(coordinates.wgs_depth_to_nztm(point_coordinates)))


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
    strike: float,
    dip: float,
    dip_dir: float,
    other_point: np.ndarray,
):
    assume(valid_coordinates(point_coordinates))

    point = sources.Point.from_lat_lon_depth(
        point_coordinates, length_m=length_m, strike=strike, dip=dip, dip_dir=dip_dir
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
    strike=st.floats(0, 360),
    dip=st.floats(0, 180),
    dip_dir=st.floats(0, 360),
    distance=st.floats(1, 1000),
)
def test_point_rjb(
    point_coordinates: np.ndarray,
    length_m: float,
    strike: float,
    dip: float,
    dip_dir: float,
    distance: float,
):
    assume(valid_coordinates(point_coordinates))

    point = sources.Point.from_lat_lon_depth(
        point_coordinates, length_m=length_m, strike=strike, dip=dip, dip_dir=dip_dir
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
    assume(valid_coordinates(centroid))
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


def fault_plane(
    length: float,
    projected_width: float,
    strike: float,
    dip_dir: float,
    top: float,
    depth: float,
    centroid: np.ndarray,
) -> Plane:
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
    distance=st.floats(1, 1000),
)
def test_plane_rrup(plane: Plane, distance: float):
    centre = np.mean(plane.bounds, axis=0)
    normal = np.cross(
        plane.bounds[1] - plane.bounds[0], plane.bounds[-1] - plane.bounds[0]
    )
    normal /= np.linalg.norm(normal)
    point = coordinates.nztm_to_wgs_depth(centre + distance * 1000 * normal)
    assert np.isclose(plane.rrup_distance(point), distance * 1000, atol=1e-4)


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
    distance=st.floats(1, 1000),
)
def test_plane_rjb(plane: Plane, distance: float):
    # if dip dir is too close to strike it will create a degenerate geometry that rjb distance isn't designed for anyway.
    assume(plane.dip_dir >= plane.strike + 1)
    buffer = shapely.buffer(plane.geometry, distance * 1000)
    for point in coordinates.nztm_to_wgs_depth(np.array(buffer.exterior.coords)):
        assert np.isclose(
            plane.rjb_distance(point),
            distance * 1000,
            atol=1e-4,
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
    # The fault rrup should be equal to the small rrup among the planes in the fault.
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
def test_fault_closest_point_comparison(fault: Fault, other_fault: float):
    pairwise_distance = sp.spatial.distance.cdist(fault.bounds, other_fault.bounds)
    assume(pairwise_distance.min() > 1)
    point_a, point_b = sources.closest_point_between_sources(fault, other_fault)
    X, Y = np.meshgrid(np.linspace(0, 1, num=10), np.linspace(0, 1, num=10))
    local_coords = np.c_[X.ravel(), Y.ravel()]
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
