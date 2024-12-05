from typing import Optional

import numpy as np
import pytest
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
    assume(valid_coordinates(centroid))
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
    assert np.allclose(
        shapely.get_coordinates(plane.geometry, include_z=True)[:-1], plane.bounds
    )
    assert np.isclose(plane.strike_nztm, strike, atol=1e-6)
    if plane.dip != 90:
        assert np.isclose(plane.dip_dir_nztm, dip_dir, atol=1e-6)
    assert np.allclose(plane.centroid, centroid * np.array([1, 1, 1000]), atol=1e-6)


@pytest.mark.parametrize(
    "centroid, strike, dip, dip_dir, length, width, dtop, dbottom, exception_message",
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
            r"Top and bottom depths are not consistent with dip and width parameters\.",
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
            r"Top and bottom depths are not consistent with centroid depth\.",
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
            r"At least one of top, bottom, or centroid depth must be given\.",
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
            r"Centroid depth and dtop are inconsistent\.",
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
            r"Centroid depth and dbottom are inconsistent\.",
        ),
    ],
)
def test_from_centroid_strike_dip_failure_cases(
    centroid,
    strike,
    dip,
    dip_dir,
    length,
    width,
    dtop,
    dbottom,
    exception_message,
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
    centroid,
    strike,
    dip,
    dip_dir,
    length,
    width,
    dtop,
    dbottom,
    expected_dtop,
    expected_dbottom,
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
    assume(plane.dip_dir >= plane.strike + 1)
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
    assert fault.width == fault.planes[0].width
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
