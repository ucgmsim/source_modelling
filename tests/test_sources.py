from typing import Optional

import numpy as np
import pandas as pd
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from qcore import coordinates
from source_modelling import sources
from source_modelling.sources import Plane


def coordinate(lat: float, lon: float, depth: Optional[float] = None) -> np.ndarray:
    if depth is not None:
        return np.array([lat, lon, depth])
    return np.array([lat, lon])


def valid_coordinates(point_coordinates: np.ndarray) -> bool:
    return np.all(np.isfinite(coordinates.wgs_depth_to_nztm(point_coordinates)))


@given(
    point_coordinates=st.builds(
        coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180), depth=st.floats(0)
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
    assert np.isclose(point.width_m, point.width * 1000)
    assert np.allclose(point.centroid, point_coordinates)


@given(
    point_coordinates=st.builds(
        coordinate, lat=st.floats(-50, -31), lon=st.floats(160, 180), depth=st.floats(0)
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
    local_coordinates=nst.arrays(
        float, (2,), elements={"min_value": 0, "max_value": 1}
    ),
)
def test_fault_coordinate_inversion(plane: Plane, local_coordinates: np.ndarray):
    assume(not np.isclose(plane.dip_dir, plane.strike))
    assert np.allclose(
        plane.wgs_depth_coordinates_to_fault_coordinates(
            plane.fault_coordinates_to_wgs_depth_coordinates(local_coordinates)
        ),
        local_coordinates,
    )
