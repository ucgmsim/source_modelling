import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from source_modelling import community_fault_model
from source_modelling.community_fault_model import MovementSense, NodalPlane


def test_can_load_community_fault_model():
    model = community_fault_model.get_community_fault_model()
    assert len(model) == 880
    # Traces are oriented so that the fault dips to the right of the trace.
    for fault in model:
        if fault.dip_dir is None:
            continue
        coords = np.asarray(fault.trace.coords)
        strike = community_fault_model.line_segment_strike(coords[0], coords[-1])
        misfit = (strike + 90 - fault.dip_dir.value + 180) % 360 - 180
        assert abs(misfit) <= 90, fault.name
    gdf = community_fault_model.community_fault_model_as_geodataframe()
    assert len(gdf) == 880


def test_loaded_faults_have_sense():
    model = community_fault_model.get_community_fault_model()
    for fault in model:
        assert fault.sense != MovementSense(0), fault.name


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("", None),
        ("yrs", 1),
        ("100 yrs", 100),
        ("1,000s yrs", 1000),
        ("Ma", 1_000_000),
    ],
)
def test_parse_timeframe_unit(unit: str, expected: int | None):
    assert community_fault_model.parse_timeframe_unit(unit) == expected


@pytest.mark.parametrize("unit", ["days", "kyr", "1000"])
def test_parse_timeframe_unit_unknown(unit: str):
    with pytest.raises(ValueError, match="Unknown time unit"):
        community_fault_model.parse_timeframe_unit(unit)


@pytest.mark.parametrize(
    "sense, expected",
    [
        ("", MovementSense(0)),
        ("dextral", MovementSense.DEXTRAL),
        ("sinistral", MovementSense.SINISTRAL),
        ("normal", MovementSense.NORMAL),
        ("reverse", MovementSense.REVERSE),
        ("reverse dextral", MovementSense.DEXTRAL | MovementSense.REVERSE),
        ("normal sinistral", MovementSense.NORMAL | MovementSense.SINISTRAL),
    ],
)
def test_fault_sense_parse(sense: str, expected: MovementSense):
    assert community_fault_model.fault_sense_parse(sense) == expected


@pytest.mark.parametrize(
    "point_b, expected",
    [
        ([5_010_000, 1_500_000], 0.0),
        ([5_000_000, 1_510_000], 90.0),
        ([4_990_000, 1_500_000], 180.0),
        ([5_000_000, 1_490_000], 270.0),
    ],
)
def test_line_segment_strike(point_b: list[float], expected: float):
    strike = community_fault_model.line_segment_strike([5_000_000, 1_500_000], point_b)
    # Allow for NZTM grid convergence (< ~3 degrees in New Zealand).
    misfit = (strike - expected + 180) % 360 - 180
    assert abs(misfit) < 3


def test_most_likely_nodal_plane():
    solutions = pd.read_csv("tests/data/nodal_plane_labels.csv")
    model = community_fault_model.get_community_fault_model()
    correct = 0
    for _, solution in solutions.iterrows():
        nodal_plane_1 = NodalPlane(
            solution["strike1"], solution["dip1"], solution["rake1"]
        )
        nodal_plane_2 = NodalPlane(
            solution["strike2"], solution["dip2"], solution["rake2"]
        )
        if (
            community_fault_model.most_likely_nodal_plane(
                model,
                np.array([solution["Latitude"], solution["Longitude"], solution["CD"]]),
                nodal_plane_1,
                nodal_plane_2,
            )
            == nodal_plane_1
        ):
            correct += 1

    # The model's grouped cross-validation accuracy is ~0.86 and it scores
    # 316/364 on its own training labels.
    assert correct >= 305


_CFM_SCHEMA = {
    "geometry": "LineString",
    "properties": {
        "Name": "str:50",
        "Fault_stat": "str:50",
        "Domain_No": "int:10",
        "Dip_pref": "int:10",
        "Dip_min": "float:19.11",
        "Dip_max": "float:19.11",
        "Dip_dir": "str:254",
        "Dom_sense": "str:50",
        "Sub_sense": "str:50",
        "Rake_pref": "int:10",
        "Rake_minus": "int:10",
        "Rake_plus": "int:10",
        "SR_pref": "float:19.11",
        "SR_min": "float:19.11",
        "SR_max": "float:19.11",
        "SRT_pref": "str:50",
        "SRT_gen": "str:50",
        "UpdDth_prf": "float:19.11",
        "UpdDth_min": "float:19.11",
        "UpdDth_max": "float:19.11",
        "Depth_Dfc": "float:19.11",
        "Method_Dfc": "int:10",
        "Depth_D90": "float:19.11",
        "Method_D90": "float:19.11",
        "QualCode": "int:10",
        "References": "str:200",
        "Comments": "str:254",
    },
}


def _cfm_properties(**overrides: object) -> dict[str, object]:
    properties: dict[str, object] = {
        "Name": "Test Fault",
        "Fault_stat": "A-US",
        "Domain_No": 18,
        "Dip_pref": 60,
        "Dip_min": 45.0,
        "Dip_max": 75.0,
        "Dip_dir": "SE",
        "Dom_sense": "dextral",
        "Sub_sense": "reverse",
        "Rake_pref": 160,
        "Rake_minus": 140,
        "Rake_plus": 180,
        "SR_pref": 1.5,
        "SR_min": 0.5,
        "SR_max": 2.5,
        "SRT_pref": "12",
        "SRT_gen": "1,000s yrs",
        "UpdDth_prf": 1.0,
        "UpdDth_min": 0.5,
        "UpdDth_max": 2.0,
        "Depth_Dfc": 14.5,
        "Method_Dfc": 3,
        "Depth_D90": 12.25,
        "Method_D90": 2.0,
        "QualCode": 2,
        "References": "Somebody et al. 2020",
        "Comments": "a comment",
    }
    properties.update(overrides)
    return properties


def _write_cfm(path: Path, features: list[tuple[np.ndarray, dict[str, object]]]):
    """Write (northing, easting) traces and properties as a CFM-like shapefile."""
    import fiona

    with fiona.open(
        path, "w", driver="ESRI Shapefile", schema=_CFM_SCHEMA, crs="EPSG:2193"
    ) as writer:
        for trace, properties in features:
            writer.write(
                {
                    "geometry": {
                        "type": "LineString",
                        # Shapefiles store (x=easting, y=northing).
                        "coordinates": [tuple(p) for p in trace[:, ::-1]],
                    },
                    "properties": properties,
                }
            )


def _trace_with_strike(target_strike: float) -> np.ndarray:
    """Return a 10 km (northing, easting) trace whose strike is target_strike."""
    start = np.array([5_180_000.0, 1_570_000.0])
    grid_bearing = target_strike
    end = start
    error = np.inf
    # Correct the NZTM grid bearing for grid convergence until the
    # great-circle strike matches the target.
    for _ in range(10):
        direction = np.radians(grid_bearing)
        end = start + 10_000.0 * np.array([np.cos(direction), np.sin(direction)])
        strike = community_fault_model.line_segment_strike(start, end)
        error = (strike - target_strike + 180) % 360 - 180
        grid_bearing -= error
    assert abs(error) < 1e-6
    return np.array([start, end])


def test_load_community_fault_model_maps_properties(tmp_path: Path):
    trace = _trace_with_strike(30.0)
    shp = tmp_path / "cfm.shp"
    _write_cfm(
        shp,
        [
            (trace, _cfm_properties()),
            (
                trace,
                _cfm_properties(
                    Name="Other Fault",
                    Fault_stat="N-PS",
                    Domain_No=5,
                    Dip_dir="Subvertical and variable",
                    Dom_sense="normal",
                    Sub_sense="",
                    SRT_gen="",
                ),
            ),
        ],
    )

    faults = community_fault_model.load_community_fault_model(shp)
    assert len(faults) == 2
    fault, other = faults

    assert fault.name == "Test Fault"
    assert fault.fault_status == community_fault_model.FaultStatus.ACTIVE_NON_SEISOGENIC
    assert fault.domain == community_fault_model.NeotectonicDomain.CENTRAL_CANTERBURY
    assert fault.dip_range == community_fault_model.Range(45.0, 60, 75.0)
    assert fault.dip_dir == community_fault_model.CompassDirection.SE
    assert fault.sense == MovementSense.DEXTRAL
    assert fault.secondary_sense == MovementSense.REVERSE
    assert fault.rake_range == community_fault_model.Range(140, 160, 180)
    assert fault.slip_rate == community_fault_model.Range(0.5, 1.5, 2.5)
    assert fault.slip_rate_timeframe_pref == "12"
    assert fault.slip_rate_timeframe_unit == 1000
    assert fault.up_dip_depth == community_fault_model.Range(0.5, 1.0, 2.0)
    assert fault.down_dip_depth90 == pytest.approx(12.25)
    assert fault.down_dip_depth90_method == community_fault_model.DipMethod.OTHER
    assert fault.down_dip_depth_dfc == pytest.approx(14.5)
    assert (
        fault.down_dip_dfc_method
        == community_fault_model.DipMethod.DOWN_DIP_INTERSECTION
    )
    assert fault.quality_code == 2
    assert fault.reference == "Somebody et al. 2020"
    assert fault.comments == "a comment"
    # Strike 30 with dip direction SE (135) already dips to the right.
    np.testing.assert_allclose(np.asarray(fault.trace.coords), trace)

    assert other.name == "Other Fault"
    assert other.fault_status == community_fault_model.FaultStatus.INACTIVE
    assert (
        other.domain
        == community_fault_model.NeotectonicDomain.NORTH_ISLAND_DEXTRAL_FAULT_BELT
    )
    assert other.dip_dir is None
    assert other.sense == MovementSense.NORMAL
    assert other.secondary_sense is None
    assert other.slip_rate_timeframe_unit is None
    # Without a compass dip direction the trace is left as digitised.
    np.testing.assert_allclose(np.asarray(other.trace.coords), trace)


@pytest.mark.parametrize(
    "strike, reversed_",
    [
        # Dip direction N, so misfit = (strike + 270) % 360 - 180.
        (359.5, False),  # misfit 89.5
        (0.5, True),  # misfit 90.5
        (180.5, False),  # misfit -89.5
        (179.5, True),  # misfit -90.5
    ],
)
def test_load_community_fault_model_trace_orientation_boundary(
    tmp_path: Path, strike: float, reversed_: bool
):
    trace = _trace_with_strike(strike)
    shp = tmp_path / "cfm.shp"
    _write_cfm(shp, [(trace, _cfm_properties(Dip_dir="N"))])
    (fault,) = community_fault_model.load_community_fault_model(shp)
    expected = trace[::-1] if reversed_ else trace
    np.testing.assert_allclose(np.asarray(fault.trace.coords), expected)


def test_load_community_fault_model_without_fiona(monkeypatch: pytest.MonkeyPatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "fiona", None)
    with pytest.raises(ImportError) as excinfo:
        community_fault_model.get_community_fault_model()
    assert excinfo.value.__notes__ == [
        "Fiona can be installed using the optional dependency group fiona, pip install source-modelling[fiona]."
    ]


def test_community_fault_model_as_geodataframe_without_geopandas(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "geopandas", None)
    with pytest.raises(ImportError) as excinfo:
        community_fault_model.community_fault_model_as_geodataframe()
    assert excinfo.value.__notes__ == [
        "GeoPandas can be installed using the optional dependency group fiona, pip install source-modelling[fiona]."
    ]
