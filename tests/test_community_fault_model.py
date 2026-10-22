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
