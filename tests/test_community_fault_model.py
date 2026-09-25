import numpy as np
import pandas as pd

from source_modelling import community_fault_model
from source_modelling.community_fault_model import NodalPlane


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


def test_most_likely_nodal_plane():
    solutions = pd.read_csv("tests/data/GeoNet_Test_Solutions.csv")
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
                solution["Mw"],
            )
            == nodal_plane_1
        ):
            correct += 1

    # The previous strike-only vote scored 81/98; grouped cross-validation of
    # the current model is ~0.87 so the in-sample score must be at least 85.
    assert correct >= 85
