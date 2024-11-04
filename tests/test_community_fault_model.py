from source_modelling import community_fault_model
from source_modelling.community_fault_model import NodalPlane
import numpy as np
import pandas as pd



def test_can_load_community_fault_model():
    model = community_fault_model.get_community_fault_model()
    assert len(model) == 880
    gdf = community_fault_model.community_fault_model_as_geodataframe()
    assert len(gdf) == 880

def test_most_likely_nodal_plane():
    solutions = pd.read_csv('tests/data/GeoNet_Test_Solutions.csv')
    model = community_fault_model.get_community_fault_model()
    correct = 0
    for _, solution in solutions.iterrows():
        nodal_plane_1 = NodalPlane(solution['strike1'], solution['dip1'], solution['rake1'])
        nodal_plane_2 = NodalPlane(solution['strike2'], solution['dip2'], solution['rake2'])
        if community_fault_model.most_likely_nodal_plane(model, np.array([solution['Latitude'], solution['Longitude']]), nodal_plane_1, nodal_plane_2) == nodal_plane_1:
            correct += 1

    assert correct > 75
