import numpy as np
import pytest

from source_modelling import focal_mechanism
from source_modelling.community_fault_model import NodalPlane
from source_modelling.focal_mechanism import TectonicType

# 2003 Fiordland earthquake (GeoNet 2103645), a Puysegur interface event.
CENTROID = np.array([-45.1929, 166.83, 22.0])
PLANE_1 = NodalPlane(strike=20, dip=35, rake=79)
PLANE_2 = NodalPlane(strike=213, dip=56, rake=98)


@pytest.fixture(scope="module")
def classifier() -> focal_mechanism.CMTClassifier:
    return focal_mechanism.CMTClassifier.load()


def test_nodal_plane_probability_is_antisymmetric(
    classifier: focal_mechanism.CMTClassifier,
):
    p = classifier.nodal_plane_1_probability(CENTROID, PLANE_1, PLANE_2)
    swapped = classifier.nodal_plane_1_probability(CENTROID, PLANE_2, PLANE_1)
    assert p + swapped == pytest.approx(1.0)
    assert classifier.most_likely_nodal_plane(CENTROID, PLANE_1, PLANE_2) == PLANE_1
    assert classifier.most_likely_nodal_plane(CENTROID, PLANE_2, PLANE_1) == PLANE_1


def test_contributions_sum_to_log_odds(classifier: focal_mechanism.CMTClassifier):
    contributions = classifier.nodal_plane_contributions(CENTROID, PLANE_1, PLANE_2)
    assert list(contributions) == focal_mechanism.NODAL_PLANE_FEATURE_NAMES
    p = classifier.nodal_plane_1_probability(CENTROID, PLANE_1, PLANE_2)
    assert sum(contributions.values()) == pytest.approx(np.log(p / (1 - p)))


def test_centroid_without_depth_warns(classifier: focal_mechanism.CMTClassifier):
    with pytest.warns(UserWarning, match="without depth"):
        classifier.most_likely_nodal_plane(CENTROID[:2], PLANE_1, PLANE_2)


def test_slab_query():
    slab_model = focal_mechanism.SlabModel.load()
    # Hikurangi interface beneath Wellington, in either longitude convention.
    slab = slab_model.query(-41.3, 174.8)
    assert slab.present
    assert 15 < slab.depth < 35
    assert slab_model.query(-41.3, 174.8 - 360) == slab
    assert not slab_model.query(-43.5, 172.6).present  # Christchurch


def test_tectonic_type(classifier: focal_mechanism.CMTClassifier):
    assert (
        classifier.tectonic_type(CENTROID, PLANE_1, PLANE_2) == TectonicType.INTERFACE
    )
    # Same mechanism well above the interface is crustal.
    shallow = np.array([-45.1929, 166.83, 2.0])
    assert classifier.tectonic_type(shallow, PLANE_1, PLANE_2) == TectonicType.CRUSTAL


@pytest.mark.parametrize(
    "depth_below_slab, interface_like, normal_like, expected",
    [
        (float("nan"), True, False, TectonicType.CRUSTAL),
        (-20.0, True, False, TectonicType.CRUSTAL),
        (-5.0, True, False, TectonicType.INTERFACE),
        (5.0, True, False, TectonicType.INTERFACE),
        (-5.0, False, False, TectonicType.CRUSTAL),
        (5.0, False, False, TectonicType.SLAB),
        (-5.0, False, True, TectonicType.SLAB),
        (20.0, True, False, TectonicType.SLAB),
    ],
)
def test_tectonic_type_probabilities(
    depth_below_slab: float,
    interface_like: bool,
    normal_like: bool,
    expected: TectonicType,
):
    rule = focal_mechanism.tectonic_type_probabilities(
        depth_below_slab, interface_like, normal_like, sigma=0.0
    )
    assert rule[expected] == 1.0
    probabilities = focal_mechanism.tectonic_type_probabilities(
        depth_below_slab, interface_like, normal_like
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert max(probabilities, key=probabilities.__getitem__) == expected
