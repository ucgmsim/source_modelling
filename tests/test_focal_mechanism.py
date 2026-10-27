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


def test_deeper_centroid_is_less_confident(
    classifier: focal_mechanism.CMTClassifier,
):
    depths = [0.0, 50.0, 200.0]
    temperature = focal_mechanism.depth_temperature(
        depths, classifier.temperature_scale, classifier.temperature_depth_rate
    )
    assert np.all(np.diff(temperature) < 0)
    untempered = focal_mechanism.CMTClassifier(
        classifier.segments, classifier.slab_model, classifier.weights
    )
    for depth, expected in zip(depths, temperature):
        centroid = [*CENTROID[:2], depth]
        log_odds = sum(
            classifier.nodal_plane_contributions(centroid, PLANE_1, PLANE_2).values()
        )
        untempered_log_odds = sum(
            untempered.nodal_plane_contributions(centroid, PLANE_1, PLANE_2).values()
        )
        assert log_odds == pytest.approx(expected * untempered_log_odds)


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


@pytest.mark.parametrize(
    "plane_a, plane_b, expected",
    [
        (PLANE_1, PLANE_1, 0.0),
        (NodalPlane(0, 45, 90), NodalPlane(180, 45, 90), 0.0),  # Auxiliary plane
        (NodalPlane(0, 45, 90), NodalPlane(0, 45, -90), 90.0),  # Thrust vs normal
        (NodalPlane(0, 45, 90), NodalPlane(0, 45, 60), 30.0),  # Rake rotation
        (NodalPlane(0, 90, 0), NodalPlane(90, 90, 0), 90.0),  # Strike rotation
    ],
)
def test_kagan_angle(plane_a: NodalPlane, plane_b: NodalPlane, expected: float):
    assert focal_mechanism.kagan_angle(plane_a, plane_b) == pytest.approx(
        expected, abs=1e-4
    )
    assert focal_mechanism.kagan_angle(plane_b, plane_a) == pytest.approx(
        expected, abs=1e-4
    )


def test_kagan_angle_is_at_most_120():
    rng = np.random.default_rng(0)
    planes = rng.uniform([0, 0, -180], [360, 90, 180], size=(500, 3))
    angles = [
        focal_mechanism.kagan_angle(NodalPlane(*a), NodalPlane(*b))
        for a, b in zip(planes[::2], planes[1::2])
    ]
    assert 0.0 <= min(angles) and max(angles) <= 120.0


def test_plate_motion_azimuth():
    # Pacific moves west relative to Australia at the Hikurangi margin.
    assert 250 < focal_mechanism.plate_motion_azimuth(-38.7, 178.5) < 280
    assert 230 < focal_mechanism.plate_motion_azimuth(-47.0, 165.5) < 250


def test_subduction_slip_planes():
    slab_model = focal_mechanism.SlabModel.load()
    for lat, lon in [(-41.3, 174.8), (-38.7, 177.5), (-46.0, 166.0)]:
        slab = slab_model.query(lat, lon)
        pure, oblique = focal_mechanism.subduction_slip_planes(lat, lon, slab)
        assert pure == NodalPlane(slab.strike, slab.dip, 90.0)
        assert (oblique.strike, oblique.dip) == (slab.strike, slab.dip)
        assert 0 < oblique.rake < 180  # Reverse sense
        # The horizontal slip is parallel to the plate motion.
        slip = focal_mechanism._slip_vector(*oblique)
        slip_azimuth = np.degrees(np.arctan2(slip[1], slip[0]))
        difference = focal_mechanism._angular_difference(
            slip_azimuth, focal_mechanism.plate_motion_azimuth(lat, lon), 180.0
        )
        assert difference == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "depth_above_slab, depth, kagan, fixed_depth, expected",
    [
        (float("nan"), 10.0, 0.0, False, {TectonicType.CRUSTAL: 1.0}),
        (0.0, 20.0, 0.0, False, {TectonicType.INTERFACE: 1.0}),
        (0.0, 20.0, None, False, {TectonicType.INTERFACE: 1.0}),
        (-1.0, 80.0, 0.0, False, {TectonicType.SLAB: 1.0}),  # Too deep
        (
            10.0,
            20.0,
            30.0,
            False,
            {
                TectonicType.CRUSTAL: 1.0 - np.exp(-1.0),
                TectonicType.INTERFACE: np.exp(-1.0),
            },
        ),
        (
            -10.0,
            40.0,
            None,
            False,
            {
                TectonicType.SLAB: 1.0 - np.exp(-0.5),
                TectonicType.INTERFACE: np.exp(-0.5),
            },
        ),
        # A fixed depth says nothing about the distance to the interface.
        (-30.0, 50.0, 0.0, True, {TectonicType.INTERFACE: 1.0}),
        (-30.0, 50.0, 90.0, True, {TectonicType.SLAB: 1.0 - np.exp(-4.5)}),
    ],
)
def test_tectonic_type_probabilities(
    depth_above_slab: float,
    depth: float,
    kagan: float | None,
    fixed_depth: bool,
    expected: dict[TectonicType, float],
):
    probabilities = focal_mechanism.tectonic_type_probabilities(
        depth_above_slab, depth, kagan, fixed_depth
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)
    for tectonic_type, probability in expected.items():
        assert probabilities[tectonic_type] == pytest.approx(probability)


def test_tectonic_type(classifier: focal_mechanism.CMTClassifier):
    # Thrust on the Hikurangi interface beneath Wellington.
    slab = classifier.slab_model.query(-41.3, 174.8)
    on_interface = np.array([-41.3, 174.8, slab.depth])
    thrust = NodalPlane(slab.strike, slab.dip, 100.0)
    assert classifier.tectonic_type(on_interface, thrust) == (TectonicType.INTERFACE)
    # Either nodal plane gives the same probabilities.
    np.testing.assert_allclose(
        list(classifier.tectonic_type_probabilities(on_interface, PLANE_1).values()),
        list(classifier.tectonic_type_probabilities(on_interface, PLANE_2).values()),
        atol=0.01,
    )
    # Strike-slip near the interface is not interface slip.
    strike_slip = NodalPlane(40.0, 85.0, 175.0)
    shallower = on_interface - [0, 0, 2.0]
    assert classifier.tectonic_type(shallower, strike_slip) == (TectonicType.CRUSTAL)
    deeper = on_interface + [0, 0, 2.0]
    assert classifier.tectonic_type(deeper, strike_slip) == TectonicType.SLAB
    # Outside the slab footprint (Christchurch).
    assert (
        classifier.tectonic_type([-43.5, 172.6, 10.0], thrust) == TectonicType.CRUSTAL
    )


def test_tectonic_type_with_both_planes(classifier: focal_mechanism.CMTClassifier):
    # The call gcmt-to-realisation makes, which must not depend on plane order.
    forward = classifier.tectonic_type_probabilities(CENTROID, PLANE_1, PLANE_2)
    backward = classifier.tectonic_type_probabilities(CENTROID, PLANE_2, PLANE_1)
    assert forward == backward
    assert sum(forward.values()) == pytest.approx(1.0)
    # 9 km below the Slab2 surface, so mostly intraslab.
    assert classifier.tectonic_type(CENTROID, PLANE_1, PLANE_2) == TectonicType.SLAB


def test_tectonic_type_without_mechanism_or_depth(
    classifier: focal_mechanism.CMTClassifier,
):
    slab = classifier.slab_model.query(-41.3, 174.8)
    on_interface = np.array([-41.3, 174.8, slab.depth])
    assert classifier.tectonic_type(on_interface) == TectonicType.INTERFACE
    # A fixed depth leaves the interface probability to the mechanism.
    deep = on_interface + [0, 0, 30.0]
    assert classifier.tectonic_type(deep) == TectonicType.SLAB
    assert classifier.tectonic_type(deep, fixed_depth=True) == TectonicType.INTERFACE
    with pytest.warns(UserWarning, match="without depth"):
        probabilities = classifier.tectonic_type_probabilities(on_interface[:2])
    assert probabilities[TectonicType.INTERFACE] == 1.0
