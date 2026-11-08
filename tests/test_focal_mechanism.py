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


def test_centroid_with_nan_depth_uses_default_depth(
    classifier: focal_mechanism.CMTClassifier,
):
    nan_centroid = [*CENTROID[:2], np.nan]
    default_centroid = [*CENTROID[:2], focal_mechanism.DEFAULT_CENTROID_DEPTH_KM]
    with pytest.warns(UserWarning, match="without depth"):
        probabilities = classifier.tectonic_type_probabilities(
            nan_centroid, PLANE_1, PLANE_2
        )
    assert probabilities == classifier.tectonic_type_probabilities(
        default_centroid, PLANE_1, PLANE_2
    )
    with pytest.warns(UserWarning, match="without depth"):
        tectonic_type = classifier.tectonic_type(nan_centroid, PLANE_1, PLANE_2)
    assert tectonic_type == classifier.tectonic_type(default_centroid, PLANE_1, PLANE_2)
    with pytest.warns(UserWarning, match="without depth"):
        plane = classifier.most_likely_nodal_plane(nan_centroid, PLANE_1, PLANE_2)
    assert plane == classifier.most_likely_nodal_plane(
        default_centroid, PLANE_1, PLANE_2
    )


@pytest.mark.parametrize(
    "centroid",
    [[np.nan, 166.83, 22.0], [-45.1929, np.inf, 22.0], [np.nan, 166.83]],
)
def test_centroid_with_non_finite_lat_lon_raises(
    classifier: focal_mechanism.CMTClassifier, centroid: list[float]
):
    with pytest.raises(ValueError, match="latitude and longitude must be finite"):
        classifier.tectonic_type(centroid, PLANE_1, PLANE_2)


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
    planes = [
        NodalPlane(strike, dip, rake)
        for strike, dip, rake in rng.uniform(
            [0, 0, -180], [360, 90, 180], size=(500, 3)
        ).tolist()
    ]
    angles = [
        focal_mechanism.kagan_angle(a, b) for a, b in zip(planes[::2], planes[1::2])
    ]
    assert 0.0 <= min(angles) and max(angles) <= 120.0


@pytest.mark.parametrize(
    "depth, slab_depth, kagan, moho_depth, expected",
    [
        # Outside the slab footprint.
        (10.0, float("nan"), None, np.inf, (1.0, 0.0, 0.0)),
        # On the interface, above the seismogenic limit.
        (20.0, 20.0, None, np.inf, (0.0, 1.0, 0.0)),
        (20.0, 20.0, 41.5, np.inf, (0.1875, 0.625, 0.1875)),
        (20.0, 20.0, 90.0, np.inf, (0.375, 0.25, 0.375)),
        # Halfway down the ramp below the seismogenic zone.
        (37.5, 30.0, None, np.inf, (0.15625, 0.5, 0.34375)),
        # Above and below the flex zone.
        (5.0, 30.0, None, np.inf, (1.0, 0.0, 0.0)),
        (60.0, 30.0, None, np.inf, (0.0, 0.0, 1.0)),
        # In the mantle wedge, between the Moho and the top of the flex zone.
        (10.0, 30.0, None, 5.0, (0.64, 0.2, 0.16)),
        # Below the seismogenic zone and below a Moho that is below the slab.
        (48.0, 40.0, None, np.inf, (0.3, 0.0, 0.7)),
        (48.0, 40.0, None, 45.0, (0.0, 0.0, 1.0)),
    ],
)
def test_tectonic_type_probabilities(
    depth: float,
    slab_depth: float,
    kagan: float | None,
    moho_depth: float,
    expected: tuple[float, float, float],
):
    probabilities = focal_mechanism.tectonic_type_probabilities(
        depth, slab_depth, 30.0, kagan, moho_depth
    )
    assert [
        probabilities[TectonicType.CRUSTAL],
        probabilities[TectonicType.INTERFACE],
        probabilities[TectonicType.SLAB],
    ] == pytest.approx(expected)


def test_seismogenic_zone(classifier: focal_mechanism.CMTClassifier):
    slab_model = classifier.slab_model
    assert slab_model.query(-41.3, 174.8).seismogenic_zone.depth == 47.0  # Hikurangi
    assert slab_model.query(*CENTROID[:2]).seismogenic_zone.depth == 30.0  # Puysegur


def test_moho_model():
    moho_model = focal_mechanism.MohoModel.load()
    wellington = moho_model.depth(-41.3, 174.8)
    assert 15 < wellington < 45
    assert moho_model.depth(-37.0, 179.5 - 360) == moho_model.depth(-37.0, 179.5)
    assert moho_model.depth(0.0, 0.0) == np.inf


def test_tectonic_type(classifier: focal_mechanism.CMTClassifier):
    # The 2003 Fiordland earthquake ruptured the Puysegur interface, and the
    # result does not depend on the order of the planes.
    assert classifier.tectonic_type(CENTROID, PLANE_1, PLANE_2) == (
        TectonicType.INTERFACE
    )
    assert classifier.tectonic_type_probabilities(
        CENTROID, PLANE_1, PLANE_2
    ) == classifier.tectonic_type_probabilities(CENTROID, PLANE_2, PLANE_1)

    # Thrusting on the Hikurangi interface beneath Wellington.
    slab = classifier.slab_model.query(-41.3, 174.8)
    on_interface = np.array([-41.3, 174.8, slab.depth])
    thrust = NodalPlane(slab.strike, slab.dip, 95.0)
    assert classifier.tectonic_type(on_interface, thrust) == TectonicType.INTERFACE
    # Strike-slip near the interface is crustal above it and intraslab below.
    strike_slip = NodalPlane(40.0, 85.0, 175.0)
    above = on_interface - [0, 0, 10.0]
    below = on_interface + [0, 0, 10.0]
    assert classifier.tectonic_type(above, strike_slip) == TectonicType.CRUSTAL
    assert classifier.tectonic_type(below, strike_slip) == TectonicType.SLAB
    # Without a mechanism, location alone decides.
    assert classifier.tectonic_type(on_interface) == TectonicType.INTERFACE
    # Outside the slab footprint (Christchurch).
    assert (
        classifier.tectonic_type([-43.5, 172.6, 10.0], thrust) == TectonicType.CRUSTAL
    )


@pytest.mark.parametrize(
    "a, b, period, expected",
    [
        (350.0, 10.0, 360.0, 20.0),
        (10.0, 350.0, 360.0, 20.0),
        (0.0, 180.0, 360.0, 180.0),
        (720.0, 0.0, 360.0, 0.0),
        (170.0, 10.0, 180.0, 20.0),
        (90.0, 270.0, 180.0, 0.0),
    ],
)
def test_angular_difference(a: float, b: float, period: float, expected: float):
    assert focal_mechanism._angular_difference(a, b, period) == pytest.approx(expected)


def test_angular_difference_default_period_is_360():
    assert focal_mechanism._angular_difference(350.0, 10.0) == pytest.approx(20.0)
    assert focal_mechanism._angular_difference(
        [0.0, 359.0, 180.0], 1.0
    ) == pytest.approx([1.0, 2.0, 179.0])


@pytest.mark.parametrize(
    "strike, dip, rake",
    [(30.0, 40.0, 60.0), (120.0, 70.0, -45.0), (250.0, 20.0, 135.0)],
)
def test_slip_vector_lies_in_plane_along_rake(strike: float, dip: float, rake: float):
    phi, delta, lam = np.radians([strike, dip, rake])
    along_strike = np.array([np.cos(phi), np.sin(phi), 0.0])
    up_dip = np.array(
        [np.sin(phi) * np.cos(delta), -np.cos(phi) * np.cos(delta), -np.sin(delta)]
    )
    slip = focal_mechanism._slip_vector(strike, dip, rake)
    assert slip == pytest.approx(np.cos(lam) * along_strike + np.sin(lam) * up_dip)
    assert np.linalg.norm(slip) == pytest.approx(1.0)
    assert slip @ focal_mechanism._plane_normal(strike, dip) == pytest.approx(
        0.0, abs=1e-12
    )


def test_kagan_angle_is_invariant_to_strike_rotation():
    # Rotating both mechanisms about the vertical leaves the Kagan angle unchanged.
    plane_a = NodalPlane(0.0, 40.0, 60.0)
    plane_b = NodalPlane(0.0, 40.0, 100.0)
    expected = focal_mechanism.kagan_angle(plane_a, plane_b)
    assert expected == pytest.approx(40.0, abs=1e-4)
    for rotation in [37.0, 123.0, 211.0]:
        rotated_a = NodalPlane(rotation, plane_a.dip, plane_a.rake)
        rotated_b = NodalPlane(rotation, plane_b.dip, plane_b.rake)
        assert focal_mechanism.kagan_angle(rotated_a, rotated_b) == pytest.approx(
            expected, abs=1e-4
        )


def _synthetic_slab_grids() -> dict[str, np.ndarray]:
    lats = np.array([-42.0, -41.0])
    return {
        # A region in 0..360 longitudes whose strike wraps through north.
        "aaa_lat": lats,
        "aaa_lon": np.array([181.0, 182.0]),
        "aaa_depth": np.array([[10.0, 20.0], [30.0, 40.0]]),
        "aaa_dip": np.array([[10.0, 10.0], [30.0, 30.0]]),
        "aaa_strike": np.array([[350.0, 10.0], [350.0, 10.0]]),
        # A region in -180..180 longitudes.
        "bbb_lat": lats,
        "bbb_lon": np.array([170.0, 171.0]),
        "bbb_depth": np.full((2, 2), 50.0),
        "bbb_dip": np.full((2, 2), 45.0),
        "bbb_strike": np.full((2, 2), 90.0),
    }


def test_slab_model_from_grids():
    zone = focal_mechanism.SeismogenicZone(25.0, 80.0)
    slab_model = focal_mechanism.SlabModel(_synthetic_slab_grids(), {"aaa": zone})

    # Bilinear interpolation at the cell centre; strike averages through north.
    centre = slab_model.query(-41.5, 181.5)
    assert centre.present
    assert centre.depth == pytest.approx(25.0)
    assert centre.dip == pytest.approx(20.0)
    assert min(centre.strike, 360.0 - centre.strike) == pytest.approx(0.0, abs=1e-9)
    assert centre.seismogenic_zone == zone
    # Corners reproduce the grid values, in either longitude convention.
    corner = slab_model.query(-42.0, 182.0)
    assert (corner.depth, corner.dip, corner.strike) == pytest.approx(
        (20.0, 10.0, 10.0)
    )
    assert slab_model.query(-42.0, 182.0 - 360.0) == corner

    # A region without a seismogenic zone takes the default.
    other = slab_model.query(-41.5, 170.5)
    assert (other.depth, other.dip, other.strike) == pytest.approx((50.0, 45.0, 90.0))
    assert other.seismogenic_zone == focal_mechanism.DEFAULT_SEISMOGENIC_ZONE
    assert slab_model.query(-41.5, 170.5 + 360.0) == other

    # Outside both regions.
    outside = slab_model.query(-30.0, 175.0)
    assert not outside.present
    assert np.isnan(outside.dip) and np.isnan(outside.strike)


@pytest.mark.parametrize("centroid", [[-45.0], [-45.0, 166.0, 10.0, 1.0], []])
def test_centroid_with_wrong_size_raises(
    classifier: focal_mechanism.CMTClassifier, centroid: list[float]
):
    with pytest.raises(ValueError, match=r"\(lat, lon\) or \(lat, lon, depth_km\)"):
        classifier.tectonic_type(centroid, PLANE_1, PLANE_2)


def test_from_faults_rejects_mismatched_features(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        focal_mechanism,
        "NODAL_PLANE_FEATURE_NAMES",
        focal_mechanism.NODAL_PLANE_FEATURE_NAMES[::-1],
    )
    with pytest.raises(ValueError, match="does not match the features"):
        focal_mechanism.CMTClassifier.from_faults([])
