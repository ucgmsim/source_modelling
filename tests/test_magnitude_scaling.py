import functools
import itertools
import random
from collections.abc import Callable
from typing import Any, Protocol
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
import scipy as sp
from hypothesis import assume, given

from source_modelling import magnitude_scaling

MAGNITUDE_TO_AREA = {
    magnitude_scaling.ScalingRelation.LEONARD2014: magnitude_scaling.leonard_magnitude_to_area,
    magnitude_scaling.ScalingRelation.CONTRERAS_INTERFACE2017: magnitude_scaling.contreras_interface_magnitude_to_area,
    magnitude_scaling.ScalingRelation.CONTRERAS_SLAB2020: magnitude_scaling.strasser_slab_magnitude_to_area,
}

AREA_TO_MAGNITUDE = {
    magnitude_scaling.ScalingRelation.LEONARD2014: magnitude_scaling.leonard_area_to_magnitude,
    magnitude_scaling.ScalingRelation.CONTRERAS_INTERFACE2017: magnitude_scaling.contreras_interface_area_to_magnitude,
    magnitude_scaling.ScalingRelation.CONTRERAS_SLAB2020: magnitude_scaling.strasser_slab_area_to_magnitude,
}


def seed(seed: int):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> Any:
            random.seed(seed)
            np.random.seed(seed)
            return func(*args, **kwargs)

        return wrapper

    return decorator


@pytest.mark.parametrize(
    "rake, expected",
    [
        (-30, magnitude_scaling.RakeType.STRIKE_SLIP),
        (0, magnitude_scaling.RakeType.STRIKE_SLIP),
        (30, magnitude_scaling.RakeType.STRIKE_SLIP),
        (150, magnitude_scaling.RakeType.STRIKE_SLIP),
        (210, magnitude_scaling.RakeType.STRIKE_SLIP),
        (60, magnitude_scaling.RakeType.REVERSE),
        (90, magnitude_scaling.RakeType.REVERSE),
        (120, magnitude_scaling.RakeType.REVERSE),
        (-120, magnitude_scaling.RakeType.NORMAL),
        (-90, magnitude_scaling.RakeType.NORMAL),
        (-60, magnitude_scaling.RakeType.NORMAL),
        (-149, magnitude_scaling.RakeType.NORMAL_OBLIQUE),
        (-121, magnitude_scaling.RakeType.NORMAL_OBLIQUE),
        (-59, magnitude_scaling.RakeType.NORMAL_OBLIQUE),
        (-31, magnitude_scaling.RakeType.NORMAL_OBLIQUE),
        (31, magnitude_scaling.RakeType.REVERSE_OBLIQUE),
        (59, magnitude_scaling.RakeType.REVERSE_OBLIQUE),
        (121, magnitude_scaling.RakeType.REVERSE_OBLIQUE),
        (149, magnitude_scaling.RakeType.REVERSE_OBLIQUE),
    ],
)
def test_rake_type(rake: float, expected: magnitude_scaling.RakeType):
    assert magnitude_scaling.rake_type(rake) == expected


def relation_with_magnitude(
    relations: list[magnitude_scaling.ScalingRelation] = list(MAGNITUDE_TO_AREA),
):
    @st.composite
    def sampler(
        draw: st.DrawFn,
    ) -> tuple[magnitude_scaling.ScalingRelation, float | None, float]:
        scaling_relation = draw(st.sampled_from(relations))
        min_magnitude, max_magnitude = magnitude_scaling.MAGNITUDE_BOUNDS[
            scaling_relation
        ]
        if scaling_relation == magnitude_scaling.ScalingRelation.LEONARD2014:
            rake = draw(st.floats(min_value=-180, max_value=180))
        else:
            rake = None

        magnitude = draw(st.floats(min_value=min_magnitude, max_value=max_magnitude))
        return scaling_relation, rake, magnitude

    return sampler()


# The contreras slab 2020 relation (which is the same as Strasser 2010) is not invertible. See the coefficients of Table 2 in the paper
# Strasser, Fleur O., M. C. Arango, and Julian J. Bommer.
# "Scaling of the source dimensions of interface and intraslab
# subduction-zone earthquakes with moment magnitude." Seismological
# Research Letters 81.6 (2010): 941-950.
# The coefficients are not invertible, so we cannot test the inversion of the area to magnitude function.
@given(
    relation_with_magnitude(
        [
            magnitude_scaling.ScalingRelation.LEONARD2014,
            magnitude_scaling.ScalingRelation.CONTRERAS_INTERFACE2017,
        ]
    )
)
def test_inversion(
    relation_with_magnitude: tuple[
        magnitude_scaling.ScalingRelation, float | None, float
    ],
):
    """When executed with best-fit values, the magnitude to area function is an inverse of the area to magnitude function."""
    scaling_relation, rake, magnitude = relation_with_magnitude
    if scaling_relation == magnitude_scaling.ScalingRelation.LEONARD2014:
        mag_to_area = functools.partial(MAGNITUDE_TO_AREA[scaling_relation], rake=rake)
        area_to_mag = functools.partial(AREA_TO_MAGNITUDE[scaling_relation], rake=rake)
    else:
        mag_to_area = MAGNITUDE_TO_AREA[scaling_relation]
        area_to_mag = AREA_TO_MAGNITUDE[scaling_relation]
    assert area_to_mag(mag_to_area(magnitude)) == pytest.approx(magnitude)


@pytest.mark.parametrize(
    "mw, rake, expected_area",
    [
        (4.0, -180.0, 1.0232929922807537),
        (4.0, -90.0, 1.0),
        (4.0, 0.0, 1.0232929922807537),
        (4.0, 90.0, 1.0),
        (4.0, 180.0, 1.0232929922807537),
        (5.25, -180.0, 18.197008586099827),
        (5.25, -90.0, 17.78279410038923),
        (5.25, 0.0, 18.197008586099827),
        (5.25, 90.0, 17.78279410038923),
        (5.25, 180.0, 18.197008586099827),
        (6.5, -180.0, 323.5936569296281),
        (6.5, -90.0, 316.22776601683796),
        (6.5, 0.0, 323.5936569296281),
        (6.5, 90.0, 316.22776601683796),
        (6.5, 180.0, 323.5936569296281),
        (7.75, -180.0, 5754.399373371566),
        (7.75, -90.0, 5623.413251903491),
        (7.75, 0.0, 5754.399373371566),
        (7.75, 90.0, 5623.413251903491),
        (7.75, 180.0, 5754.399373371566),
        (9.0, -180.0, 102329.29922807537),
        (9.0, -90.0, 100000.0),
        (9.0, 0.0, 102329.29922807537),
        (9.0, 90.0, 100000.0),
        (9.0, 180.0, 102329.29922807537),
    ],
)
def test_leonard_area(mw: float, rake: float, expected_area: float):
    """Test the Leonard 2014 area calculation against values from the old implementation in qcore.

    NOTE: this combined with the inversion test for leonard ensure that the area calculation is compatible with the old implementation as well."""
    assert magnitude_scaling.magnitude_to_area(
        magnitude_scaling.ScalingRelation.LEONARD2014, mw, rake
    ) == pytest.approx(expected_area)


@pytest.mark.parametrize(
    "mw, expected_area",
    [
        (6.0, 130.31667784522986),
        (6.036734693877551, 140.5056788423239),
        (6.073469387755102, 151.49132185819374),
        (6.110204081632653, 163.33589351998378),
        (6.146938775510204, 176.10655042633059),
        (6.183673469387755, 189.87569991324222),
        (6.220408163265306, 204.72141059071683),
        (6.257142857142857, 220.72785497777087),
        (6.293877551020408, 237.98578674553698),
        (6.330612244897959, 256.5930552743158),
        (6.36734693877551, 276.6551604420244),
        (6.404081632653061, 298.2858507896005),
        (6.440816326530612, 321.6077684548422),
        (6.477551020408163, 346.7531445313512),
        (6.514285714285714, 373.86454879513593),
        (6.551020408163265, 403.0956980496915),
        (6.587755102040816, 434.6123276727276),
        (6.624489795918367, 468.5931313060583),
        (6.661224489795918, 505.23077401652745),
        (6.697959183673469, 544.7329846724095),
        (6.73469387755102, 587.3237337288654),
        (6.771428571428571, 633.244503100285),
        (6.808163265306122, 682.7556553194657),
        (6.844897959183673, 736.1379097465132),
        (6.881632653061224, 793.6939341973044),
        (6.918367346938775, 855.7500610157631),
        (6.955102040816326, 922.6581373197666),
        (6.9918367346938775, 994.7975199112492),
        (7.0285714285714285, 1072.577226161284),
        (7.0653061224489795, 1156.4382530652758),
        (7.1020408163265305, 1246.8560776168943),
        (7.1387755102040815, 1344.3433526774159),
        (7.1755102040816325, 1449.452813625574),
        (7.2122448979591836, 1562.7804122681011),
        (7.248979591836735, 1684.9686957796698),
        (7.285714285714286, 1816.7104498302217),
        (7.322448979591837, 1958.7526265555612),
        (7.359183673469388, 2111.9005796421025),
        (7.395918367346939, 2277.0226305379424),
        (7.43265306122449, 2455.054991679859),
        (7.469387755102041, 2647.0070746500464),
        (7.506122448979592, 2853.9672133588942),
        (7.542857142857143, 3077.1088347031946),
        (7.579591836734694, 3317.697111686388),
        (7.616326530612245, 3577.0961367227364),
        (7.653061224489796, 3856.7766557968594),
        (7.689795918367347, 4158.324407329888),
        (7.726530612244898, 4483.449113032134),
        (7.763265306122449, 4833.994171718779),
        (7.8, 5211.947111050806),
    ],
)
def test_strasser_slab_expected_area(mw: float, expected_area: float):
    """Test the Strasser 2010 slab area calculation against values from the old implementation in qcore."""
    assert magnitude_scaling.strasser_slab_magnitude_to_area(mw) == pytest.approx(
        expected_area
    )


@given(st.floats(min_value=5.9, max_value=7.7))
def test_strasser_monotonicity(mag1: float):
    mag2 = mag1 + 0.1  # Slightly higher magnitude
    assert magnitude_scaling.strasser_slab_magnitude_to_area(
        mag2
    ) > magnitude_scaling.strasser_slab_magnitude_to_area(mag1)


@given(relation_with_magnitude())
def test_monotonicity_mag_to_area(
    relation_with_magnitude: tuple[
        magnitude_scaling.ScalingRelation, float | None, float
    ],
):
    """Test that the area is monotonic with respect to the magnitude."""

    scaling_relation, rake, magnitude = relation_with_magnitude
    assume(
        scaling_relation != magnitude_scaling.ScalingRelation.CONTRERAS_SLAB2020
        or magnitude <= 7.7
    )
    if scaling_relation == magnitude_scaling.ScalingRelation.LEONARD2014:
        mag_to_area = functools.partial(MAGNITUDE_TO_AREA[scaling_relation], rake=rake)
    else:
        mag_to_area = MAGNITUDE_TO_AREA[scaling_relation]
    assert mag_to_area(magnitude + 0.1) > mag_to_area(magnitude)


class RandomFunction(Protocol):
    __name__: str

    def __call__(
        self,
        x: magnitude_scaling.Array,
        random: bool = False,
        rake: float | None = None,
    ) -> float: ...


@pytest.mark.parametrize(
    "area_to_mag, area",
    list(
        itertools.product(
            [magnitude_scaling.contreras_interface_area_to_magnitude],
            np.linspace(
                magnitude_scaling.contreras_interface_magnitude_to_area(6.0),
                magnitude_scaling.contreras_interface_magnitude_to_area(9.0),
                10,
            ),
        )
    ),
)
@seed(1)
def test_normal_error_contreras_interface(area_to_mag: RandomFunction, area: float):
    """Generate samples with random = True set on area_to_mag and check that it approximates the value with random = False."""
    samples = area_to_mag(np.full(300, area), random=True)
    result = sp.stats.goodness_of_fit(
        sp.stats.norm,
        samples,
        statistic="ad",
        known_params=dict(loc=area_to_mag(area), scale=0.73 / np.log(10)),
    )
    assert result.pvalue > 0.05


@pytest.mark.parametrize(
    "aspect_ratio, magnitude",
    list(
        itertools.product(
            [magnitude_scaling.contreras_interface_magnitude_to_aspect_ratio],
            np.linspace(
                6.0,
                9.0,
                10,
            ),
        )
    ),
)
@seed(1)
def test_normal_error_contreras_interface_aspect_ratio(
    aspect_ratio: RandomFunction, magnitude: float
):
    """Generate samples with random = True set on area_to_mag and check that it approximates the value with random = False."""
    samples = aspect_ratio(np.full(100, magnitude), random=True)
    sigma = 0.32 if magnitude < 7.25 else 0.47
    result = sp.stats.goodness_of_fit(
        sp.stats.norm,
        np.log(samples),
        statistic="ad",
        known_params=dict(loc=np.log(aspect_ratio(magnitude)), scale=sigma),
    )
    assert result.pvalue > 0.05


# NOTE: Because magnitude is normally distributed, the area is log-normally distributed. We test for normality of the log of the area.


@pytest.mark.parametrize(
    "mag_to_area, magnitude",
    itertools.product(
        [magnitude_scaling.contreras_interface_magnitude_to_area],
        np.linspace(
            6.0,
            9.0,
            5,
        ),
    ),
)
@seed(1)
def test_normal_error_contreras_interface_mag_to_area(
    mag_to_area: RandomFunction, magnitude: float
):
    """Generate samples with random = True set on mag_to_area and check that it approximates the value with random = False."""
    samples = mag_to_area(np.full(100, magnitude), random=True)

    result = sp.stats.goodness_of_fit(
        sp.stats.lognorm,
        samples,
        statistic="ks",
    )
    assert result.pvalue > 0.05


@pytest.mark.parametrize(
    "area_to_mag, area",
    list(
        itertools.product(
            [magnitude_scaling.strasser_slab_area_to_magnitude],
            np.linspace(
                130,
                5212,
                10,
            ),
        )
    ),
)
@seed(1)
def test_normal_error_strasser_slab(area_to_mag: RandomFunction, area: float):
    """Generate samples with random = True set on area_to_mag and check that it approximates the value with random = False."""
    samples = area_to_mag(np.full(100, area), random=True)
    result = sp.stats.goodness_of_fit(
        sp.stats.norm,
        samples,
        statistic="ad",
        known_params=dict(loc=area_to_mag(area)),
    )
    assert result.pvalue > 0.05


@pytest.mark.parametrize(
    "mag_to_area, magnitude",
    itertools.product(
        [magnitude_scaling.strasser_slab_magnitude_to_area],
        np.linspace(
            5.9,
            7.8,
            5,
        ),
    ),
)
@seed(1)
def test_normal_error_strasser_slab_mag_to_area(
    mag_to_area: RandomFunction, magnitude: float
):
    """Generate samples with random = True set on mag_to_area and check that it approximates the value with random = False."""
    samples = mag_to_area(np.full(100, magnitude), random=True)
    result = sp.stats.goodness_of_fit(
        sp.stats.lognorm,
        samples,
        statistic="ks",
    )
    assert result.pvalue > 0.05


@pytest.mark.parametrize(
    "aspect_ratio, magnitude",
    list(
        itertools.product(
            [magnitude_scaling.contreras_slab_magnitude_to_aspect_ratio],
            np.linspace(
                6.0,
                9.0,
                10,
            ),
        )
    ),
)
@seed(1)
def test_normal_error_contreras_slab_aspect_ratio(
    aspect_ratio: RandomFunction, magnitude: float
):
    """Generate samples with random = True set on area_to_mag and check that it approximates the value with random = False."""
    samples = aspect_ratio(np.full(100, magnitude), random=True)
    sigma = 0.24 if magnitude < 6.5 else 0.38
    result = sp.stats.goodness_of_fit(
        sp.stats.norm,
        np.log(samples),
        statistic="ad",
        known_params=dict(loc=np.log(aspect_ratio(magnitude)), scale=sigma),
    )
    assert result.pvalue > 0.05


@pytest.mark.parametrize(
    "area_to_mag, rake, area",
    list(
        itertools.product(
            [magnitude_scaling.leonard_area_to_magnitude],
            np.linspace(-180, 180, 10),
            np.linspace(
                10,
                10**5,
                10,
            ),
        )
    ),
)
@seed(1)
def test_normal_error_leonard(area_to_mag: RandomFunction, rake: float, area: float):
    """Generate samples with random = True set on area_to_mag and check that it approximates the value with random = False."""
    samples = area_to_mag(np.full(100, area), rake=rake, random=True)
    result = sp.stats.goodness_of_fit(
        sp.stats.norm,
        samples,
        statistic="ad",
        # NOTE: We cannot assume a mean and standard deviation because these are assymetric values. So we just test for normality.
    )
    assert result.pvalue > 0.05


@pytest.mark.parametrize(
    "relations, magnitude",
    itertools.product(
        [
            (
                magnitude_scaling.contreras_slab_magnitude_to_length_width,
                magnitude_scaling.strasser_slab_magnitude_to_area,
            ),
            (
                magnitude_scaling.contreras_interface_magnitude_to_length_width,
                magnitude_scaling.contreras_interface_magnitude_to_area,
            ),
        ],
        np.linspace(6, 7.8, 10),
    ),
)
def test_area_preservation_lw(
    relations: tuple[Callable[[float], tuple[float, float]], RandomFunction],
    magnitude: float,
):
    """Test that the area is preserved when converting between area and length/width."""
    magnitude_to_lw, magnitude_to_area = relations
    area = magnitude_to_area(
        magnitude,
    )
    length, width = magnitude_to_lw(
        magnitude,
    )
    assert length * width == pytest.approx(area)


@pytest.mark.parametrize(
    "rake, magnitude",
    itertools.product(
        np.linspace(-180, 180, 10),
        np.linspace(5.5, 7, 10),
    ),
)
def test_area_preservation_lw_leonard(
    rake: float,
    magnitude: float,
):
    """Test that the area is preserved when converting between area and length/width."""
    area = magnitude_scaling.leonard_magnitude_to_area(magnitude, rake)
    length, width = magnitude_scaling.leonard_magnitude_to_length_width(
        magnitude,
        rake,
    )
    assert length * width == pytest.approx(area)


@given(
    area=st.floats(1, 10**5),
    aspect_ratio=st.floats(0.1, 10),
)
def test_length_width(area: float, aspect_ratio: float):
    """Test that the length and width are correct (Sanity check!)."""
    length, width = magnitude_scaling.area_aspect_ratio_to_length_width(
        area, aspect_ratio
    )
    assert length * width == pytest.approx(area)
    assert length / width == pytest.approx(aspect_ratio)


@pytest.mark.parametrize(
    "scaling_relation, func_name, rake_required",
    [
        (
            magnitude_scaling.ScalingRelation.LEONARD2014,
            "leonard_magnitude_to_length_width",
            True,
        ),
        (
            magnitude_scaling.ScalingRelation.CONTRERAS_INTERFACE2017,
            "contreras_interface_magnitude_to_length_width",
            False,
        ),
        (
            magnitude_scaling.ScalingRelation.CONTRERAS_SLAB2020,
            "contreras_slab_magnitude_to_length_width",
            False,
        ),
    ],
)
def test_magnitude_to_length_width_calls_correct_function(
    scaling_relation: magnitude_scaling.ScalingRelation,
    func_name: str,
    rake_required: bool,
):
    magnitude = 7.0
    rake = 90.0 if rake_required else None
    random = True

    with patch(f"source_modelling.magnitude_scaling.{func_name}") as mock_func:
        magnitude_scaling.magnitude_to_length_width(
            scaling_relation, magnitude, rake, random
        )
        if rake_required:
            mock_func.assert_called_once_with(magnitude, rake=rake, random=random)
        else:
            mock_func.assert_called_once_with(magnitude, random=random)


@pytest.mark.parametrize(
    "scaling_relation, func_name, rake_required",
    [
        (
            magnitude_scaling.ScalingRelation.LEONARD2014,
            "leonard_magnitude_to_area",
            True,
        ),
        (
            magnitude_scaling.ScalingRelation.CONTRERAS_INTERFACE2017,
            "contreras_interface_magnitude_to_area",
            False,
        ),
        (
            magnitude_scaling.ScalingRelation.CONTRERAS_SLAB2020,
            "strasser_slab_magnitude_to_area",
            False,
        ),
    ],
)
def test_magnitude_to_area_calls_correct_function(
    scaling_relation: magnitude_scaling.ScalingRelation,
    func_name: str,
    rake_required: bool,
):
    magnitude = 7.0
    rake = 90.0 if rake_required else None
    random = True

    with patch(f"source_modelling.magnitude_scaling.{func_name}") as mock_func:
        magnitude_scaling.magnitude_to_area(scaling_relation, magnitude, rake, random)
        if rake_required:
            mock_func.assert_called_once_with(magnitude, rake=rake, random=random)
        else:
            mock_func.assert_called_once_with(magnitude, random=random)


@pytest.mark.parametrize(
    "scaling_relation, func_name, rake_required",
    [
        (
            magnitude_scaling.ScalingRelation.LEONARD2014,
            "leonard_area_to_magnitude",
            True,
        ),
        (
            magnitude_scaling.ScalingRelation.CONTRERAS_INTERFACE2017,
            "contreras_interface_area_to_magnitude",
            False,
        ),
        (
            magnitude_scaling.ScalingRelation.CONTRERAS_SLAB2020,
            "strasser_slab_area_to_magnitude",
            False,
        ),
    ],
)
def test_area_to_magnitude_calls_correct_function(
    scaling_relation: magnitude_scaling.ScalingRelation,
    func_name: str,
    rake_required: bool,
):
    area = 1000.0  # Example fault area in km^2
    rake = 90.0 if rake_required else None
    random = True

    with patch(f"source_modelling.magnitude_scaling.{func_name}") as mock_func:
        magnitude_scaling.area_to_magnitude(scaling_relation, area, rake, random)
        if rake_required:
            mock_func.assert_called_once_with(area, rake=rake, random=random)
        else:
            mock_func.assert_called_once_with(area, random=random)


RETURN_TYPE_FUNCS = [
    magnitude_scaling.leonard_area_to_magnitude,
    magnitude_scaling.leonard_magnitude_to_area,
    magnitude_scaling.leonard_magnitude_to_length,
    magnitude_scaling.leonard_magnitude_to_width,
    magnitude_scaling.contreras_interface_area_to_magnitude,
    magnitude_scaling.contreras_interface_magnitude_to_area,
    magnitude_scaling.contreras_interface_magnitude_to_aspect_ratio,
    magnitude_scaling.strasser_slab_area_to_magnitude,
    magnitude_scaling.strasser_slab_magnitude_to_area,
    magnitude_scaling.contreras_slab_magnitude_to_aspect_ratio,
    magnitude_scaling.leonard_magnitude_to_length_width,
    magnitude_scaling.contreras_interface_magnitude_to_length_width,
    magnitude_scaling.contreras_slab_magnitude_to_length_width,
]


@pytest.mark.parametrize("func", RETURN_TYPE_FUNCS)
@pytest.mark.parametrize("random", [True, False], ids=["randomised", "mean"])
@pytest.mark.parametrize(
    "input_data, expected_type",
    [
        (7.0, float),
        (np.array([6.5, 7.5]), np.ndarray),
        (pd.Series([6.5, 7.5]), pd.Series),
    ],
    ids=["float", "numpy array", "pandas series"],
)
def test_scaling_output_types(
    func: RandomFunction,
    random: bool,
    input_data: tuple[magnitude_scaling.Array, type],
    expected_type: type,
) -> None:
    """Checks all single-output functions for type preservation."""
    # Leonard functions require 'rake', others do not.
    kwargs = {"random": random}
    if "leonard" in func.__name__:
        kwargs["rake"] = 0.0

    result = func(input_data, **kwargs)
    if isinstance(result, tuple):
        a, b = result
        assert isinstance(a, expected_type) and isinstance(b, expected_type)
    else:
        assert isinstance(result, expected_type)
