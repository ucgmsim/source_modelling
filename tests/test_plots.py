from pathlib import Path
from typing import Callable

import diffimg
import pytest

from source_modelling.scripts import (
    plot_rakes,
    plot_rise,
    plot_srf,
    plot_srf_cumulative_moment,
    plot_srf_moment,
)

PLOT_IMAGE_DIRECTORY = Path("wiki/images")
SRF_FFP = Path(__file__).parent / "srfs" / "rupture_1.srf"


@pytest.mark.parametrize(
    "plot_function, expected_image_name",
    [
        (plot_srf.plot_srf, "srf_plot_example.png"),
        (plot_srf_moment.plot_srf_moment, "srf_moment_rate_example.png"),
        (
            plot_srf_cumulative_moment.plot_srf_cumulative_moment,
            "srf_cumulative_moment_rate_example.png",
        ),
        (plot_rise.plot_rise, "rise_example.png"),
        (plot_rakes.plot_rakes, "rakes_example.png"),
    ],
)
def test_plot_functions(
    tmp_path: Path, plot_function: Callable, expected_image_name: str
):
    """Check that the plotting scripts produce the wiki images within the expected tolerance."""
    output_image_path = tmp_path / "output.png"

    # plot-rakes expects a seed parameter that controls the distribution of rake vectors.
    # We set this seed to 1 to match the seed in the output image.
    if plot_function == plot_rakes.plot_rakes:
        plot_function(SRF_FFP, output_image_path, seed=1)
    else:
        plot_function(SRF_FFP, output_image_path)

    original = PLOT_IMAGE_DIRECTORY / expected_image_name
    generated = output_image_path

    diff = diffimg.diff(original, generated)
    assert diff <= 0.05
