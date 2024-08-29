import hashlib
import tempfile
from pathlib import Path

from source_modelling.scripts import (
    plot_rakes,
    plot_rise,
    plot_srf,
    plot_srf_cumulative_moment,
    plot_srf_moment,
)


def md5sum(file: Path) -> str:
    with open(file, "rb") as file_handle:
        return hashlib.file_digest(file_handle, "md5").hexdigest()


PLOT_IMAGE_DIRECTORY = Path("wiki/images")
SRF_FFP = Path(__file__).parent / "srfs" / "rupture_1.srf"


def test_plot_srf():
    """Check that the plot-srf script produces the same output as the wiki still."""
    with tempfile.NamedTemporaryFile(suffix=".png") as output_path:
        plot_srf.plot_srf(SRF_FFP, Path(output_path.name))
        assert md5sum(PLOT_IMAGE_DIRECTORY / "srf_plot_example.png") == md5sum(
            output_path.name
        )


# NOTE: subsequent tests are identical so we have not documented them
def test_plot_srf_moment_rate():
    with tempfile.NamedTemporaryFile(suffix=".png") as output_path:
        plot_srf_moment.plot_srf_moment(SRF_FFP, Path(output_path.name))
        assert md5sum(PLOT_IMAGE_DIRECTORY / "srf_moment_rate_example.png") == md5sum(
            output_path.name
        )


def test_plot_cumulative_srf_moment_rate():
    with tempfile.NamedTemporaryFile(suffix=".png") as output_path:
        plot_srf_cumulative_moment.plot_srf_cumulative_moment(
            SRF_FFP, Path(output_path.name)
        )
        assert md5sum(
            PLOT_IMAGE_DIRECTORY / "srf_cumulative_moment_rate_example.png"
        ) == md5sum(output_path.name)


def test_plot_rise():
    with tempfile.NamedTemporaryFile(suffix=".png") as output_path:
        plot_rise.plot_rise(SRF_FFP, Path(output_path.name))
        assert md5sum(PLOT_IMAGE_DIRECTORY / "rise_example.png") == md5sum(
            output_path.name
        )


def test_plot_rake():
    with tempfile.NamedTemporaryFile(suffix=".png") as output_path:
        # Supplying seed = 1 to get the same output as the wiki.
        plot_rakes.plot_rakes(SRF_FFP, Path(output_path.name), seed=1)
        assert md5sum(PLOT_IMAGE_DIRECTORY / "rakes_example.png") == md5sum(
            output_path.name
        )
