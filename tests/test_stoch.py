import io
import os
import tempfile
from pathlib import Path
from random import sample

import numpy as np
import pytest

from qcore import coordinates
from source_modelling import parse_utils
from source_modelling.stoch import (
    StochFile,
    StochHeader,
    _read_stoch_header,
    _read_stoch_plane,
)

STOCH_PATH = Path("tests") / "stoch" / "realisation.stoch"


@pytest.fixture
def bad_header_file(tmp_path: Path) -> Path:
    stoch_filepath = tmp_path / "bad_header_file"
    with open(stoch_filepath, "w") as f:
        f.write("-1")
    return stoch_filepath


@pytest.fixture
def sample_header() -> io.StringIO:
    return io.StringIO("174.5  -41.3  10  5  1.0  1.0  45  60 90  0.5  2.5  1.5")


@pytest.fixture
def sample_stoch_file(tmp_path: Path) -> Path:
    """Create a temporary stoch file for testing."""
    stoch_filepath = tmp_path / "stoch_file"
    with open(stoch_filepath, "w") as f:
        f.write("""1
        174.5 -41.3 3 2 1.0 1.0 45 60 90 0.5 2.5 1.5
        1.0 2.0 3.0 4.0 5.0 6.0
        0.1 0.2 0.3 0.4 0.5 0.6
        0.01 0.02 0.03 0.04 0.05 0.06
        """)
    return stoch_filepath


@pytest.fixture
def sample_stoch_file_multi_plane(tmp_path: Path) -> Path:
    """Create a temporary stoch file with multiple planes for testing."""
    stoch_filepath = tmp_path / "stoch_file"
    with open(stoch_filepath, "w") as f:
        f.write("""2
        174.5 -41.3 2 2 1.0 1.0 45 60 90 0.5 2.5 1.5
        1.0 2.0 3.0 4.0
        0.1 0.2 0.3 0.4
        0.01 0.02 0.03 0.04
        175.0 -42.0 2 2 2.0 2.0 30 40 80 1.0 3.0 2.0
        5.0 6.0 7.0 8.0
        0.5 0.6 0.7 0.8
        0.05 0.06 0.07 0.08
        """)
    return stoch_filepath


@pytest.fixture
def sample_stoch_file_plane(tmp_path: Path) -> Path:
    plane_file = tmp_path / "stoch_file"
    with open(plane_file, "w") as f:
        f.write(
            "174.5  -41.3  2  2  1.0  1.0  45  60 90  0.5  2.5 1.5 1.0 2.0 3.0 4.0 1.0 2.0 3.0 4.0 1.0 2.0 3.0 4.0"
        )
    return plane_file


def test_read_stoch_header_from_file(sample_header: io.StringIO):
    """Test reading a header from an actual file-like object."""
    # Read the header
    header = _read_stoch_header(sample_header)

    # Verify the header values
    assert header.longitude == 174.5
    assert header.latitude == -41.3
    assert header.nx == 10
    assert header.ny == 5
    assert header.dx == 1.0
    assert header.dy == 1.0
    assert header.strike == 45
    assert header.dip == 60
    assert header.average_rake == 90
    assert header.dtop == 0.5
    assert header.shypo == 2.5
    assert header.dhypo == 1.5


def test_read_stoch_plane_from_file(sample_stoch_file_plane: Path):
    """Test reading a plane from an actual file-like object."""
    # Prepare a file-like object with plane data
    # Read the plane
    with open(sample_stoch_file_plane, "r") as handle:
        plane = _read_stoch_plane(handle)

    # Verify the plane header
    assert plane.header.longitude == 174.5
    assert plane.header.latitude == -41.3
    assert plane.header.nx == 2
    assert plane.header.ny == 2

    # Verify the arrays
    assert plane.slip.shape == (2, 2)
    assert plane.rise.shape == (2, 2)
    assert plane.trup.shape == (2, 2)

    # Check specific values

    assert plane.slip == pytest.approx(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )

    plane.rise == pytest.approx(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    plane.trup == pytest.approx(
        np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    )


def test_stoch_file_initialization(sample_stoch_file: Path):
    """Test initializing a StochFile from a real file."""
    # Initialize the StochFile with the sample file
    stoch_file = StochFile(sample_stoch_file)

    # Verify the file was read correctly
    assert len(stoch_file._planes) == 1

    # Check the plane header
    plane = stoch_file._planes[0]
    assert plane.header.longitude == 174.5
    assert plane.header.latitude == -41.3
    assert plane.header.nx == 3
    assert plane.header.ny == 2

    # Check the arrays
    assert plane.slip.shape == (2, 3)
    assert plane.rise.shape == (2, 3)
    assert plane.trup.shape == (2, 3)

    # Check specific values in slip array
    expected_slip = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    plane.slip == pytest.approx(expected_slip)


def test_stoch_file_properties(sample_stoch_file):
    """Test the properties of a StochFile."""
    # Initialize the StochFile with the sample file
    stoch_file = StochFile(sample_stoch_file)

    # Test the slip property
    slip_arrays = stoch_file.slip
    assert len(slip_arrays) == 1
    assert slip_arrays[0].shape == (2, 3)

    # Test the rise property
    rise_arrays = stoch_file.rise
    assert len(rise_arrays) == 1
    assert rise_arrays[0].shape == (2, 3)

    # Test the trup property
    trup_arrays = stoch_file.trup
    assert len(trup_arrays) == 1
    assert trup_arrays[0].shape == (2, 3)

    # Test the planes property
    planes = stoch_file.planes
    assert len(planes) == 1

    assert planes[0].centroid[:2] == pytest.approx([-41.3, 174.5])

    assert planes[0].bounds[0, 2] == pytest.approx(500)
    assert planes[0].width == pytest.approx(2)
    assert planes[0].dip == pytest.approx(60)
    assert planes[0].length == pytest.approx(3)
    assert planes[0].strike == pytest.approx(45, abs=0.1)


def test_multiple_planes(sample_stoch_file_multi_plane):
    """Test reading a file with multiple planes."""
    # Initialize the StochFile with the multi-plane sample file
    stoch_file = StochFile(sample_stoch_file_multi_plane)

    # Verify the file was read correctly
    assert len(stoch_file._planes) == 2

    # Check the first plane
    plane1 = stoch_file._planes[0]
    assert plane1.header.longitude == 174.5
    assert plane1.header.latitude == -41.3
    assert plane1.header.nx == 2
    assert plane1.header.ny == 2

    # Check the second plane
    plane2 = stoch_file._planes[1]
    assert plane2.header.longitude == 175.0
    assert plane2.header.latitude == -42.0
    assert plane2.header.nx == 2
    assert plane2.header.ny == 2

    # Check the slip arrays
    slip_arrays = stoch_file.slip
    assert len(slip_arrays) == 2
    assert slip_arrays[0] == pytest.approx(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert slip_arrays[1] == pytest.approx(
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    )


def test_real_world_stoch():
    """Test reading a real-world stochastic file."""
    stoch_file = StochFile(STOCH_PATH)

    # Check the plane headers
    headers = [plane.header for plane in stoch_file._planes]
    assert headers == [
        StochHeader(
            172.1811, -43.5095, 4, 4, 2.00, 2.00, 150, 54, 255, 1.40, 1.83, 5.77
        ),
        StochHeader(
            172.1255, -43.5728, 4, 5, 2.00, 2.00, 35, 70, 39, 0.97, -999.00, -999.00
        ),
        StochHeader(
            172.0362, -43.5808, 8, 5, 2.00, 2.00, 303, 75, 127, 0.48, -999.00, -999.00
        ),
        StochHeader(
            172.1871, -43.5920, 9, 5, 2.00, 2.00, 86, 80, 55, 0.49, -999.00, -999.00
        ),
        StochHeader(
            172.3556, -43.5759, 6, 4, 2.00, 2.00, 86, 78, 65, 0.49, -999.00, -999.00
        ),
        StochHeader(
            172.3115, -43.5509, 6, 5, 2.00, 2.00, 40, 80, 64, 1.49, -999.00, -999.00
        ),
        StochHeader(
            171.9397, -43.5815, 3, 4, 2.00, 2.00, 216, 50, 79, 0.88, -999.00, -999.00
        ),
    ]

    # Check the arrays
    for slip, trise, trup, plane_data in zip(
        stoch_file.slip, stoch_file.rise, stoch_file.trup, stoch_file._planes
    ):
        assert slip.shape == (plane_data.header.ny, plane_data.header.nx)
        assert trise.shape == (plane_data.header.ny, plane_data.header.nx)
        assert trup.shape == (plane_data.header.ny, plane_data.header.nx)

    # Check specific values in first slip array
    assert stoch_file.slip[0] == pytest.approx(
        np.array(
            [
                [59, 139, 157, 200],
                [195, 153, 103, 223],
                [221, 161, 159, 217],
                [284, 179, 312, 115],
            ],
            dtype=np.float32,
        )
    )
    for patches, plane_data in zip(stoch_file.patch_centres, stoch_file._planes):
        assert patches.shape == (plane_data.header.ny, plane_data.header.nx, 3)
        nztm_patches = coordinates.wgs_depth_to_nztm(patches)
        dip_diff = np.linalg.norm(np.diff(nztm_patches, axis=0), axis=2)
        strike_diff = np.linalg.norm(np.diff(nztm_patches, axis=1), axis=2)
        assert dip_diff == pytest.approx(np.full_like(dip_diff, 2000))
        assert strike_diff == pytest.approx(np.full_like(strike_diff, 2000))


def test_stoch_file_invalid_planes(bad_header_file: Path):
    """Test handling of invalid number of planes."""
    # Create a temporary file with invalid n_planes value

    file_path = Path(bad_header_file)

    # Test that ValueError is raised
    with pytest.raises(
        parse_utils.ParseError, match="Expected non-negative integer number of planes"
    ):
        StochFile(file_path)
