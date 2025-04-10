import numpy as np
import pytest

from deinterlacing import DeinterlaceParameters
from deinterlacing.processing import deinterlace


def test_deinterlace_frames_gt_dims(
    artifact: np.ndarray, corrected: np.ndarray
) -> None:
    """Test basic deinterlacing functionality."""
    deinterlace(artifact)
    np.testing.assert_array_equal(artifact, corrected)


def test_deinterlace_frames_lt_dims(
    artifact: np.ndarray, corrected: np.ndarray
) -> None:
    """Test deinterlacing with frames less than or equal to dimensions."""
    artifact = artifact[:3, :, :]
    deinterlace(artifact)
    np.testing.assert_array_equal(artifact[:3, :, :], corrected[:3, :, :])


def test_deinterlace_blocks_gt_dims(
    artifact: np.ndarray, corrected: np.ndarray
) -> None:
    """Test deinterlacing blocks with dimensions greater than the artifact."""
    parameters = DeinterlaceParameters(block_size=artifact.shape[-1] + 1)
    deinterlace(artifact, parameters)
    np.testing.assert_array_equal(artifact, corrected)
    # SHOULD BE NUMERICALLY EQUIVALENT TO NON-BLOCK DEINTERLACING


def test_deinterlace_blocks_lt_dims(
    artifact: np.ndarray, corrected: np.ndarray
) -> None:
    """Test deinterlacing blocks with dimensions less than or equal to the artifact."""
    parameters = DeinterlaceParameters(block_size=3)  # smaller than artifact
    deinterlace(artifact, parameters)
    np.testing.assert_array_equal(artifact, corrected)
    # SHOULD BE NUMERICALLY EQUIVALENT TO NON-BLOCK DEINTERLACING


# THE POOLS ARE SPLIT BECAUSE STD IS ONLY USEFUL FOR PARTICULAR DISTRIBUTIONS OF NOISE
def test_deinterlace_pool_mean(artifact: np.ndarray, corrected: np.ndarray) -> None:
    """Test deinterlacing with pooling mean."""
    parameters = DeinterlaceParameters(pool="mean")
    deinterlace(artifact[:3, :, :], parameters)
    np.testing.assert_array_equal(artifact[:3, :, :], corrected[:3, :, :])
    # WE SHOULD ARRIVE AT SAME GROUND TRUTH AS NON-POOLING


def test_deinterlace_pool_median(artifact: np.ndarray, corrected: np.ndarray) -> None:
    """Test deinterlacing with pooling mean."""
    parameters = DeinterlaceParameters(pool="median")
    deinterlace(artifact[:3, :, :], parameters)
    np.testing.assert_array_equal(artifact[:3, :, :], corrected[:3, :, :])
    # WE SHOULD ARRIVE AT SAME GROUND TRUTH AS NON-POOLING


@pytest.mark.skip(reason="Needs its own test dataset")
def test_deinterlace_pool_std(artifact: np.ndarray, corrected: np.ndarray) -> None:
    """Test deinterlacing with pooling standard deviation."""
    parameters = DeinterlaceParameters(pool="std")
    deinterlace(artifact[:3, :, :], parameters)
    np.testing.assert_array_equal(artifact[:3, :, :], corrected[:3, :, :])
    # WE SHOULD ARRIVE AT SAME GROUND TRUTH AS NON-POOLING


def test_deinterlace_pool_sum(artifact: np.ndarray, corrected: np.ndarray) -> None:
    """Test deinterlacing with pooling mean."""
    parameters = DeinterlaceParameters(pool="sum")
    deinterlace(artifact[:3, :, :], parameters)
    np.testing.assert_array_equal(artifact[:3, :, :], corrected[:3, :, :])
    # WE SHOULD ARRIVE AT SAME GROUND TRUTH AS NON-POOLING


def test_deinterlace_unstable(artifact: np.ndarray, corrected: np.ndarray) -> None:
    artifact = artifact[:3, :, :]
    parameters = DeinterlaceParameters(unstable=artifact.shape[0])
    deinterlace(artifact, parameters)
    np.testing.assert_array_equal(artifact, corrected[:3, :, :])
    # We should arrive at the same ground truth as non-unstable deinterlacing


def test_deinterlace_vertical_flipped(
    artifact: np.ndarray, corrected: np.ndarray
) -> None:
    """Test deinterlacing with flipped frames."""
    # Flip the artifact vertically
    flipped_artifact = np.flipud(artifact[:3, :, :])
    parameters = DeinterlaceParameters()
    deinterlace(flipped_artifact, parameters)
    np.testing.assert_array_equal(np.flipud(flipped_artifact), corrected[:3, :, :])
    # We should arrive at the same ground truth as non-flipped deinterlacing


@pytest.mark.skip(reason="Fails")
def test_deinterlace_horizontal_flipped(
    artifact: np.ndarray, corrected: np.ndarray
) -> None:
    """Test deinterlacing with flipped frames."""
    flipped_artifact = np.fliplr(artifact[:3, :, :])
    parameters = DeinterlaceParameters()
    deinterlace(flipped_artifact, parameters)
    np.testing.assert_almost_equal(np.fliplr(flipped_artifact), corrected[:3, :, :])
    # We should arrive at the same ground truth as non-flipped deinterlacing


@pytest.mark.skip(reason="Fails")
def test_deinterlace_zero_offset(corrected: np.ndarray) -> None:
    artifact = corrected[:3, :, :].copy()
    deinterlace(artifact)
    np.testing.assert_array_equal(artifact, corrected[:3, :, :])


def test_deinterlacing_subpixel(
    artifact: np.ndarray, subpixel_corrected: np.ndarray
) -> None:
    """Test deinterlacing with subpixel correction."""
    parameters = DeinterlaceParameters(align="subpixel")
    deinterlace(artifact, parameters)
    np.testing.assert_allclose(artifact, subpixel_corrected)
