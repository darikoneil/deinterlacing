import numpy as np
import pytest

from deinterlacing import DeinterlaceParameters
from deinterlacing.processing import deinterlace
from tests.conftest import acquire_gpu, release_gpu


def test_deinterlacing_gpu(
    artifact: np.ndarray,
    corrected: np.ndarray,
    missing_cupy: bool,  # noqa: FBT001
) -> None:
    """Test deinterlacing with GPU support."""
    if missing_cupy:
        pytest.skip("CuPy not installed, skipping GPU test")
    # noinspection PyGlobalUndefined
    _ = acquire_gpu()
    try:
        parameters = DeinterlaceParameters(use_gpu=True)
        deinterlace(artifact, parameters)
        np.testing.assert_allclose(artifact, corrected)
    finally:
        release_gpu()


def test_deinterlacing_gpu_subpixel(
    artifact: np.ndarray,
    subpixel_corrected: np.ndarray,
    missing_cupy: bool,  # noqa: FBT001
) -> None:
    """Test deinterlacing with GPU support and subpixel correction."""
    if missing_cupy:
        pytest.skip("CuPy not installed, skipping GPU test")
    _ = acquire_gpu()
    try:
        parameters = DeinterlaceParameters(align="subpixel", use_gpu=True)
        deinterlace(artifact, parameters)
        np.testing.assert_allclose(artifact, subpixel_corrected)
    finally:
        release_gpu()
