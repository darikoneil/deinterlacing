import numpy as np

from deinterlacing.processing import deinterlace


def test_basic_deinterlacing(artifact: np.ndarray, corrected: np.ndarray) -> None:
    """Test basic deinterlacing functionality."""
    deinterlace(artifact)
    np.testing.assert_array_equal(artifact, corrected)
