import numpy as np
import pytest
from pydantic import ValidationError

from deinterlacing.processing import DeinterlaceParameters
from deinterlacing.tools import ParameterError


def test_init() -> None:
    """
    Test initialization with custom parameters.

    :returns: None
    """
    params = DeinterlaceParameters(
        block_size=20,
        pool="mean",
        unstable=5,
        subsearch=10,
        align="subpixel",
        use_gpu=True,
    )
    assert params.block_size == 20
    assert params.pool == "mean"
    assert params.unstable == 5
    assert params.subsearch == 10
    assert params.align == "subpixel"
    assert params.use_gpu is True


def test_init_with_images(large_artifact: np.ndarray) -> None:
    """
    Test initialization with images parameter.

    :param large_images: Sample image stack fixture
    :returns: None
    """
    params = DeinterlaceParameters(images=large_artifact)
    assert params.block_size == large_artifact.shape[0]


def test_validate_positive_integer() -> None:
    """
    Test validation of positive integers.

    :returns: None
    """
    # Valid values
    params = DeinterlaceParameters(block_size=10, unstable=5, subsearch=8)
    assert params.block_size == 10
    assert params.unstable == 5
    assert params.subsearch == 8

    # Invalid values
    with pytest.raises(ValidationError):
        DeinterlaceParameters(block_size=-5)

    with pytest.raises(ValidationError):
        DeinterlaceParameters(unstable=0)

    with pytest.raises(ValidationError):
        DeinterlaceParameters(subsearch=-3)


def test_validate_with_images(large_artifact: np.ndarray) -> None:
    """
    Test parameter validation against images.

    :param large_images: Sample image stack fixture
    :returns: None
    """
    # Auto-setting block_size
    params = DeinterlaceParameters()
    params.validate_with_images(large_artifact)
    assert params.block_size == large_artifact.shape[0]

    # Auto-setting subsearch
    params = DeinterlaceParameters(subsearch=None)
    params.validate_with_images(large_artifact)
    min_dim = min(large_artifact.shape[1:])
    assert params.subsearch == min_dim // 16


def test_block_size_too_large(large_artifact: np.ndarray) -> None:
    """
    Test validation when block_size exceeds image dimensions.

    :param large_images: Sample image stack fixture
    :returns: None
    """
    params = DeinterlaceParameters(block_size=large_artifact.shape[0] + 1)
    with pytest.raises(ParameterError) as exc_info:
        params.validate_with_images(large_artifact)
    assert "block_size" in str(exc_info.value)


def test_subsearch_too_large(large_artifact: np.ndarray) -> None:
    """
    Test validation when subsearch exceeds image dimensions.

    :param large_images: Sample image stack fixture
    :returns: None
    """
    min_dim = min(large_artifact.shape[1:])
    params = DeinterlaceParameters(subsearch=min_dim + 1)
    with pytest.raises(ParameterError) as exc_info:
        params.validate_with_images(large_artifact)
    assert "subsearch" in str(exc_info.value)


def test_unstable_too_large(large_artifact: np.ndarray) -> None:
    """
    Test validation when unstable exceeds frame count.

    :param large_images: Sample image stack fixture
    :returns: None
    """
    params = DeinterlaceParameters(unstable=large_artifact.shape[0] + 1)
    with pytest.raises(ParameterError) as exc_info:
        params.validate_with_images(large_artifact)
    assert "unstable" in str(exc_info.value)


def test_gpu_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test validation when GPU requested but not available.

    :param monkeypatch: pytest's monkeypatch fixture
    :returns: None
    """
    import deinterlacing.processing as proc

    # Simulate CuPy not available
    monkeypatch.setattr(proc, "cp", proc.np)

    params = DeinterlaceParameters(use_gpu=True)
    with pytest.raises(ValueError, match="CuPy is not available"):
        params.validate_with_images(np.zeros((10, 64, 64)))


def test_pool_options() -> None:
    """
    Test all valid pool options.

    :returns: None
    """
    for pool in ["mean", "median", "std", "sum", None]:
        params = DeinterlaceParameters(pool=pool)
        assert params.pool == pool


def test_align_options() -> None:
    """
    Test valid alignment options.

    :returns: None
    """
    params1 = DeinterlaceParameters(align="pixel")
    assert params1.align == "pixel"

    params2 = DeinterlaceParameters(align="subpixel")
    assert params2.align == "subpixel"


def test_small_image_handling(small_artifact: np.ndarray) -> None:
    """
    Test parameter handling with small images.

    :param small_images: Small image stack fixture
    :returns: None
    """
    params = DeinterlaceParameters(subsearch=None)
    params.validate_with_images(small_artifact)

    # Check subsearch is properly scaled down for small images
    min_dim = min(small_artifact.shape[1:])
    assert params.subsearch == min_dim // 16
