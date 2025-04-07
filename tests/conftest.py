from importlib.util import find_spec
from pathlib import Path
from threading import Semaphore

import numpy as np
import pytest

# NOTE: Since GPU's are stateful, we need to ensure that only one test is using the
#  GPU at a time
USING_GPU = Semaphore(value=1)


def acquire_gpu() -> Semaphore:
    """
    Acquire GPU semaphore.

    :returns: None
    """
    global USING_GPU
    USING_GPU.acquire()


def release_gpu() -> None:
    """
    Release GPU semaphore.

    :returns: None
    """
    global USING_GPU
    USING_GPU.release()
    # NOTE: This is essentially no cost, just a lookup in sys.modules
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    cp.cuda.runtime.deviceSynchronize()


@pytest.fixture(scope="session")
def missing_cupy() -> bool:
    """
    Check if GPU is available.

    :returns: True if GPU is available, False otherwise
    """
    # Kind of anti-pattern to return True if cupy is not available, but it's clearer
    # within the tests. I could put skip here, but then it's not clear what the point
    # of having the fixture is in test signatures.
    try:
        assert find_spec("cupy")
    except (
        AssertionError,
        AttributeError,
        ModuleNotFoundError,
        ValueError,
        ImportError,
    ):
        return True
    else:
        return False


@pytest.fixture
def artifact() -> np.ndarray:
    if (cached_value := getattr(artifact, "_cached_value", None)) is None:
        images = np.load(
            Path(__file__).parent.joinpath("artifact.npy"), allow_pickle=False
        )
        cached_value = np.vstack(
            [np.reshape(images, (1, *images.shape)) for _ in range(1024)]
        )
        artifact._cached_value = cached_value  # noqa: SLF001
    return cached_value.copy()


@pytest.fixture
def corrected() -> np.ndarray:
    if (cached_value := getattr(corrected, "_cached_value", None)) is None:
        images = np.load(
            Path(__file__).parent.joinpath("corrected.npy"), allow_pickle=False
        )
        cached_value = np.vstack(
            [np.reshape(images, (1, *images.shape)) for _ in range(1024)]
        )
        corrected._cached_value = cached_value  # noqa: SLF001
    return cached_value.copy()


@pytest.fixture
def subpixel_corrected() -> np.ndarray:
    if (cached_value := getattr(subpixel_corrected, "_cached_value", None)) is None:
        images = np.load(
            Path(__file__).parent.joinpath("subpixel_corrected.npy"), allow_pickle=False
        )
        cached_value = np.vstack(
            [np.reshape(images, (1, *images.shape)) for _ in range(1024)]
        )
        subpixel_corrected._cached_value = cached_value  # noqa: SLF001
    return cached_value.copy()


@pytest.fixture
def large_artifact(artifact: pytest.FixtureDef[None]) -> None:
    """
    Create large image stack for testing.

    :returns: 3D numpy array representing an image stack
    """
    return artifact[:10, 192:320, 192:320].copy()


@pytest.fixture
def small_artifact(artifact: pytest.FixtureDef[None]) -> None:
    """
    Create small image stack for edge case testing.

    :returns: 3D numpy array with small dimensions
    """
    return artifact[:3, 224:288, 224:288].copy()
