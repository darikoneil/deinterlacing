from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def artifact() -> np.ndarray:
    if (cached_value := getattr(artifact, "_cached_value", None)) is None:
        images = np.load(
            Path(__file__).parent.joinpath("artifact.npy"), allow_pickle=False
        )
        cached_value = np.vstack(
            [np.reshape(images, (1, *images.shape)) for _ in range(2048)]
        )
        artifact._cached_value = cached_value  # noqa: SLF001
    return cached_value


@pytest.fixture
def corrected() -> np.ndarray:
    if (cached_value := getattr(corrected, "_cached_value", None)) is None:
        images = np.load(
            Path(__file__).parent.joinpath("corrected.npy"), allow_pickle=False
        )
        cached_value = np.vstack(
            [np.reshape(images, (1, *images.shape)) for _ in range(2048)]
        )
        corrected._cached_value = cached_value  # noqa: SLF001
    return cached_value
