import inspect
from collections.abc import Callable, Generator
from functools import wraps
from itertools import chain
from typing import Any, Literal, TypeAlias

import numpy as np
from boltons.iterutils import chunk_ranges

try:
    import cupy as cp
except ImportError:
    cp = np
else:
    pass

__all__ = [
    "align_pixels",
    "align_subpixels",
    "extract_image_block",
    "find_pixel_offset",
    "find_subpixel_offset",
    "index_image_blocks",
]

#: Type alias for numpy array-like data structures.
NDArrayLike: TypeAlias = np.ndarray | cp.ndarray

#: Type alias for a generator of image blocks
ImageBlockGenerator: TypeAlias = Generator[tuple[int, int], None, None]


class ParameterError(ValueError):
    """Custom exception for parameter validation errors in deinterlacing."""

    def __init__(self, parameter: str, value: Any, limits: tuple[Any, Any]) -> None:
        self.value = value
        self.limits = limits
        message = (
            f"Parameter '{parameter}' with value {value} is "
            f"not within the appropriate bounds {limits}."
        )
        super().__init__(message)


def calculate_offset_matrix(
    images: NDArrayLike, fft_module: Literal[np, cp] = np
) -> NDArrayLike:
    # offset, avoid division by zero in normalization
    OFFSET = 1e-5  # noqa: N806

    forward = fft_module.fft.fft(images[..., 1::2, :], axis=-1)
    forward /= np.abs(forward) + OFFSET

    backward = fft_module.fft.fft(images[..., ::2, :], axis=-1)
    np.conj(backward, out=backward)
    backward /= np.abs(backward) + OFFSET
    backward = backward[..., : forward.shape[-2], :]

    # inverse
    comp_conj = fft_module.fft.ifft(forward * backward, axis=-1)
    comp_conj = np.real(comp_conj)
    if comp_conj.ndim == 3:
        comp_conj = comp_conj.mean(axis=1)
    if comp_conj.ndim == 2:
        comp_conj = comp_conj.mean(axis=0)
    return fft_module.fft.fftshift(comp_conj)  # ifftshift?


def extract_image_block(
    images: NDArrayLike,
    start: int,
    stop: int,
    pool: Literal["mean", "median", "std", "sum", None],
) -> NDArrayLike:
    image_block = images[start:stop, ...]
    if pool == "mean":
        return image_block.mean(axis=0).astype(images.dtype)
    if pool == "median":
        return np.median(image_block, axis=0).astype(images.dtype)
    if pool == "std":
        return image_block.std(axis=0, ddof=1).astype(images.dtype)
    if pool == "sum":
        return image_block.sum(axis=0).astype(images.dtype)
    return image_block


def find_pixel_offset(
    images: NDArrayLike,
    offset_matrix: NDArrayLike,
    subsearch: int,
) -> int:
    # search only subspace to save computation time and avoid any artifacts from edge
    # of image. extremely important!
    peak = np.argmax(
        offset_matrix[-subsearch + images.shape[-2] // 2 : images.shape[-1] // 2]
        + subsearch
        + 1
    )

    # If the image is very sparse, the peak here could be determined by the statistics
    # of PMT noise, which is unrelated to scanning artifacts. To avoid this, we can
    # check if the second highest peak is significantly lower than the zeroth peak in
    # the case that the calculated phase offset is 0
    if peak == subsearch:
        # argpart is log(n) complexity, so it is faster than sorting the entire array
        pk0, pk1 = np.argpartition(
            -offset_matrix[-subsearch + images.shape[-2] // 2 : images.shape[-1] // 2]
            + subsearch
            + 1,
            2,
        )[:2]
        # If peak is +/- 1 from the first peak, it is likely not genuine
        if (new_peak := pk1) - pk0 != 1:
            peak = new_peak

    return -(peak - subsearch)


def find_subpixel_offset(
    images: NDArrayLike,
    offset_matrix: NDArrayLike,
    subsearch: int,
) -> float:
    peak = find_pixel_offset(images, offset_matrix, subsearch)
    y0, y1, y2 = offset_matrix[peak - 1], offset_matrix[peak], offset_matrix[peak + 1]
    subpixel_offset = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    # Combine integer offset with sub-pixel refinement
    return peak - subpixel_offset


def index_image_blocks(
    images: NDArrayLike,
    block_size: int,
    unstable: int | None = None,
) -> ImageBlockGenerator:
    """
    Index the image blocks for batch processing during deinterlacing. This function
    returns a generator yielding tuples of start and end
    indices for each block of images to be processed. It takes into account the
    `unstable` parameter, which specifies how many frames should be processed
    individually before switching to batch-wise processing.

    :param images:
    :param block_size:
    :param unstable:
    :returns: A generator yielding tuples of
        (start_index, end_index) for each block.
    """
    if unstable:
        stable_frames = images.shape[0] - unstable
        blocks = chain(
            chunk_ranges(unstable, 1),
            chunk_ranges(stable_frames, block_size, input_offset=unstable),
        )
    else:
        blocks = chunk_ranges(images.shape[0], block_size)
    return blocks


def align_pixels(images: NDArrayLike, start: int, stop: int, phase_offset: int) -> None:
    if phase_offset > 0:
        images[start:stop, 1::2, phase_offset:] = images[
            start:stop, 1::2, :-phase_offset
        ]
    elif phase_offset < 0:
        images[start:stop, 1::2, :phase_offset] = images[
            start:stop, 1::2, -phase_offset:
        ]


def align_subpixels(
    images: NDArrayLike,
    start: int,
    stop: int,
    offset: int,
    fft_module: Literal[np, cp] = np,
) -> None:
    backward_lines = images[start:stop, 1::2, ...]
    fft_lines = fft_module.fft.fft(backward_lines, axis=-1)
    n = fft_lines.shape[-1]
    freq = fft_module.fft.fftfreq(n)
    phase = -2.0 * fft_module.pi * offset * freq
    fft_lines *= fft_module.exp(1j * phase)
    images[start:stop, 1::2, :] = fft_module.real(
        fft_module.fft.ifft(fft_lines, axis=-1)
    )


def wrap_cupy(
    function: Callable[[cp.ndarray], cp.ndarray], *parameter: str
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Convenience decorator that wraps a CuPy function such that incoming numpy arrays
    are converted to cupy arrays and swapped back on return.

    :param function: any CuPy function that accepts a CuPy array
    :param parameter: name/s of the parameter to be converted
    :returns: wrapped function
    """

    @wraps(function)
    def decorator(*args, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        sig = inspect.signature(function)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        bound_args.arguments = {**bound_args.kwargs, **bound_args.arguments}
        for param in parameter:
            # noinspection PyUnresolvedReferences
            bound_args.arguments[param] = cp.asarray(bound_args.arguments[param])
        return function(**bound_args.arguments).get()

    # noinspection PyTypeChecker
    return decorator
