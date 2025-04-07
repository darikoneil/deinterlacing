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
    # offset used simply to avoid division by zero in normalization
    OFFSET = 1e-10  # noqa: N806

    backward = fft_module.fft.fft(images[..., 1::2, :], axis=-1)
    backward /= fft_module.abs(backward) + OFFSET

    forward = fft_module.fft.fft(images[..., ::2, :], axis=-1)
    fft_module.conj(forward, out=forward)
    forward /= fft_module.abs(forward) + OFFSET
    forward = forward[..., : backward.shape[-2], :]

    # inverse
    comp_conj = fft_module.fft.ifft(backward * forward, axis=-1)
    comp_conj = fft_module.real(comp_conj)
    if comp_conj.ndim == 3:
        comp_conj = comp_conj.mean(axis=1)
    if comp_conj.ndim == 2:
        comp_conj = comp_conj.mean(axis=0)
    return fft_module.fft.ifftshift(comp_conj)
    # REVIEW: Should this be ifftshift or fftshift?


# This is to use dictionary dispatch in extract_image_block
POOL_FUNCS = {
    "mean": lambda x: x.mean(axis=0).astype(x.dtype),
    "median": lambda x: np.median(x, axis=0).astype(x.dtype),
    "std": lambda x: x.std(axis=0, ddof=1).astype(x.dtype),
    "sum": lambda x: x.sum(axis=0).astype(x.dtype),
    None: lambda x: x,
}


def extract_image_block(
    images: NDArrayLike,
    start: int,
    stop: int,
    pool: Literal["mean", "median", "std", "sum", None],
) -> NDArrayLike:
    image_block = images[start:stop, ...]
    return POOL_FUNCS[pool](image_block).astype(images.dtype)


def find_pixel_offset(
    images: NDArrayLike,
    offset_matrix: NDArrayLike,
    subsearch: int,
) -> int:
    # search only subspace to save computation time and avoid any artifacts from edge
    # of image. Extremely important for avoiding artifacts, not sure about about
    # performance impact in practice.
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
    if peak <= 0 or peak >= offset_matrix.shape[0] - 1:
        return float(peak)  # Just a boundary check here; return it as is

    # this part is just a manual implementation of quadratic interpolation
    # to find sub-pixel offset. Something more sophisticated might be more appropriate,
    # but this is the first thing that came to mind.
    y0, y1, y2 = offset_matrix[peak - 1], offset_matrix[peak], offset_matrix[peak + 1]
    denominator = y0 - 2 * y1 + y2
    if abs(denominator) < 1e-10:
        # If the denominator is too close to zero, interpolation is not reliable.
        return float(peak)
    subpixel_offset = 0.5 * (y0 - y2) / denominator
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


def align_pixels(images: NDArrayLike, start: int, stop: int, offset: int) -> None:
    if offset > 0:
        images[start:stop, 1::2, offset:] = images[start:stop, 1::2, :-offset]
    elif offset < 0:
        images[start:stop, 1::2, :offset] = images[start:stop, 1::2, -offset:]


def correct_subpixel_offset(
    backward_lines: NDArrayLike,
    offset: float,
    fft_module: Literal[np, cp] = np,
) -> None:
    vectorized = backward_lines.reshape(-1, backward_lines.shape[-1])
    fft_lines = fft_module.fft.fft(vectorized, axis=-1)

    # FREQUENCY CACHE
    n = fft_lines.shape[-1]
    if (freq := getattr(align_subpixels, "freq", None)) is None:
        # Cache the frequencies for the first time
        freq = fft_module.fft.fftfreq(n)
        align_subpixels.freq = {n: freq}
    elif (freq := freq.get(n)) is None:
        freq = fft_module.fft.fftfreq(n)
        align_subpixels.freq[n] = freq
    # HACK: This hack makes sure the frequency cache is an appropriate type, because
    #  the test suite will fail in an order-dependent way if the cache is not the
    #  appropriate type
    phase = -2.0 * fft_module.pi * offset * fft_module.asarray(freq)
    fft_lines *= fft_module.exp(1j * phase)
    return fft_module.real(fft_module.fft.ifft(fft_lines, axis=-1))


def align_subpixels(
    images: NDArrayLike,
    start: int,
    stop: int,
    offset: float,
    fft_module: Literal[np, cp] = np,
) -> None:
    backward_lines = images[start:stop, 1::2, ...]
    if fft_module == cp:
        corrector = wrap_cupy(correct_subpixel_offset, "backward_lines", "offset")
    else:
        corrector = correct_subpixel_offset
    vectorized_correction = corrector(backward_lines, offset, fft_module=fft_module)
    images[start:stop, 1::2, ...] = vectorized_correction.reshape(backward_lines.shape)


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
