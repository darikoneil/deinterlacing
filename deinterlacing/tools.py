import inspect
from collections.abc import Callable, Generator
from functools import wraps
from itertools import chain
from typing import Literal, TypeAlias

import numpy as np
from boltons.iterutils import chunk_ranges

try:
    import cupy as cp
except ImportError:
    cp = np

__all__ = [
    "ImageBlockGenerator",
    "NDArrayLike",
    "extract_image_block",
    "index_image_blocks",
    "wrap_cupy",
]

# NOTE: Using type aliasing instead of type for backwards compatibility to 3.10
#: Type alias for numpy array-like data structures.
NDArrayLike: TypeAlias = np.ndarray | cp.ndarray

#: Type alias for a generator of image blocks
ImageBlockGenerator: TypeAlias = Generator[tuple[int, int], None, None]

# This is to use dictionary dispatch in extract_image_block
_POOL_FUNCS = {
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
    return _POOL_FUNCS[pool](image_block).astype(images.dtype)


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
