import inspect
from functools import wraps
from itertools import chain
from typing import TYPE_CHECKING, Callable, TypeAlias

import numpy as np
from boltons.iterutils import chunk_ranges
from numpy.fft import fft as _fft
from numpy.fft import ifft as _ifft
from tqdm import tqdm

__all__ = [
    "deinterlace",
]


try:
    import cupy as cp
except ImportError:
    cp = None
    if TYPE_CHECKING:
        cp = np
    print("CuPy not found. Falling back to NumPy implementation.")
else:
    from cupy.fft import fft as _fft  # noqa: F811
    from cupy.fft import ifft as _ifft  # noqa: F811


#: Type alias for CuPy or NumPy arrays
CupyArray: TypeAlias = cp.ndarray if cp else np.ndarray


def _wrap_cupy(
    function: Callable[CupyArray, CupyArray], *parameter: str
) -> Callable[np.ndarray, np.ndarray]:
    """
    Convinence decorater that wraps a cupy function such that incoming numpy arrays are
    converting to cupy arrays and swapped back on return.

    :param function: any cupy function that accepts a cupy array
    :param parameter: name/s of the parameter to be converted
    :return: wrapped function
    """

    @wraps(function)
    def decorator(*args, **kwargs) -> Callable[np.ndarray, np.ndarray]:
        sig = inspect.signature(function)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        bound_args.arguments = {**bound_args.kwargs, **bound_args.arguments}
        for param in parameter:
            # noinspection PyUnresolvedReferences
            bound_args.arguments[param] = cp.asarray(bound_args.arguments[param])
        return function(**bound_args.arguments).get()

    return decorator


def _calculate_phase_offset(images: np.ndarray | CupyArray) -> int:
    # offset
    offset = 1e-5

    forward = _fft(images[:, 1::2, :], axis=2)
    forward /= np.abs(forward) + offset

    backward = _fft(images[:, ::2, :], axis=2)
    np.conj(backward, out=backward)
    backward /= np.abs(backward) + offset
    backward = backward[:, : forward.shape[1], ...]

    # inverse
    comp_conj = _ifft(forward * backward, axis=2)
    comp_conj = np.real(comp_conj)
    comp_conj = comp_conj.mean(axis=1).mean(axis=0)
    comp_conj = np.fft.fftshift(comp_conj)

    # find peak
    return -(
        np.argmax(comp_conj[-10 + images.shape[1] // 2 : 11 + images.shape[2] // 2])
        - 10
    )


def deinterlace(
    images: np.ndarray,
    block_size: int | None = None,
    subsample: bool = False,
    unstable: int | None = None,
) -> None:
    """
    Deinterlace images collected using resonance-scanning microscopes such that the
    forward and backward-scanned lines are properly aligned. A fourier-approach is utilized:
    the fourier transform of the two sets of lines is computed to calculate thecross-power 
    spectral density. Taking the inverse fourier transform of the cross-power spectral density 
    yields a matrix whose peak corresponds to the sub-pixel offset between the two sets of lines. 
    This translative offset was then discretized and used to shift the backward-scanned lines.

    Unfortunately, the fast-fourier transform methods that underlie the implementation
    of the deinterlacing algorithm have poor spatial complexity
    (i.e., large memory constraints). This weakness is particularly problematic when
    using GPU-parallelization. To mitigate these issues, deinterlacing can be performed
    batch-wise while maintaining numerically identical results (see `block_size`).

    To improve performance, the deinterlacing algorithm can be applied to a subsample of
    the images while maintaining efficacy. Specifically, setting the `subsample`
    parameter will apply the deinterlacing algorithm to the the standard deviation of
    each pixel across a block of images. This approach is better suited to images with
    limited signal-to-noise or sparse activity than simply operating on every n-th
    frame.

    Finally, it is often the case that the auto-alignment algorithms used in microscopy
    software are unstable until a sufficient number of frames have been collected.]
    Therefore, the `unstable` parameter can be used to specify the number of frames
    that should be deinterlaced individually before switching to batch-wise processing.


    :param images: Images to deinterlace (frames, y-pixels, x-pixels)

    :param block_size: Number of frames included per FFT calculation.

    :param subsample: Whether to apply the deinterlacing algorithm to the standard
        deviation of each pixel across a block of images.

    :param unstable: Number of frames to deinterlace individually before switching to
        batch-wise processing.

    .. note::
        This function operates in-place.

    .. warning::
        The number of frames included in each fourier transform must be several times
        smaller than the maximum number of frames that fit within your GPU's VRAM
        (`CuPy <https://cupy.dev>`_) or RAM (`NumPy <https://numpy.org>`_). This
        function will not automatically revert to the NumPy implementation if there is
        not sufficient VRAM. Instead, an out of memory error will be raised.
    """
    original_type = images.dtype
    block_size = block_size or images.shape[0]

    if cp:
        calculate_phase_offset = _wrap_cupy(_calculate_phase_offset, "images")
    else:
        calculate_phase_offset = _calculate_phase_offset

    if unstable:
        stable_frames = images.shape[0] - unstable
        num_blocks = (
            unstable
            + stable_frames // block_size
            + bool(stable_frames % block_size) * 1
        )
        blocks = chain(
            chunk_ranges(unstable, 1),
            chunk_ranges(stable_frames, block_size, input_offset=unstable),
        )
    else:
        num_blocks = (
            images.shape[0] // block_size + bool(images.shape[0] % block_size) * 1
        )
        blocks = chunk_ranges(images.shape[0], block_size)

    pbar = tqdm(total=num_blocks, desc="Deinterlacing resonant images", colour="blue")
    for start, stop in blocks:
        if subsample:
            block = np.std(images[start:stop, ...], axis=0).astype(original_type)
            block = np.dstack([block for _ in range(2)]).swapaxes(0, 2).swapaxes(1, 2)
        else:
            block = images[start:stop, ...]
        phase_offset = calculate_phase_offset(block)
        if phase_offset > 0:
            images[start:stop, 1::2, phase_offset:] = images[
                start:stop, 1::2, :-phase_offset
            ]
        elif phase_offset < 0:
            images[start:stop, 1::2, :phase_offset] = images[
                start:stop, 1::2, -phase_offset:
            ]
          # TODO: Implement subpixel-interpolation for sub-pixel alignment
        pbar.update(1)
    pbar.close()
