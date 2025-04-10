from typing import Literal

import numpy as np

from deinterlacing.tools import NDArrayLike, wrap_cupy

try:
    import cupy as cp
except ImportError:
    cp = np

__all__ = [
    "align_pixels",
    "align_subpixels",
    "correct_subpixel_offset",
]


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
