from functools import partial
from typing import Literal

import numpy as np
from boltons.iterutils import chunk_ranges
from scipy.optimize import Bounds, curve_fit

from deinterlacing.tools import NDArrayLike

try:
    import cupy as cp
except ImportError:
    cp = np

__all__ = [
    "calculate_offset_matrix",
    "find_pixel_offset",
    "find_subpixel_offset",
]


def find_pixel_offset(
    images: NDArrayLike,
    offset_matrix: NDArrayLike,
    subsearch: int,
) -> int:
    # search only subspace to save computation time and avoid any artifacts from edge
    # of image. Extremely important for avoiding artifacts, not sure about about
    # performance impact in practice.
    peak = np.argmax(
        offset_matrix[
            -subsearch + images.shape[-1] // 2 : images.shape[-1] // 2 + subsearch + 1
        ]
    )

    # If the image is very sparse, the peak here could be determined by the statistics
    # of PMT noise, which is unrelated to scanning artifacts. To avoid this, we can
    # check if the second highest peak is significantly lower than the zeroth peak in
    # the case that the calculated phase offset is 0
    if peak == subsearch:
        # argpart is log(n) complexity, so it is faster than sorting the entire array
        pk0, pk1 = np.argpartition(
            -offset_matrix[
                -subsearch + images.shape[-1] // 2 : images.shape[-1] // 2
                + subsearch
                + 1
            ],
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


def _parabolic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Parabolic function for curve fitting."""
    return a * (x**2) + b * x + c


def find_variable_offset(
    images: np.ndarray, line_subsections: int = 8, subsearch: int = 15
) -> 0:
    # TODO: Review argument defaults
    pixels_per_line = images.shape[-1]
    pixels_per_chunk = pixels_per_line // line_subsections
    indices = list(chunk_ranges(pixels_per_line, pixels_per_chunk))
    # REVIEW: This should probably be a parameter someplace
    offset_matrices = [
        calculate_offset_matrix(images[:, idx[0] : idx[1]]) for idx in indices
    ]
    phase_offsets = [
        find_pixel_offset(images[:, idx[0] : idx[1]], offset_matrices[n], subsearch)
        for n, idx in enumerate(indices)
    ]
    bounds = Bounds([-np.inf, -np.inf, -np.inf], [0, np.inf, np.inf])
    xtrain = np.asarray([sum(idx) // 2 for idx in indices])
    coefs = curve_fit(_parabolic, xtrain, phase_offsets, bounds=bounds)[0]
    a_, b_, c_ = coefs
    fit = partial(_parabolic, a=a_, b=b_, c=c_)
    return fit
