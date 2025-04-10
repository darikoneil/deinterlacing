from typing import Literal

import numpy as np

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
