from functools import partial

import numpy as np
from tqdm import tqdm

from deinterlacing.alignment import align_pixels, align_subpixels
from deinterlacing.offsets import (
    calculate_offset_matrix,
    find_pixel_offset,
    find_subpixel_offset,
)
from deinterlacing.parameters import DeinterlaceParameters
from deinterlacing.tools import (
    NDArrayLike,
    compose,
    extract_image_block,
    index_image_blocks,
    wrap_cupy,
)

try:
    import cupy as cp
except ImportError:
    cp = np


__all__ = [
    "deinterlace",
]


def deinterlace(
    images: NDArrayLike,
    parameters: DeinterlaceParameters | None = None,
) -> None:
    """
    Deinterlace images collected using resonance-scanning microscopes such that the
    forward and backward-scanned lines are properly aligned. A fourier-approach is
    utilized: the fourier transform of the two sets of lines is computed to calculate
    the cross-power spectral density. Taking the inverse fourier transform of the
    cross-power spectral density yields a matrix whose peak corresponds to the
    sub-pixel offset between the two sets of lines. This translative offset was then
    discretized and used to shift the backward-scanned lines.

    Unfortunately, the fast-fourier transform methods that underlie the implementation
    of the deinterlacing algorithm have poor spatial complexity
    (i.e., large memory constraints). This weakness is particularly problematic when
    using GPU-parallelization. To mitigate these issues, deinterlacing can be performed
    batch-wise while maintaining numerically identical results (see `block_size`).

    To improve performance, the deinterlacing algorithm can be applied to a pool
    of the images while maintaining efficacy. Specifically, setting the `pool`
    parameter will apply the deinterlacing algorithm to the the standard deviation of
    each pixel across a block of images. This approach is better suited to images with
    limited signal-to-noise or sparse activity than simply operating on every n-th
    frame.

    Finally, it is often the case that the auto-alignment algorithms used in microscopy
    software are unstable until a sufficient number of frames have been collected.
    Therefore, the `unstable` parameter can be used to specify the number of frames
    that should be deinterlaced individually before switching to batch-wise processing.

    .. note::
        This function operates in-place.

    .. warning::
        The number of frames included in each fourier transform must be several times
        smaller than the maximum number of frames that fit within your GPU's VRAM
        (`CuPy <https://cupy.dev>`_) or RAM (`NumPy <https://numpy.org>`_). This
        function will not automatically revert to the NumPy implementation if there is
        not sufficient VRAM. Instead, an out of memory error will be raised.
    """
    parameters = parameters or DeinterlaceParameters()
    parameters.validate_with_images(images)

    # Set implementations for calculations
    match (parameters.align, parameters.use_gpu):
        case ("pixel", False):
            calculate_matrix = partial(calculate_offset_matrix, fft_module=np)
            find_peak = partial(find_pixel_offset, subsearch=parameters.subsearch)
            calculate_offset = compose(calculate_matrix)(find_peak)
            align_images = align_pixels
        case ("pixel", True):
            calculate_matrix = wrap_cupy(
                partial(calculate_offset_matrix, fft_module=cp), "images"
            )
            find_peaks = partial(find_pixel_offset, subsearch=parameters.subsearch)
            calculate_offset = compose(calculate_matrix)(find_peaks)
            align_images = align_pixels
        case ("subpixel", False):
            calculate_matrix = partial(calculate_offset_matrix, fft_module=np)
            find_peak = partial(find_subpixel_offset, subsearch=parameters.subsearch)
            calculate_offset = compose(calculate_matrix)(find_peak)
            align_images = partial(align_subpixels, fft_module=np)
        case ("subpixel", True):
            calculate_matrix = wrap_cupy(
                partial(calculate_offset_matrix, fft_module=cp), "images"
            )
            find_peak = partial(find_subpixel_offset, subsearch=parameters.subsearch)
            calculate_offset = compose(calculate_matrix)(find_peak)
            align_images = partial(align_subpixels, fft_module=cp)
        case ("variable", False):
            calculate_offset = print
            align_images = print
        case ("variable", True):
            calculate_offset = print
            align_images = print
        case _:  # pragma: no cover
            # NOTE: This should never be reached due to the validation in
            #  DeinterlaceParameters
            msg = (
                f"Invalid combination of align='{parameters.align}' and use_gpu={parameters.use_gpu}. "
                "Align must be either 'pixel' or 'subpixel', and use_gpu must be a boolean."
            )
            raise ValueError(msg)

    # now iterate
    pbar = tqdm(total=images.shape[0], desc="Deinterlacing Images", colour="blue")
    for start, stop in index_image_blocks(
        images, parameters.block_size, parameters.unstable
    ):
        # NOTE: Extraction isn't done inline due to the 'pool' parameter potentially
        #  changing the shape of the images being processed. In some cases this means
        #  the returned block_images will not be views of the original images, but
        #  currently this only occurs when reducing the number of frames to process
        #  through pool.If adding a feature here in the future (e.g., upscaling), one
        #  will need to remember this is no view guarantee here.
        block_images = extract_image_block(images, start, stop, parameters.pool)
        # offset_matrix = calculate_matrix(block_images)
        # offset = find_peak(block_images, offset_matrix, parameters.subsearch)
        offset = calculate_offset(block_images)
        align_images(images, start, stop, offset)
        pbar.update(stop - start)
    pbar.close()
