from dataclasses import InitVar
from math import inf
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass

from deinterlacing.tools import NDArrayLike

try:
    import cupy as cp
except ImportError:
    cp = np

__all__ = [
    "DeinterlaceParameters",
]


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class DeinterlaceParameters:
    """
    DeinterlaceParameters encapsulates the parameters utilized in
    :func:`deinterlacing <deinterlacing.deinterlacing>` to control the deinterlacing
    process. The structure also contains validation logic to ensure that the
    parameters are appropriate for the provided images.

    :var block_size: f
    :var pool:  f
    :var unstable: f
    :var subsearch: f
    :var align: f
    :var has_turnaround: f
    :var null_edges: f
    :var use_gpu: f
    :var images: f
    """

    block_size: int | None = None
    pool: Literal["mean", "median", "std", "sum", None] = None
    unstable: int | None = None
    subsearch: int | None = 15
    align: Literal["pixel", "subpixel", "variable"] = "pixel"
    # has_turnaround: bool = False
    # null_edges: bool = False
    use_gpu: bool = False
    images: InitVar[NDArrayLike | None] = None

    def __post_init__(self, images: NDArrayLike | None) -> None:
        if images is not None:
            self.validate_with_images(images)

    @field_validator("block_size", "unstable", "subsearch", mode="after")
    @classmethod
    def _validate_positive_integer(cls, value: int | None, ctx: Field) -> int | None:
        """
        Validate that the given value is a positive integer or None.

        :param value: The value to validate, which can be an integer or None.
        :returns: The validated value, or None if the input was None.
        """
        if value is not None and value <= 0:
            raise ParameterError(parameter=ctx.field_name, value=value, limits=(0, inf))
        return value

    def validate_with_images(self, images: NDArrayLike) -> None:
        """
        Validate the parameters against the provided images..

        :param images: The images to validate against.
        :returns: None
        """
        # BLOCK SIZE
        if self.block_size is None:
            self.block_size = images.shape[0]
        if self.block_size > images.shape[0]:
            raise ParameterError(
                parameter="block_size",
                value=self.block_size,
                limits=(1, images.shape[0]),
            )

        # SUBSEARCH
        if self.subsearch is None:
            min_dim = min(images.shape[1:])  # Get the minimum spatial dimension
            self.subsearch = min_dim // 16
        if self.subsearch > min(images.shape[1:]):
            raise ParameterError(
                parameter="subsearch",
                value=self.subsearch,
                limits=(1, min(images.shape[1:]) - 1),
            )

        # UNSTABLE
        if self.unstable is not None and self.unstable > images.shape[0]:
            raise ParameterError(
                parameter="unstable",
                value=self.unstable,
                limits=(0, images.shape[0]),
            )

        # USE GPU
        if self.use_gpu and cp == np:
            msg = "CuPy is not available. GPU acceleration cannot be used."
            raise ValueError(msg)


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
