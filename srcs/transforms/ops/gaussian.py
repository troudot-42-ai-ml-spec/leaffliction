import numpy as np
from plantcv import plantcv as pcv  # type: ignore
from typing import Dict, Any
from ..registry import register


@register("gaussian_blur")
class GaussianBlur:
    name = "gaussian_blur"

    def __init__(self, ksize: int = 5) -> None:
        self.ksize = (ksize, ksize)

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if ctx.keys() != set(["_images"]):
            raise ValueError(
                "Gaussian blur requires be called right after image loading"
            )
        ctx["gaussian_blur"] = pcv.gaussian_blur(img=img, ksize=self.ksize)
        return img
