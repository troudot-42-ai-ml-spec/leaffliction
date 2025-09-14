import numpy as np
from plantcv import plantcv as pcv
from typing import Dict, Any
from ..registry import register


@register("gaussian_blur")
class GaussianBlur:
    name = "gaussian_blur"

    def __init__(self, ksize: int = 3) -> None:
        self.ksize = (ksize, ksize)

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        return pcv.gaussian_blur(img=img, ksize=self.ksize)
