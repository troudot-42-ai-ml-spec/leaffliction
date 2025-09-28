import numpy as np
from typing import Dict, Any
from ..registry import register
from plantcv import plantcv as pcv


@register("crop_blur")
class CropBlur:
    name = "crop_blur"

    def __init__(self, blur_kernel: int = 7) -> None:
        self.blur_kernel = blur_kernel

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        kernel_size = (
            self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
        )

        blurred_img = pcv.gaussian_blur(img, (kernel_size, kernel_size))

        return blurred_img
