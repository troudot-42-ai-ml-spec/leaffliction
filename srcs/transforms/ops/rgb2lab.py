import numpy as np
from plantcv import plantcv as pcv  # type: ignore
from typing import Dict, Any
from ..registry import register


@register("rgb2lab")
class Rgb2Lab:
    name = "rgb2lab"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "lab" not in ctx:
            ctx["lab"] = {}
            _img = img.copy() if "gaussian_blur" not in ctx else ctx["gaussian_blur"]
            for channel in "lab":
                ctx["lab"][channel] = pcv.rgb2gray_lab(_img, channel)
        return img
