import numpy as np
from plantcv import plantcv as pcv
from typing import Dict, Any, Literal
from ..registry import register


@register("rgb2lab")
class Rgb2Lab:
    name = "rgb2lab"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "lab" not in ctx:
            ctx["lab"] = {}
            for channel in "lab":
                ctx["lab"][channel] = pcv.rgb2gray_lab(img, channel)
        return img
