import numpy as np
from plantcv import plantcv as pcv
from typing import Dict, Any, Literal
from ..registry import register


@register("otsu")
class Otsu:
    name = "otsu"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "lab" not in ctx:
            raise Exception("Rgb2Lab has to be called before Otsu!")
        lab: Dict[str, np.ndarray] = ctx["lab"]
        ctx["mask"] = {}
        for channel, img in lab.items():
            type = "light" if channel == "b" else "dark"
            ctx["mask"][channel] = pcv.threshold.otsu(img, type)
        return img
