import numpy as np
from plantcv import plantcv as pcv  # type: ignore
from typing import Dict, Any
from ..registry import register


@register("mask")
class Mask:
    name = "mask"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "lab" not in ctx:
            raise Exception("Rgb2Lab has to be called before MaskOtsu!")
        lab: Dict[str, np.ndarray] = ctx["lab"]
        ctx["mask"] = {}
        for channel, _img in lab.items():
            type = "light" if channel == "b" else "dark"
            ctx["mask"][channel] = pcv.threshold.otsu(_img.astype(np.uint16), type)
        return img
