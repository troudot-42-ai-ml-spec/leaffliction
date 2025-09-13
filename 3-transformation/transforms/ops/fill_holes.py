import numpy as np
from plantcv import plantcv as pcv
from typing import Dict, Any
from ..registry import register


@register("fill_holes")
class FillHoles:
    name = "fill_holes"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "mask" not in ctx:
            raise Exception("Otsu has to be called before FillHoles!")
        otsu: Dict[str, np.ndarray] = ctx["mask"]
        for channel, _img in otsu.items():
            ctx["mask"][channel] = pcv.fill_holes(_img)
        return img
