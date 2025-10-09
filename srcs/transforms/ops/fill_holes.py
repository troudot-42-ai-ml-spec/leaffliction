import numpy as np
from plantcv import plantcv as pcv  # type: ignore
from typing import Dict, Any
from ..registry import register


@register("fill_holes")
class FillHoles:
    name = "fill_holes"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "mask" not in ctx:
            raise Exception("OtsuMask has to be called before FillHoles!")
        masks: Dict[str, np.ndarray] = ctx["mask"]
        ctx["fill_holes"] = {}
        for channel, _img in masks.items():
            ctx["fill_holes"][channel] = pcv.fill_holes(_img)
        return img
