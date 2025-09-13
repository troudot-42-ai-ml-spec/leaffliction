import numpy as np
from plantcv import plantcv as pcv
from typing import Dict, Any
from ..registry import register


@register("analyse")
class Analyse:
    name = "analyse"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "mask" not in ctx:
            raise Exception("OtsuMask has to be called before Analyse!")
        masks: Dict[str, np.ndarray] = ctx["mask"]
        pcv.params.sample_label = "leaf"
        ctx["analyse"] = {}
        ctx["analyse_value"] = {}
        for i, (channel, mask) in enumerate(masks.items(), start=1):
            ctx["analyse_value"][channel] = f"{pcv.params.sample_label}_{i}"
            ctx["analyse"][channel] = pcv.analyze.size(img, mask, n_labels=i)
        return img
