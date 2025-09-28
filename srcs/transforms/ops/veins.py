import numpy as np
from plantcv import plantcv as pcv  # type: ignore
from typing import Dict, Any
from ..registry import register


@register("veins")
class Veins:
    name = "veins"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "lab" not in ctx or "mask" not in ctx:
            raise Exception("Rgb2Lab and Mask has to be called before Veins!")
        if "l" not in ctx["lab"]:
            raise Exception("Lab has to have l channel!")
        labs: Dict[str, np.ndarray] = ctx["lab"]
        ct: Dict[str, np.ndarray] = {}
        ctx["veins"] = ct
        for channel, _img in labs.items():
            mask_applied_otsu = pcv.apply_mask(
                ctx["lab"]["l"], ctx["mask"][channel], "black"
            )

            viz = pcv.stdev_filter(img=mask_applied_otsu, ksize=7, borders="nearest")
            _, masked_img = pcv.threshold.custom_range(viz, [9], [255])
            ct[channel] = masked_img
        return img
