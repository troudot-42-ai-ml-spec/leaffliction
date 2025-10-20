import numpy as np
from typing import Dict, Any
from plantcv import plantcv as pcv  # type: ignore
from ..registry import register


@register("remove_background")
class RemoveBackground:
    name = "remove_background"

    def __init__(self, mask_color: str = "white") -> None:
        self.mask_color = mask_color

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "mask" not in ctx or "selected_mask" not in ctx:
            raise Exception("SelectMask has to be called before RemoveBackground!")
        selected_channel: str = ctx["selected_mask"]
        mask = (
            ctx["mask"][selected_channel]
            if "corrected_mask" not in ctx
            else ctx["corrected_mask"]
        )
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "Cannot remove background from image which has been resized.",
                "Please adjust the order of operations.",
            )
        masked_img = pcv.apply_mask(img=img, mask=mask, mask_color=self.mask_color)
        ctx["remove_background"] = masked_img
        return masked_img
