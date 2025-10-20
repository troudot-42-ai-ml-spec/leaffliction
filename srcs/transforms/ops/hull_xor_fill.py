import numpy as np
import cv2
from typing import Dict, Any
from ..registry import register


@register("hull_xor_fill")
class HullXorFill:
    name = "hull_xor_fill"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "mask" not in ctx or "selected_mask" not in ctx:
            raise Exception("SelectMask has to be called before HullXorFill!")

        selected_channel: str = ctx["selected_mask"]
        original_mask: np.ndarray = ctx["mask"][selected_channel]

        original_dtype = original_mask.dtype
        mask_uint8 = original_mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise Exception(f"No contours found in selected mask '{selected_channel}'")

        hull_mask = np.zeros_like(mask_uint8)

        for contour in contours:
            hull = cv2.convexHull(contour)
            cv2.drawContours(hull_mask, [hull], 0, 255, cv2.FILLED)

        xor_result = cv2.bitwise_xor(hull_mask, mask_uint8)

        corrected_mask = cv2.bitwise_or(mask_uint8, xor_result)

        corrected_mask = corrected_mask.astype(original_dtype)
        hull_mask = hull_mask.astype(original_dtype)
        xor_result = xor_result.astype(original_dtype)

        ctx["mask"][selected_channel] = corrected_mask

        ctx["hull_mask"] = hull_mask
        ctx["hull_xor_result"] = xor_result

        return img
