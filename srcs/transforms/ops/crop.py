import numpy as np
from typing import Dict, Any
from ..registry import register


@register("crop")
class Crop:
    name = "crop"

    def __init__(self, margin: int = 0) -> None:
        self.margin = margin

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        h, w = img.shape[:2]

        if "selected_mask" not in ctx or "analyse_results" not in ctx:
            raise ValueError(
                "Crop operation requires selected_mask and analyse_results in context"
            )

        selected_mask = ctx["selected_mask"]
        mask_data = ctx["analyse_results"][selected_mask]

        center_x = int(mask_data["centroid_x"])
        center_y = int(mask_data["centroid_y"])
        mask_width = int(mask_data["width"])
        mask_height = int(mask_data["height"])

        half_width = (mask_width // 2) + self.margin
        half_height = (mask_height // 2) + self.margin

        x1 = max(0, center_x - half_width)
        y1 = max(0, center_y - half_height)
        x2 = min(w, center_x + half_width)
        y2 = min(h, center_y + half_height)

        cropped_img = img[y1:y2, x1:x2]

        return cropped_img
