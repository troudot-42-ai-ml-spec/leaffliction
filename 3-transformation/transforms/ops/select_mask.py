import numpy as np
from plantcv import plantcv as pcv
from typing import Dict, Any, Literal
from ..registry import register


@register("select_mask")
class SelectMask:
    name = "select_mask"

    def __init__(self, key="largest"):
        self.key = key

    def apply(self, img, ctx):
        masks = ctx["mask"]
        # for channel, mask in masks.items():
        #     pass
        return img
