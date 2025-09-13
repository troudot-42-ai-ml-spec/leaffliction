import numpy as np
from typing import Dict
from ..registry import register


def calculate_compactness(area: float, perimeter: float) -> float:
    return (4 * np.pi * area) / (perimeter**2 + 1e-6)


@register("select_mask")
class SelectMask:
    name = "select_mask"

    def __init__(self): ...

    def apply(self, img, ctx):
        if "analyse" not in ctx or "analyse_value" not in ctx:
            raise Exception("Analyse has to be called before SelectMask!")
        analyse_values: Dict[str, str] = ctx["analyse_value"]
        ctx["selected_mask"] = {}
        for channel, val in analyse_values.items():
            # TODO: fix that only one analysed img is stored in pcv.obs\
            # TODO:     instead of a dict of analysed img
            break
            # area: float = obs[val]["area"]["value"]
            # perimeter: float = obs[val]["perimeter"]["value"]
            # compactness: float = calculate_compactness(area, perimeter)
            # compatness_values[channel] = compactness

        # ctx["selected_mask"][channel] = np.ones([1, 1, 3])
        return img
