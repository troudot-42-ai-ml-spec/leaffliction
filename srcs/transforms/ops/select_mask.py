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
        if "analyse" not in ctx or "analyse_results" not in ctx:
            raise Exception("Analyse has to be called before SelectMask!")
        analyse_results: Dict[str, Dict[str, float]] = ctx["analyse_results"]
        ctx["selected_mask"] = {}
        compatness_values = {}

        for channel, results in analyse_results.items():
            area: float = results["area"]
            perimeter: float = results["perimeter"]
            compactness: float = calculate_compactness(area, perimeter)
            compatness_values[channel] = compactness

            print(f"compactness for {channel}: {area}")

        # ctx["selected_mask"][channel] = np.ones([1, 1, 3])
        return img
