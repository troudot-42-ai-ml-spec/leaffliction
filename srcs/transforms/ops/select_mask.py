import numpy as np
from typing import Dict
from ..registry import register


WEIGHTS: Dict[str, float] = {"compactness": 1, "smoothness": 0.8}


def calculate_compactness(area: float, perimeter: float) -> float:
    return (4 * np.pi * area) / (perimeter**2 + 1e-6)


def calculate_smoothness(area: float, perimeter: float) -> float:
    return area / (perimeter + 1e-6)


@register("select_mask")
class SelectMask:
    name = "select_mask"

    def __init__(self): ...

    def apply(self, img, ctx):
        if "analyse" not in ctx or "analyse_results" not in ctx:
            raise Exception("Analyse has to be called before SelectMask!")
        analyse_results: Dict[str, Dict[str, float]] = ctx["analyse_results"]
        ctx["selected_mask"] = {}
        scores: Dict[str, float] = {}

        for channel, results in analyse_results.items():
            area: float = results["area"]
            perimeter: float = results["perimeter"]
            compactness: float = calculate_compactness(area, perimeter)
            smoothness: float = calculate_smoothness(area, perimeter)

            score: float = (compactness * WEIGHTS["compactness"]) + (
                smoothness * WEIGHTS["smoothness"]
            )
            scores[channel] = score

        if not scores:
            raise Exception("No valid masks found after analysis!")

        best_channel = max(scores, key=scores.get)
        ctx["selected_mask"] = best_channel
        return img
