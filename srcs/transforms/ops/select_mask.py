import numpy as np
from typing import Dict, Any
from ..registry import register


WEIGHTS: Dict[str, float] = {"compactness": 1, "smoothness": 0.8}


def calculate_compactness(area: float, perimeter: float) -> float:
    return (4 * np.pi * area) / (perimeter**2 + 1e-6)


def calculate_smoothness(area: float, perimeter: float) -> float:
    return area / (perimeter + 1e-6)


@register("select_mask")
class SelectMask:
    name = "select_mask"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "analyse" not in ctx or "analyse_results" not in ctx:
            raise Exception("Analyse has to be called before SelectMask!")
        analyse_results: Dict[str, Dict[str, float]] = ctx["analyse_results"]
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

        best_channel: str = max(scores, key=lambda k: scores[k])
        ctx["selected_mask"] = best_channel
        return img
