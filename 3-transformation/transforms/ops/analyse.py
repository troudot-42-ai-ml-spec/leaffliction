import numpy as np
from plantcv import plantcv as pcv
from typing import Dict, Any
from ..registry import register


@register("analyse")
class Analyse:
    name = "analayse"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        # return pcv.analyze.s
        return np.zeros(0)
