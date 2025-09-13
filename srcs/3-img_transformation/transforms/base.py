from typing import Protocol, Dict, Any
import numpy as np


class Transformation(Protocol):
    name: str

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray: ...
