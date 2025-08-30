import transforms.ops
from transforms.base import Transformation
from transforms.registry import build, available_ops
from typing import List, Dict, Any
from plantcv import plantcv as pcv
from viz.plotting import show_grid

img, _, _ = pcv.readimage(
    filename=f"/Users/troudot/42-cursus/leaffliction/images/Apple/Apple_rust/image (10).JPG",
    mode="rgb",
)

ops: List[Transformation] = [
    build("gaussian_blur", ksize=11),
    build("rgb2lab"),
    build("otsu"),
    build("fill_holes"),
]

ctx: Dict[str, Any] = {}

for op in ops:
    img = op.apply(img, ctx)

show_grid(
    list(ctx["lab"].values()) + list(ctx["otsu"].values()),
    list(ctx["lab"].keys()) + list(ctx["otsu"].keys()),
)
