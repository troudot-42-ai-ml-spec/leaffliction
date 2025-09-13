import numpy as np
from pathlib import Path
from argparse import Namespace
from transforms.base import Transformation
from transforms.registry import build, available_ops
from typing import List, Optional, Dict, Any
from plantcv import plantcv as pcv
from viz.plotting import show_grid

_OPS: List[Transformation] = []


def build_ops(ops_list: List[str]) -> None:
    for op in ops_list:
        if op not in available_ops():
            raise Exception(f"{op} is not a valid operation!")
        _OPS.append(build(op))


def get_ops() -> List[Transformation]:
    return _OPS


def pipeline(
    path_list: List[Path], args: Namespace, dst: Optional[Path] = None
) -> None:
    if not _OPS:
        raise Exception("You have build first the ops!")

    for path in path_list:
        img, _, _ = pcv.readimage(filename=str(path.absolute()))
        ctx: Dict[str, Any] = {}
        for op in _OPS:
            img = op.apply(img, ctx)

        if args.show:
            imgs: List[np.ndarray] = []
            labels: List[str] = []
            for op in ctx:
                for ch, img in ctx[op].items():
                    if type(img) is not np.ndarray:
                        continue
                    imgs.append(img)
                    labels.append(f"{op}_{ch}")
            show_grid(imgs, labels, max_cols=len(ctx["lab"]))

        pcv.outputs.clear()
