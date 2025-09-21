from pathlib import Path
import numpy as np
from argparse import Namespace
from transforms.base import Transformation
from transforms.registry import build, available_ops
from typing import List, Optional, Dict, Any
from plantcv import plantcv as pcv
from utils.plotting.grid import show_grid
import time

_OPS: List[Transformation] = []


def build_ops(ops_list: List[str]) -> None:
    for op in ops_list:
        if op not in available_ops():
            raise Exception(f"{op} is not a valid operation!")
        _OPS.append(build(op))


def get_ops() -> List[Transformation]:
    return _OPS


def extract_variants(
    original_img: np.ndarray, ctx: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    if (
        "lab" not in ctx
        or "mask" not in ctx
        or "selected_mask" not in ctx
        or "veins" not in ctx
    ):
        raise RuntimeError("Context does not contain required keys.")

    selected_ch = ctx["selected_mask"]
    lab_l = ctx["lab"]["l"]
    mask_selected = ctx["mask"][selected_ch]
    veins_selected = ctx["veins"][selected_ch]
    selected_analysed_img = ctx["analyse"][selected_ch]
    remove_background = pcv.apply_mask(original_img, mask_selected, mask_color="white")
    ctx["remove_background"] = remove_background
    crop_img = build("crop").apply(remove_background, ctx)
    crop_blur_img = build("gaussian_blur").apply(crop_img, ctx)

    return {
        "original": original_img,
        "lab_l": lab_l,
        "mask_selected": mask_selected,
        "veins_selected": veins_selected,
        "selected_analysed": selected_analysed_img,
        "remove_background": remove_background,
        "crop": crop_img,
        "crop_blur": crop_blur_img,
    }


def pipeline(
    path_list: List[Path], args: Namespace, dst: Optional[Path] = None
) -> None:
    if not _OPS:
        raise Exception("You have build first the ops!")

    for path in path_list:
        start_time = time.time()
        img, _, _ = pcv.readimage(filename=str(path.absolute()))
        ctx: Dict[str, Any] = {}
        for op in _OPS:
            img = op.apply(img, ctx)

        variants = extract_variants(img, ctx)

        if args.show:
            show_grid(
                list(variants.values()), list(variants.keys()), max_cols=len(variants)
            )
        if dst:
            for name, variant in variants.items():
                # dst/<classe>/<variant>/...
                save_path = dst / path.parent.name / name
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path / f"{path.stem}{path.suffix}"
                pcv.print_image(variant, str(save_path.absolute()))
            print(f"Saved {path.name} in {time.time() - start_time:.2f}s")
        pcv.outputs.clear()
