import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from transforms.base import Transformation
from transforms.registry import build, available_ops
from typing import List, Literal, Dict, Any, Set
from plantcv import plantcv as pcv
from utils.plotting.grid import show_grid
from tqdm import tqdm


_OP_DEPS: Dict[str, List[str]] = {
    "rgb2lab": [],
    "gaussian_blur": [],
    "otsu": ["rgb2lab"],
    "fill_holes": ["otsu"],
    "analyse": ["fill_holes"],
    "select_mask": ["analyse"],
    "veins": ["select_mask"],
    "remove_background": ["select_mask"],
    "crop": ["select_mask"],
    "crop_blur": ["crop"],
}


def _resolve_deps_for(op: str, ordered: List[str], seen: Set[str]) -> None:
    if op in seen:
        return
    for dep in _OP_DEPS.get(op, []):
        _resolve_deps_for(dep, ordered, seen)
    if op not in ordered:
        ordered.append(op)
    seen.add(op)


def resolve_ops(requested_ops: List[str]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    for op in requested_ops:
        if op not in available_ops():
            raise Exception(f"{op} is not a valid operation!")

    for op in requested_ops:
        _resolve_deps_for(op, ordered, seen)

    return ordered


def _build_ops(ops_list: List[str]) -> List[Transformation]:
    ops: List[Transformation] = []
    ordered_ops = resolve_ops(ops_list)
    for op in ordered_ops:
        ops.append(build(op))
    return ops


def extract_variants(  # noqa: C901
    original_img: np.ndarray,
    ctx: Dict[str, Any],
    applied_ops: List[str],
    requested_ops: List[str],
) -> Dict[str, np.ndarray]:
    variants: Dict[str, np.ndarray] = {"original": original_img}

    if "gaussian_blur" in applied_ops and "gaussian_blur" in ctx:
        variants["gaussian_blur"] = ctx["gaussian_blur"]

    if "lab" in ctx:
        variants["lab_l"] = ctx["lab"]["l"]

    if "mask" in ctx:
        ch = ctx["selected_mask"] if "selected_mask" in ctx else "b"
        variants["mask"] = ctx["mask"][ch]

        if "fill_holes" in ctx:
            variants["fill_holes"] = ctx["fill_holes"][ch]

    if "mask" in ctx and "selected_mask" in ctx:
        selected_ch = ctx["selected_mask"]
        if "fill_holes" in ctx:
            variants["select_mask"] = ctx["fill_holes"][selected_ch]
        else:
            variants["select_mask"] = ctx["mask"][selected_ch]

    if "analyse" in ctx and "selected_mask" in ctx:
        variants["analyse"] = ctx["analyse"][ctx["selected_mask"]]

    if "veins" in ctx:
        variants["veins"] = ctx["veins"]

    images = ctx.get("_images", {})

    if "remove_background" in applied_ops and "remove_background" in images:
        variants["remove_background"] = images["remove_background"]

    if "crop" in applied_ops and "crop" in images:
        variants["crop"] = images["crop"]

    if "crop_blur" in requested_ops:
        variants["crop_blur"] = images["crop_blur"]

    return variants


def process_single_image(  # noqa: C901
    img: np.ndarray,
    show: Literal["all", "one"],
    ops: List[Transformation],
    applied_ops: List[str],
    requested_ops: List[str],
) -> None:
    ctx: Dict[str, Any] = {"_images": {"original": img}}
    for op in ops:
        img = op.apply(img, ctx)
        try:
            op_name = getattr(op, "name", None) or op.__class__.__name__
            ctx["_images"][op_name] = img
        except Exception:
            pass

    variants = extract_variants(
        ctx["_images"]["original"], ctx, applied_ops, requested_ops
    )

    if not variants:
        return

    if show == "all":
        show_grid(
            list(variants.values()), list(variants.keys()), max_cols=len(variants)
        )
    elif show == "one":
        pcv.plot_image(variants[requested_ops[-1]])


def transform_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Transform a single image.

    Args:
        image: Input image
        label: Input label
        ops: List of transformations to apply
        requested_ops: List of requested operations
        applied_ops: List of applied operations

    Returns:
        Transformed image and label
    """
    _img = image.numpy()
    ctx: Dict[str, Any] = {"_images": {"original": _img}}
    for op in _ops:
        _img = op.apply(_img, ctx)
        try:
            op_name = getattr(op, "name", None) or op.__class__.__name__
            ctx["_images"][op_name] = _img
        except Exception:
            pass

    variants = extract_variants(
        ctx["_images"]["original"], ctx, applied_ops, requested_ops
    )

    if requested_ops[-1] in variants:
        _img = variants[requested_ops[-1]]

    return _img


def transform_dataset(dataset: tf.data.Dataset, ops: list[str]) -> tf.data.Dataset:
    """
    Transform the dataset to create preprocessed samples.

    Args:
        dataset: Original tf.data.Dataset

    Returns:
        Transformed tf.data.Dataset
    """
    global _ops, applied_ops, requested_ops
    _ops = _build_ops(ops)
    applied_ops = [getattr(op, "name", op.__class__.__name__) for op in _ops]
    requested_ops = [op for op in ops]

    os.makedirs(".tf-cache/transformation", exist_ok=True)

    # Cache first, then iterate
    transformed_dataset = dataset.map(
        lambda img, label: (
            tf.py_function(transform_image, [img], tf.uint8),
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    for img, label in tqdm(transformed_dataset):
        pass

    transformed_dataset = transformed_dataset.cache(".tf-cache/transformation")

    transformed_dataset.class_names = dataset.class_names

    return transformed_dataset


def transform_one_image(
    path: Path, ops: list[str], show: Literal["all", "one"]
) -> None:
    _ops = _build_ops(ops)
    applied_ops: list[str] = [getattr(op, "name", op.__class__.__name__) for op in _ops]
    requested_ops: list[str] = [op for op in ops]

    img, _, _ = pcv.readimage(filename=str(path.absolute()))
    process_single_image(img, show, _ops, applied_ops, requested_ops)
