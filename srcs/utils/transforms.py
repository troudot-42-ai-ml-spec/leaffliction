import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from transforms.base import Transformation
from transforms.registry import build, available_ops
from typing import List, Literal, Dict, Any, Set, Generator, Tuple
from plantcv import plantcv as pcv
from utils.plotting.grid import show_grid


_OP_DEPS: Dict[str, List[str]] = {
    "rgb2lab": [],
    "gaussian_blur": [],
    "mask": ["rgb2lab"],
    "fill_holes": ["mask"],
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
        ch = ctx["selected_mask"] if "selected_mask" in ctx else "l"
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


def create_transformed_generator(
    dataset: tf.data.Dataset, ops: List[str]
) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    Generator that yields transformed images.

    Args:
        dataset: Original tf.data.Dataset
        ops: List of transformation operations to apply

    Returns:
        Generator yielding tuples of (image, label)
    """
    _ops = _build_ops(ops)
    requested_ops = [op for op in ops]

    print(f"⏳ Applying transformations: {', '.join(requested_ops)}")

    # Yield transformed samples
    for image, label in dataset.as_numpy_iterator():
        ctx: Dict[str, Any] = {"_images": {"original": image}}
        pcv.outputs.clear()

        _img = image
        for op in _ops:
            _img = op.apply(_img, ctx)
            if _img.shape == (5, 5, 3):
                print(f"⚠️  Transformation {op} returned a dummy image, skipping.")
                _img = image
            try:
                op_name = getattr(op, "name", None) or op.__class__.__name__
                ctx["_images"][op_name] = _img
            except Exception as e:
                print(f"Could not store image for operation {op}: {e}")
                pass

        yield _img, label


def transform_dataset(dataset: tf.data.Dataset, ops: List[str]) -> tf.data.Dataset:
    """
    Transform the dataset to create preprocessed samples.

    Args:
        dataset: Original tf.data.Dataset
        ops: List of transformation operations to apply

    Returns:
        Transformed tf.data.Dataset
    """

    os.makedirs(".tf-cache/transformation", exist_ok=True)
    transformed_dataset = tf.data.Dataset.from_generator(
        lambda: create_transformed_generator(dataset, ops),
        output_signature=dataset.element_spec,
    ).cache(filename=".tf-cache/transformation/")

    # Force evaluation to apply transformations immediately (and apply cardinality)
    cardinality = 0
    for _ in transformed_dataset.as_numpy_iterator():
        cardinality += 1

    transformed_dataset = transformed_dataset.apply(
        tf.data.experimental.assert_cardinality(cardinality)
    )

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
