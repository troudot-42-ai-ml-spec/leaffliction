from pathlib import Path
import numpy as np
from argparse import Namespace
from transforms.base import Transformation
from transforms.registry import build, available_ops
from typing import List, Optional, Dict, Any, Set
from plantcv import plantcv as pcv
from utils.plotting.grid import show_grid
import multiprocessing as mp
from tqdm import tqdm


_OP_DEPS: Dict[str, Set[str]] = {
    "rgb2lab": set(),
    "otsu": {"rgb2lab"},
    "fill_holes": {"otsu"},
    "analyse": {"fill_holes"},
    "select_mask": {"analyse"},
    "crop": {"select_mask"},
    "veins": {"otsu"},
    "gaussian_blur": set(),
    "remove_background": {"select_mask"},
}


_ALIASES: Dict[str, Set[str]] = {
    "crop_blur": {"remove_background", "crop", "gaussian_blur"},
}


def _expand_aliases(requested_ops: List[str]) -> List[str]:
    expanded: List[str] = []
    for op in requested_ops:
        expanded.extend(_ALIASES.get(op, [op]))
    return expanded


def _resolve_deps_for(op: str, ordered: List[str], seen: Set[str]) -> None:
    if op in seen:
        return
    for dep in _OP_DEPS.get(op, []):
        _resolve_deps_for(dep, ordered, seen)
    if op not in ordered:
        ordered.append(op)
    seen.add(op)


def resolve_ops(requested_ops: List[str]) -> List[str]:
    requested_ops = _expand_aliases(requested_ops)
    ordered: List[str] = []
    seen: Set[str] = set()

    valid_ops = set(available_ops()) | set(_ALIASES.keys())
    for op in requested_ops:
        if op not in valid_ops:
            raise Exception(f"{op} is not a valid operation!")

    for op in requested_ops:
        for real_op in _ALIASES.get(op, [op]):
            _resolve_deps_for(real_op, ordered, seen)

    return ordered


_OPS: List[Transformation] = []


def build_ops(ops_list: List[str]) -> None:
    ordered_ops = resolve_ops(ops_list)
    for op in ordered_ops:
        _OPS.append(build(op))


def _build_ops(ops_list: List[str]) -> List[Transformation]:
    ops: List[Transformation] = []
    ordered_ops = resolve_ops(ops_list)
    for op in ordered_ops:
        ops.append(build(op))
    return ops


def get_ops() -> List[Transformation]:
    return _OPS


def extract_variants(
    original_img: np.ndarray,
    ctx: Dict[str, Any],
    applied_ops: Set[str],
    requested_ops: Set[str],
) -> Dict[str, np.ndarray]:
    variants: Dict[str, np.ndarray] = {"original": original_img}

    if "lab" in ctx:
        variants["lab_l"] = ctx["lab"]["l"]

    if "mask" in ctx and "selected_mask" in ctx:
        selected_ch = ctx["selected_mask"]
        variants["mask_selected"] = ctx["mask"][selected_ch]

    if "analyse" in ctx and "selected_mask" in ctx:
        variants["selected_analysed"] = ctx["analyse"][ctx["selected_mask"]]

    if "veins" in ctx and "selected_mask" in ctx:
        variants["veins_selected"] = ctx["veins"][ctx["selected_mask"]]

    images = ctx.get("_images", {})

    if "remove_background" in applied_ops and "remove_background" in images:
        variants["remove_background"] = images["remove_background"]

    if "crop" in applied_ops and "crop" in images:
        variants["crop"] = images["crop"]

    if "crop_blur" in requested_ops and "gaussian_blur" in images:
        variants["crop_blur"] = images["gaussian_blur"]
    elif "gaussian_blur" in applied_ops and "gaussian_blur" in images:
        variants["gaussian_blur"] = images["gaussian_blur"]

    return variants


def process_single_image(
    img: np.ndarray,
    path: Path,
    args: Namespace,
    ops: List[Transformation],
    applied_ops: Set[str],
    requested_ops: Set[str],
    dst: Optional[Path] = None,
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
        ctx["_images"]["original"],
        ctx,
        applied_ops,
        requested_ops
    )

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


def process(
    task_queue: mp.Queue,
    results_queue: mp.Queue,
    args: Namespace,
    dst: Optional[Path] = None
) -> None:
    ops = _build_ops(args.ops)
    applied_ops: Set[str] = {getattr(op, "name", op.__class__.__name__) for op in ops}
    requested_ops: Set[str] = {op for op in args.ops}
    while True:
        path = task_queue.get()
        if path is None:
            break
        pcv.outputs.clear()
        img, _, _ = pcv.readimage(filename=str(path.absolute()))
        process_single_image(img, path, args, ops, applied_ops, requested_ops, dst)
        results_queue.put(path)


def pipeline(
    path_list: List[Path],
    args: Namespace,
    dst: Optional[Path] = None
) -> None:
    if not _OPS:
        build_ops(args.ops)

    task_queue: mp.Queue = mp.Queue()
    results_queue: mp.Queue = mp.Queue()

    for path in path_list:
        task_queue.put(path)

    num_processes = mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1
    for _ in range(num_processes):
        task_queue.put(None)

    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=process, args=(task_queue, results_queue, args, dst))
        p.start()
        processes.append(p)

    for _ in tqdm(range(len(path_list)), desc="Processing images"):
        results_queue.get()

    for p in processes:
        p.join()
