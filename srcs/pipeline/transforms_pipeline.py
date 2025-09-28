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


_OP_DEPS: Dict[str, List[str]] = {
    "rgb2lab": [],
    "gaussian_blur": [],
    "otsu": ["rgb2lab"],
    "fill_holes": ["otsu"],
    "analyse": ["fill_holes"],
    "select_mask": ["analyse"],
    "veins": ["otsu"],
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


def extract_variants(
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

    if "crop_blur" in requested_ops:
        variants["crop_blur"] = images["crop_blur"]

    return variants


def process_single_image(  # noqa: C901
    img: np.ndarray,
    path: Path,
    args: Namespace,
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

    if args.mode == "single":
        if args.show == "all":
            show_grid(
                list(variants.values()), list(variants.keys()), max_cols=len(variants)
            )
        elif args.show == "one":
            pcv.plot_image(variants[requested_ops[-1]])
    elif args.mode == "multi":
        # dst/<classe>/<variant>/...
        if args.save == "all":
            for name, variant in variants.items():
                save_path = args.dst / path.parent.name / name
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path / f"{path.stem}{path.suffix}"
                pcv.print_image(variant, str(save_path.absolute()))
        elif args.save == "one":
            save_path = args.dst / path.parent.name / requested_ops[-1]
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / f"{path.stem}{path.suffix}"
            pcv.print_image(variants[requested_ops[-1]], str(save_path.absolute()))


def process(
    task_queue: mp.Queue,
    results_queue: mp.Queue,
    args: Namespace,
    dst: Optional[Path] = None,
) -> None:
    ops = _build_ops(args.ops)
    applied_ops: List[str] = [getattr(op, "name", op.__class__.__name__) for op in ops]
    requested_ops: List[str] = [op for op in args.ops]
    try:
        while True:
            path = task_queue.get()
            if path is None:
                break
            pcv.outputs.clear()
            img, _, _ = pcv.readimage(filename=str(path.absolute()))
            process_single_image(img, path, args, ops, applied_ops, requested_ops)
            results_queue.put(path)
    except Exception as e:
        print(f"Error processing {path}: {e}")
    finally:
        results_queue.put(None)


def pipeline(
    path_list: List[Path],
    args: Namespace,
) -> None:
    task_queue: mp.Queue = mp.Queue()
    results_queue: mp.Queue = mp.Queue()

    try:
        for path in path_list:
            task_queue.put(path)

        num_processes = mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1
        for _ in range(num_processes):
            task_queue.put(None)

        processes = []
        for _ in range(num_processes):
            p = mp.Process(target=process, args=(task_queue, results_queue, args))
            p.start()
            processes.append(p)

        for _ in tqdm(range(len(path_list)), desc="Processing images"):
            results_queue.get()

        for p in processes:
            p.join()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        task_queue.close()
        results_queue.close()
