import argparse
from typing import Literal
from ...transforms.registry import available_ops

SrcType = Literal["multi", "single"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply leaf image transformations (Part 3)."
    )

    # input / output
    p.add_argument("-path", help="Path to an image")
    p.add_argument("-src", help="Source directory (alternative to single file)")
    p.add_argument("-dst", help="Destination directory (for saving results)")

    # ops
    p.add_argument(
        "--ops",
        type=lambda s: s.split(","),
        default=[
            "gaussian_blur",
            "rgb2lab",
            "otsu",
            "fill_holes",
            "analyse",
            "veins",
        ],
        help=f"Comma-separated list of ops. Available: {', '.join(available_ops())}",
    )

    # display / save
    p.add_argument(
        "--show",
        default=False,
        help="Display results in a grid (if input is single file)",
    )
    p.add_argument("--save", default=False, help="Save the analytics")

    return p.parse_args()


def check_args(args: argparse.Namespace) -> SrcType:
    if not args.path and args.src and args.dst:
        if args.show:
            raise Exception("Impossible to use --show with multiple images.")
        return "multi"
    if args.path and not args.src and not args.dst:
        return "single"
    raise Exception("Invalid -path, -dst, -src handling.")
