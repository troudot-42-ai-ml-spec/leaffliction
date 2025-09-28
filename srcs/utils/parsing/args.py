import argparse
from pathlib import Path
from transforms.registry import available_ops


def validate_src_path(s: str, should_exist: bool = True) -> Path:
    path = Path(s).expanduser().resolve()
    if path.exists():
        return path
    elif not should_exist and not path.exists():
        return path
    else:
        raise ValueError(f"Path {s} does not exist")


def config_single_parser(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "-path", type=validate_src_path, required=True, help="Path to an image"
    )
    _ = parser.add_argument(
        "--ops",
        type=lambda s: s.split(","),
        default=[
            "gaussian_blur",
            "rgb2lab",
            "veins",
            "otsu",
            "fill_holes",
            "analyse",
            "select_mask",
            "remove_background",
            "crop",
            "crop_blur",
        ],
        help=f"Comma-separated list of ops. Available: {', '.join(available_ops())}",
    )
    _ = parser.add_argument(
        "--show",
        type=str,
        choices=["all", "one"],
        default="all",
        help="Choice to display all ops or just the last one",
    )


def config_multi_parser(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "-src",
        type=validate_src_path,
        required=True,
        help="Source directory (for loading images)",
    )
    _ = parser.add_argument(
        "-dst",
        type=lambda path: validate_src_path(path, should_exist=False),
        required=True,
        help="Destination directory (for saving results)",
    )
    _ = parser.add_argument(
        "--ops",
        type=lambda s: s.split(","),
        default=[
            "gaussian_blur",
            "rgb2lab",
            "otsu",
            "fill_holes",
            "analyse",
            "select_mask",
            "remove_background",
            "crop",
            "crop_blur",
        ],
        help=f"Comma-separated list of ops. Available: {', '.join(available_ops())}",
    )
    _ = parser.add_argument(
        "--save",
        type=str,
        choices=["all", "one"],
        default="one",
        help="Choice to save all ops or just the last one",
    )
    _ = parser.add_argument(
        "--split",
        action="store_true",
        help="Split mode for multiple images",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply leaf image transformations (Part 3)."
    )
    subparsers = p.add_subparsers(
        title="mode", description="modes", help="additional help"
    )
    parser_single = subparsers.add_parser("single", help="Single image mode")
    parser_multi = subparsers.add_parser("multi", help="Multiple images mode")

    config_single_parser(parser_single)
    config_multi_parser(parser_multi)

    args = p.parse_args()

    try:
        if not args.path or args.path:
            pass
    except:  # noqa: E722
        args.mode = "multi"
    else:
        args.mode = "single"

    return args
