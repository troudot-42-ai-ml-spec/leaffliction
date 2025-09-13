from argparse import Namespace
from pathlib import Path
from typing import List
from pipeline.pipeline import pipeline, build_ops
from parsing.args import parse_args, check_args, SrcType
from parsing.parse import get_all_images_path, find_dir_path

if __name__ == "__main__":
    args: Namespace = parse_args()
    src_type: SrcType = check_args(args)

    build_ops(args.ops)

    if src_type == "multi":
        paths: List[Path] = get_all_images_path(args.src)
        try:
            dst_path: Path = find_dir_path(args.dst)
        except Exception:
            Path.mkdir(args.dst)
            dst_path: Path = Path(args.dst)
        pipeline(paths, args=args, dst=dst_path)
    else:
        path: Path = Path(args.path)
        if path.exists():
            paths: List[Path] = [path]
            pipeline(paths, args=args)
        else:
            raise Exception("Image path have to exists.")
