from argparse import Namespace
from pathlib import Path
from typing import List
from pipeline.transforms_pipeline import pipeline
from utils.parsing.args import parse_args
from utils.parsing.get_all_images_path import get_all_images_path

if __name__ == "__main__":
    args: Namespace = parse_args()

    if args.mode == "multi":
        paths: List[Path] = get_all_images_path(args.src)
        pipeline(paths, args=args)
    elif args.mode == "single":
        path: Path
        path = Path(args.path)
        if path.exists():
            paths = [path]
            pipeline(paths, args=args)
        else:
            raise Exception("Image path have to exists.")
    else:
        raise Exception("Invalid source type.")
