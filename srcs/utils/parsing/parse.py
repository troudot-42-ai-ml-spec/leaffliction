from pathlib import Path
from argparse import Namespace
from typing import List, Dict
import attrs
import shutil
from collections import defaultdict


@attrs.define(frozen=True, slots=True)
class Config:
    types: List[str] = attrs.field(default=["Grape", "Apple"])


@attrs.define(slots=True)
class Type:
    name: str
    labels: Dict[str, int] = attrs.field(factory=dict)


def parse(args: Namespace, cfg: Config) -> List[Type]:
    types: Dict[str, Type] = {type: Type(name=type) for type in cfg.types}

    for root, dirs, files in Path(args.path).walk(on_error=print):
        separator_i: int = root.name.find("_")
        name: str = root.name[:separator_i]
        if name in cfg.types:
            types[name].labels[root.name[separator_i + 1 :]] = len(files)
    return [type for type in types.values()]


def parse_file(input_path: Path):
    """
    Verify that the file we are trying to augment is a JPG image.
    """
    file_extension = input_path.suffix
    if file_extension != ".JPG":
        raise Exception(
            f"Input path {input_path} leads to an unsupported file with\
                 {file_extension} extension. Supported extensions: .JPG"
        )


def parse_dir(input_path: Path) -> Path:
    """
    Scans a directory for subfolders, groups them by a common prefix
    (e.g., 'Apple_' from 'Apple_healthy'), and copies them into a new,
    organized 'augmented_directory'.
    """
    augmented_dir = input_path.parent / "augmented_directory"
    augmented_dir.mkdir(exist_ok=True)

    grouped_dirs = defaultdict(list)
    for item in input_path.iterdir():
        if item.is_dir() and "_" in item.name:
            prefix = item.name.split("_")[0]
            grouped_dirs[prefix].append(item)
        elif item.name != ".DS_Store":
            raise ValueError(
                f"❌ Expected only subdirectories of format\
 <Apple/Grape>_<condition> in '{input_path.name}', but found: '{item.name}'"
            )
    if len(grouped_dirs) < 2:
        raise ValueError(
            f"❌ Expected at least 2 groups of subdirectories \
                (e.g., 'Apple_*', 'Grape_*'), but found {len(grouped_dirs)}."
        )
    print("✅ Directory structure is valid.")
    print(f"⏳ Copying folders to '{augmented_dir.name}'...")
    for prefix, dir_list in grouped_dirs.items():
        for source_dir in dir_list:
            destination = augmented_dir / source_dir.name
            shutil.copytree(source_dir, destination, dirs_exist_ok=True)

    return augmented_dir
