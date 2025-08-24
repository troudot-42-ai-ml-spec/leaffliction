import shutil
from pathlib import Path
from collections import defaultdict


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
                f"❌ Expected only subdirectories in '{input_path.name}',\
                     but found file: '{item.name}'"
            )

    if len(grouped_dirs) < 2:
        raise ValueError(
            f"❌ Expected at least 2 groups of subdirectories \
                (e.g., 'Apple_*', 'Grape_*'), but found {len(grouped_dirs)}."
        )

    print("✅ Directory structure is valid. Organizing...")

    for prefix, dir_list in grouped_dirs.items():
        destination_folder = augmented_dir / prefix
        destination_folder.mkdir(exist_ok=True)

        print(f"\nProcessing group '{prefix}':")
        for source_dir in dir_list:
            destination = destination_folder / source_dir.name
            shutil.copytree(source_dir, destination, dirs_exist_ok=True)
            print(f"  -> Copied '{source_dir.name}' to '{destination_folder.name}'")

    return augmented_dir