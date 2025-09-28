from pathlib import Path
from typing import List


def get_all_images_path(dir_path: str) -> List[Path]:
    """
    Given a directory path (string), find all .JPG images inside it (recursively).

    Args:
        dir_path (str): Input directory path (can be relative, absolute, or ~ expanded).

    Returns:
        List[Path]: A list of Path objects pointing to .JPG files.
    """
    base: Path = Path(dir_path).expanduser().resolve()
    paths_list: List[Path] = []

    for root, _, files in base.walk():
        if not files:
            continue

        for file in files:
            if not file.upper().endswith(".JPG"):
                continue
            img_path: Path = root / file
            paths_list.append(img_path)

    return paths_list
