from pathlib import Path
from typing import List, Optional


def find_dir_path(path_str: str, search_root: Optional[Path] = None) -> Path:
    """
    Resolve a given path string to an existing directory,
    handling ~, relative, and absolute paths. Falls back to searching.

    Args:
        path_str (str): Input path (can be absolute, relative, or with ~).
        search_root (Path, optional): Root to search if path_str is ambiguous.
                                      Defaults to current working dir.
    Returns:
        Path: Resolved directory path if found, else an exception is raised!
    """
    p = Path(path_str).expanduser()

    if p.exists():
        return p.resolve()

    if not search_root:
        search_root = Path.cwd()

    for found in search_root.rglob(p.name):
        if found.is_dir():
            return found.resolve()

    raise Exception("Directory not found!")


def get_all_images_path(dir_path: str) -> List[Path]:
    """
    Given a directory path (string), find all .JPG images inside it (recursively).

    Args:
        dir_path (str): Input directory path (can be relative, absolute, or ~ expanded).

    Returns:
        List[Path]: A list of Path objects pointing to .JPG files.
    """
    base: Path = find_dir_path(dir_path)
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
