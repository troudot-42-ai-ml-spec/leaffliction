from pathlib import Path


def check_dir(input_path: str) -> None:
    """
    Check if the directory passed as parameter is correctly structured.
    """
    directory = Path(input_path)
    max_depth = 0
    subdir_cats = ["Apple", "Grape"]
    for root, dirs, files in directory.walk():
        depth = len(root.relative_to(directory).parts)
        if depth > max_depth:
            max_depth = depth
        if max_depth > 2:
            raise Exception(f"❌ Directory {input_path} is not formated correctly:\n\
Needs only one layer of subdirectories containing images.")
        for file in files:
            file_path = root / file
            if file_path.suffix != ".JPG":
                raise Exception(f"❌ File {file} is not .JPG")
        for subdir in dirs:
            subdir_path = root / subdir
            if "_" in subdir_path.name:
                prefix = subdir_path.name.split("_")[0]
                if prefix not in subdir_cats:
                    raise Exception(f"❌ Subdir {subdir} doesn't respect the naming\
 convention:\nPrefix neither Apple nor Grape.")
            else:
                raise Exception(f"❌ Subdir {subdir} doesn't respect the naming\
 convention:\nNeeds <Apple/Grape>_<condition>.")
