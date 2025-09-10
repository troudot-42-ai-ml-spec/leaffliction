import math
import shutil
import random
from pathlib import Path


def split_data(dataset_path: str) -> Path:
    """
        Split dataset into a training set and a test set.
    """
    input_path = Path(dataset_path)
    test_set = input_path.parent / "test_set"
    test_set.mkdir(exist_ok=True)
    train_set = input_path.parent / "train_set"
    train_set.mkdir(exist_ok=True)

    split_ratio = 0.8

    print("Starting dataset split...")

    for dirpath, dirnames, filenames in input_path.walk():
        current_dir = Path(dirpath)
        relative_path = current_dir.relative_to(input_path)
        train_set_dir = train_set / relative_path
        test_set_dir = test_set / relative_path
        train_set_dir.mkdir(exist_ok=True)
        test_set_dir.mkdir(exist_ok=True)

        if not filenames:
            continue

        print(f"Processing {len(filenames)} files in '{current_dir}'...")

        random.shuffle(filenames)
        split_point = math.ceil(len(filenames) * split_ratio)
        train_files = filenames[:split_point]
        print(f"{len(train_files)} files reserved for train_set.")
        test_files = filenames[split_point:]
        print(f"{len(test_files)} files reserved for test_set.")

        for filename in train_files:
            source_file = current_dir / filename
            dest_file = train_set_dir / filename
            shutil.move(source_file, dest_file)

        for filename in test_files:
            source_file = current_dir / filename
            dest_file = test_set_dir / filename
            shutil.move(source_file, dest_file)

    print("\nDataset split complete! ✨")
    print(f"Train set created at: {train_set}")
    print(f"Test set created at: {test_set}")
    return train_set, test_set
