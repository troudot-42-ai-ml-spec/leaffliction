import argparse
import cv2
import shutil
import random
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def get_augmentations():
    """
    Returns a dictionary of specific, named augmentations.
    """
    return {
        "Flip": A.HorizontalFlip(p=1.0),
        "Rotate": A.Rotate(limit=45, p=1.0, border_mode=cv2.BORDER_CONSTANT),
        "Skew": A.Affine(
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            translate_percent={"x": (-0.1, 0.1)},
            p=1.0,
        ),
        "Shear": A.Affine(shear={"x": (-25, 25)}, p=1.0),
        "Crop": A.CenterCrop(height=200, width=200, p=1.0),
        "Distortion": A.Perspective(scale=(0.05, 0.1), p=1.0),
    }


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


def process_file(input_path: Path):
    """
    Augment image using different Albumentations methods, then display it.
    """
    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise Exception(f"Could not read image file: {input_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    augs = get_augmentations()
    augmented_images = {}
    for name, aug in augs.items():
        transform = A.Compose([aug])
        augmented_images[name] = transform(image=image_rgb)["image"]
        augmented_bgr = cv2.cvtColor(augmented_images[name], cv2.COLOR_RGB2BGR)
        path_without_extension = input_path.parent / input_path.stem
        cv2.imwrite(
            f"{path_without_extension}_{name}{input_path.suffix}", augmented_bgr
        )

    print("✅ Saved all augmented images.")

    fig, axes = plt.subplots(1, len(augmented_images) + 1, figsize=(12, 3))
    fig.suptitle(f"Augmentations for {input_path.name}")
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    img_list = list(augmented_images.items())
    for i in range(len(img_list)):
        axes[i + 1].imshow(img_list[i][1])
        axes[i + 1].set_title(img_list[i][0])
        axes[i + 1].axis("off")

    plt.show()


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
        if item.is_dir():
            if "_" in item.name:
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


def process_dir(new_path: Path):
    """
    Defines the biggest subdirectory, then balances the others using\
         augmentation to match the number of elements.
    """
    max_count = 0
    folder_counts = {}
    for root, dirs, files in new_path.walk():
        if not dirs:
            folder_name = Path(root).name
            image_files = [f for f in files if f != ".DS_Store"]
            current_count = len(image_files)
            folder_counts[folder_name] = current_count
            if current_count > max_count:
                max_count = current_count

    aug = list(get_augmentations().items())
    for root, dirs, files in new_path.walk():
        if not dirs:
            current_path = Path(root)
            folder_name = current_path.name
            num_to_generate = max_count - folder_counts[folder_name]
            if num_to_generate <= 0:
                continue
            image_paths = [current_path / f for f in files if f != ".DS_Store"]
            for i in range(num_to_generate):
                source_image_path = random.choice(image_paths)
                aug_name, aug_obj = random.choice(aug)
                image_bgr = cv2.imread(str(source_image_path))
                if image_bgr is None:
                    raise Exception(f"Could not read image file: {current_path}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                transform = A.Compose([aug_obj])
                new_image = transform(image=image_rgb)["image"]
                augmented_bgr = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                counter = 1
                new_path = (
                    current_path
                    / f"{source_image_path.stem}_{aug_name}_{counter}\
                        {source_image_path.suffix}"
                )
                while new_path.exists():
                    counter += 1
                    new_path = (
                        current_path
                        / f"{source_image_path.stem}_{aug_name}_{counter}\
                            {source_image_path.suffix}"
                    )
                cv2.imwrite(str(new_path), augmented_bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        help="The path to the image you are trying to augment,\
             or directory you are trying to balance.",
    )
    args = parser.parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: The path '{input_path}' does not exist.")
        return
    try:
        if input_path.is_file():
            parse_file(input_path)
            process_file(input_path)
        elif input_path.is_dir():
            new_path = parse_dir(input_path)
            process_dir(new_path)
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
