import argparse
import cv2
import numpy as np
import shutil
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path

def get_random_augmentation():
    """
        Applies one random augmentation from the dict.
    """
    return A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=45, p=1.0, border_mode=cv2.BORDER_CONSTANT),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent={'x': (-0.1, 0.1)}, p=1.0),
            A.Affine(shear={'x': (-25, 25)}, p=1.0),
            A.CenterCrop(height=200, width=200, p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0)
        ], p=1.0)
    ])

def get_augmentations():
    """
    Returns a dictionary of specific, named augmentations.
    """
    return {
        "Flip": A.HorizontalFlip(p=1.0),
        "Rotate": A.Rotate(limit=45, p=1.0, border_mode=cv2.BORDER_CONSTANT),
        "Skew": A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent={'x': (-0.1, 0.1)}, p=1.0),
        "Shear": A.Affine(shear={'x': (-25, 25)}, p=1.0),
        "Crop": A.CenterCrop(height=200, width=200, p=1.0),
        "Distortion": A.Perspective(scale=(0.05, 0.1), p=1.0),
    }

def parse_file(input_path: Path):
    """
        Verify that the file we are trying to augment is a JPG image.
    """
    file_extension = input_path.suffix
    if file_extension != '.JPG':
        raise Exception(f"Input path {input_path} leads to an unsupported file with {file_extension} extension. Supported extensions: .JPG")

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
        augmented_images[name] = transform(image=image_rgb)['image']
        augmented_bgr = cv2.cvtColor(augmented_images[name], cv2.COLOR_RGB2BGR)
        path_without_extension = input_path.parent / input_path.stem
        cv2.imwrite(f"{path_without_extension}_{name}{input_path.suffix}", augmented_bgr)

    print("✅ Saved all augmented images.")

    fig, axes = plt.subplots(1, len(augmented_images) + 1, figsize=(12,3))
    fig.suptitle(f"Augmentations for {input_path.name}")
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')

    img_list = list(augmented_images.items())
    for i in range(len(img_list)):
        axes[i+1].imshow(img_list[i][1])
        axes[i+1].set_title(img_list[i][0])
        axes[i+1].axis('off')
    
    plt.show()
    

def parse_dir(input_path: Path):
    """
        Verify directory validity, to reorganise it for processing.
    """
    items_list = list(input_path.iterdir())
    for item in items_list:
        if item.is_file():
            if item.name == ".DS_Store":
                continue
            raise Exception(f"❌ An item in {input_path} directory is not a subdirectory.")
    subdirs = items_list
    i: int = 0
    if len(subdirs) != 8:
        for subdir in subdirs:
            if subdir.name == ".DS_Store":
                i += 1
                break
        if i == len(subdirs):
            raise Exception(f"Expected 8 subdirectories in {input_path} but found {len(subdirs)}")

    main_dir = input_path.parent / "augmented_directory"
    main_dir.mkdir(exist_ok=True)
    apple_dir = main_dir / "Apple"
    grape_dir = main_dir / "Grape"
    apple_dir.mkdir(exist_ok=True)
    grape_dir.mkdir(exist_ok=True)

    ignore_list = [".DS_Store", "Apple", "Grape"]
    for subdir in subdirs:
        if subdir.name.startswith("Apple_"):
            for image in subdir.iterdir():
                parse_file(input_path / subdir.name / image)
            destination = apple_dir / subdir.name
            shutil.copytree(subdir, destination, dirs_exist_ok=True)
        elif subdir.name.startswith("Grape_"):
            for image in subdir.iterdir():
                parse_file(input_path / subdir.name / image)
            destination = grape_dir / subdir.name
            shutil.copytree(subdir, destination, dirs_exist_ok=True)
        else:
            if subdir.name not in ignore_list:
                raise Exception(f"Expected 4 Apple_ and 4 Grape_ subdirs in {input_path}, but one name doesnt match.")
    return main_dir

def process_dir(new_path: Path):
    """
        Defines the biggest subdirectory, then balances the others using augmentation to match the number of elements.
    """
    
    print(new_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="The path to the image you are trying to augment, or directory you are trying to balance.")
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

if __name__ == '__main__':
    main()