import albumentations as A
from pathlib import Path
import cv2
import random


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


def augment_one_image(current_path: Path, image_paths: Path, aug: dict):
    """
    Augments one single random image in the folder passed as argument
    """
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
