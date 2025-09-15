from typing import List, Dict, Tuple, Generator
import tensorflow as tf
import albumentations as A
import cv2
import numpy as np
import random
from .hyperparams import IMG_HEIGHT, IMG_WIDTH


def augmentations() -> Dict[str, A.BasicTransform]:
    """
    Returns a dictionary of specific, named augmentations.
    """
    return {
        "Flip": A.HorizontalFlip(p=0.5),
        "Rotate": A.Affine(
            rotate=(-30, 30),
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        "Skew": A.Affine(
            scale=(0.92, 1.08),
            rotate=(-10, 10),
            translate_percent={"x": (-0.1, 0.1)},
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        "Shear": A.Affine(
            shear={"x": (-20, 20)},
            p=0.4,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        "Crop": A.Compose(
            [
                A.CenterCrop(height=115, width=115, p=1.0),
                # To maintain consistent image size after cropping
                A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),
            ],
            p=0.5
        ),
        "Distortion": A.Perspective(
            scale=(0.05, 0.1),
            p=0.3,
            border_mode=cv2.BORDER_REFLECT_101
        ),
    }


def augment_image(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Apply all available augmentations to a single image and return the results.

    Args:
        image: Input image as numpy array

    Returns:
        List of augmented images
    """

    augmentation_dict = augmentations()
    augmented_images = []

    for name, augmentation in augmentation_dict.items():
        transform = A.Compose([augmentation])
        augmented = transform(image=image)["image"]
        augmented_images.append((name, augmented))

    return augmented_images


def create_augmented_generator(
    dataset: tf.data.Dataset,
) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    Generator that yields augmented images to balance the dataset.

    Args:
        dataset: Original tf.data.Dataset

    Returns:
        Generator yielding tuples of (image, label)
    """

    # Count the number of samples per class
    class_counts: Dict[int, int] = {}
    for _, label in dataset.as_numpy_iterator():
        class_counts[label] = class_counts.get(label, 0) + 1

    max_count = max(class_counts.values())
    augmentation_list = list(augmentations().items())

    # Yield original samples
    for image, label in dataset.as_numpy_iterator():
        yield image, label

    # Augment each class to match the max count
    for class_label, count in class_counts.items():
        class_name = dataset.class_names[class_label]
        num_to_generate = max_count - count
        if num_to_generate <= 0:
            print(f"✅ Class {class_name:<15} is already balanced.")
            continue

        print(
            f"⏳ Augmenting class {class_name:<15} - "
            f"Generating {num_to_generate:<4} new images..."
        )

        class_dataset = dataset.filter(lambda _, label: label == class_label)
        class_images = list(class_dataset.as_numpy_iterator())

        # Randomly apply augmentations to generate new images
        for _ in range(num_to_generate):
            source_image, _ = random.choice(class_images)
            _, augmentation = random.choice(augmentation_list)

            transform = A.Compose([augmentation])
            augmented = transform(image=source_image)["image"]

            yield augmented, class_label


def augment_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Augments the dataset to balance class distributions.

    Args:
        dataset: Original tf.data.Dataset

    Returns:
        Augmented tf.data.Dataset
    """

    augmented_dataset = tf.data.Dataset.from_generator(
        lambda: create_augmented_generator(dataset),
        output_signature=dataset.element_spec
    )
    augmented_dataset.class_names = dataset.class_names
    return augmented_dataset
