from typing import List, Dict, Tuple, Generator
import tensorflow as tf
import albumentations as A
import cv2
import numpy as np
import random


def augmentations() -> Dict[str, A.BasicTransform]:
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
        "Crop": A.Compose(
            [
                A.CenterCrop(height=200, width=200, p=1.0),
                A.Resize(height=256, width=256, p=1.0),
            ]
        ),  # To maintain consistent image size
        "Distortion": A.Perspective(scale=(0.05, 0.1), p=1.0),
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
        num_to_generate = max_count - count
        if num_to_generate <= 0:
            continue

        class_name = dataset.class_names[class_label]
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

    def generator():
        return create_augmented_generator(dataset)

    augmented_dataset = tf.data.Dataset.from_generator(
        generator, output_signature=dataset.element_spec
    )
    augmented_dataset.class_names = dataset.class_names

    print("✅ Dataset augmentation setup complete.")
    return augmented_dataset


def main():
    print("⏳ Loading original dataset...")
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory="./data",
        labels="inferred",
        label_mode="int",
        image_size=(256, 256),
        batch_size=None,
    )
    print("✅ Original dataset loaded.")

    augmented_dataset = augment_dataset(dataset)

    print("⏳ Saving augmented dataset...")

    tf.data.Dataset.save(augmented_dataset, "./augmented_dataset")

    print("✅ Augmented dataset saved.")


if __name__ == "__main__":
    main()
