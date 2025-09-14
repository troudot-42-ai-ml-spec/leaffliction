from typing import List, Tuple
import math
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.augmentation import augment_image


def display_augmented_image(
    image: np.ndarray, augmented_images: List[Tuple[str, np.ndarray]]
) -> None:
    """
    Display original image alongside all augmented versions.

    Args:
        image: Original image
        augmented_images: List of (name, augmented_image) tuples
    """

    IMAGE_FIGSIZE = (3, 3)
    IMAGE_COLS = 4

    total_images = len(augmented_images) + 1
    rows = math.ceil(total_images / IMAGE_COLS)

    _, axes = plt.subplots(
        rows,
        IMAGE_COLS,
        figsize=(IMAGE_FIGSIZE[0] * IMAGE_COLS, IMAGE_FIGSIZE[1] * rows),
    )
    axes = axes.flatten() if rows > 1 else [axes] if IMAGE_COLS == 1 else axes

    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for i, (name, aug_image) in enumerate(augmented_images, 1):
        if i < len(axes):
            axes[i].imshow(aug_image.astype(np.uint8))
            axes[i].set_title(f"{name}", fontsize=12)
            axes[i].axis("off")

    for i in range(total_images, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Image Augmentation Results", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function to parse arguments and display augmented images.
    """
    # https://github.com/tensorflow/tensorflow/issues/68593
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path", help="The path to the image you are trying to augment"
    )
    args = parser.parse_args()

    try:
        image = tf.keras.utils.load_img(args.image_path, target_size=(256, 256))
        image_array = tf.keras.utils.img_to_array(image).astype(np.uint8)

        augmented_images = augment_image(image_array)
        display_augmented_image(image_array, augmented_images)
    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    main()
