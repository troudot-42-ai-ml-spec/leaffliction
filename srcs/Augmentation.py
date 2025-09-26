from typing import List, Tuple, Optional
import math
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.augmentation import augment_image
from utils.hyperparams import IMG_HEIGHT, IMG_WIDTH
import traceback
from pathlib import Path


def display_augmented_image(
    images: List[Tuple[np.ndarray, Optional[str]]]
) -> None:
    """
    Display original image alongside all augmented versions.

    Args:
        image: Original image
        augmented_images: List of (name, augmented_image) tuples
    """

    IMAGE_FIGSIZE = (3, 3)
    IMAGE_COLS = 4

    total_images = len(images)
    rows = math.ceil(total_images / IMAGE_COLS)

    _, axes = plt.subplots(
        rows,
        IMAGE_COLS,
        figsize=(IMAGE_FIGSIZE[0] * IMAGE_COLS, IMAGE_FIGSIZE[1] * rows),
    )
    axes = axes.flatten() if rows > 1 else [axes] if IMAGE_COLS == 1 else axes

    images[0] = (images[0][0], "Original Image")
    for i, (image, augmentation_name) in enumerate(images):
        if i < len(axes):
            axes[i].imshow(image.astype(np.uint8))
            axes[i].set_title(f"{augmentation_name}", fontsize=12)
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
        path = Path(args.image_path)
        if not path.exists():
            raise FileNotFoundError("The specified path does not exist")

        if path.is_dir():
            print(f"📁 Processing directory: {path}")
            dataset = tf.keras.utils.image_dataset_from_directory(
                directory=str(path),
                labels="inferred",
                label_mode="int",
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=None,
            )
            class_names = dataset.class_names  # noqa: F841

            # TODO: Augment the dataset and save it to `augmented_directory`
            return

        print(f"🖼️  Processing single image: {path}")
        image = tf.keras.utils.load_img(
            str(path),
            target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        image_array = tf.keras.utils.img_to_array(image).astype(np.uint8)

        images = augment_image(image_array)
        display_augmented_image(images)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
