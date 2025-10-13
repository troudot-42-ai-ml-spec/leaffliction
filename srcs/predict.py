import argparse
import numpy as np
import tensorflow as tf
from utils.parsing.model import load_model_from_zip
from utils.hyperparams import IMG_HEIGHT, IMG_WIDTH
from utils.plotting.grid import show_grid
from utils.transforms import transform_one_image


def main() -> None:
    """
    Load a trained model and predict the class of a given image.
    """
    # https://github.com/tensorflow/tensorflow/issues/68593
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path", help="The path to the image you are trying to augment"
    )
    args = parser.parse_args()

    try:
        image = tf.keras.utils.load_img(
            args.image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        image_array = tf.keras.utils.img_to_array(image).astype(np.uint8)

        transformed_images = transform_one_image(
            image_array, ["remove_background", "crop_blur"]
        )
        transformed_image = next(
            (img for img, name in transformed_images if name == "crop_blur"), None
        )

        model = load_model_from_zip()
        prediction = model.predict(
            np.array([transformed_image.astype(np.uint8)]),
        )

        indices = np.argsort(prediction[0])[::-1]
        print("Predictions (sorted by confidence):")
        for i, idx in enumerate(indices, 1):
            print(f"{i}. {model.class_names[idx]}: {prediction[0][idx] * 100:.2f}%")

        show_grid([image_array, transformed_image], ["Original", "Transformed"])

    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    main()
