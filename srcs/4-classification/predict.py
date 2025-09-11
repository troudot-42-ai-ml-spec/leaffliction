import argparse
from keras.saving import load_model
from keras.utils import load_img, img_to_array
import numpy as np

from utils.hyperparams import IMG_HEIGHT, IMG_WIDTH, CLASS_LABELS

MODEL_PATH = "models/leaffliction.keras"


def main(args: argparse.Namespace) -> None:
    """
    Load a trained model and predict the class of a given image.
    """

    model = load_model(MODEL_PATH)
    if model is None:
        raise ValueError(f"Failed to load a valid model from {MODEL_PATH}.")

    image = load_img(args.image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)

    indices = np.argsort(prediction[0])[::-1]
    print("Predictions (sorted by confidence):")
    for i, idx in enumerate(indices, 1):

        print(
            f"{i}. {CLASS_LABELS[idx]}: {prediction[0][idx] * 100:.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(e)
