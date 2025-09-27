from typing import List
import tensorflow as tf
import zipfile
import io
from PIL import Image
import numpy as np
import json
import os


ZIP_PATH = "models/model.zip"


def _save_dataset_to_zip(
    zipf: zipfile.ZipFile, dataset: tf.data.Dataset, name: str, class_names: List[str]
) -> None:
    """
    Save a TensorFlow dataset to a zip file in image format without temp files.

    Args:
        zipf: ZipFile object to write the dataset files to
        dataset: TensorFlow dataset
    """
    dataset = dataset.unbatch()
    class_counters = {class_name: 0 for class_name in class_names}

    for image, label in dataset.as_numpy_iterator():
        image = image.astype(np.uint8)

        class_name = class_names[label]
        filename = f"{class_name}_{class_counters[class_name]:05d}.jpg"

        img_buffer = io.BytesIO()
        Image.fromarray(image).save(img_buffer, format="JPEG")

        zipf.writestr(f"data/{name}/{class_name}/{filename}", img_buffer.getvalue())
        class_counters[class_name] += 1


def _save_model_to_zip(zipf: zipfile.ZipFile, model: tf.keras.Model) -> None:
    """
    Save a TensorFlow model to a zip file without temp files using h5py.

    Args:
        zipf: ZipFile object to write the model files to
        model: TensorFlow model
    """
    model_json = model.to_json()
    zipf.writestr("model/architecture.json", model_json)

    weights_buffer = io.BytesIO()
    weights = model.get_weights()

    weights_dict = {}
    for i, weight_array in enumerate(weights):
        weights_dict[f"weight_{i}"] = weight_array

    # np requires named arrays
    np.savez_compressed(weights_buffer, **weights_dict)
    zipf.writestr("model/weights.npz", weights_buffer.getvalue())

    summary_buffer = io.StringIO()
    model.summary(print_fn=lambda x: summary_buffer.write(x + "\n"))
    zipf.writestr("model/summary.txt", summary_buffer.getvalue())

    class_names = json.dumps(model.class_names)
    zipf.writestr("model/classes.json", class_names)


def save_to_zip(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
) -> None:
    """
    Save a TensorFlow model and multiple datasets to a zip file.

    Args:
        model: The TensorFlow model to save.
        train_dataset: The training dataset to save.
        validation_dataset: The validation dataset to save.
        test_dataset: The test dataset to save.
    """
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "w") as zipf:
        _save_model_to_zip(zipf, model)

        _save_dataset_to_zip(zipf, train_dataset, "train", model.class_names)
        _save_dataset_to_zip(zipf, validation_dataset, "validation", model.class_names)
        _save_dataset_to_zip(zipf, test_dataset, "test", model.class_names)


def load_model_from_zip() -> tf.keras.Model:
    """
    Load a TensorFlow model from a zip file.
    """

    with zipfile.ZipFile(ZIP_PATH) as zf:
        model_architecture = zf.read("model/architecture.json").decode("utf-8")
        model = tf.keras.models.model_from_json(model_architecture)

        weights_buffer = io.BytesIO(zf.read("model/weights.npz"))
        weights_dict = np.load(weights_buffer)

        weights = []
        for i in range(len(weights_dict.files)):
            weights.append(weights_dict[f"weight_{i}"])

        model.set_weights(weights)

        class_names_json = zf.read("model/classes.json").decode("utf-8")
        model.class_names = json.loads(class_names_json)

        return model
