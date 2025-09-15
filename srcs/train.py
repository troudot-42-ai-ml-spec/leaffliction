import argparse
import tensorflow as tf
from utils.augmentation import augment_dataset
from utils.hyperparams import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
from utils.build_model import build_model
from utils.train_model import train_model


def main() -> None:
    """
    Main function to load dataset, augment it, build and train the model.
    """
    # https://github.com/tensorflow/tensorflow/issues/68593
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path", help="The path to the dataset directory."
    )
    args = parser.parse_args()

    try:
        print("⏳ Loading original dataset...")
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=args.dataset_path,
            labels="inferred",
            label_mode="int",
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=None,
        )
        print("✅ Original dataset loaded.")
        class_names = dataset.class_names

        # Split into train (70%), val (20%), test (10%)
        print("⏳ Splitting dataset...")
        train_dataset, remaining_dataset = tf.keras.utils.split_dataset(
            dataset, left_size=0.7, shuffle=True, seed=42
        )
        validation_dataset, test_dataset = tf.keras.utils.split_dataset(
            remaining_dataset, left_size=0.67, shuffle=True, seed=42
        )

        train_dataset.class_names = class_names
        validation_dataset.class_names = class_names
        test_dataset.class_names = class_names

        train_dataset = augment_dataset(train_dataset)
        train_dataset = train_dataset.shuffle(
            10_000, reshuffle_each_iteration=True
        )

        train_dataset = train_dataset.batch(BATCH_SIZE)     \
            .prefetch(tf.data.AUTOTUNE)

        os.makedirs(".tf-cache/validation", exist_ok=True)
        validation_dataset = validation_dataset             \
            .cache(filename=".tf-cache/validation/")        \
            .batch(BATCH_SIZE)                              \
            .prefetch(tf.data.AUTOTUNE)

        os.makedirs(".tf-cache/test", exist_ok=True)
        test_dataset = test_dataset                         \
            .cache(filename=".tf-cache/test/")              \
            .batch(BATCH_SIZE)                              \
            .prefetch(tf.data.AUTOTUNE)

        # Force evaluation to fill the cache
        for _ in validation_dataset.as_numpy_iterator():
            pass
        for _ in test_dataset.as_numpy_iterator():
            pass

        model = build_model()
        train_model(
            model,
            train_dataset,
            validation_dataset,
            test_dataset,
        )

        # TODO: Save the model in a zip with the dataset
        os.makedirs("models", exist_ok=True)
        model.save("models/leaffliction.keras", overwrite=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    main()
