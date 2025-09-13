from .hyperparams import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
import tensorflow as tf
from pathlib import Path


def load_datasets(train_set: Path, test_set: Path):
    """
        Takes the train set to split it into training and validation sets.
        Turns train, validation and test sets as tf.data.Dataset.
    """
    print("⏳ Loading dataset for training...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_set,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='training',
        seed=123
    )
    print("⏳ Loading dataset for validation...")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        train_set,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=123
    )
    print("⏳ Loading dataset for testing...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_set,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    print("✅ Data loading complete.")
    print("⏳ Caching datasets for performance...")
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print("✅ Done.\n")

    return train_dataset, validation_dataset, test_dataset
