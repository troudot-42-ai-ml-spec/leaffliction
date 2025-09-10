from .hyperparams import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
import tensorflow as tf
from pathlib import Path


def load_datasets(train_set: Path, test_set: Path):
    # --- Setup training and validation sets ---
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
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_set,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --- Setup caching for performance ---
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset
