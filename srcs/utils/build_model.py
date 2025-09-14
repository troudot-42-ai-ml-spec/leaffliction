from tensorflow import keras
from keras import Sequential, layers
from .hyperparams import IMG_HEIGHT, IMG_WIDTH, CLASSES


def build_model() -> Sequential:
    """
    Build the CNN structure here.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            layers.SeparableConv2D(16, (3, 3), activation="relu", padding="same"),
            layers.SeparableConv2D(16, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.SeparableConv2D(32, (3, 3), activation="relu", padding="same"),
            layers.SeparableConv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same"),
            layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(CLASSES, activation="softmax"),
        ]
    )
    return model
