from tensorflow import keras
from keras import Sequential, layers
from .hyperparams import IMG_HEIGHT, IMG_WIDTH, CLASSES


def build_model() -> Sequential:
    model = keras.Sequential(
        [
            keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            layers.SeparableConv2D(24, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.SeparableConv2D(24, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(CLASSES, activation='softmax')
        ]
    )
    return model
