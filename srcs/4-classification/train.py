# from utils import split_data
# from pathlib import Path
import argparse
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from keras.optimizers import Adam


# Hyperparameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_DEPTH = 3
NUM_CLASSES = 8
EPOCHS = 20


# def build_model(input_path: str):
#     model = keras.Sequential(
#         [
#             layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)),
#             layers.Conv2D(),
#             layers.MaxPool2D()
#         ]
#     )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    # args = parser.parse_args()
    # build_model(args.input_path)


if __name__ == "__main__":
    main()
