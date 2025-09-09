# from utils import split_data
from utils import check_dir
# from pathlib import Path
import argparse
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from keras.optimizers import Adam


# Hyperparameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
CLASSES = 8
EPOCHS = 20


# def build_model(input_path: str):

#     model = keras.Sequential(
#         [
#             layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
#             layers.SeparableConv2D(24, 3, activation='relu'),
#             layers.MaxPooling2D()
#         ]
#     )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    args = parser.parse_args()
    try:
        check_dir(args.input_path)
        # build_model(args.input_path)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
