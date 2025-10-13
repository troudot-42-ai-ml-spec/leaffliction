import tensorflow as tf
from argparse import Namespace
from utils.transforms import transform_one_image, transform_dataset
from utils.parsing.args import parse_args
from utils.hyperparams import IMG_HEIGHT, IMG_WIDTH
from utils.cache import tf_cache
from Augmentation import save_dataset
from utils.plotting.grid import show_grid
import numpy as np


def main() -> None:
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    args: Namespace = parse_args()

    try:
        if args.mode == "multi":
            path = args.src
            if path.is_dir():
                print(f"📁 Processing directory: {path}")
                dataset = tf.keras.utils.image_dataset_from_directory(
                    directory=str(path),
                    labels="inferred",
                    label_mode="int",
                    image_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=None,
                )
                class_names = dataset.class_names

                with tf_cache() as cache_dirs:
                    transformed_dataset = transform_dataset(
                        dataset,
                        args.ops,
                        cache_dirs.transformation,
                    )
                    transformed_dataset = transformed_dataset.prefetch(tf.data.AUTOTUNE)
                    save_dataset(transformed_dataset, args.dst.name, class_names)

                return

        image = tf.keras.utils.load_img(
            str(args.path),
            target_size=(IMG_HEIGHT, IMG_WIDTH),
        )
        image_array = tf.keras.utils.img_to_array(image).astype(np.uint8)

        images = transform_one_image(image_array, args.ops)
        show_grid([img for img, _ in images], [name for _, name in images])

    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    main()
