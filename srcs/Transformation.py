import tensorflow as tf
from argparse import Namespace
from pathlib import Path
from utils.transforms import transform_one_image, transform_dataset
from utils.parsing.args import parse_args
from utils.hyperparams import IMG_HEIGHT, IMG_WIDTH
from Augmentation import save_dataset
import shutil

if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    args: Namespace = parse_args()

    if args.mode == "multi":
        try:
            path = args.src
            if path.is_dir():
                # TEMPORARY FIX: Clean up old cache
                cache_dir = Path(".tf-cache/transformation")
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)

                print(f"📁 Processing directory: {path}")
                dataset = tf.keras.utils.image_dataset_from_directory(
                    directory=str(path),
                    labels="inferred",
                    label_mode="int",
                    image_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=None,
                )
                class_names = dataset.class_names  # noqa: F841
                transformed_dataset = transform_dataset(dataset, args.ops)
                transformed_dataset = transformed_dataset.prefetch(tf.data.AUTOTUNE)

                save_dataset(transformed_dataset, args.dst.name, class_names)
        except Exception as e:
            print(f"An error occurred: {e}")
    elif args.mode == "single":
        path = Path(args.path)
        if path.exists():
            transform_one_image(path, args.ops, args.show)
        else:
            raise Exception("Image path have to exists.")
    else:
        raise Exception("Invalid source type.")
