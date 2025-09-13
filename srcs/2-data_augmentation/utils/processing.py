import cv2
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path
from .augmentations import get_augmentations, augment_one_image


def process_file(input_path: Path):
    """
    Augment image using different Albumentations methods, then display it.
    """
    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise Exception(f"Could not read image file: {input_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    augs = get_augmentations()
    augmented_images = {}
    for name, aug in augs.items():
        transform = A.Compose([aug])
        augmented_images[name] = transform(image=image_rgb)["image"]
        augmented_bgr = cv2.cvtColor(augmented_images[name], cv2.COLOR_RGB2BGR)
        path_without_extension = input_path.parent / input_path.stem
        cv2.imwrite(
            f"{path_without_extension}_{name}{input_path.suffix}", augmented_bgr
        )

    print("✅ Saved all augmented images.")

    fig, axes = plt.subplots(1, len(augmented_images) + 1, figsize=(12, 3))
    fig.suptitle(f"Augmentations for {input_path.name}")
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    img_list = list(augmented_images.items())
    for i in range(len(img_list)):
        axes[i + 1].imshow(img_list[i][1])
        axes[i + 1].set_title(img_list[i][0])
        axes[i + 1].axis("off")

    plt.show()


def process_dir(new_path: Path):
    """
    Defines the biggest subdirectory, then balances the others using\
         augmentation to match the number of elements.
    """
    folder_info = {}
    for root, dirs, files in new_path.walk():
        if not dirs:
            current_path = Path(root)
            image_paths = [current_path / f for f in files if f != ".DS_Store"]
            folder_info[current_path] = image_paths
    if not folder_info:
        raise Exception(f"❌ No images found in {new_path}")

    max_count = 0
    for images in folder_info.values():
        if len(images) > max_count:
            max_count = len(images)

    aug = list(get_augmentations().items())

    for current_path, image_paths in folder_info.items():

        num_to_generate = max_count - len(image_paths)
        if num_to_generate <= 0:
            continue

        print(
            f"⏳ Augmenting '{current_path.name}':\n\
Generating {num_to_generate} new images..."
        )

        for i in range(num_to_generate):
            augment_one_image(current_path, image_paths, aug)
    print("✅ Dataset augmentation is complete.")
