import math
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Optional

Array = np.ndarray


def show_grid(
    images: List[Array],
    titles: Optional[List[str]] = None,
    max_cols: int = 3,
    dpi: int = 120,
):
    """
    Display a list of images in a grid using matplotlib.

    Args:
        images (List[np.ndarray]): List of images (each as a numpy array).
        titles (List[str], optional): Optional list of titles, one per image.
        max_cols (int, default=5): Maximum number of columns in the grid.
        dpi (int, default=120): Resolution (dots per inch) of the figure.
    """
    n = len(images)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(2 * cols, 2 * rows), dpi=dpi)

    for i, img in enumerate(images, 1):
        plt.subplot(rows, cols, i)
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            to_show = img if img.ndim == 2 else img[..., 0]
            plt.imshow(to_show, cmap="gray")
        else:
            plt.imshow(img)
        if titles and i - 1 < len(titles):
            plt.title(titles[i - 1], fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
